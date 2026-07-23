import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from model.loss import LapLoss
from config import *


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if 'attn_mask' not in k and 'HW' not in k
    }


class Model:
    def __init__(self, local_rank=0, loss_type='lap', flow_loss_weight=0.01,
                 flow_stage_gamma=0.8, flow_motion_threshold=1.0,
                 flow_motion_balance=0.5, flow_motion_gain=1.0,
                 flow_motion_scale=10.0, flow_motion_weight_cap=4.0,
                 flow_charbonnier_eps=1e-3):
        """
        loss_type        : 'l1' | 'lap' | 'l1+lap'
        flow_loss_weight : teacher flow 监督的总权重, 0=关闭
        flow_stage_gamma : 多阶段监督的递增系数; 越后级权重越高
        """
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.local = LOCAL

        self.device()

        assert loss_type in ('l1', 'lap', 'l1+lap'), \
            f"loss_type 须为 'l1'/'lap'/'l1+lap'，got {loss_type}"
        self.loss_type = loss_type
        self.lap_loss = (
            LapLoss(max_levels=5, channels=3).to(self._dev)
            if loss_type in ('lap', 'l1+lap') else None
        )
        self.flow_loss_weight = flow_loss_weight
        self.flow_stage_gamma = float(flow_stage_gamma)
        self.flow_loss_kwargs = {
            'motion_threshold': float(flow_motion_threshold),
            'motion_balance': min(max(float(flow_motion_balance), 0.0), 1.0),
            'motion_gain': float(flow_motion_gain),
            'motion_scale': float(flow_motion_scale),
            'motion_weight_cap': float(flow_motion_weight_cap),
            'charbonnier_eps': float(flow_charbonnier_eps),
        }
        self.last_flow_stage_losses = None

        self.optimG = optim.AdamW(self.net.parameters(), lr=1e-6, weight_decay=1e-4)

    def train(self): self.net.train()
    def eval(self):  self.net.eval()

    def device(self):
        if torch.cuda.is_available():
            self._dev = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self._dev = torch.device('mps')
        else:
            self._dev = torch.device('cpu')
        self.net.to(self._dev)

    # ── pad / unpad ───────────────────────────────────────────────────────────

    @staticmethod
    def pad_to_multiple(tensor, multiple=16):
        _, _, h, w = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return tensor, (0, 0)
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect'), (pad_w, pad_h)

    @staticmethod
    def unpad(tensor, pad_right, pad_bottom):
        h, w = tensor.shape[-2], tensor.shape[-1]
        return tensor[...,
                      :h - pad_bottom if pad_bottom else h,
                      :w - pad_right if pad_right else w]

    # ── checkpoint ────────────────────────────────────────────────────────────

    # def load_model(self, ckpt_path, rank=0, real=False):
    #     self.net.load_state_dict(
    #         convert(torch.load(ckpt_path, map_location='cpu')), strict=False
    #     )

    def load_model(self, ckpt_path, resume=False):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt['net'] if 'net' in ckpt else ckpt          # 兼容新旧格式
        state = convert(state)

        # 过滤形状不匹配的键 (strict=False不处理size mismatch, 必须手动剔除)
        model_state = self.net.state_dict()
        dropped = []
        filtered = {}
        for k, v in state.items():
            if k in model_state and model_state[k].shape != v.shape:
                dropped.append(f'{k}: ckpt{tuple(v.shape)} vs model{tuple(model_state[k].shape)}')
            else:
                filtered[k] = v
        if dropped:
            print(f'[load_model] {len(dropped)} 个键形状不匹配, 保持新初始化:')
            for d in dropped:
                print(f'    {d}')

        missing, unexpected = self.net.load_state_dict(filtered, strict=False)
        if missing:
            print(f'[load_model] WARNING: {len(missing)} 个模型键未从checkpoint加载'
                  f'(保持新初始化): {missing[:8]}'
                  f'{"..." if len(missing) > 8 else ""}')
        if unexpected:
            print(f'[load_model] WARNING: {len(unexpected)} 个checkpoint键未被当前模型使用: '
                  f'{unexpected[:8]}{"..." if len(unexpected) > 8 else ""}')

        # version=2 的 refine UNet 比 version=1 多 bottleneck attention。
        # 老checkpoint没有结构元数据，只能通过这些特征键提示版本切换。
        attention_prefix = 'unet.bottleneck_attn.'
        missing_attention = [k for k in missing if k.startswith(attention_prefix)]
        unused_attention = [k for k in unexpected if k.startswith(attention_prefix)]
        if missing_attention:
            print('[load_model] WARNING: 当前模型启用了version=2 attention，'
                  '但checkpoint不含对应权重；该模块将从随机初始化开始训练')
        elif unused_attention:
            print('[load_model] WARNING: checkpoint含version=2 attention权重，'
                  '但当前模型为version=1；这些attention权重已忽略')

        # finetune安全检查: 明确告知原有最后一级(全分辨率)IFBlock是否成功复用。
        fullres_index = len(self.net.local_block) - 1
        fullres_prefix = f'local_block.{fullres_index}.'
        fullres_loaded = sum(k.startswith(fullres_prefix) for k in filtered)
        if fullres_loaded:
            print(f'[load_model] 全分辨率IFBlock[{fullres_index}] '
                  f'已复用checkpoint权重 ({fullres_loaded} tensors)')
        else:
            print(f'[load_model] WARNING: checkpoint中未找到全分辨率'
                  f'IFBlock[{fullres_index}], 该分支将使用零残差初始化')

        if resume and 'optim' in ckpt:
            self.optimG.load_state_dict(ckpt['optim'])
        return ckpt.get('epoch', 0), ckpt.get('step', 0)

    def save_model(self, sp, epoch, rank=0):
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'ckpt/{sp}/{self.name}_{str(epoch)}.pkl'
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(self.net.state_dict(), ckpt_path)
        print(f"model saved -> {ckpt_path}")

    # ── loss ──────────────────────────────────────────────────────────────────

    def _pixel_loss(self, pred, gt):
        if self.loss_type == 'l1':
            return (pred - gt).abs().mean()
        if self.loss_type == 'lap':
            return self.lap_loss(pred, gt)
        return (pred - gt).abs().mean() + self.lap_loss(pred, gt)   # l1+lap

    @staticmethod
    def _flow_region_weights(flow_gt, has_mv, motion_threshold=1.0,
                             motion_gain=1.0, motion_scale=10.0,
                             motion_weight_cap=4.0):
        """预计算与预测无关的前景/背景权重, 供所有flow阶段复用。"""
        valid = flow_gt[:, 4].clamp(0, 1)                                # [B,H,W]
        global_flow = torch.zeros(
            flow_gt.shape[0], 4, 1, 1, dtype=flow_gt.dtype, device=flow_gt.device)
        for b in range(flow_gt.shape[0]):
            sparse_valid = valid[b, ::8, ::8] > 0
            if sparse_valid.any():
                sparse_flow = flow_gt[b, :4, ::8, ::8]
                global_flow[b, :, 0, 0] = sparse_flow[:, sparse_valid].median(dim=1).values
        relative_mag = 0.5 * (
            torch.linalg.vector_norm(flow_gt[:, 0:2] - global_flow[:, 0:2], dim=1)
            + torch.linalg.vector_norm(flow_gt[:, 2:4] - global_flow[:, 2:4], dim=1))
        moving = valid * (relative_mag >= motion_threshold).to(valid.dtype)
        static = valid * (relative_mag < motion_threshold).to(valid.dtype)

        scale = max(float(motion_scale), 1e-6)
        motion_factor = torch.clamp(
            1.0 + motion_gain * relative_mag / scale,
            max=max(float(motion_weight_cap), 1.0))
        motion_weight = moving * motion_factor
        sample_valid = (has_mv > 0) & (valid.sum(dim=(1, 2)) > 0)
        return motion_weight, static, sample_valid

    @staticmethod
    def _flow_epe_masked(flow_pred, flow_gt, has_mv,
                         motion_threshold=1.0, motion_balance=0.5,
                         motion_gain=1.0, motion_scale=10.0,
                         motion_weight_cap=4.0, charbonnier_eps=1e-3,
                         _regions=None):
        """
        flow_pred : [B, 4, H, W] 像素位移 (前2通道 F_t→0, 后2通道 F_t→1)
        flow_gt   : [B, 5, H, W] 前4通道同上, 第5通道为遮挡有效性mask
                    (valid=0 的uncover区无合法对应点, 监督病态, 逐像素跳过)
        has_mv    : [B] 1=该样本flow_gt有效
        移动区与静态区分别归一化, 防止小运动物体被背景像素数量淹没。
        移动区内再按GT位移大小软加权。Charbonnier EPE不做硬clamp,
        因此大误差像素仍然有梯度。
        """
        diff0 = flow_pred[:, 0:2] - flow_gt[:, 0:2]
        diff1 = flow_pred[:, 2:4] - flow_gt[:, 2:4]
        eps2 = charbonnier_eps ** 2
        epe0 = torch.sqrt(diff0.square().sum(dim=1) + eps2)
        epe1 = torch.sqrt(diff1.square().sum(dim=1) + eps2)
        epe = (epe0 + epe1) * 0.5                                       # [B,H,W]

        if _regions is None:
            _regions = Model._flow_region_weights(
                flow_gt, has_mv, motion_threshold=motion_threshold,
                motion_gain=motion_gain, motion_scale=motion_scale,
                motion_weight_cap=motion_weight_cap)
        motion_weight, static, sample_valid = _regions

        def region_mean(region_weight):
            denom = region_weight.sum(dim=(1, 2))
            mean = (epe * region_weight).sum(dim=(1, 2)) / denom.clamp(min=1e-6)
            return mean, denom > 0

        moving_mean, has_moving = region_mean(motion_weight)
        static_mean, has_static = region_mean(static)
        both = has_moving & has_static
        per_sample = torch.where(
            both,
            motion_balance * moving_mean + (1.0 - motion_balance) * static_mean,
            torch.where(has_moving, moving_mean, static_mean))

        n_valid = sample_valid.sum()
        if n_valid < 1:
            return flow_pred.sum() * 0.0
        return (per_sample * sample_valid.to(per_sample.dtype)).sum() / n_valid

    def _multistage_flow_loss(self, flow_list, flow_gt, has_mv, pr, pb):
        """对所有 learned-feature/local flow 阶段做深监督。

        gamma 权重按阶段递增并归一化, 使 flow_loss_weight 的整体量级
        不因阶段数改变。
        """
        n = len(flow_list)
        if n == 0:
            self.last_flow_stage_losses = None
            return flow_gt.sum() * 0.0
        gamma = min(max(self.flow_stage_gamma, 1e-6), 1.0)
        weights = flow_gt.new_tensor([gamma ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        regions = self._flow_region_weights(
            flow_gt, has_mv,
            motion_threshold=self.flow_loss_kwargs['motion_threshold'],
            motion_gain=self.flow_loss_kwargs['motion_gain'],
            motion_scale=self.flow_loss_kwargs['motion_scale'],
            motion_weight_cap=self.flow_loss_kwargs['motion_weight_cap'])
        stage_losses = []
        for flow_pred in flow_list:
            flow_pred = self.unpad(flow_pred, pr, pb)
            stage_losses.append(self._flow_epe_masked(
                flow_pred, flow_gt, has_mv, _regions=regions,
                **self.flow_loss_kwargs))
        stage_losses = torch.stack(stage_losses)
        self.last_flow_stage_losses = stage_losses.detach()
        return (stage_losses * weights).sum()

    # ── train / eval step ─────────────────────────────────────────────────────

    def update(self, imgs, gt, timestep=0.5, learning_rate=0, training=True,
               scaler=None, flow_gt=None, has_mv=None):
        """
        imgs     : [B, 6, H, W] float 0~1  (img0, img1)
        gt       : [B, 3, H, W] float 0~1
        timestep : float 或 [B,1,1,1] 张量 (来自Dataset, 任意timestep训练必须传)
        flow_gt  : [B, 5, H, W] 前4通道为像素位移, 第5通道valid mask
        has_mv   : [B] flow_gt有效性mask, 可为 None
        """
        if torch.is_tensor(timestep):
            timestep = timestep.to(self._dev)

        imgs_pad, (pr, pb) = self.pad_to_multiple(imgs, 16)

        if training:
            self.train()
            for pg in self.optimG.param_groups:
                pg['lr'] = learning_rate

            flow_list, _, _, _, _, merged, pred_pad = self.net(
                imgs_pad, timestep=timestep, scale=0, local=self.local)

            pred = self.unpad(pred_pad, pr, pb)
            loss = self._pixel_loss(pred, gt)
            for merge in merged:
                loss = loss + self._pixel_loss(self.unpad(merge, pr, pb), gt)

            # teacher flow 多阶段监督 (仅对 has_mv=1 的样本生效)
            loss_flow = torch.zeros((), device=self._dev)
            self.last_flow_stage_losses = None
            if (self.flow_loss_weight > 0 and flow_gt is not None
                    and has_mv is not None and has_mv.sum() > 0):
                flow_gt = flow_gt.to(self._dev)
                has_mv = has_mv.to(self._dev)
                loss_flow = self._multistage_flow_loss(
                    flow_list, flow_gt, has_mv, pr, pb)
                loss = loss + self.flow_loss_weight * loss_flow

            self.optimG.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimG)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                scaler.step(self.optimG)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimG.step()

            return pred.detach(), loss.item(), loss_flow.item()

        else:
            self.eval()
            with torch.no_grad():
                _, _, _, _, _, _, pred_pad = self.net(
                    imgs_pad, timestep=timestep, scale=0, local=self.local)
                pred = self.unpad(pred_pad, pr, pb)
                loss = self._pixel_loss(pred, gt)
            return pred, loss.item(), 0.0

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(self, img0, img1, local, TTA=False, timestep=0.5,
                  scale=0, fast_TTA=False):
        imgs = torch.cat((img0, img1), 1)
        imgs, (pr, pb) = self.pad_to_multiple(imgs, 16)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            flow_list, mask_list, res, warp0, warp1, merged, preds = self.net(
                torch.cat((imgs, imgs_), 0), local=local, timestep=timestep, scale=scale)
            return (self.unpad((preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2., pr, pb),
                    flow_list[-1], mask_list[-1], merged[-1], res, warp0, warp1)
        flow_list, mask_list, res, warp0, warp1, merged, preds = self.net(
            imgs, timestep=timestep, scale=scale, local=local)
        if not TTA:
            return self.unpad(preds, pr, pb), None, None, None, None, None, None
        _, _, _, _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep,
                                           scale=scale, local=local)
        return (self.unpad((preds + pred2.flip(2).flip(3)) / 2, pr, pb),
                flow_list[-1], mask_list[-1], merged[-1], res, warp0, warp1)
