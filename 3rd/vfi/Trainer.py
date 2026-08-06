import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from model.loss import CensusLoss, CharbonnierLoss, LapLoss
from config import *


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if 'attn_mask' not in k and 'HW' not in k
    }


def load_checkpoint_file(path):
    """优先使用受限加载器；仅为旧式含NumPy对象的本地checkpoint兼容回退。"""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # 旧版PyTorch没有weights_only参数
        return torch.load(path, map_location='cpu')
    except Exception as safe_error:
        print('[load_model] WARNING: checkpoint无法由weights_only安全加载器读取；'
              '仅应对可信的本地旧checkpoint继续加载 '
              f'({type(safe_error).__name__})')
        return torch.load(path, map_location='cpu', weights_only=False)


class Model:
    def __init__(self, local_rank=0, loss_type='lap', flow_loss_weight=0.01,
                 flow_stage_gamma=0.2, flow_motion_threshold=1.0,
                 flow_motion_balance=0.1, flow_motion_gain=0.0,
                 flow_motion_scale=10.0, flow_motion_weight_cap=4.0,
                 flow_charbonnier_eps=1e-3, flow_loss_warmup_steps=0,
                 merge_loss_gamma=0.5, merge_loss_weights=None,
                 normalize_pixel_loss=True, residual_loss_weight=0.0,
                 lc_charbonnier_eps=1e-3, lc_census_weight=1.0,
                 lc_lap_weight=1.0, lc_warp_weight=0.5,
                 pervfi_mask_loss_weight=0.0):
        """
        loss_type        : 'l1' | 'lap' | 'l1+lap' | 'lc'
        flow_loss_weight : teacher flow 监督的总权重, 0=关闭
        flow_stage_gamma : 多阶段监督的递增系数; 越后级权重越高
        """
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.local = bool(LOCAL)

        self.device()

        assert loss_type in ('l1', 'lap', 'l1+lap', 'lc'), \
            f"loss_type 须为 'l1'/'lap'/'l1+lap'/'lc'，got {loss_type}"
        self.loss_type = loss_type
        self.lap_loss = (
            LapLoss(max_levels=5, channels=3).to(self._dev)
            if loss_type in ('lap', 'l1+lap', 'lc') else None
        )
        self.charbonnier_loss = (
            CharbonnierLoss(eps=lc_charbonnier_eps).to(self._dev)
            if loss_type == 'lc' else None)
        self.census_loss = (
            CensusLoss(patch_size=7).to(self._dev)
            if loss_type == 'lc' else None)
        self.lc_census_weight = max(float(lc_census_weight), 0.0)
        self.lc_lap_weight = max(float(lc_lap_weight), 0.0)
        self.lc_warp_weight = max(float(lc_warp_weight), 0.0)
        self.pervfi_mask_loss_weight = max(
            float(pervfi_mask_loss_weight), 0.0)
        self.flow_loss_weight = flow_loss_weight
        self.flow_loss_warmup_steps = max(int(flow_loss_warmup_steps), 0)
        self.flow_stage_gamma = float(flow_stage_gamma)
        self.merge_loss_gamma = min(max(float(merge_loss_gamma), 1e-6), 1.0)
        self.merge_loss_weights = (
            [float(value) for value in merge_loss_weights]
            if merge_loss_weights is not None else None)
        self.normalize_pixel_loss = bool(normalize_pixel_loss)
        self.residual_loss_weight = max(float(residual_loss_weight), 0.0)
        self.flow_loss_kwargs = {
            'motion_threshold': float(flow_motion_threshold),
            'motion_balance': min(max(float(flow_motion_balance), 0.0), 1.0),
            'motion_gain': max(float(flow_motion_gain), 0.0),
            'motion_scale': max(float(flow_motion_scale), 1e-6),
            'motion_weight_cap': max(float(flow_motion_weight_cap), 1.0),
            'charbonnier_eps': max(float(flow_charbonnier_eps), 1e-12),
        }
        self.last_flow_stage_losses = None
        self.last_flow_stage_sums = None
        self.last_flow_valid_count = None
        self.last_loss_components = {}
        self.last_image_loss_components = {}
        self.last_grad_norm = None
        self.loaded_checkpoint = None
        self.optimizer_restored = False
        self.grad_clip = 1.0

        self.configure_optimizer({})

    def train(self): self.net.train()
    def eval(self):  self.net.eval()

    def set_local_enabled(self, enabled):
        """Enable/bypass local IFBlocks for the current training phase.

        Bypassed parameters receive no gradient and AdamW therefore leaves
        both their weights and optimizer state untouched.  This permits the
        official-style coarse-to-local schedule without changing checkpoint
        structure or rebuilding the optimizer between phases.
        """
        self.local = bool(enabled)

    @staticmethod
    def _quasi_binary_mask_loss(mask_list, reference):
        """Mean uncertainty m*(1-m); zero for binary masks, max at 0.5."""
        if not mask_list:
            return reference.new_zeros(())
        return torch.stack([
            (mask * (1.0 - mask)).mean() for mask in mask_list
        ]).mean()

    @staticmethod
    def _mask_soft_ratio(mask_list, reference, low=0.1, high=0.9):
        """Fraction of mask pixels that still perform meaningful soft mixing."""
        if not mask_list:
            return reference.new_zeros(())
        return torch.stack([
            ((mask > low) & (mask < high)).to(mask.dtype).mean()
            for mask in mask_list
        ]).mean()

    def device(self):
        if torch.cuda.is_available():
            self._dev = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self._dev = torch.device('mps')
        else:
            self._dev = torch.device('cpu')
        self.net.to(self._dev)

    def configure_optimizer(self, opt):
        """建立可恢复的 AdamW 参数组，并尊重 Mamba 的免衰减标记。"""
        weight_decay = float(opt.get('weight_decay', 1e-4))
        finetune = bool(opt.get('finetune', False))
        scales = {
            'backbone': float(opt.get(
                'backbone_lr_scale', 0.25 if finetune else 1.0)),
            'flow': float(opt.get('flow_lr_scale', 1.0)),
            'refine': float(opt.get('refine_lr_scale', 1.0)),
        }
        self.grad_clip = float(opt.get('grad_clip', 1.0))
        if self.grad_clip <= 0:
            raise ValueError(f'grad_clip must be > 0, got {self.grad_clip}')

        grouped = {}
        for name, parameter in self.net.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('feature_bone.'):
                family = 'backbone'
            elif name.startswith(('block.', 'local_block.')):
                family = 'flow'
            else:
                family = 'refine'
            no_decay = (
                parameter.ndim <= 1
                or name.endswith('.bias')
                or bool(getattr(parameter, '_no_weight_decay', False)))
            key = (family, no_decay)
            grouped.setdefault(key, []).append(parameter)

        param_groups = []
        for (family, no_decay), parameters in grouped.items():
            param_groups.append({
                'params': parameters,
                'lr': 1e-6 * scales[family],
                'lr_scale': scales[family],
                'weight_decay': 0.0 if no_decay else weight_decay,
                'group_name': f'{family}_{"no_decay" if no_decay else "decay"}',
            })
        betas = tuple(float(value) for value in opt.get('betas', (0.9, 0.999)))
        if len(betas) != 2:
            raise ValueError(f'optim.betas must contain 2 values, got {betas}')
        self.optimG = optim.AdamW(
            param_groups, lr=1e-6, betas=betas,
            eps=float(opt.get('eps', 1e-8)))

    def set_learning_rate(self, learning_rate):
        for group in self.optimG.param_groups:
            group['lr'] = learning_rate * float(group.get('lr_scale', 1.0))

    def flow_weight_at_step(self, step):
        if self.flow_loss_warmup_steps <= 0 or step is None:
            return float(self.flow_loss_weight)
        progress = min(max((int(step) + 1) / self.flow_loss_warmup_steps, 0.0), 1.0)
        return float(self.flow_loss_weight) * progress

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
        ckpt = load_checkpoint_file(ckpt_path)
        state = ckpt['net'] if 'net' in ckpt else ckpt          # 兼容新旧格式
        state = convert(state)
        self.loaded_checkpoint = ckpt if 'net' in ckpt else None
        current_version = int(getattr(self.net, 'version', 1))
        checkpoint_version = (
            ckpt.get('model_version') if isinstance(ckpt, dict) else None)
        if (checkpoint_version is not None
                and int(checkpoint_version) != current_version):
            print('[load_model] WARNING: refine版本发生变化: '
                  f'checkpoint version={checkpoint_version}, '
                  f'current version={current_version}')

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

        # 老checkpoint没有结构元数据，通过特征键提示 refine 版本切换。
        v2_attention_prefix = 'unet.bottleneck_attn.'
        v3_attention_prefix = 'unet.bottleneck_attention.'
        missing_v2_attention = [
            k for k in missing if k.startswith(v2_attention_prefix)]
        unused_v2_attention = [
            k for k in unexpected if k.startswith(v2_attention_prefix)]
        missing_v3_attention = [
            k for k in missing if k.startswith(v3_attention_prefix)]
        if current_version == 2 and missing_v2_attention:
            print('[load_model] WARNING: 当前模型启用了version=2 attention，'
                  '但checkpoint不含对应权重；该模块将从随机初始化开始训练')
        elif unused_v2_attention:
            print(f'[load_model] WARNING: checkpoint含version=2 attention权重，'
                  f'但当前模型为version={current_version}；这些权重已忽略')
        if current_version == 3 and missing_v3_attention:
            print('[load_model] INFO: version=3 residual attention本身使用identity初始化；'
                  '但v3同时改变了refiner残差映射和幅度限制，切换版本时loss可出现跳变，'
                  '应使用新的exp_name进行finetune')

        # finetune安全检查: 明确告知原有最后一级(全分辨率)IFBlock是否成功复用。
        fullres_index = len(self.net.local_block) - 1
        fullres_prefix = f'local_block.{fullres_index}.'
        fullres_loaded = sum(
            k.startswith(fullres_prefix) and k in model_state
            for k in filtered)
        if fullres_loaded:
            print(f'[load_model] 全分辨率IFBlock[{fullres_index}] '
                  f'已复用checkpoint权重 ({fullres_loaded} tensors)')
        else:
            print(f'[load_model] WARNING: checkpoint中未找到全分辨率'
                  f'IFBlock[{fullres_index}], 该分支将使用零残差初始化')

        if resume and 'optim' in ckpt:
            try:
                self.optimG.load_state_dict(ckpt['optim'])
                self.optimizer_restored = True
            except ValueError as exc:
                print('[load_model] WARNING: optimizer参数组与当前代码不兼容，'
                      f'仅恢复模型权重并重建optimizer ({exc})')
        return ckpt.get('epoch', 0), ckpt.get('step', 0)

    def save_model(self, sp, epoch, rank=0, step=0, scaler=None,
                   tag=None, extra_state=None):
        filename = f'{self.name}_{tag or str(epoch)}.pkl'
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f'ckpt/{sp}/{filename}')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        payload = {
            'checkpoint_version': 2,
            'model_version': int(getattr(self.net, 'version', 1)),
            'net': self.net.state_dict(),
            'optim': self.optimG.state_dict(),
            'epoch': int(epoch),
            'step': int(step),
        }
        if scaler is not None and scaler.is_enabled():
            payload['scaler'] = scaler.state_dict()
        if extra_state:
            payload.update(extra_state)
        tmp_path = ckpt_path + '.tmp'
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        print(f"model saved -> {ckpt_path}")
        return ckpt_path

    # ── loss ──────────────────────────────────────────────────────────────────

    def _pixel_loss(self, pred, gt):
        if self.loss_type == 'l1':
            return (pred - gt).abs().mean()
        if self.loss_type == 'lap':
            return self.lap_loss(pred, gt)
        if self.loss_type == 'l1+lap':
            return (pred - gt).abs().mean() + self.lap_loss(pred, gt)
        reconstruction, _ = self._lc_reconstruction_loss(pred, gt)
        return reconstruction

    def _lc_reconstruction_loss(self, pred, gt, target_lap_pyramid=None):
        """LC-Mamba final reconstruction: Charbonnier + Census + Lap."""
        charbonnier = self.charbonnier_loss(pred, gt)
        census = self.census_loss(pred, gt)
        lap = (
            self.lap_loss(pred, gt)
            if target_lap_pyramid is None
            else self.lap_loss.forward_with_target_pyramid(
                pred, target_lap_pyramid))
        total = (
            charbonnier
            + self.lc_census_weight * census
            + self.lc_lap_weight * lap)
        return total, {
            'lc_charbonnier': charbonnier.detach(),
            'lc_census': census.detach(),
            'lc_census_weighted': (
                self.lc_census_weight * census).detach(),
            'lc_final_lap': lap.detach(),
            'lc_final_lap_weighted': (
                self.lc_lap_weight * lap).detach(),
        }

    def _pixel_supervision(self, pred, merged, gt):
        """稳定归一化的 final + 多级merge像素监督。"""
        if self.loss_type == 'lc':
            # GT金字塔在final和全部warp stage间复用，避免重复构建。
            target_lap_pyramid = self.lap_loss.pyramid(gt)
            final_loss, components = self._lc_reconstruction_loss(
                pred, gt, target_lap_pyramid=target_lap_pyramid)
            merge_losses = [
                self.lap_loss.forward_with_target_pyramid(
                    stage_merge, target_lap_pyramid)
                for stage_merge in merged]
            weights = final_loss.new_full(
                (len(merge_losses),), self.lc_warp_weight)
            warp_loss = sum(
                (weight * value
                 for weight, value in zip(weights, merge_losses)),
                final_loss.new_zeros(()))
            pixel_loss = final_loss + warp_loss
            components['lc_reconstruction'] = final_loss.detach()
            components['lc_warp'] = warp_loss.detach()
            for index, value in enumerate(merge_losses):
                components[f'lc_warp_stage_{index}'] = value.detach()
            self.last_image_loss_components = components
            return pixel_loss, final_loss, merge_losses, weights

        self.last_image_loss_components = {}
        final_loss = self._pixel_loss(pred, gt)
        merge_losses = [
            self._pixel_loss(merge, gt) for merge in merged]
        if self.merge_loss_weights is not None:
            if len(self.merge_loss_weights) != len(merge_losses):
                raise ValueError(
                    f'merge_loss_weights长度({len(self.merge_loss_weights)}) '
                    f'必须等于local merge数量({len(merge_losses)})')
            weights = final_loss.new_tensor(self.merge_loss_weights)
        else:
            count = len(merge_losses)
            weights = final_loss.new_tensor([
                self.merge_loss_gamma ** (count - 1 - index)
                for index in range(count)])

        weighted_merges = sum(
            (weight * value for weight, value in zip(weights, merge_losses)),
            final_loss.new_zeros(()))
        denominator = (
            1.0 + weights.sum() if self.normalize_pixel_loss else 1.0)
        pixel_loss = (final_loss + weighted_merges) / denominator
        return pixel_loss, final_loss, merge_losses, weights

    @staticmethod
    def _safe_flow(flow_gt):
        finite = torch.isfinite(flow_gt[:, :4]).all(dim=1)
        flow = torch.nan_to_num(
            flow_gt[:, :4], nan=0.0, posinf=0.0, neginf=0.0)
        valid = flow_gt[:, 4].clamp(0, 1)
        valid = valid * finite.to(valid.dtype)
        return flow, valid

    @staticmethod
    def _flow_region_weights(flow_gt, has_mv, motion_threshold=1.0,
                             motion_gain=1.0, motion_scale=10.0,
                             motion_weight_cap=4.0):
        """预计算与预测无关的前景/背景权重, 供所有flow阶段复用。"""
        flow, valid = Model._safe_flow(flow_gt)
        global_flow = torch.zeros(
            flow_gt.shape[0], 4, 1, 1, dtype=flow_gt.dtype, device=flow_gt.device)
        for b in range(flow_gt.shape[0]):
            sparse_valid = valid[b, ::8, ::8] > 0
            if sparse_valid.any():
                sparse_flow = flow[b, :, ::8, ::8]
                global_flow[b, :, 0, 0] = sparse_flow[:, sparse_valid].median(dim=1).values
        relative_mag = 0.5 * (
            torch.linalg.vector_norm(flow[:, 0:2] - global_flow[:, 0:2], dim=1)
            + torch.linalg.vector_norm(flow[:, 2:4] - global_flow[:, 2:4], dim=1))
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
                         motion_threshold=1.0, motion_balance=0.1,
                         motion_gain=0.0, motion_scale=10.0,
                         motion_weight_cap=4.0, charbonnier_eps=1e-3,
                         _regions=None, return_stats=False):
        """
        flow_pred : [B, 4, H, W] 像素位移 (前2通道 F_t→0, 后2通道 F_t→1)
        flow_gt   : [B, 5, H, W] 前4通道同上, 第5通道为有效性/置信度mask
                    (非有限MV、原图/crop终点越界处valid=0)
        has_mv    : [B] 1=该样本flow_gt有效
        移动区与静态区分别归一化, 防止小运动物体被背景像素数量淹没。
        移动区内再按GT位移大小软加权。Charbonnier EPE不做硬clamp,
        因此大误差像素仍然有梯度。
        """
        safe_flow, _ = Model._safe_flow(flow_gt)
        diff0 = flow_pred[:, 0:2] - safe_flow[:, 0:2]
        diff1 = flow_pred[:, 2:4] - safe_flow[:, 2:4]
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

        count = sample_valid.sum().to(per_sample.dtype)
        loss_sum = (per_sample * sample_valid.to(per_sample.dtype)).sum()
        if return_stats:
            return loss_sum, count
        return loss_sum / count.clamp(min=1.0)

    def _multistage_flow_loss_stats(self, flow_list, flow_gt, has_mv, pr, pb):
        """对所有 learned-feature/local flow 阶段做深监督。

        gamma 权重按阶段递增并归一化, 使 flow_loss_weight 的整体量级
        不因阶段数改变。
        """
        n = len(flow_list)
        if n == 0:
            self.last_flow_stage_losses = None
            self.last_flow_stage_sums = None
            self.last_flow_valid_count = flow_gt.new_zeros(())
            zero = flow_gt.sum() * 0.0
            return zero, zero, self.last_flow_valid_count
        gamma = min(max(self.flow_stage_gamma, 1e-6), 1.0)
        weights = flow_gt.new_tensor([gamma ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        regions = self._flow_region_weights(
            flow_gt, has_mv,
            motion_threshold=self.flow_loss_kwargs['motion_threshold'],
            motion_gain=self.flow_loss_kwargs['motion_gain'],
            motion_scale=self.flow_loss_kwargs['motion_scale'],
            motion_weight_cap=self.flow_loss_kwargs['motion_weight_cap'])
        stage_sums = []
        valid_count = flow_gt.new_zeros(())
        for flow_pred in flow_list:
            flow_pred = self.unpad(flow_pred, pr, pb)
            stage_sum, valid_count = self._flow_epe_masked(
                flow_pred, flow_gt, has_mv, _regions=regions,
                return_stats=True, **self.flow_loss_kwargs)
            stage_sums.append(stage_sum)
        stage_sums = torch.stack(stage_sums)
        stage_losses = stage_sums / valid_count.clamp(min=1.0)
        self.last_flow_stage_sums = stage_sums.detach()
        self.last_flow_stage_losses = stage_losses.detach()
        self.last_flow_valid_count = valid_count.detach()
        weighted_sum = (stage_sums * weights).sum()
        weighted_mean = weighted_sum / valid_count.clamp(min=1.0)
        return weighted_mean, weighted_sum, valid_count

    def _multistage_flow_loss(self, flow_list, flow_gt, has_mv, pr, pb):
        mean, _, _ = self._multistage_flow_loss_stats(
            flow_list, flow_gt, has_mv, pr, pb)
        return mean

    def flow_metric_sums(self, flow_pred, flow_gt, has_mv):
        """验证用原始像素EPE统计，不使用motion_balance重加权。"""
        safe_flow, valid = self._safe_flow(flow_gt)
        sample_mask = (has_mv > 0).to(valid.dtype)[:, None, None]
        valid = valid * sample_mask
        diff0 = torch.linalg.vector_norm(
            flow_pred[:, 0:2] - safe_flow[:, 0:2], dim=1)
        diff1 = torch.linalg.vector_norm(
            flow_pred[:, 2:4] - safe_flow[:, 2:4], dim=1)
        epe = 0.5 * (diff0 + diff1)

        motion_weight, static_weight, _ = self._flow_region_weights(
            flow_gt, has_mv,
            motion_threshold=self.flow_loss_kwargs['motion_threshold'],
            motion_gain=0.0, motion_scale=1.0, motion_weight_cap=1.0)
        moving = (motion_weight > 0).to(valid.dtype) * valid
        static = (static_weight > 0).to(valid.dtype) * valid
        return {
            'sum': (epe * valid).sum(),
            'count': valid.sum(),
            'moving_sum': (epe * moving).sum(),
            'moving_count': moving.sum(),
            'static_sum': (epe * static).sum(),
            'static_count': static.sum(),
        }

    # ── train / eval step ─────────────────────────────────────────────────────

    def update(self, imgs, gt, timestep=0.5, learning_rate=0, training=True,
               scaler=None, flow_gt=None, has_mv=None, loss_step=None):
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
            self.set_learning_rate(learning_rate)

            outputs = self.net(
                imgs_pad, timestep=timestep, scale=0, local=self.local,
                return_all_merges=self.loss_type == 'lc')
            if self.loss_type == 'lc':
                (flow_list, mask_list, res_pad, _, _, merged_pad, pred_pad,
                 all_merged_pad) = outputs
                supervised_merged_pad = all_merged_pad
            else:
                (flow_list, mask_list, res_pad, _, _, merged_pad,
                 pred_pad) = outputs
                supervised_merged_pad = merged_pad

            pred = self.unpad(pred_pad, pr, pb)
            res = self.unpad(res_pad, pr, pb)
            merged = [
                self.unpad(merge, pr, pb)
                for merge in supervised_merged_pad]
            loss_pixel, loss_final, merge_losses, merge_weights = (
                self._pixel_supervision(pred, merged, gt))
            loss_residual = res.abs().mean()
            loss_mask = self._quasi_binary_mask_loss(mask_list, loss_pixel)
            mask_soft_ratio = self._mask_soft_ratio(mask_list, loss_pixel)
            loss = (
                loss_pixel
                + self.residual_loss_weight * loss_residual
                + self.pervfi_mask_loss_weight * loss_mask)

            # teacher flow 多阶段监督 (仅对 has_mv=1 的样本生效)
            loss_flow = torch.zeros((), device=self._dev)
            flow_weight = self.flow_weight_at_step(loss_step)
            self.last_flow_stage_losses = None
            self.last_flow_stage_sums = None
            self.last_flow_valid_count = None
            if (flow_weight > 0 and flow_gt is not None
                    and has_mv is not None and has_mv.sum() > 0):
                flow_gt = flow_gt.to(self._dev)
                has_mv = has_mv.to(self._dev)
                loss_flow = self._multistage_flow_loss(
                    flow_list, flow_gt, has_mv, pr, pb)
                loss = loss + flow_weight * loss_flow

            components = {
                'total': loss.detach(),
                'pixel': loss_pixel.detach(),
                'final': loss_final.detach(),
                'residual': loss_residual.detach(),
                'residual_weighted': (
                    self.residual_loss_weight * loss_residual).detach(),
                'pervfi_mask_binary': loss_mask.detach(),
                'pervfi_mask_soft_ratio': mask_soft_ratio.detach(),
                'pervfi_mask_binary_weighted': (
                    self.pervfi_mask_loss_weight * loss_mask).detach(),
                'pervfi_mask_loss_weight': loss.new_tensor(
                    self.pervfi_mask_loss_weight),
                'flow_raw': loss_flow.detach(),
                'flow_weighted': (flow_weight * loss_flow).detach(),
                'flow_weight': loss.new_tensor(flow_weight),
            }
            components.update(self.last_image_loss_components)
            for index, (merge_loss, merge_weight) in enumerate(
                    zip(merge_losses, merge_weights)):
                components[f'merge_{index}'] = merge_loss.detach()
                components[f'merge_weight_{index}'] = merge_weight.detach()
            self.last_loss_components = components

            self.optimG.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimG)
                self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=self.grad_clip)
                scaler.step(self.optimG)
                scaler.update()
            else:
                loss.backward()
                self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=self.grad_clip)
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
                  scale=0, fast_TTA=False, return_debug=True):
        self.eval()
        imgs = torch.cat((img0, img1), 1)
        imgs, (pr, pb) = self.pad_to_multiple(imgs, 16)
        if fast_TTA:
            batch = imgs.shape[0]
            imgs_ = imgs.flip(2).flip(3)
            flow_list, mask_list, res, warp0, warp1, merged, preds = self.net(
                torch.cat((imgs, imgs_), 0), local=local, timestep=timestep, scale=scale)
            pred = (preds[:batch] + preds[batch:].flip(2).flip(3)) / 2.0
            pred = self.unpad(pred, pr, pb)
            if not return_debug:
                return pred
            # 调试张量统一返回未翻转的主分支；TTA只融合最终预测图像。
            return (pred,
                    flow_list[-1][:batch], mask_list[-1][:batch],
                    merged[-1][:batch], res[:batch],
                    warp0[:batch], warp1[:batch])
        flow_list, mask_list, res, warp0, warp1, merged, preds = self.net(
            imgs, timestep=timestep, scale=scale, local=local)
        if not TTA:
            pred = self.unpad(preds, pr, pb)
            if not return_debug:
                return pred
            return (pred, flow_list[-1], mask_list[-1], merged[-1],
                    res, warp0, warp1)

        if not return_debug:
            # The first pass auxiliary tensors are not needed for a normal
            # release inference. Free them before evaluating the TTA pass.
            del flow_list, mask_list, res, warp0, warp1, merged
        second_outputs = self.net(
            imgs.flip(2).flip(3), timestep=timestep, scale=scale, local=local)
        pred2 = second_outputs[-1]
        del second_outputs
        pred = self.unpad((preds + pred2.flip(2).flip(3)) / 2, pr, pb)
        if not return_debug:
            return pred
        return (pred,
                flow_list[-1], mask_list[-1], merged[-1], res, warp0, warp1)
