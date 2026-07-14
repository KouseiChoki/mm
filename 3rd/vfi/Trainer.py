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
    def __init__(self, local_rank=0, loss_type='lap', flow_loss_weight=0.01):
        """
        loss_type        : 'l1' | 'lap' | 'l1+lap'
        flow_loss_weight : teacher flow 监督 (masked EPE) 的权重, 0=关闭
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

    def save_model(self, sp, epoch, step=0):
        ckpt = {'net': self.net.state_dict(),
                'optim': self.optimG.state_dict(),
                'epoch': epoch, 'step': step}
        torch.save(ckpt, ckpt_path)

    def load_model(self, ckpt_path, resume=False):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'net' in ckpt:                      # 新格式
            self.net.load_state_dict(convert(ckpt['net']), strict=True)
            if resume and 'optim' in ckpt:
                self.optimG.load_state_dict(ckpt['optim'])
            return ckpt.get('epoch', 0), ckpt.get('step', 0)
        self.net.load_state_dict(convert(ckpt), strict=True)   # 旧格式兼容
        return 0, 0

    # ── loss ──────────────────────────────────────────────────────────────────

    def _pixel_loss(self, pred, gt):
        if self.loss_type == 'l1':
            return (pred - gt).abs().mean()
        if self.loss_type == 'lap':
            return self.lap_loss(pred, gt)
        return (pred - gt).abs().mean() + self.lap_loss(pred, gt)   # l1+lap

    @staticmethod
    def _flow_epe_masked(flow_pred, flow_gt, has_mv):
        """
        flow_pred/flow_gt : [B, 4, H, W] 像素位移 (前2通道 F_t→0, 后2通道 F_t→1)
        has_mv            : [B] 1=该样本flow_gt有效
        返回 masked EPE (仅在有效样本上平均); 无有效样本返回 0。
        """
        n_valid = has_mv.sum()
        if n_valid < 1:
            return flow_pred.sum() * 0.0
        epe0 = torch.norm(flow_pred[:, 0:2] - flow_gt[:, 0:2], dim=1)   # [B,H,W]
        epe1 = torch.norm(flow_pred[:, 2:4] - flow_gt[:, 2:4], dim=1)
        per_sample = (epe0 + epe1).mean(dim=(1, 2)) * 0.5               # [B]
        return (per_sample * has_mv).sum() / n_valid

    # ── train / eval step ─────────────────────────────────────────────────────

    def update(self, imgs, gt, timestep=0.5, learning_rate=0, training=True,
               scaler=None, flow_gt=None, has_mv=None):
        """
        imgs     : [B, 6, H, W] float 0~1  (img0, img1)
        gt       : [B, 3, H, W] float 0~1
        timestep : float 或 [B,1,1,1] 张量 (来自Dataset, 任意timestep训练必须传)
        flow_gt  : [B, 4, H, W] 像素位移 (teacher样本), 可为 None
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

            # teacher flow 监督 (masked EPE, 仅对 has_mv=1 的样本生效)
            loss_flow = torch.zeros((), device=self._dev)
            if (self.flow_loss_weight > 0 and flow_gt is not None
                    and has_mv is not None and has_mv.sum() > 0):
                flow_pred = self.unpad(flow_list[-1], pr, pb)
                loss_flow = self._flow_epe_masked(
                    flow_pred, flow_gt.to(self._dev), has_mv.to(self._dev))
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