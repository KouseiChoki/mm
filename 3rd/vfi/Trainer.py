import torch
import torch.nn.functional as F
import torch.optim as optim
from model.warplayer import warp
from model.loss import LapLoss
# from border_mask import compute_border_mask, masked_loss
import os
from config import *


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if 'attn_mask' not in k and 'HW' not in k
    }


class Model:
    def __init__(self, local_rank=0, loss_type='lap'):
        """
        loss_type: 'l1' | 'lap' | 'l1+lap'
        """
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg  = MODEL_CONFIG['MODEL_ARCH']
        self.net  = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.local = LOCAL

        self.device()

        assert loss_type in ('l1', 'lap', 'l1+lap'), \
            f"loss_type 须为 'l1'/'lap'/'l1+lap'，got {loss_type}"
        self.loss_type = loss_type
        self.lap = LapLoss()
        self.lap_loss  = (
            LapLoss(max_levels=5, channels=3).to(self._dev)
            if loss_type in ('lap', 'l1+lap') else None
        )

        self.optimG = optim.AdamW(self.net.parameters(), lr=1e-6, weight_decay=1e-4)

    def train(self): self.net.train()
    def eval(self):  self.net.eval()

    def device(self):
        # self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                      :w - pad_right  if pad_right  else w]

    # ── checkpoint ────────────────────────────────────────────────────────────

    def load_model(self, ckpt_path, rank=0, real=False):
        self.net.load_state_dict(
            convert(torch.load(ckpt_path, map_location='cpu')), strict=True
        )

    def save_model(self,sp, epoch,rank=0):
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f'ckpt/{sp}/{self.name}_{str(epoch)}.pkl'
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(self.net.state_dict(), ckpt_path)
        print(f"model saved -> {ckpt_path}")

    @classmethod
    def from_pretrained(cls, model_id, local_rank=-1):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")
        if "/" not in model_id:
            model_id = "MCG-NJU/" + model_id
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.pkl")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        from transformers import PretrainedConfig
        cfg = PretrainedConfig.from_pretrained(model_id)
        MODEL_CONFIG['MODEL_ARCH'] = init_model_config(F=cfg.F, depth=cfg.depth)
        model = cls()
        model.net.load_state_dict(convert(checkpoint), strict=True)
        return model

    # ── train / eval step ─────────────────────────────────────────────────────

    def update(self, imgs, gt, learning_rate=0, training=True, scaler=None):
        """
        imgs : [B, 6, H, W]  float 0~1
        gt   : [B, 3, H, W]  float 0~1

        流程：
        1. 检测 gt 中的 film border → mask [B,1,H,W]
        2. pad 到 64 倍数后送入网络
        3. unpad pred 回原始尺寸
        4. 用 mask 屏蔽 border 区域后计算 loss
        """
        # ── 1. film border mask（在原始尺寸上检测）──
        with torch.no_grad():
            border_mask = compute_border_mask(gt)   # [B, 1, H, W]  0/1

        # ── 2. pad → 网络前向 ──
        imgs_pad, (pr, pb) = self.pad_to_multiple(imgs, 16)

        if training:
            self.train()
            for pg in self.optimG.param_groups:
                pg['lr'] = learning_rate

            _,_,_,_,_, merged, pred_pad = self.net(imgs_pad, timestep=0.5, scale=0, local=self.local)

            # ── 3. unpad 回原始尺寸 ──
            pred = self.unpad(pred_pad, pr, pb)

            # ── 4. masked loss ──
            loss = masked_loss(pred, gt, border_mask, self.loss_type, self.lap_loss)
            # ── 5. warp loss ──
            for merge in merged:
                merge = self.unpad(merge, pr, pb)
                loss += masked_loss(merge, gt, border_mask, self.loss_type, self.lap_loss)

            self.optimG.zero_grad()

            if scaler is not None:
                # ── AMP 反向 ──
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimG)          # unscale 后才能正确 clip
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                scaler.step(self.optimG)
                scaler.update()
            else:
                # ── 原始 FP32 反向 ──
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimG.step()

            return pred.detach(), loss.item()

        else:
            self.eval()
            with torch.no_grad():
                _,_,_,_,_,_, pred_pad = self.net(imgs_pad, timestep=0.5, scale=0, local=self.local)
                pred = self.unpad(pred_pad, pr, pb)
                loss = masked_loss(pred, gt, border_mask, self.loss_type, self.lap_loss)
            return pred, loss.item()

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def hr_inference(self, img0, img1, local, TTA=False, down_scale=1.0,
                     timestep=0.5, fast_TTA=False):
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale,
                                      mode="bilinear", align_corners=False)
            flow, mask = self.net.calculate_flow(imgs_down, timestep, local=local)
            flow = F.interpolate(flow, scale_factor=1/down_scale,
                                 mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor=1/down_scale,
                                 mode="bilinear", align_corners=False)
            af = self.net.feature_bone(img0, img1)
            pred, warp0, warp1, mask_ = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred, warp0, warp1, flow, mask_

        imgs = torch.cat((img0, img1), 1)
        imgs, (pr, pb) = self.pad_to_multiple(imgs, 16)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            preds = infer(torch.cat((imgs, imgs_), 0))
            out = (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.
            return self.unpad(out, pr, pb)
        if not TTA:
            result = infer(imgs)
            return (self.unpad(result[0], pr, pb),) + result[1:]
        r1, r2 = infer(imgs), infer(imgs.flip(2).flip(3))
        return self.unpad((r1[0] + r2[0].flip(2).flip(3)) / 2, pr, pb)

    @torch.no_grad()
    def inference(self, img0, img1, local, TTA=False, timestep=0.5,
                  scale=0, fast_TTA=False):
        imgs = torch.cat((img0, img1), 1)
        imgs, (pr, pb) = self.pad_to_multiple(imgs, 16)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            flow_list, mask_list,res ,warp0,warp1,merged, preds = self.net(torch.cat((imgs, imgs_), 0),
                                      local=local, timestep=timestep, scale=scale)
            return self.unpad((preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2., pr, pb),flow_list[-1], mask_list[-1], merged[-1],res,warp0,warp1
        flow_list, mask_list,res,warp0,warp1,merged, preds = self.net(imgs, timestep=timestep, scale=scale, local=local)
        if not TTA:
            return self.unpad(preds, pr, pb),None,None,None,None,None,None
        _,_,_,_,_,_, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep,
                                  scale=scale, local=local)
        return self.unpad((preds + pred2.flip(2).flip(3)) / 2, pr, pb),flow_list[-1], mask_list[-1],merged[-1],res,warp0,warp1


