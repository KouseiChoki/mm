'''
VFIMamba 训练入口 (yaml配置版)
================================
对接:
  kousei_dataset.MixedTierDataset / TierDataset  (四元组: frames, timestep, flow_gt, has_mv)
  Trainer.Model.update                            (三返回: pred, loss, loss_flow)
  build_lists.py 生成的可扩展分类清单              (<lists_dir>/*_train.txt, val.txt)

全部超参来自 yaml (train_config.yaml), 命令行只留 --config 与 --restore_ckpt。

用法:
  python train.py --config train_config.yaml
  python train.py --config train_config.yaml --restore_ckpt ckpt/base/VFIMamba_100.pkl
'''
import os
import math
import time
import random
import argparse

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from Trainer import Model
from kousei_dataset import MixedTierDataset, TierDataset, resolve_train_lists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_learning_rate(step, total_steps, opt):
    """warmup + cosine, 跨全部phase统一调度。"""
    warmup = opt['warmup_steps']
    lr_max, lr_min = float(opt['lr_max']), float(opt['lr_min'])
    if opt.get('finetune', False):
        # finetune保留旧分支权重, 使用1/4基础LR; 新增零初始化head仍需warmup。
        lr_max *= 0.25
        lr_min *= 0.25
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine = math.cos(math.pi * min(progress, 1.0)) * 0.5 + 0.5
    return lr_min + (lr_max - lr_min) * cosine


def get_grad_norm(net):
    total = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


class SpikeDetector:
    def __init__(self, window=50, spike_ratio=3.0, spike_dir='spike_samples'):
        self.window = window
        self.spike_ratio = spike_ratio
        self.history = []
        self.spike_count = 0
        os.makedirs(spike_dir, exist_ok=True)

    def check(self, loss, step, writer):
        self.history.append(loss)
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < 10:
            return False
        mean_loss = np.mean(self.history[:-1])
        is_spike = (mean_loss > 0) and (loss > mean_loss * self.spike_ratio)
        if is_spike:
            self.spike_count += 1
            print(f'\n[SPIKE #{self.spike_count}] step={step}  '
                  f'loss={loss:.4f}  window_mean={mean_loss:.4f}  '
                  f'ratio={loss / mean_loss:.1f}x')
            writer.add_scalar('spike/loss', loss, step)
        return is_spike


class AnomalyDumper:
    """异常batch落盘: 图像(img0/gt/img1/pred) + flow_gt(.npy) + 元信息, 用于事后归因。"""

    def __init__(self, dump_dir='anomaly_dumps', max_dumps=500):
        self.dump_dir = dump_dir
        self.max_dumps = max_dumps
        self.count = 0
        os.makedirs(dump_dir, exist_ok=True)

    def dump(self, step, reason, frames, timestep, flow_gt, has_mv,
             pred=None, loss=None, loss_flow=None, flow_stage_losses=None):
        """frames: [B,9,H,W] float 0~1; flow_gt: [B,5,H,W]; has_mv: [B]。"""
        if self.count >= self.max_dumps:
            if self.count == self.max_dumps:
                print(f'[AnomalyDumper] 已达上限 {self.max_dumps}, 停止落盘')
                self.count += 1
            return
        self.count += 1
        d = os.path.join(self.dump_dir, f'step{step:07d}_{reason}')
        os.makedirs(d, exist_ok=True)

        import cv2
        frames_np = (frames.detach().cpu().clamp(0, 1) * 255).byte().numpy()
        pred_np = (pred.detach().cpu().clamp(0, 1) * 255).byte().numpy() \
            if pred is not None else None
        t_np = timestep.detach().cpu().numpy().reshape(-1)
        hm_np = has_mv.detach().cpu().numpy().reshape(-1)

        meta = [f'step={step} reason={reason} loss={loss} loss_flow={loss_flow}']
        if flow_stage_losses is not None:
            values = flow_stage_losses.detach().cpu().tolist()
            meta.append('flow_stage_epe=' + ','.join(f'{v:.6f}' for v in values))
        for b in range(frames_np.shape[0]):
            for name, sl in (('img0', slice(0, 3)), ('img1', slice(3, 6)),
                             ('gt', slice(6, 9))):
                img = frames_np[b, sl].transpose(1, 2, 0)[..., ::-1]   # RGB→BGR
                cv2.imwrite(os.path.join(d, f'b{b}_{name}.png'), img)
            if pred_np is not None:
                cv2.imwrite(os.path.join(d, f'b{b}_pred.png'),
                            pred_np[b].transpose(1, 2, 0)[..., ::-1])
            if hm_np[b] > 0:                                           # 仅有效mv落盘
                np.save(os.path.join(d, f'b{b}_flow_gt.npy'),
                        flow_gt[b].detach().cpu().numpy())
            meta.append(f'b{b}: t={t_np[b]:.4f} has_mv={int(hm_np[b])}')
        with open(os.path.join(d, 'meta.txt'), 'w') as f:
            f.write('\n'.join(meta) + '\n')
        print(f'[AnomalyDumper #{self.count}] {reason} → {d}')


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, nr_eval, writer, use_amp, amp_dtype):
    psnr_list = []
    for frames, timestep, _, _ in val_loader:
        frames = frames.to(device, non_blocking=True).float() / 255.
        timestep = timestep.to(device, non_blocking=True)
        imgs, gt = frames[:, :6], frames[:, 6:9]
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            pred, _, _ = model.update(imgs, gt, timestep=timestep, training=False)
        for j in range(gt.shape[0]):
            mse = ((gt[j] - pred[j]) ** 2).mean().cpu().item()
            if mse > 0:
                psnr_list.append(-10 * math.log10(mse))
    if psnr_list:
        psnr = float(np.mean(psnr_list))
        print(f'[eval {nr_eval}] PSNR: {psnr:.4f}')
        writer.add_scalar('val/psnr', psnr, nr_eval)


# ─────────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────────

def train(C, restore_ckpt=None, config_path=None):
    exp = C['exp_name']
    # 约定: 配置yaml随checkpoint归档, 推理侧读同目录model.yaml构建同构模型
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'ckpt/{exp}')
    os.makedirs(ckpt_dir, exist_ok=True)
    if config_path:
        import shutil
        shutil.copy(config_path, os.path.join(ckpt_dir, 'model.yaml'))
        print(f'[archive] 配置已归档 → {ckpt_dir}/model.yaml')
    d, opt, mon = C['data'], C['optim'], C['monitor']
    writer = SummaryWriter(f'log/train_{exp}')
    spike_det = SpikeDetector(spike_ratio=mon['spike_ratio'],
                              spike_dir=mon['spike_dir'])
    dumper = AnomalyDumper(dump_dir=mon.get('dump_dir', 'anomaly_dumps'),
                           max_dumps=mon.get('dump_max', 50))
    flow_dump_thresh = mon.get('flow_loss_dump_threshold', 30.0)

    amp_name = str(opt.get('amp_dtype', 'bf16')).lower()
    if amp_name not in ('bf16', 'fp16'):
        raise ValueError(f'amp_dtype must be bf16 or fp16, got {amp_name}')
    amp_dtype = torch.bfloat16 if amp_name == 'bf16' else torch.float16
    use_amp = bool(opt.get('amp', False)) and torch.cuda.is_available()
    # BF16 的指数范围与 FP32 相同，不需要 loss scaling；仅 FP16 使用 scaler。
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    if use_amp:
        print(f'[AMP] {amp_name.upper()} 混合精度训练已启用 '
              f'(GradScaler={"on" if use_scaler else "off"})')

    # ── 模型 ────────────────────────────────────────────────────────────────
    m = C['model']
    cfg.MODEL_CONFIG['LOGNAME'] = exp
    _train_keys = (
        'loss_type', 'flow_loss_weight', 'flow_stage_gamma',
        'flow_motion_threshold', 'flow_motion_balance', 'flow_motion_gain',
        'flow_motion_scale', 'flow_motion_weight_cap', 'flow_charbonnier_eps',
    )                                                        # 训练侧消费, 不进结构
    _extra = {k: v for k, v in m.items()
              if k not in ('F', 'depth', 'M', 'version') + _train_keys}
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=m['F'], depth=m['depth'], M=m.get('M', False), version=m['version'],
        **_extra)
    model = Model(
        0, loss_type=m['loss_type'],
        flow_loss_weight=m.get('flow_loss_weight', 0.0),
        flow_stage_gamma=m.get('flow_stage_gamma', 0.8),
        flow_motion_threshold=m.get('flow_motion_threshold', 1.0),
        flow_motion_balance=m.get('flow_motion_balance', 0.5),
        flow_motion_gain=m.get('flow_motion_gain', 1.0),
        flow_motion_scale=m.get('flow_motion_scale', 10.0),
        flow_motion_weight_cap=m.get('flow_motion_weight_cap', 4.0),
        flow_charbonnier_eps=m.get('flow_charbonnier_eps', 1e-3))
    if restore_ckpt:
        model.load_model(restore_ckpt)
        print(f'restore ckpt from {restore_ckpt}')

    # ── 数据 ────────────────────────────────────────────────────────────────
    lists_dir = d['lists_dir']
    lists = resolve_train_lists(
        lists_dir, C['phases'], tiers=d.get('tiers'))
    crop_sizes = [tuple(c) for c in d['crop_sizes']]

    train_set = MixedTierDataset(
        d['root'], lists,
        ratios=C['phases'][0]['ratios'],
        crop_hw=crop_sizes[0],
        framesteps=tuple(d['framesteps']),
        t_half_prob=d['t_half_prob'],
        mv_prob=d['mv_prob'],
        mv_sign=tuple(d['mv_sign']),
        motion_aware_crop_prob=d.get('motion_aware_crop_prob', 0.0),
        motion_crop_threshold=d.get('motion_crop_threshold', 1.0),
        small_motion_min_pixels=d.get('small_motion_min_pixels', 8),
        small_motion_max_ratio=d.get('small_motion_max_ratio', 0.05),
        motion_crop_jitter=d.get('motion_crop_jitter', 0.2),
    )
    val_set = TierDataset(d['root'], os.path.join(lists_dir, 'val.txt'), split='val')
    val_loader = DataLoader(val_set, batch_size=1,
                            num_workers=d['num_workers'], pin_memory=True)

    total_epochs = sum(p['epochs'] for p in C['phases'])
    # 名义 step/epoch: 固定值保证LR调度可预期 (Mixed数据集名义长度过大, 不直接用)
    steps_per_epoch = C.get('steps_per_epoch',
                            min(len(train_set), 2000 * d['batch_size'])) // d['batch_size']
    total_steps = total_epochs * steps_per_epoch
    print(f'training... phases={[(p["name"], p["epochs"]) for p in C["phases"]]} '
          f'steps/epoch={steps_per_epoch} total_steps={total_steps}')

    step, nr_eval, loss_ema = 0, 0, None
    epoch_global = 0
    time_stamp = time.time()

    for phase in C['phases']:
        print(f'\n════════ Phase [{phase["name"]}] {phase["epochs"]} epochs ════════')
        train_set.set_ratios(phase['ratios'])

        for _ in range(phase['epochs']):
            # 多尺度: 每epoch换一次crop尺寸 (worker副本重建时生效)
            sel = random.choice(crop_sizes)          # (h, w, bs)
            train_set.set_crop_size((sel[0], sel[1]))
            train_loader = DataLoader(train_set, batch_size=sel[2],
                                    num_workers=d['num_workers'], pin_memory=True,
                                    drop_last=True, shuffle=True)
            it = iter(train_loader)

            for i in range(steps_per_epoch):
                try:
                    frames, timestep, flow_gt, has_mv = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    frames, timestep, flow_gt, has_mv = next(it)

                data_time = time.time() - time_stamp
                time_stamp = time.time()

                frames = frames.to(device, non_blocking=True).float() / 255.
                timestep = timestep.to(device, non_blocking=True)
                imgs, gt = frames[:, :6], frames[:, 6:9]

                lr = get_learning_rate(step, total_steps, opt)

                with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    pred, loss, loss_flow = model.update(
                        imgs, gt, timestep=timestep, learning_rate=lr,
                        training=True, scaler=scaler if use_scaler else None,
                        flow_gt=flow_gt, has_mv=has_mv)

                train_time = time.time() - time_stamp
                time_stamp = time.time()

                loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
                # is_spike = spike_det.check(loss, step, writer)

                # 异常样本落盘: flow loss超阈值 或 loss spike
                if step > 40000 and loss_flow > flow_dump_thresh:
                    dumper.dump(step, f'flowloss{loss_flow:.0f}', frames, timestep,
                                flow_gt, has_mv, pred=pred, loss=loss, loss_flow=loss_flow,
                                flow_stage_losses=model.last_flow_stage_losses)

                if step % mon['log_every_steps'] == 0:
                    writer.add_scalar('loss/raw', loss, step)
                    writer.add_scalar('loss/ema', loss_ema, step)
                    writer.add_scalar('loss/flow_epe', loss_flow, step)
                    if model.last_flow_stage_losses is not None:
                        for si, stage_loss in enumerate(
                                model.last_flow_stage_losses.cpu().tolist()):
                            writer.add_scalar(f'loss/flow_stage_{si}', stage_loss, step)
                    writer.add_scalar('train/lr', lr, step)
                    writer.add_scalar('train/grad_norm', get_grad_norm(model.net), step)
                    if use_scaler:
                        writer.add_scalar('amp/scale', scaler.get_scale(), step)

                print(f'[{phase["name"]}] epoch:{epoch_global} {i}/{steps_per_epoch} '
                      f'time:{data_time:.2f}+{train_time:.2f} '
                      f'loss:{loss:.4f} ema:{loss_ema:.4f} flow:{loss_flow:.3f}')
                step += 1

            epoch_global += 1
            nr_eval += 1
            if nr_eval % mon['eval_every_epochs'] == 0:
                evaluate(model, val_loader, nr_eval, writer, use_amp, amp_dtype)
            if epoch_global % mon['save_every_epochs'] == 0:
                model.save_model(exp, epoch_global, 0)

    model.save_model(exp, epoch_global, 0)          # 收尾存档
    print('════════ 训练完成 ════════')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='yaml配置文件 (全部超参来源)')
    parser.add_argument('--restore_ckpt', type=str, default=None)
    args = parser.parse_args()

    C = load_config(args.config)

    seed = C.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs('log', exist_ok=True)
    train(C, restore_ckpt=args.restore_ckpt, config_path=args.config)
