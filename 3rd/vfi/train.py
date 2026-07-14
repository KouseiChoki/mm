'''
VFIMamba 训练入口 (yaml配置版)
================================
对接:
  kousei_dataset.MixedTierDataset / TierDataset  (四元组: frames, timestep, flow_gt, has_mv)
  Trainer.Model.update                            (三返回: pred, loss, loss_flow)
  build_lists.py 生成的清单                        (<lists_dir>/{easy,normal,hard,teacher}_train.txt, val.txt)

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
from torch.cuda.amp import GradScaler, autocast

import config as cfg
from Trainer import Model
from dataset import MixedTierDataset, TierDataset

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
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cosine = math.cos(math.pi * min(progress, 1.0)) * 0.5 + 0.5
        return (lr_min + (lr_max - lr_min) * cosine) / 4
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


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, nr_eval, writer, use_amp):
    psnr_list = []
    for frames, timestep, _, _ in val_loader:
        frames = frames.to(device, non_blocking=True).float() / 255.
        timestep = timestep.to(device, non_blocking=True)
        imgs, gt = frames[:, :6], frames[:, 6:9]
        with autocast(enabled=use_amp):
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

def train(C, restore_ckpt=None):
    exp = C['exp_name']
    d, opt, mon = C['data'], C['optim'], C['monitor']
    writer = SummaryWriter(f'log/train_{exp}')
    spike_det = SpikeDetector(spike_ratio=mon['spike_ratio'],
                              spike_dir=mon['spike_dir'])

    use_amp = opt.get('amp', False) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print('[AMP] 半精度训练已启用')

    # ── 模型 ────────────────────────────────────────────────────────────────
    m = C['model']
    cfg.MODEL_CONFIG['LOGNAME'] = exp
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=m['F'], depth=m['depth'], M=m.get('M', False), version=m['version'])
    model = Model(0, loss_type=m['loss_type'],
                  flow_loss_weight=m.get('flow_loss_weight', 0.0))
    start_epoch, start_step = 0, 0
    if restore_ckpt:
        start_epoch, start_step = model.load_model(restore_ckpt, resume=True)
    step = start_step
    # 种子按起始epoch偏移: 保证可复现性的同时, 续训不重放同一序列
    effective_seed = seed + start_epoch * 10007
    random.seed(effective_seed); np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)

    # ── 数据 ────────────────────────────────────────────────────────────────
    lists_dir = d['lists_dir']
    lists = {t: os.path.join(lists_dir, f'{t}_train.txt')
             for t in ('easy', 'normal', 'hard', 'teacher')
             if os.path.exists(os.path.join(lists_dir, f'{t}_train.txt'))}
    crop_sizes = [tuple(c) for c in d['crop_sizes']]

    train_set = MixedTierDataset(
        d['root'], lists,
        ratios=C['phases'][0]['ratios'],
        crop_hw=crop_sizes[0],
        framesteps=tuple(d['framesteps']),
        t_half_prob=d['t_half_prob'],
        mv_prob=d['mv_prob'],
        mv_sign=tuple(d['mv_sign']),
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
            train_set.set_crop_size(random.choice(crop_sizes))
            train_loader = DataLoader(
                train_set, batch_size=d['batch_size'],
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

                with autocast(enabled=use_amp):
                    pred, loss, loss_flow = model.update(
                        imgs, gt, timestep=timestep, learning_rate=lr,
                        training=True, scaler=scaler if use_amp else None,
                        flow_gt=flow_gt, has_mv=has_mv)

                train_time = time.time() - time_stamp
                time_stamp = time.time()

                loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss
                # spike_det.check(loss, step, writer)

                if step % mon['log_every_steps'] == 0:
                    writer.add_scalar('loss/raw', loss, step)
                    writer.add_scalar('loss/ema', loss_ema, step)
                    writer.add_scalar('loss/flow_epe', loss_flow, step)
                    writer.add_scalar('train/lr', lr, step)
                    writer.add_scalar('train/grad_norm', get_grad_norm(model.net), step)
                    if use_amp:
                        writer.add_scalar('amp/scale', scaler.get_scale(), step)

                print(f'[{phase["name"]}] epoch:{epoch_global} {i}/{steps_per_epoch} '
                      f'time:{data_time:.2f}+{train_time:.2f} '
                      f'loss:{loss:.4f} ema:{loss_ema:.4f} flow:{loss_flow:.3f}')
                step += 1

            epoch_global += 1
            nr_eval += 1
            if nr_eval % mon['eval_every_epochs'] == 0:
                evaluate(model, val_loader, nr_eval, writer, use_amp)
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
    train(C, restore_ckpt=args.restore_ckpt)