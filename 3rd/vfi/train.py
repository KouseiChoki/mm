'''
VFIMamba 训练入口 (yaml配置版)
================================
对接:
  kousei_dataset.MixedTierDataset / TierDataset  (四元组: frames, timestep, flow_gt, has_mv)
  Trainer.Model.update                            (三返回: pred, loss, loss_flow)
  build_lists.py 生成的可扩展分类清单              (<lists_dir>/*_train.txt, val.txt)

全部超参来自 yaml (train_config.yaml), 命令行只留配置/checkpoint/恢复模式。

用法:
  python train.py --config train_config.yaml
  python train.py --config train_config.yaml --restore_ckpt ckpt/base/VFIMamba_100.pkl
  python train.py --config train_config.yaml --restore_ckpt ckpt/exp/latest.pkl --resume
'''
import os
import math
import time
import random
import argparse
import shutil

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from Trainer import Model
from kousei_dataset import MixedTierDataset, TierDataset, resolve_train_lists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def archive_config(path, checkpoint_dir):
    """同一实验目录禁止静默覆盖不同配置，保护旧checkpoint的release语义。"""
    destination = os.path.join(checkpoint_dir, 'model.yaml')
    if os.path.exists(destination):
        if load_config(destination) != load_config(path):
            raise ValueError(
                f'{destination} 已存在且内容与当前配置不同。'
                '请更换 exp_name，避免旧checkpoint关联到错误model.yaml。')
        print(f'[archive] 复用已归档配置 → {destination}')
        return
    shutil.copy(path, destination)
    print(f'[archive] 配置已归档 → {destination}')


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        'python_rng_state': random.getstate(),
        # ndarray改存list，使PyTorch weights_only加载器可安全读取新checkpoint。
        'numpy_rng_state': (
            numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]),
        'torch_rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(payload):
    if not isinstance(payload, dict):
        return
    if 'python_rng_state' in payload:
        random.setstate(payload['python_rng_state'])
    if 'numpy_rng_state' in payload:
        numpy_state = payload['numpy_rng_state']
        np.random.set_state((
            numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2], numpy_state[3], numpy_state[4]))
    if 'torch_rng_state' in payload:
        torch.set_rng_state(payload['torch_rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state' in payload:
        torch.cuda.set_rng_state_all(payload['cuda_rng_state'])


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def apply_motion_curriculum(train_set, curriculum, phase_epoch):
    """应用VFIMamba式X-TRAIN运动课程，并返回当前阶段信息。"""
    if not curriculum:
        return None
    every = int(curriculum.get('every_epochs', 50))
    if every <= 0:
        raise ValueError('curriculum.every_epochs必须为正整数')
    stage = phase_epoch // every
    max_stage = curriculum.get('max_stage')
    if max_stage is not None:
        stage = min(stage, int(max_stage))
    source = str(curriculum.get('source', 'xtrain'))
    framestep = int(round(
        float(curriculum.get('framestep_start', 1))
        * float(curriculum.get('framestep_multiplier', 2)) ** stage))
    resize = int(round(
        float(curriculum.get('resize_start', 256))
        * float(curriculum.get('resize_multiplier', 1.1)) ** stage))
    train_set.configure_source(
        source, framesteps=(framestep,), resize_hw=(resize, resize))
    return {
        'stage': stage,
        'source': source,
        'framestep': framestep,
        'resize': resize,
    }


def metric_improved(value, best, metric_name, monitor_cfg):
    mode = str(monitor_cfg.get('best_mode', 'auto')).lower()
    if mode == 'auto':
        mode = 'min' if any(
            token in metric_name.lower()
            for token in ('loss', 'epe', 'error')) else 'max'
    if mode not in ('min', 'max'):
        raise ValueError(f'monitor.best_mode must be auto/min/max, got {mode}')
    return value < best if mode == 'min' else value > best, mode


def get_grad_norm(net):
    total = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def build_record_paths(exp, monitor_cfg, project_dir=PROJECT_DIR):
    """Resolve all non-checkpoint training artifacts under record/<exp>."""
    record_root = str(monitor_cfg.get('record_root', 'record')).strip()
    if not record_root:
        raise ValueError('monitor.record_root不能为空')
    if not os.path.isabs(record_root):
        record_root = os.path.join(project_dir, record_root)
    exp = str(exp).strip()
    if not exp or os.path.isabs(exp) or '..' in exp.split(os.sep):
        raise ValueError(f'exp_name必须是record内的相对目录名, got {exp!r}')
    experiment_dir = os.path.abspath(os.path.join(record_root, exp))

    def subdir(key, default):
        value = str(monitor_cfg.get(key, default)).strip()
        if not value:
            value = default
        if os.path.isabs(value) or '..' in value.split(os.sep):
            raise ValueError(
                f'monitor.{key}必须是record/<exp>内的相对目录名, got {value!r}')
        return os.path.join(experiment_dir, value)

    paths = {
        'root': experiment_dir,
        'tensorboard': subdir('tensorboard_dir', 'tensorboard'),
        'spikes': subdir('spike_dir', 'spike_samples'),
        'anomalies': subdir('dump_dir', 'anomaly_dumps'),
        'mv_comparisons': subdir('mv_compare_dir', 'mv_comparisons'),
    }
    os.makedirs(paths['root'], exist_ok=True)
    return paths


def _flow_to_bgr(flow, valid, max_magnitude):
    """Convert a [2,H,W] pixel flow to a direction/magnitude HSV image."""
    import cv2

    flow = np.nan_to_num(np.asarray(flow, dtype=np.float32), copy=False)
    dx, dy = flow[0], flow[1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod((angle + np.pi) * (179.0 / (2.0 * np.pi)), 180).astype(
        np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        magnitude / max(float(max_magnitude), 1e-6) * 255.0, 0, 255).astype(
            np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result[~valid] = 0
    return result


def _write_original_style_mv_exr(path, pixel_flow):
    """Write XY like cal_mv.py: normalized R/G, zero B/A, half PIZ EXR."""
    from file_utils import mvwrite

    pixel_flow = np.nan_to_num(
        np.asarray(pixel_flow, dtype=np.float32), copy=True)
    if pixel_flow.ndim != 3 or pixel_flow.shape[0] != 2:
        raise ValueError(f'MV EXR输入应为[2,H,W], got {pixel_flow.shape}')
    height, width = pixel_flow.shape[1:]
    normalized = np.zeros((height, width, 4), dtype=np.float32)
    normalized[..., 0] = pixel_flow[0] / width
    normalized[..., 1] = pixel_flow[1] / height
    mvwrite(path, normalized, compress='piz', precision='half')


class MVComparisonDumper:
    """Persist exact predicted/GT MV plus directly comparable visualizations."""

    def __init__(self, output_dir, every_epochs=0, max_samples=8):
        self.output_dir = output_dir
        self.every_epochs = int(every_epochs)
        self.max_samples = int(max_samples)
        if self.every_epochs < 0:
            raise ValueError('monitor.mv_compare_every_epochs必须>=0')
        if self.every_epochs > 0 and self.max_samples <= 0:
            raise ValueError('monitor.mv_compare_max_samples必须为正整数')

    def due(self, epoch):
        return self.every_epochs > 0 and int(epoch) % self.every_epochs == 0

    def dump_sample(self, epoch, sample_index, frames, pred_frame,
                    flow_pred, flow_gt, timestep, metadata=None):
        import cv2

        sample_dir = os.path.join(
            self.output_dir, f'epoch_{int(epoch):04d}',
            f'sample_{int(sample_index):04d}')
        os.makedirs(sample_dir, exist_ok=True)

        frames_np = frames.detach().float().cpu().clamp(0, 1).numpy()
        pred_frame_np = pred_frame.detach().float().cpu().clamp(0, 1).numpy()
        pred_mv = flow_pred.detach().float().cpu().numpy().astype(np.float32)
        safe_pred_mv = np.nan_to_num(pred_mv, copy=True)
        gt_all = flow_gt.detach().float().cpu().numpy().astype(np.float32)
        gt_mv = np.nan_to_num(gt_all[:4], copy=True)
        valid_weight = np.nan_to_num(
            gt_all[4], nan=0.0, posinf=0.0, neginf=0.0).clip(0, 1)
        valid = valid_weight > 0

        np.savez_compressed(
            os.path.join(sample_dir, 'mv.npz'),
            pred_mv1=pred_mv[:2], pred_mv0=pred_mv[2:4],
            gt_mv1=gt_mv[:2], gt_mv0=gt_mv[2:4],
            valid=valid_weight)
        for name, value in (
                ('pred_mv1.exr', safe_pred_mv[:2]),
                ('gt_mv1.exr', gt_mv[:2]),
                ('pred_mv0.exr', safe_pred_mv[2:4]),
                ('gt_mv0.exr', gt_mv[2:4])):
            _write_original_style_mv_exr(
                os.path.join(sample_dir, name), value)

        for name, channel_slice in (
                ('img0', slice(0, 3)), ('img1', slice(3, 6)),
                ('middle_gt', slice(6, 9))):
            image = (frames_np[channel_slice].transpose(1, 2, 0)[..., ::-1]
                     * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(sample_dir, f'{name}.png'), image)
        pred_image = (pred_frame_np.transpose(1, 2, 0)[..., ::-1]
                      * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(sample_dir, 'middle_pred.png'), pred_image)
        cv2.imwrite(
            os.path.join(sample_dir, 'valid.png'),
            (valid_weight * 255.0).round().astype(np.uint8))

        epe_values = []
        scales = []
        for direction, channel_slice in (
                ('mv1_to_previous', slice(0, 2)),
                ('mv0_to_next', slice(2, 4))):
            pred_direction = safe_pred_mv[channel_slice]
            gt_direction = gt_mv[channel_slice]
            pred_mag = np.linalg.norm(pred_direction, axis=0)
            gt_mag = np.linalg.norm(gt_direction, axis=0)
            if valid.any():
                shared = np.concatenate((pred_mag[valid], gt_mag[valid]))
                scale = max(float(np.percentile(shared, 99.0)), 1.0)
            else:
                scale = 1.0
            scales.append(scale)
            cv2.imwrite(
                os.path.join(sample_dir, f'pred_{direction}.png'),
                _flow_to_bgr(pred_direction, valid, scale))
            cv2.imwrite(
                os.path.join(sample_dir, f'gt_{direction}.png'),
                _flow_to_bgr(gt_direction, valid, scale))

            epe = np.linalg.norm(pred_direction - gt_direction, axis=0)
            weighted_count = float(valid_weight.sum())
            mean_epe = (
                float((epe * valid_weight).sum() / weighted_count)
                if weighted_count > 0 else None)
            epe_values.append(mean_epe)
            error_scale = max(
                float(np.percentile(epe[valid], 99.0)), 1.0) if valid.any() else 1.0
            error_u8 = np.clip(epe / error_scale * 255.0, 0, 255).astype(np.uint8)
            error_bgr = cv2.applyColorMap(error_u8, cv2.COLORMAP_TURBO)
            error_bgr[~valid] = 0
            cv2.imwrite(
                os.path.join(sample_dir, f'error_{direction}.png'), error_bgr)

        meta = {
            'epoch': int(epoch),
            'sample_index': int(sample_index),
            'timestep': float(timestep),
            'valid_ratio': float(valid.mean()),
            'prediction_finite_ratio': float(np.isfinite(pred_mv).all(axis=0).mean()),
            'mv1_epe': epe_values[0],
            'mv0_epe': epe_values[1],
            'mean_epe': (
                0.5 * (epe_values[0] + epe_values[1])
                if all(value is not None for value in epe_values) else None),
            'visualization_p99_magnitude': {
                'mv1_to_previous': scales[0], 'mv0_to_next': scales[1]},
            'metadata': metadata or {},
            'convention': {
                'mv1': 'middle/current -> previous input',
                'mv0': 'middle/current -> next input',
                'mv.npz_unit': 'pixels',
                'exr_xy_unit': 'normalized: R=dx/width, G=dy/height',
                'exr_channels': 'RGBA half; B/A are zero (training uses XY only)',
            },
        }
        with open(os.path.join(sample_dir, 'meta.yaml'), 'w') as handle:
            yaml.safe_dump(meta, handle, allow_unicode=True, sort_keys=False)
        return sample_dir


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
def evaluate(model, val_loaders, nr_eval, writer, use_amp, amp_dtype):
    """按数据域评估PSNR；带MV的验证集额外评估最终flow EPE。"""
    model.eval()
    metrics = {}
    for name, val_loader in val_loaders.items():
        psnr_sum = 0.0
        sample_count = 0
        flow_totals = {
            key: 0.0 for key in (
                'sum', 'count', 'moving_sum', 'moving_count',
                'static_sum', 'static_count', 'mv_pixels', 'mv_samples')}
        teacher_groups = {}

        for sample_index, (frames, timestep, flow_gt, has_mv) in enumerate(
                val_loader):
            frames = frames.to(device, non_blocking=True).float() / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs, gt = frames[:, :6], frames[:, 6:9]
            imgs_pad, (pad_right, pad_bottom) = model.pad_to_multiple(imgs, 16)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                flow_list, _, _, _, _, _, pred_pad = model.net(
                    imgs_pad, timestep=timestep, scale=0, local=model.local)
            pred = model.unpad(pred_pad, pad_right, pad_bottom)
            mse = (gt - pred).square().mean(dim=(1, 2, 3)).clamp(min=1e-12)
            psnr_sum += (-10.0 * torch.log10(mse)).sum().item()
            sample_count += int(gt.shape[0])

            group = None
            if name == 'teacher' and hasattr(
                    val_loader.dataset, 'sample_metadata'):
                metadata = val_loader.dataset.sample_metadata(sample_index)
                group_name = f'{metadata["domain"]}_{metadata["fps"]}fps'
                group = teacher_groups.setdefault(group_name, {
                    key: 0.0 for key in (
                        'psnr_sum', 'samples', 'sum', 'count',
                        'moving_sum', 'moving_count', 'static_sum',
                        'static_count', 'mv_pixels', 'mv_samples')})
                group['psnr_sum'] += (-10.0 * torch.log10(mse)).sum().item()
                group['samples'] += int(gt.shape[0])

            mv_samples = float(has_mv.sum().item())
            flow_totals['mv_samples'] += mv_samples
            if group is not None:
                group['mv_samples'] += mv_samples

            if flow_list and has_mv.sum().item() > 0:
                flow_gt = flow_gt.to(device, non_blocking=True)
                has_mv = has_mv.to(device, non_blocking=True)
                final_flow = model.unpad(
                    flow_list[-1], pad_right, pad_bottom)
                sums = model.flow_metric_sums(final_flow, flow_gt, has_mv)
                for key, value in sums.items():
                    flow_totals[key] += value.item()
                    if group is not None:
                        group[key] += value.item()
                mv_pixels = float(
                    has_mv.sum().item() * flow_gt.shape[-2] * flow_gt.shape[-1])
                flow_totals['mv_pixels'] += mv_pixels
                if group is not None:
                    group['mv_pixels'] += mv_pixels

        if sample_count:
            psnr = psnr_sum / sample_count
            metrics[f'{name}/psnr'] = psnr
            writer.add_scalar(f'val/{name}_psnr', psnr, nr_eval)
            message = f'[eval {nr_eval}] {name}: PSNR={psnr:.4f}'
            if flow_totals['count'] > 0:
                epe = flow_totals['sum'] / flow_totals['count']
                metrics[f'{name}/flow_epe'] = epe
                writer.add_scalar(f'val/{name}_flow_epe', epe, nr_eval)
                message += f' flow_EPE={epe:.4f}'
                for region in ('moving', 'static'):
                    count = flow_totals[f'{region}_count']
                    if count > 0:
                        region_epe = flow_totals[f'{region}_sum'] / count
                        metrics[f'{name}/{region}_flow_epe'] = region_epe
                        writer.add_scalar(
                            f'val/{name}_{region}_flow_epe',
                            region_epe, nr_eval)
                        message += f' {region}_EPE={region_epe:.4f}'
                if flow_totals['mv_pixels'] > 0:
                    valid_ratio = (
                        flow_totals['count'] / flow_totals['mv_pixels'])
                    metrics[f'{name}/flow_valid_ratio'] = valid_ratio
                    writer.add_scalar(
                        f'val/{name}_flow_valid_ratio', valid_ratio, nr_eval)
                    message += f' flow_valid={valid_ratio:.3f}'
                flow_sample_coverage = (
                    flow_totals['mv_samples'] / sample_count)
                metrics[f'{name}/flow_sample_coverage'] = flow_sample_coverage
                writer.add_scalar(
                    f'val/{name}_flow_sample_coverage',
                    flow_sample_coverage, nr_eval)
                message += f' flow_samples={flow_sample_coverage:.3f}'
            print(message)

            if name == 'teacher':
                for group_name, group in sorted(teacher_groups.items()):
                    if not group['samples']:
                        continue
                    group_psnr = group['psnr_sum'] / group['samples']
                    prefix = f'{name}/{group_name}'
                    metrics[f'{prefix}/psnr'] = group_psnr
                    writer.add_scalar(
                        f'val/{name}_{group_name}_psnr', group_psnr, nr_eval)
                    group_message = (
                        f'[eval {nr_eval}] {prefix}: PSNR={group_psnr:.4f}')
                    if group['count'] > 0:
                        group_epe = group['sum'] / group['count']
                        group_valid = group['count'] / max(group['mv_pixels'], 1.0)
                        group_coverage = (
                            group['mv_samples'] / group['samples'])
                        metrics[f'{prefix}/flow_epe'] = group_epe
                        metrics[f'{prefix}/flow_valid_ratio'] = group_valid
                        metrics[f'{prefix}/flow_sample_coverage'] = group_coverage
                        writer.add_scalar(
                            f'val/{name}_{group_name}_flow_epe',
                            group_epe, nr_eval)
                        writer.add_scalar(
                            f'val/{name}_{group_name}_flow_valid_ratio',
                            group_valid, nr_eval)
                        writer.add_scalar(
                            f'val/{name}_{group_name}_flow_sample_coverage',
                            group_coverage, nr_eval)
                        group_message += (
                            f' flow_EPE={group_epe:.4f}'
                            f' flow_valid={group_valid:.3f}'
                            f' flow_samples={group_coverage:.3f}')
                    print(group_message)
    model.train()
    return metrics


@torch.no_grad()
def dump_mv_comparisons(model, val_loaders, epoch, dumper,
                        use_amp, amp_dtype):
    """Run a small deterministic teacher-val subset and dump both MV fields."""
    if not dumper.due(epoch):
        return 0
    val_loader = val_loaders.get('teacher')
    if val_loader is None:
        print('[mv-compare] WARNING: 没有teacher验证集，跳过MV输出')
        return 0

    was_training = bool(model.net.training)
    model.eval()
    saved = 0
    for sample_index, (frames, timestep, flow_gt, has_mv) in enumerate(val_loader):
        if saved >= dumper.max_samples:
            break
        if has_mv.sum().item() <= 0:
            continue

        frames_gpu = frames.to(device, non_blocking=True).float() / 255.0
        timestep_gpu = timestep.to(device, non_blocking=True)
        imgs = frames_gpu[:, :6]
        imgs_pad, (pad_right, pad_bottom) = model.pad_to_multiple(imgs, 16)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            flow_list, _, _, _, _, _, pred_pad = model.net(
                imgs_pad, timestep=timestep_gpu, scale=0, local=model.local)
        if not flow_list:
            print('[mv-compare] WARNING: 模型没有输出flow，跳过MV输出')
            break
        pred = model.unpad(pred_pad, pad_right, pad_bottom)
        final_flow = model.unpad(flow_list[-1], pad_right, pad_bottom)
        metadata = (
            val_loader.dataset.sample_metadata(sample_index)
            if hasattr(val_loader.dataset, 'sample_metadata') else {})
        output = dumper.dump_sample(
            epoch=epoch, sample_index=sample_index,
            frames=frames_gpu[0], pred_frame=pred[0],
            flow_pred=final_flow[0], flow_gt=flow_gt[0],
            timestep=timestep.reshape(-1)[0].item(), metadata=metadata)
        saved += 1
        print(f'[mv-compare {saved}/{dumper.max_samples}] → {output}')

    if was_training:
        model.train()
    print(f'[mv-compare] epoch={epoch} saved={saved} root={dumper.output_dir}')
    return saved


def build_val_loaders(data_cfg, lists_dir):
    """支持 data.val_lists: {all: val.txt, easy: easy_val.txt, teacher: teacher_val.txt}。"""
    configured = data_cfg.get('val_lists')
    if configured is None:
        configured = {'all': 'val.txt'}
    common = {
        'mv_sign': tuple(data_cfg.get('mv_sign', (1, 1))),
        'mv_symmetry_confidence': bool(
            data_cfg.get('mv_symmetry_confidence', False)),
        'occ_alpha': float(data_cfg.get('occ_alpha', 0.05)),
        'occ_beta': float(data_cfg.get('occ_beta', 1.0)),
        'mv_cache_dirname': data_cfg.get('mv_cache_dirname'),
        'mv_cache_required': bool(data_cfg.get('mv_cache_required', False)),
        'mv_cache_preview_stride': int(
            data_cfg.get('mv_cache_preview_stride', 4)),
        'mv_cycle_confidence': data_cfg.get(
            'val_mv_cycle_confidence',
            data_cfg.get('mv_cycle_confidence', 'none')),
        'mv_cycle_cache_required': bool(
            data_cfg.get(
                'val_mv_cycle_cache_required',
                data_cfg.get('mv_cycle_cache_required', False))),
        'mv_cycle_cache_root': data_cfg.get('mv_cycle_cache_root'),
        'mv_cycle_on_the_fly': bool(data_cfg.get(
            'val_mv_cycle_on_the_fly',
            data_cfg.get('mv_cycle_on_the_fly', False))),
        'val_samples_per_scene': data_cfg.get('val_samples_per_scene'),
    }
    loaders = {}
    for name, value in configured.items():
        path = value if os.path.isabs(value) else os.path.join(lists_dir, value)
        if not os.path.isfile(path):
            if name == 'all':
                raise FileNotFoundError(f'验证清单不存在: {path}')
            train_path = os.path.join(lists_dir, f'{name}_train.txt')
            if (os.path.isfile(train_path)
                    and os.path.getsize(train_path) == 0):
                print(f'[eval] 空训练分类，跳过验证: {name}')
                continue
            print(f'[eval] WARNING: 验证清单不存在，跳过 {name}: {path}')
            continue
        is_teacher = name == 'teacher' or name.startswith('teacher_')
        dataset = TierDataset(
            data_cfg['root'], path, split='val',
            val_with_mv=is_teacher, **common)
        cap = data_cfg.get('val_samples_per_scene')
        print(f'[eval] {name}: {len(dataset)} samples / '
              f'{len(dataset.scenes)} scenes '
              f'(val_samples_per_scene={cap if cap is not None else "all"})')
        loaders[name] = DataLoader(
            dataset, batch_size=1,
            num_workers=int(data_cfg.get(
                'val_num_workers', min(int(data_cfg['num_workers']), 4))),
            pin_memory=True, shuffle=False, worker_init_fn=seed_worker)
    if not loaders:
        raise ValueError('没有可用的验证集')
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────────

def train(C, restore_ckpt=None, config_path=None, resume=False):
    exp = C['exp_name']
    # 约定: 配置yaml随checkpoint归档, 推理侧读同目录model.yaml构建同构模型
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'ckpt/{exp}')
    os.makedirs(ckpt_dir, exist_ok=True)
    if config_path:
        archive_config(config_path, ckpt_dir)
    d, opt, mon = C['data'], C['optim'], C['monitor']
    record_paths = build_record_paths(exp, mon)
    writer = SummaryWriter(record_paths['tensorboard'])
    spike_det = SpikeDetector(spike_ratio=mon['spike_ratio'],
                              spike_dir=record_paths['spikes'])
    dumper = AnomalyDumper(dump_dir=record_paths['anomalies'],
                           max_dumps=mon.get('dump_max', 50))
    mv_dumper = MVComparisonDumper(
        output_dir=record_paths['mv_comparisons'],
        every_epochs=mon.get('mv_compare_every_epochs', 0),
        max_samples=mon.get('mv_compare_max_samples', 8))
    print(f'[record] {record_paths["root"]}')
    if mv_dumper.every_epochs > 0:
        print('[mv-compare] enabled: '
              f'every={mv_dumper.every_epochs} epochs, '
              f'max_samples={mv_dumper.max_samples}')
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
        'flow_loss_warmup_steps', 'merge_loss_gamma', 'merge_loss_weights',
        'normalize_pixel_loss', 'residual_loss_weight',
        'lc_charbonnier_eps', 'lc_census_weight', 'lc_lap_weight',
        'lc_warp_weight', 'pervfi_mask_loss_weight',
    )                                                        # 训练侧消费, 不进结构
    _extra = {k: v for k, v in m.items()
              if k not in ('F', 'depth', 'M', 'version') + _train_keys}
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=m['F'], depth=m['depth'], M=m.get('M', False), version=m['version'],
        **_extra)
    model = Model(
        loss_type=m['loss_type'],
        flow_loss_weight=m.get('flow_loss_weight', 0.0),
        flow_stage_gamma=m.get('flow_stage_gamma', 0.2),
        flow_motion_threshold=m.get('flow_motion_threshold', 1.0),
        flow_motion_balance=m.get('flow_motion_balance', 0.1),
        flow_motion_gain=m.get('flow_motion_gain', 0.0),
        flow_motion_scale=m.get('flow_motion_scale', 10.0),
        flow_motion_weight_cap=m.get('flow_motion_weight_cap', 4.0),
        flow_charbonnier_eps=m.get('flow_charbonnier_eps', 1e-3),
        flow_loss_warmup_steps=m.get('flow_loss_warmup_steps', 0),
        merge_loss_gamma=m.get('merge_loss_gamma', 0.5),
        merge_loss_weights=m.get('merge_loss_weights'),
        normalize_pixel_loss=m.get('normalize_pixel_loss', True),
        residual_loss_weight=m.get('residual_loss_weight', 0.0),
        lc_charbonnier_eps=m.get('lc_charbonnier_eps', 1e-3),
        lc_census_weight=m.get('lc_census_weight', 1.0),
        lc_lap_weight=m.get('lc_lap_weight', 1.0),
        lc_warp_weight=m.get('lc_warp_weight', 0.5),
        pervfi_mask_loss_weight=m.get('pervfi_mask_loss_weight', 0.0))
    if m.get('blend_mode', 'soft') == 'pervfi':
        print('[blend] PerVFI-inspired quasi-binary asymmetric blending: '
              f'temperature={m.get("pervfi_mask_temperature", 0.5)} '
              f'disagreement={m.get("pervfi_disagreement_threshold", 0.03)} '
              f'strength={m.get("pervfi_blend_strength", 1.0)} '
              f'mask_loss={m.get("pervfi_mask_loss_weight", 0.0)}')
    if m['loss_type'] == 'lc':
        print('[loss] LC-Mamba: Charbonnier + Census7x7 + final Lap '
              f'+ {m.get("lc_warp_weight", 0.5)} * Σ(stage warp Lap)')
        if (m.get('flow_loss_weight', 0.0) != 0.0
                or m.get('residual_loss_weight', 0.0) != 0.0):
            print('[loss] WARNING: 当前仍启用了teacher flow或residual附加loss；'
                  '严格LC对照应将flow_loss_weight和residual_loss_weight设为0')
    model.configure_optimizer(opt)
    start_epoch, restored_step = 0, 0
    if restore_ckpt:
        start_epoch, restored_step = model.load_model(
            restore_ckpt, resume=resume)
        if resume and not model.optimizer_restored:
            raise RuntimeError(
                'checkpoint未能恢复optimizer，不能安全--resume；'
                '请移除--resume按finetune方式启动')
        payload = model.loaded_checkpoint
        if resume and isinstance(payload, dict):
            if use_scaler and payload.get('scaler') is not None:
                scaler.load_state_dict(payload['scaler'])
            restore_rng_state(payload)
        print(f'{"resume" if resume else "finetune"} ckpt from {restore_ckpt}')

    # ── 数据 ────────────────────────────────────────────────────────────────
    lists_dir = d['lists_dir']
    lists = resolve_train_lists(
        lists_dir, C['phases'], tiers=d.get('tiers'))
    crop_sizes = [tuple(c) for c in d['crop_sizes']]
    if d.get('mv_cache_dirname'):
        print('[data] teacher MV cache: '
              f'{d["mv_cache_dirname"]} (mmap crop, '
              f'required={bool(d.get("mv_cache_required", False))}, '
              f'preview_stride={int(d.get("mv_cache_preview_stride", 4))}, '
              f'cycle={d.get("mv_cycle_confidence", "none")}, '
              f'cycle_required={bool(d.get("mv_cycle_cache_required", False))})')

    train_set = MixedTierDataset(
        d['root'], lists,
        ratios=C['phases'][0]['ratios'],
        source_options=d.get('source_options'),
        crop_hw=crop_sizes[0],
        framesteps=tuple(d['framesteps']),
        t_half_prob=d['t_half_prob'],
        mv_prob=d['mv_prob'],
        mv_sign=tuple(d['mv_sign']),
        mv_symmetry_confidence=d.get('mv_symmetry_confidence', False),
        occ_alpha=d.get('occ_alpha', 0.05),
        occ_beta=d.get('occ_beta', 1.0),
        motion_aware_crop_prob=d.get('motion_aware_crop_prob', 0.0),
        motion_crop_threshold=d.get('motion_crop_threshold', 1.0),
        small_motion_min_pixels=d.get('small_motion_min_pixels', 8),
        small_motion_max_ratio=d.get('small_motion_max_ratio', 0.05),
        motion_crop_jitter=d.get('motion_crop_jitter', 0.2),
        mv_cache_dirname=d.get('mv_cache_dirname'),
        mv_cache_required=d.get('mv_cache_required', False),
        mv_cache_preview_stride=d.get('mv_cache_preview_stride', 4),
        mv_cycle_confidence=d.get('mv_cycle_confidence', 'none'),
        mv_cycle_cache_required=d.get('mv_cycle_cache_required', False),
        mv_cycle_cache_root=d.get('mv_cycle_cache_root'),
        mv_cycle_on_the_fly=d.get('mv_cycle_on_the_fly', False),
        augment_profile=d.get('augment_profile', 'legacy'),
    )
    val_loaders = build_val_loaders(d, lists_dir)

    # 名义 step/epoch: 固定值保证LR调度可预期 (Mixed数据集名义长度过大, 不直接用)
    default_steps_per_epoch = max(
        1, C.get('steps_per_epoch',
                 min(len(train_set), 2000 * d['batch_size']))
        // d['batch_size'])
    total_steps = sum(
        int(phase['epochs'])
        * int(phase.get('steps_per_epoch', default_steps_per_epoch))
        for phase in C['phases'])
    phase_summary = [
        (phase['name'], phase['epochs'],
         phase.get('steps_per_epoch', default_steps_per_epoch),
         phase.get('grad_accum_steps', opt.get('grad_accum_steps', 1)))
        for phase in C['phases']]
    print(f'training... phases(name,epochs,steps,accum)={phase_summary} '
          f'total_optimizer_steps={total_steps}')

    step = restored_step if resume else 0
    nr_eval = start_epoch if resume else 0
    loss_ema = None
    best_metric_name = mon.get('best_metric', 'all/psnr')
    _, best_mode = metric_improved(0.0, 0.0, best_metric_name, mon)
    initial_best = float('inf') if best_mode == 'min' else -float('inf')
    saved_best = (
        model.loaded_checkpoint.get('best_metric_value')
        if resume and isinstance(model.loaded_checkpoint, dict) else None)
    best_metric_value = (
        initial_best if saved_best is None else float(saved_best))
    epoch_global = 0
    time_stamp = time.time()

    for phase in C['phases']:
        local_enabled = bool(phase.get('local', True))
        model.set_local_enabled(local_enabled)
        model.set_trainable_scope(phase.get('trainable', 'all'))
        phase_steps_per_epoch = int(
            phase.get('steps_per_epoch', default_steps_per_epoch))
        if phase_steps_per_epoch <= 0:
            raise ValueError(f'{phase["name"]}.steps_per_epoch必须为正整数')
        grad_accum_steps = int(
            phase.get('grad_accum_steps', opt.get('grad_accum_steps', 1)))
        if grad_accum_steps <= 0:
            raise ValueError(f'{phase["name"]}.grad_accum_steps必须为正整数')
        phase_crop_sizes = [
            tuple(value) for value in phase.get('crop_sizes', crop_sizes)]
        phase_opt = dict(opt)
        phase_opt.update(phase.get('optim', {}))
        phase_lr_schedule = str(
            phase.get('lr_schedule', 'global')).lower()
        if phase_lr_schedule not in ('global', 'phase'):
            raise ValueError(
                f'{phase["name"]}.lr_schedule只支持global或phase')
        default_phase_lr_steps = int(phase['epochs']) * phase_steps_per_epoch
        phase_lr_total_steps = int(
            phase.get('lr_total_steps', default_phase_lr_steps))
        if phase_lr_total_steps <= 0:
            raise ValueError(f'{phase["name"]}.lr_total_steps必须为正整数')
        print(f'\n════════ Phase [{phase["name"]}] {phase["epochs"]} epochs '
              f'local={"on" if local_enabled else "off/frozen"} '
              f'trainable={phase.get("trainable", "all")} '
              f'accum={grad_accum_steps} lr_schedule={phase_lr_schedule} '
              f'lr_steps={phase_lr_total_steps if phase_lr_schedule == "phase" else total_steps} '
              f'════════')
        train_set.set_ratios(
            phase['ratios'], batch_counts=phase.get('batch_counts'))

        for phase_epoch in range(phase['epochs']):
            if resume and epoch_global < start_epoch:
                epoch_global += 1
                continue
            curriculum_state = apply_motion_curriculum(
                train_set, phase.get('curriculum'), phase_epoch)
            if (curriculum_state is not None
                    and phase_epoch % int(
                        phase['curriculum'].get('every_epochs', 50)) == 0):
                print('[curriculum] '
                      f'phase_epoch={phase_epoch} '
                      f'stage={curriculum_state["stage"]} '
                      f'{curriculum_state["source"]}: '
                      f'framestep={curriculum_state["framestep"]} '
                      f'resize={curriculum_state["resize"]}x'
                      f'{curriculum_state["resize"]}')
            # 多尺度: 每epoch换一次crop尺寸 (worker副本重建时生效)
            sel = random.choice(phase_crop_sizes)          # (h, w, bs)
            if len(sel) != 3 or min(sel) <= 0:
                raise ValueError(
                    f'{phase["name"]}.crop_sizes条目必须为(h,w,batch): {sel}')
            pattern_size = train_set.batch_pattern_size
            if pattern_size is not None and pattern_size != sel[2]:
                raise ValueError(
                    f'{phase["name"]}.batch_counts总数({pattern_size}) '
                    f'必须等于batch size({sel[2]})')
            train_set.set_crop_size((sel[0], sel[1]))
            loader_workers = int(d['num_workers'])
            loader_prefetch = int(d.get('prefetch_factor', 2))
            loader_in_order = bool(d.get('in_order', True))
            if loader_workers <= 0:
                loader_prefetch = None
            train_loader = DataLoader(train_set, batch_size=sel[2],
                                      num_workers=loader_workers, pin_memory=True,
                                      drop_last=True, shuffle=False,
                                      worker_init_fn=seed_worker,
                                      prefetch_factor=loader_prefetch,
                                      in_order=loader_in_order)
            it = iter(train_loader)

            for i in range(phase_steps_per_epoch):
                schedule_step = (
                    phase_epoch * phase_steps_per_epoch + i
                    if phase_lr_schedule == 'phase' else step)
                schedule_total = (
                    phase_lr_total_steps
                    if phase_lr_schedule == 'phase' else total_steps)
                # A fixed phase lr_total_steps preserves a short controlled
                # schedule while permitting a longer low-LR weekend extension.
                schedule_step = min(schedule_step, schedule_total)
                lr = get_learning_rate(
                    schedule_step, schedule_total, phase_opt)

                data_time = 0.0
                train_time = 0.0
                loss_sum = 0.0
                flow_loss_sum = 0.0
                for accumulation_index in range(grad_accum_steps):
                    try:
                        frames, timestep, flow_gt, has_mv = next(it)
                    except StopIteration:
                        it = iter(train_loader)
                        frames, timestep, flow_gt, has_mv = next(it)

                    data_time += time.time() - time_stamp
                    time_stamp = time.time()

                    frames = frames.to(device, non_blocking=True).float() / 255.
                    timestep = timestep.to(device, non_blocking=True)
                    imgs, gt = frames[:, :6], frames[:, 6:9]

                    with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                        pred, micro_loss, micro_flow_loss = model.update(
                            imgs, gt, timestep=timestep, learning_rate=lr,
                            training=True,
                            scaler=scaler if use_scaler else None,
                            flow_gt=flow_gt, has_mv=has_mv, loss_step=step,
                            accumulation_steps=grad_accum_steps,
                            accumulation_index=accumulation_index)
                    loss_sum += micro_loss
                    flow_loss_sum += micro_flow_loss
                    train_time += time.time() - time_stamp
                    time_stamp = time.time()

                loss = loss_sum / grad_accum_steps
                loss_flow = flow_loss_sum / grad_accum_steps

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
                    for name, value in model.last_loss_components.items():
                        writer.add_scalar(
                            f'loss_component/{name}', value.item(), step)
                    if model.last_flow_stage_losses is not None:
                        for si, stage_loss in enumerate(
                                model.last_flow_stage_losses.cpu().tolist()):
                            writer.add_scalar(f'loss/flow_stage_{si}', stage_loss, step)
                    writer.add_scalar('train/lr', lr, step)
                    if model.last_grad_norm is not None:
                        writer.add_scalar(
                            'train/grad_norm',
                            model.last_grad_norm.item(), step)
                    if use_scaler:
                        writer.add_scalar('amp/scale', scaler.get_scale(), step)

                print_every = max(int(mon.get('print_every_steps', 1)), 1)
                if i % print_every == 0 or i + 1 == phase_steps_per_epoch:
                    print(f'[{phase["name"]}] epoch:{epoch_global} '
                          f'{i}/{phase_steps_per_epoch} '
                          f'time:{data_time:.2f}+{train_time:.2f} '
                          f'loss:{loss:.4f} ema:{loss_ema:.4f} '
                          f'flow:{loss_flow:.3f} lr:{lr:.2e}')
                step += 1

            epoch_global += 1
            nr_eval += 1
            if nr_eval % mon['eval_every_epochs'] == 0:
                metrics = evaluate(
                    model, val_loaders, nr_eval, writer, use_amp, amp_dtype)
                metric_value = metrics.get(best_metric_name)
                if metric_value is None:
                    print(f'[best] WARNING: 指标 {best_metric_name} 不存在，'
                          f'可用指标={sorted(metrics)}')
                elif metric_improved(
                        metric_value, best_metric_value,
                        best_metric_name, mon)[0]:
                    best_metric_value = metric_value
                    model.save_model(
                        exp, epoch_global, step=step, scaler=scaler,
                        tag='best', extra_state={
                            'best_metric_name': best_metric_name,
                            'best_metric_value': best_metric_value,
                            **capture_rng_state(),
                        })
                    print(f'[best] {best_metric_name}={best_metric_value:.6f}')
            dump_mv_comparisons(
                model, val_loaders, nr_eval, mv_dumper, use_amp, amp_dtype)
            if epoch_global % mon['save_every_epochs'] == 0:
                model.save_model(
                    exp, epoch_global, step=step, scaler=scaler,
                    extra_state={
                        'best_metric_name': best_metric_name,
                        'best_metric_value': best_metric_value,
                        **capture_rng_state(),
                    })

    model.save_model(
        exp, epoch_global, step=step, scaler=scaler,
        extra_state={
            'best_metric_name': best_metric_name,
            'best_metric_value': best_metric_value,
            **capture_rng_state(),
        })
    writer.close()
    print('════════ 训练完成 ════════')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str,
                        help='yaml配置文件 (全部超参来源)')
    parser.add_argument('--restore_ckpt', type=str, default=None)
    parser.add_argument(
        '--resume', action='store_true',
        help='恢复optimizer/epoch/step/RNG；普通finetune不要传此参数')
    parser.add_argument(
        '--loader_workers', type=int, default=None,
        help='仅覆盖本次运行的DataLoader worker数，不改变/归档yaml')
    parser.add_argument(
        '--loader_prefetch', type=int, default=None,
        help='仅覆盖本次运行每个worker的预取batch数')
    parser.add_argument(
        '--loader_out_of_order', action='store_true',
        help='允许先返回已完成batch，避免单个慢PNG读取阻塞整个DataLoader')
    args = parser.parse_args()
    if args.resume and not args.restore_ckpt:
        parser.error('--resume requires --restore_ckpt')

    C = load_config(args.config)
    data_cfg = C['data']
    if args.loader_workers is not None:
        if args.loader_workers < 0:
            parser.error('--loader_workers必须>=0')
        data_cfg['num_workers'] = args.loader_workers
    if args.loader_prefetch is not None:
        if args.loader_prefetch <= 0:
            parser.error('--loader_prefetch必须为正整数')
        data_cfg['prefetch_factor'] = args.loader_prefetch
    if args.loader_out_of_order:
        data_cfg['in_order'] = False
    if (args.loader_workers is not None or args.loader_prefetch is not None
            or args.loader_out_of_order):
        print('[loader override] '
              f'workers={data_cfg["num_workers"]} '
              f'prefetch={data_cfg.get("prefetch_factor", 2)} '
              f'in_order={data_cfg.get("in_order", True)}')

    seed = C.get('seed', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs('log', exist_ok=True)
    train(
        C, restore_ckpt=args.restore_ckpt,
        config_path=args.config, resume=args.resume)
