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
                'static_sum', 'static_count')}

        for frames, timestep, flow_gt, has_mv in val_loader:
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

            if flow_list and has_mv.sum().item() > 0:
                flow_gt = flow_gt.to(device, non_blocking=True)
                has_mv = has_mv.to(device, non_blocking=True)
                final_flow = model.unpad(
                    flow_list[-1], pad_right, pad_bottom)
                sums = model.flow_metric_sums(final_flow, flow_gt, has_mv)
                for key, value in sums.items():
                    flow_totals[key] += value.item()

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
            print(message)
    model.train()
    return metrics


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
    }
    loaders = {}
    for name, value in configured.items():
        path = value if os.path.isabs(value) else os.path.join(lists_dir, value)
        if not os.path.isfile(path):
            if name == 'all':
                raise FileNotFoundError(f'验证清单不存在: {path}')
            print(f'[eval] WARNING: 验证清单不存在，跳过 {name}: {path}')
            continue
        dataset = TierDataset(
            data_cfg['root'], path, split='val',
            val_with_mv=(name == 'teacher'), **common)
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
        0, loss_type=m['loss_type'],
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

    train_set = MixedTierDataset(
        d['root'], lists,
        ratios=C['phases'][0]['ratios'],
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
        augment_profile=d.get('augment_profile', 'legacy'),
    )
    val_loaders = build_val_loaders(d, lists_dir)

    total_epochs = sum(p['epochs'] for p in C['phases'])
    # 名义 step/epoch: 固定值保证LR调度可预期 (Mixed数据集名义长度过大, 不直接用)
    steps_per_epoch = max(
        1, C.get('steps_per_epoch',
                 min(len(train_set), 2000 * d['batch_size']))
        // d['batch_size'])
    total_steps = total_epochs * steps_per_epoch
    print(f'training... phases={[(p["name"], p["epochs"]) for p in C["phases"]]} '
          f'steps/epoch={steps_per_epoch} total_steps={total_steps}')

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
        print(f'\n════════ Phase [{phase["name"]}] {phase["epochs"]} epochs '
              f'local={"on" if local_enabled else "off/frozen"} ════════')
        train_set.set_ratios(phase['ratios'])

        for _ in range(phase['epochs']):
            if resume and epoch_global < start_epoch:
                epoch_global += 1
                continue
            # 多尺度: 每epoch换一次crop尺寸 (worker副本重建时生效)
            sel = random.choice(crop_sizes)          # (h, w, bs)
            train_set.set_crop_size((sel[0], sel[1]))
            train_loader = DataLoader(train_set, batch_size=sel[2],
                                    num_workers=d['num_workers'], pin_memory=True,
                                    drop_last=True, shuffle=False,
                                    worker_init_fn=seed_worker)
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
                        flow_gt=flow_gt, has_mv=has_mv, loss_step=step)

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

                print(f'[{phase["name"]}] epoch:{epoch_global} {i}/{steps_per_epoch} '
                      f'time:{data_time:.2f}+{train_time:.2f} '
                      f'loss:{loss:.4f} ema:{loss_ema:.4f} flow:{loss_flow:.3f}')
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
                        exp, epoch_global, 0, step=step, scaler=scaler,
                        tag='best', extra_state={
                            'best_metric_name': best_metric_name,
                            'best_metric_value': best_metric_value,
                            **capture_rng_state(),
                        })
                    print(f'[best] {best_metric_name}={best_metric_value:.6f}')
            if epoch_global % mon['save_every_epochs'] == 0:
                model.save_model(
                    exp, epoch_global, 0, step=step, scaler=scaler,
                    extra_state={
                        'best_metric_name': best_metric_name,
                        'best_metric_value': best_metric_value,
                        **capture_rng_state(),
                    })

    model.save_model(
        exp, epoch_global, 0, step=step, scaler=scaler,
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
    args = parser.parse_args()
    if args.resume and not args.restore_ckpt:
        parser.error('--resume requires --restore_ckpt')

    C = load_config(args.config)

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
