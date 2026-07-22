"""
VFIMamba 独立多机 DDP 训练入口。

不修改原 train.py 的单机语义。每台机器通过 torchrun 启动一个进程，
使用 NCCL 同步梯度。支持 BF16/FP16、梯度累积、rank 统一 crop、
分布式验证，以及 rank0-only 日志/checkpoint/dump。
"""

import argparse
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_distributed(backend='nccl', timeout_minutes=30):
    if not torch.cuda.is_available():
        raise RuntimeError('train_ddp.py 需要 CUDA/NCCL; 未检测到 CUDA')

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    distributed = world_size > 1
    if distributed:
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
        dist.init_process_group(
            backend=backend, init_method='env://',
            timeout=timedelta(minutes=timeout_minutes))
    return rank, local_rank, world_size, distributed, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(distributed):
    if distributed:
        dist.barrier()


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader_worker_options(num_workers, data_cfg):
    options = {}
    if num_workers > 0:
        options['persistent_workers'] = bool(data_cfg.get('persistent_workers', True))
        options['prefetch_factor'] = int(data_cfg.get('prefetch_factor', 2))
    return options


def get_learning_rate(step, total_steps, opt, schedule_divisor=1):
    """warmup + cosine; 保持与单机入口一致的 finetune 1/4 LR 语义。"""
    warmup = int(math.ceil(opt['warmup_steps'] / max(schedule_divisor, 1)))
    lr_max, lr_min = float(opt['lr_max']), float(opt['lr_min'])
    if opt.get('finetune', False):
        lr_max *= 0.25
        lr_min *= 0.25
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine = math.cos(math.pi * min(max(progress, 0.0), 1.0)) * 0.5 + 0.5
    return lr_min + (lr_max - lr_min) * cosine


class AnomalyDumper:
    """rank0 异常batch落盘。"""

    def __init__(self, dump_dir='anomaly_dumps', max_dumps=500):
        self.dump_dir = dump_dir
        self.max_dumps = max_dumps
        self.count = 0
        os.makedirs(dump_dir, exist_ok=True)

    def dump(self, step, reason, frames, timestep, flow_gt, has_mv,
             pred=None, loss=None, loss_flow=None, flow_stage_losses=None):
        if self.count >= self.max_dumps:
            return
        self.count += 1
        output_dir = os.path.join(self.dump_dir, f'step{step:07d}_{reason}')
        os.makedirs(output_dir, exist_ok=True)

        import cv2
        frames_np = (frames.detach().cpu().clamp(0, 1) * 255).byte().numpy()
        pred_np = ((pred.detach().cpu().clamp(0, 1) * 255).byte().numpy()
                   if pred is not None else None)
        t_np = timestep.detach().cpu().numpy().reshape(-1)
        has_mv_np = has_mv.detach().cpu().numpy().reshape(-1)
        meta = [f'step={step} reason={reason} loss={loss} loss_flow={loss_flow}']
        if flow_stage_losses is not None:
            values = flow_stage_losses.detach().cpu().tolist()
            meta.append('flow_stage_epe=' + ','.join(f'{v:.6f}' for v in values))

        for batch_index in range(frames_np.shape[0]):
            for name, channel_slice in (
                    ('img0', slice(0, 3)), ('img1', slice(3, 6)),
                    ('gt', slice(6, 9))):
                image = frames_np[batch_index, channel_slice].transpose(1, 2, 0)[..., ::-1]
                cv2.imwrite(os.path.join(output_dir, f'b{batch_index}_{name}.png'), image)
            if pred_np is not None:
                cv2.imwrite(os.path.join(output_dir, f'b{batch_index}_pred.png'),
                            pred_np[batch_index].transpose(1, 2, 0)[..., ::-1])
            if has_mv_np[batch_index] > 0:
                np.save(os.path.join(output_dir, f'b{batch_index}_flow_gt.npy'),
                        flow_gt[batch_index].detach().cpu().numpy())
            meta.append(
                f'b{batch_index}: t={t_np[batch_index]:.4f} '
                f'has_mv={int(has_mv_np[batch_index])}')
        with open(os.path.join(output_dir, 'meta.txt'), 'w') as f:
            f.write('\n'.join(meta) + '\n')


def unwrap_network(network):
    return network.module if isinstance(network, DDP) else network


def save_checkpoint(model, checkpoint_dir, epoch, step, scaler=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'{model.name}_{epoch}.pkl')
    payload = {
        'net': unwrap_network(model.net).state_dict(),
        'optim': model.optimG.state_dict(),
        'epoch': epoch,
        'step': step,
    }
    if scaler is not None and scaler.is_enabled():
        payload['scaler'] = scaler.state_dict()
    torch.save(payload, path)
    print(f'[checkpoint] {path}')


def choose_crop(crop_sizes, rank, device, distributed):
    crop_index = random.randrange(len(crop_sizes)) if rank == 0 else 0
    crop_index_tensor = torch.tensor(crop_index, device=device, dtype=torch.long)
    if distributed:
        dist.broadcast(crop_index_tensor, src=0)
    return crop_sizes[int(crop_index_tensor.item())]


def build_train_loader(train_set, batch_size, data_cfg, rank, epoch, base_seed):
    # MixedTierDataset.__getitem__ 内部已按tier配比二次随机采样,
    # 传入index并不决定最终样本。因此不使用DistributedSampler/randperm,
    # 避免对巨大名义Dataset生成无意义的全量索引排列。
    generator = torch.Generator()
    generator.manual_seed(base_seed + rank * 100003 + epoch)
    num_workers = int(data_cfg['num_workers'])
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_worker_options(num_workers, data_cfg),
    )


def build_val_loader(val_set, data_cfg, rank, world_size, distributed, base_seed):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank,
            shuffle=False, seed=base_seed, drop_last=False)
    num_workers = int(data_cfg.get('val_num_workers', max(data_cfg['num_workers'] // 2, 1)))
    return DataLoader(
        val_set,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        **loader_worker_options(num_workers, data_cfg),
    )


def next_batch(loader, iterator):
    try:
        return iterator, next(iterator)
    except StopIteration:
        iterator = iter(loader)
        return iterator, next(iterator)


def compute_training_loss(model, imgs, gt, timestep, flow_gt, has_mv):
    """Trainer.update 的纯 forward/loss 部分，使 DDP 可控制 no_sync 与梯度累积。"""
    imgs_pad, (pad_right, pad_bottom) = model.pad_to_multiple(imgs, 16)
    flow_list, _, _, _, _, merged, pred_pad = model.net(
        imgs_pad, timestep=timestep, scale=0, local=model.local)
    pred = model.unpad(pred_pad, pad_right, pad_bottom)

    loss = model._pixel_loss(pred, gt)
    for merge in merged:
        loss = loss + model._pixel_loss(
            model.unpad(merge, pad_right, pad_bottom), gt)

    loss_flow = torch.zeros((), device=imgs.device)
    model.last_flow_stage_losses = None
    if (model.flow_loss_weight > 0 and has_mv.sum() > 0):
        loss_flow = model._multistage_flow_loss(
            flow_list, flow_gt, has_mv, pad_right, pad_bottom)
        loss = loss + model.flow_loss_weight * loss_flow
    return pred.detach(), loss, loss_flow


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp, amp_dtype,
             world_size, distributed, rank, writer, epoch):
    model.eval()
    network = unwrap_network(model.net)
    psnr_sum = torch.zeros((), device=device)
    sample_count = torch.zeros((), device=device)
    for frames, timestep, _, _ in val_loader:
        frames = frames.to(device, non_blocking=True).float() / 255.0
        timestep = timestep.to(device, non_blocking=True)
        imgs, gt = frames[:, :6], frames[:, 6:9]
        imgs_pad, (pad_right, pad_bottom) = model.pad_to_multiple(imgs, 16)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            _, _, _, _, _, _, pred_pad = network(
                imgs_pad, timestep=timestep, scale=0, local=model.local)
        pred = model.unpad(pred_pad, pad_right, pad_bottom)
        mse = (gt - pred).square().mean(dim=(1, 2, 3))
        valid = mse > 0
        psnr_sum += (-10.0 * torch.log10(mse[valid])).sum()
        sample_count += valid.sum()

    if distributed:
        dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    if rank == 0 and sample_count.item() > 0:
        psnr = (psnr_sum / sample_count).item()
        print(f'[eval epoch={epoch}] PSNR={psnr:.4f}')
        writer.add_scalar('val/psnr', psnr, epoch)
    model.train()


def run_training(config, args, rank, local_rank, world_size, distributed, device):
    # 项目模块必须在 set_device 之后导入，避免模块级 cuda:0 tensor。
    import config as model_config
    from Trainer import Model
    from kousei_dataset import (
        MixedTierDataset, TierDataset, resolve_train_lists)

    is_main = rank == 0
    exp_name = config['exp_name']
    data_cfg = config['data']
    opt_cfg = config['optim']
    monitor_cfg = config['monitor']
    ddp_cfg = config.get('distributed', {})

    seed = int(config.get('seed', 1234)) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(opt_cfg.get('cudnn_benchmark', True))
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(opt_cfg.get('matmul_precision', 'high'))

    script_dir = Path(__file__).resolve().parent
    checkpoint_dir = script_dir / 'ckpt' / exp_name
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, checkpoint_dir / 'model.yaml')
        writer = SummaryWriter(f'log/train_{exp_name}_ddp')
        dumper = AnomalyDumper(
            dump_dir=monitor_cfg.get('dump_dir', 'anomaly_dumps'),
            max_dumps=monitor_cfg.get('dump_max', 50))
    else:
        writer = None
        dumper = None
    barrier(distributed)

    model_section = config['model']
    model_config.MODEL_CONFIG['LOGNAME'] = exp_name
    train_only_keys = (
        'loss_type', 'flow_loss_weight', 'flow_stage_gamma',
        'flow_motion_threshold', 'flow_motion_balance', 'flow_motion_gain',
        'flow_motion_scale', 'flow_motion_weight_cap', 'flow_charbonnier_eps',
    )
    extra = {
        key: value for key, value in model_section.items()
        if key not in ('F', 'depth', 'M', 'version') + train_only_keys
    }
    model_config.MODEL_CONFIG['MODEL_ARCH'] = model_config.init_model_config(
        F=model_section['F'], depth=model_section['depth'],
        M=model_section.get('M', False), version=model_section['version'],
        **extra)
    model = Model(
        local_rank,
        loss_type=model_section['loss_type'],
        flow_loss_weight=model_section.get('flow_loss_weight', 0.0),
        flow_stage_gamma=model_section.get('flow_stage_gamma', 0.8),
        flow_motion_threshold=model_section.get('flow_motion_threshold', 1.0),
        flow_motion_balance=model_section.get('flow_motion_balance', 0.5),
        flow_motion_gain=model_section.get('flow_motion_gain', 1.0),
        flow_motion_scale=model_section.get('flow_motion_scale', 10.0),
        flow_motion_weight_cap=model_section.get('flow_motion_weight_cap', 4.0),
        flow_charbonnier_eps=model_section.get('flow_charbonnier_eps', 1e-3))

    start_epoch, global_step = 0, 0
    resume_scaler_state = None
    if args.restore_ckpt:
        start_epoch, restored_step = model.load_model(
            args.restore_ckpt, resume=args.resume)
        global_step = restored_step if args.resume else 0
        if args.resume:
            resume_payload = torch.load(args.restore_ckpt, map_location='cpu')
            if isinstance(resume_payload, dict):
                resume_scaler_state = resume_payload.get('scaler')
        if is_main:
            mode = 'resume' if args.resume else 'finetune'
            print(f'[{mode}] loaded {args.restore_ckpt}')

    if distributed:
        model.net = DDP(
            model.net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=bool(ddp_cfg.get('broadcast_buffers', False)),
            find_unused_parameters=bool(ddp_cfg.get('find_unused_parameters', False)),
            gradient_as_bucket_view=bool(ddp_cfg.get('gradient_as_bucket_view', True)),
            static_graph=bool(ddp_cfg.get('static_graph', False)),
            bucket_cap_mb=int(ddp_cfg.get('bucket_cap_mb', 25)),
        )
        if ddp_cfg.get('fp16_compress_hook', False):
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.net.register_comm_hook(
                state=None, hook=default_hooks.fp16_compress_hook)
            if is_main:
                print('[DDP] FP16 gradient communication compression enabled')

    amp_name = str(opt_cfg.get('amp_dtype', 'bf16')).lower()
    if amp_name not in ('bf16', 'fp16'):
        raise ValueError(f'amp_dtype must be bf16 or fp16, got {amp_name}')
    amp_dtype = torch.bfloat16 if amp_name == 'bf16' else torch.float16
    use_amp = bool(opt_cfg.get('amp', True))
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    if use_scaler and resume_scaler_state is not None:
        scaler.load_state_dict(resume_scaler_state)

    lists_dir = data_cfg['lists_dir']
    lists = resolve_train_lists(
        lists_dir, config['phases'], tiers=data_cfg.get('tiers'))
    crop_sizes = [tuple(item) for item in data_cfg['crop_sizes']]
    train_set = MixedTierDataset(
        data_cfg['root'], lists,
        ratios=config['phases'][0]['ratios'],
        crop_hw=(crop_sizes[0][0], crop_sizes[0][1]),
        framesteps=tuple(data_cfg['framesteps']),
        t_half_prob=data_cfg['t_half_prob'],
        mv_prob=data_cfg['mv_prob'],
        mv_sign=tuple(data_cfg['mv_sign']),
        motion_aware_crop_prob=data_cfg.get('motion_aware_crop_prob', 0.0),
        motion_crop_threshold=data_cfg.get('motion_crop_threshold', 1.0),
        small_motion_min_pixels=data_cfg.get('small_motion_min_pixels', 8),
        small_motion_max_ratio=data_cfg.get('small_motion_max_ratio', 0.05),
        motion_crop_jitter=data_cfg.get('motion_crop_jitter', 0.2),
    )
    val_set = TierDataset(
        data_cfg['root'], os.path.join(lists_dir, 'val.txt'), split='val')
    val_loader = build_val_loader(
        val_set, data_cfg, rank, world_size, distributed, seed)

    accumulation_steps = max(int(opt_cfg.get('grad_accum_steps', 1)), 1)
    preserve_samples = bool(ddp_cfg.get('preserve_single_node_samples_per_epoch', True))
    schedule_divisor = world_size * accumulation_steps if preserve_samples else 1
    base_steps_per_epoch = int(config.get(
        'steps_per_epoch',
        min(len(train_set), 2000 * data_cfg['batch_size']) // data_cfg['batch_size']))
    steps_per_epoch = max(1, math.ceil(base_steps_per_epoch / schedule_divisor))
    total_epochs = sum(phase['epochs'] for phase in config['phases'])
    total_steps = total_epochs * steps_per_epoch

    if is_main:
        print(
            f'[DDP] world_size={world_size} backend={ddp_cfg.get("backend", "nccl")} '
            f'amp={amp_name if use_amp else "off"} accum={accumulation_steps}')
        print(
            f'[schedule] base_steps/epoch={base_steps_per_epoch} '
            f'ddp_steps/epoch={steps_per_epoch} total_steps={total_steps} '
            f'preserve_samples={preserve_samples}')

    flow_dump_threshold = monitor_cfg.get('flow_loss_dump_threshold', 30.0)
    print_every = max(int(monitor_cfg.get('print_every_steps', 10)), 1)
    log_every = max(int(monitor_cfg['log_every_steps']), 1)
    grad_clip = float(opt_cfg.get('grad_clip', 1.0))
    epoch_global = 0
    loss_ema = None

    for phase in config['phases']:
        train_set.set_ratios(phase['ratios'])
        if is_main:
            print(f'\n=== phase={phase["name"]} epochs={phase["epochs"]} ===')

        for _ in range(phase['epochs']):
            if args.resume and epoch_global < start_epoch:
                epoch_global += 1
                continue

            crop_h, crop_w, per_rank_batch = choose_crop(
                crop_sizes, rank, device, distributed)
            train_set.set_crop_size((crop_h, crop_w))
            train_loader = build_train_loader(
                train_set, per_rank_batch, data_cfg, rank,
                epoch_global, int(config.get('seed', 1234)))
            iterator = iter(train_loader)

            for step_in_epoch in range(steps_per_epoch):
                step_start = time.time()
                model.train()
                model.optimG.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device)
                flow_loss_sum = torch.zeros((), device=device)
                last_batch = None

                lr = get_learning_rate(
                    global_step, total_steps, opt_cfg, schedule_divisor)
                for param_group in model.optimG.param_groups:
                    param_group['lr'] = lr

                for micro_step in range(accumulation_steps):
                    iterator, batch = next_batch(train_loader, iterator)
                    frames, timestep, flow_gt, has_mv = batch
                    frames = frames.to(device, non_blocking=True).float() / 255.0
                    timestep = timestep.to(device, non_blocking=True)
                    flow_gt = flow_gt.to(device, non_blocking=True)
                    has_mv = has_mv.to(device, non_blocking=True)
                    imgs, gt = frames[:, :6], frames[:, 6:9]

                    should_sync = micro_step == accumulation_steps - 1
                    sync_context = (
                        nullcontext() if should_sync or not distributed
                        else model.net.no_sync())
                    with sync_context:
                        with torch.autocast(
                                'cuda', dtype=amp_dtype, enabled=use_amp):
                            pred, loss, loss_flow = compute_training_loss(
                                model, imgs, gt, timestep, flow_gt, has_mv)
                            backward_loss = loss / accumulation_steps
                        if use_scaler:
                            scaler.scale(backward_loss).backward()
                        else:
                            backward_loss.backward()

                    loss_sum += loss.detach()
                    flow_loss_sum += loss_flow.detach()
                    last_batch = (
                        frames, timestep, flow_gt, has_mv, pred,
                        loss.detach(), loss_flow.detach())

                if use_scaler:
                    scaler.unscale_(model.optimG)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.net.parameters(), max_norm=grad_clip)
                if use_scaler:
                    scaler.step(model.optimG)
                    scaler.update()
                else:
                    model.optimG.step()

                # 合并为一次all-reduce, 减少局域网小包同步延迟。
                metrics = torch.stack([
                    (loss_sum / accumulation_steps).float(),
                    (flow_loss_sum / accumulation_steps).float(),
                    grad_norm.detach().float(),
                ])
                if distributed:
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    metrics /= world_size
                mean_loss, mean_flow_loss, mean_grad_norm = metrics
                loss_value = mean_loss.item()
                flow_loss_value = mean_flow_loss.item()
                loss_ema = (loss_value if loss_ema is None
                            else 0.98 * loss_ema + 0.02 * loss_value)

                mean_stage_losses = None
                if global_step % log_every == 0:
                    network = unwrap_network(model.net)
                    stage_count = network.flow_num_stage + network.local_num
                    stage_payload = torch.zeros(stage_count + 1, device=device)
                    if model.last_flow_stage_losses is not None:
                        count = min(stage_count, len(model.last_flow_stage_losses))
                        stage_payload[:count] = model.last_flow_stage_losses[:count].to(device)
                        stage_payload[-1] = 1.0
                    if distributed:
                        dist.all_reduce(stage_payload, op=dist.ReduceOp.SUM)
                    if stage_payload[-1].item() > 0:
                        mean_stage_losses = stage_payload[:-1] / stage_payload[-1]

                if is_main:
                    if (global_step > 40000
                            and flow_loss_value > flow_dump_threshold):
                        frames, timestep, flow_gt, has_mv, pred, loss, loss_flow = last_batch
                        # 全局flow loss可能来自其他rank; rank0当前batch无MV时不误dump。
                        if has_mv.sum().item() > 0:
                            dumper.dump(
                                global_step, f'flowloss{flow_loss_value:.0f}',
                                frames, timestep, flow_gt, has_mv, pred=pred,
                                loss=loss.item(), loss_flow=loss_flow.item(),
                                flow_stage_losses=model.last_flow_stage_losses)

                    if global_step % log_every == 0:
                        writer.add_scalar('loss/raw', loss_value, global_step)
                        writer.add_scalar('loss/ema', loss_ema, global_step)
                        writer.add_scalar('loss/flow_epe', flow_loss_value, global_step)
                        writer.add_scalar('train/lr', lr, global_step)
                        writer.add_scalar('train/grad_norm', mean_grad_norm.item(), global_step)
                        writer.add_scalar('train/crop_height', crop_h, global_step)
                        writer.add_scalar('train/crop_width', crop_w, global_step)
                        writer.add_scalar(
                            'train/global_batch',
                            per_rank_batch * world_size * accumulation_steps,
                            global_step)
                        if mean_stage_losses is not None:
                            for stage_index, stage_loss in enumerate(
                                    mean_stage_losses.tolist()):
                                writer.add_scalar(
                                    f'loss/flow_stage_{stage_index}',
                                    stage_loss, global_step)
                        if use_scaler:
                            writer.add_scalar('amp/scale', scaler.get_scale(), global_step)

                    if global_step % print_every == 0:
                        elapsed = time.time() - step_start
                        print(
                            f'[{phase["name"]}] epoch={epoch_global} '
                            f'{step_in_epoch}/{steps_per_epoch} crop={crop_h}x{crop_w} '
                            f'global_bs={per_rank_batch * world_size * accumulation_steps} '
                            f'time={elapsed:.2f}s loss={loss_value:.4f} '
                            f'ema={loss_ema:.4f} flow={flow_loss_value:.3f}')
                global_step += 1

            epoch_global += 1
            if epoch_global % monitor_cfg['eval_every_epochs'] == 0:
                evaluate(
                    model, val_loader, device, use_amp, amp_dtype,
                    world_size, distributed, rank, writer,
                    epoch_global)
            if epoch_global % monitor_cfg['save_every_epochs'] == 0:
                barrier(distributed)
                if is_main:
                    save_checkpoint(
                        model, checkpoint_dir, epoch_global, global_step, scaler)
                barrier(distributed)

    barrier(distributed)
    if is_main:
        save_checkpoint(model, checkpoint_dir, epoch_global, global_step, scaler)
        writer.close()
        print('=== DDP training complete ===')
    barrier(distributed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--restore_ckpt', default=None)
    parser.add_argument(
        '--resume', action='store_true',
        help='恢复optimizer/epoch/step; 普通finetune不要传此参数')
    args = parser.parse_args()
    if args.resume and not args.restore_ckpt:
        parser.error('--resume requires --restore_ckpt')

    config = load_config(args.config)
    distributed_config = config.get('distributed', {})
    backend = distributed_config.get('backend', 'nccl')
    timeout_minutes = int(distributed_config.get('timeout_minutes', 30))
    rank, local_rank, world_size, distributed, device = setup_distributed(
        backend, timeout_minutes)
    try:
        run_training(
            config, args, rank, local_rank, world_size, distributed, device)
    finally:
        cleanup_distributed(distributed)


if __name__ == '__main__':
    main()
