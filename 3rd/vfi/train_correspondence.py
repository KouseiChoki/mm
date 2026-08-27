#!/usr/bin/env python3
"""Pretrain and verify endpoint correspondence before VFI integration."""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kousei_dataset import TierDataset
from model.gmflow_pretrained import GMFlowFeatureEncoder
from model.warplayer import warp


class CorrespondenceEncoder(nn.Module):
    """Trainable GMFlow representation plus a dedicated 1/16 projection."""

    def __init__(self, config):
        super().__init__()
        channels = int(config.get('feature_channels', 128))
        self.encoder = GMFlowFeatureEncoder(
            checkpoint_path=config['pretrained_path'],
            checkpoint_required=True,
            feature_channels=channels,
            num_transformer_layers=int(config.get('transformer_layers', 6)),
            max_feature_tokens=int(config.get('max_feature_tokens', 2880)))
        self.coarse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
        )

    def forward(self, image0, image1):
        feature0_8, feature1_8 = self.encoder(image0, image1)
        return {
            '1/8': (feature0_8, feature1_8),
            '1/16': (self.coarse(feature0_8), self.coarse(feature1_8)),
        }


def _aligned_features(feature0, feature1, flow_gt, has_mv):
    """Warp both endpoint features to the GT middle-frame grid."""
    image_height, image_width = flow_gt.shape[-2:]
    height, width = feature0.shape[-2:]
    finite = torch.isfinite(flow_gt[:, :4]).all(dim=1, keepdim=True)
    safe_flow = torch.nan_to_num(
        flow_gt[:, :4], nan=0.0, posinf=0.0, neginf=0.0)
    flow = F.interpolate(
        safe_flow, size=(height, width), mode='bilinear',
        align_corners=False).clone()
    scale_x = image_width / width
    scale_y = image_height / height
    flow[:, 0::2] /= scale_x
    flow[:, 1::2] /= scale_y
    valid = F.interpolate(
        (flow_gt[:, 4:5] > 0).to(feature0.dtype),
        size=(height, width), mode='nearest') > 0
    finite = F.interpolate(
        finite.to(feature0.dtype), size=(height, width),
        mode='nearest') > 0
    valid = valid & finite & (has_mv[:, None, None, None] > 0)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing='ij')
    xx = xx[None, None]
    yy = yy[None, None]
    inside0 = (
        (xx + flow[:, 0:1] >= 0)
        & (xx + flow[:, 0:1] <= width - 1)
        & (yy + flow[:, 1:2] >= 0)
        & (yy + flow[:, 1:2] <= height - 1))
    inside1 = (
        (xx + flow[:, 2:3] >= 0)
        & (xx + flow[:, 2:3] <= width - 1)
        & (yy + flow[:, 3:4] >= 0)
        & (yy + flow[:, 3:4] <= height - 1))
    valid = valid & inside0 & inside1
    aligned0 = F.normalize(warp(feature0, flow[:, :2]), dim=1, eps=1e-6)
    aligned1 = F.normalize(warp(feature1, flow[:, 2:4]), dim=1, eps=1e-6)
    return aligned0, aligned1, valid[:, 0], scale_x, scale_y


def bilateral_info_nce(feature0, feature1, flow_gt, has_mv, temperature,
                        max_queries, random_queries):
    """Symmetric dense matching on GT-aligned endpoint feature grids."""
    aligned0, aligned1, valid, scale_x, scale_y = _aligned_features(
        feature0, feature1, flow_gt, has_mv)
    batch, channels, height, width = aligned0.shape
    flat0 = aligned0.flatten(2).transpose(1, 2)
    flat1 = aligned1.flatten(2).transpose(1, 2)
    valid_flat = valid.flatten(1)
    losses = []
    stats = {
        key: 0.0 for key in (
            'query_count', 'correct_count', 'epe_sum', 'rank_sum',
            'rank_fraction_sum', 'entropy_sum', 'positive_cosine_sum')}

    for batch_index in range(batch):
        candidates = torch.nonzero(
            valid_flat[batch_index], as_tuple=False).flatten()
        if candidates.numel() < 2:
            continue
        if candidates.numel() > max_queries:
            if random_queries:
                selection = torch.randperm(
                    candidates.numel(), device=candidates.device)[:max_queries]
            else:
                selection = torch.linspace(
                    0, candidates.numel() - 1, max_queries,
                    device=candidates.device).round().long()
            queries = candidates[selection]
        else:
            queries = candidates

        directions = (
            (flat0[batch_index, queries], flat1[batch_index]),
            (flat1[batch_index, queries], flat0[batch_index]),
        )
        invalid_candidates = ~valid_flat[batch_index]
        for query, candidate in directions:
            scores = torch.matmul(query, candidate.transpose(0, 1))
            scores = scores / temperature
            scores[:, invalid_candidates] = torch.finfo(scores.dtype).min
            losses.append(F.cross_entropy(scores.float(), queries))

            with torch.no_grad():
                prediction = scores.argmax(dim=1)
                positive = scores[
                    torch.arange(queries.numel(), device=scores.device),
                    queries]
                rank = (scores > positive[:, None]).sum(dim=1) + 1
                pred_y = torch.div(prediction, width, rounding_mode='floor')
                pred_x = prediction.remainder(width)
                true_y = torch.div(queries, width, rounding_mode='floor')
                true_x = queries.remainder(width)
                epe = torch.sqrt(
                    ((pred_x - true_x).float() * scale_x).square()
                    + ((pred_y - true_y).float() * scale_y).square())
                probability = torch.softmax(scores.float(), dim=1)
                entropy = -(
                    probability * probability.clamp_min(1e-8).log()
                ).sum(dim=1) / math.log(max(int(candidates.numel()), 2))
                positive_cosine = positive.float() * temperature
                count = float(queries.numel())
                stats['query_count'] += count
                stats['correct_count'] += float(
                    (prediction == queries).sum())
                stats['epe_sum'] += float(epe.sum())
                stats['rank_sum'] += float(rank.float().sum())
                stats['rank_fraction_sum'] += float(
                    (rank.float() / candidates.numel()).sum())
                stats['entropy_sum'] += float(entropy.sum())
                stats['positive_cosine_sum'] += float(
                    positive_cosine.sum())

    if not losses:
        return (feature0.sum() + feature1.sum()) * 0.0, stats
    return torch.stack(losses).mean(), stats


def finalize_stats(stats):
    count = max(stats['query_count'], 1.0)
    return {
        'queries': stats['query_count'],
        'top1_accuracy': stats['correct_count'] / count,
        'matching_epe_px': stats['epe_sum'] / count,
        'positive_rank': stats['rank_sum'] / count,
        'positive_rank_fraction': stats['rank_fraction_sum'] / count,
        'normalized_entropy': stats['entropy_sum'] / count,
        'positive_cosine': stats['positive_cosine_sum'] / count,
    }


def merge_stats(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0.0) + float(value)


def make_dataset(config, split):
    data = config['data']
    is_train = split == 'train'
    return TierDataset(
        data['root'], data[f'{split}_list'], split=split,
        crop_hw=tuple(data['crop_size']), framesteps=(1,),
        t_half_prob=1.0, mv_prob=1.0, mv_sign=(1, 1),
        mv_symmetry_confidence=False,
        motion_aware_crop_prob=(
            float(data.get('motion_aware_crop_prob', 0.0))
            if is_train else 0.0),
        motion_crop_threshold=float(data.get('motion_crop_threshold', 1.0)),
        small_motion_min_pixels=int(data.get('small_motion_min_pixels', 8)),
        small_motion_max_ratio=float(data.get('small_motion_max_ratio', 0.05)),
        motion_crop_jitter=float(data.get('motion_crop_jitter', 0.2)),
        mv_cache_dirname=data.get('mv_cache_dirname', 'mv_cache_f16'),
        mv_cache_required=True, mv_cycle_confidence='hard',
        mv_cycle_cache_required=True, mv_cycle_on_the_fly=False,
        val_with_mv=not is_train,
        val_samples_per_scene=data.get('val_samples_per_scene', 1),
        augment_profile='legacy', augment=is_train)


def make_loader(dataset, config, split):
    data = config['data']
    is_train = split == 'train'
    workers = int(data['num_workers'] if is_train
                  else data.get('val_num_workers', 2))
    return DataLoader(
        dataset, batch_size=int(data['batch_size'] if is_train else 1),
        shuffle=is_train, num_workers=workers,
        pin_memory=True, drop_last=is_train,
        persistent_workers=workers > 0,
        prefetch_factor=(int(data.get('prefetch_factor', 2))
                         if workers > 0 else None))


def compute_losses(model, batch, config, device, random_queries):
    frames, _, flow_gt, has_mv = batch
    frames = frames.to(device, non_blocking=True).float().div_(255.0)
    flow_gt = flow_gt.to(device, non_blocking=True)
    has_mv = has_mv.to(device, non_blocking=True)
    features = model(frames[:, :3], frames[:, 3:6])
    loss_config = config['loss']
    total_loss = frames.new_zeros(())
    scale_results = {}
    for scale, weight in loss_config['scale_weights'].items():
        feature0, feature1 = features[scale]
        loss, stats = bilateral_info_nce(
            feature0, feature1, flow_gt, has_mv,
            temperature=float(loss_config['temperature']),
            max_queries=int(loss_config['max_queries']),
            random_queries=random_queries)
        total_loss = total_loss + float(weight) * loss
        scale_results[scale] = (loss, stats)
    return total_loss, scale_results


@torch.no_grad()
def evaluate(model, loader, config, device):
    model.eval()
    totals = {
        scale: {} for scale in config['loss']['scale_weights']}
    loss_sums = {scale: 0.0 for scale in totals}
    batches = 0
    for batch in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            _, scale_results = compute_losses(
                model, batch, config, device, random_queries=False)
        for scale, (loss, stats) in scale_results.items():
            loss_sums[scale] += float(loss)
            merge_stats(totals[scale], stats)
        batches += 1
    result = {}
    for scale in totals:
        result[scale] = finalize_stats(totals[scale])
        result[scale]['loss'] = loss_sums[scale] / max(batches, 1)
    model.train()
    return result


def learning_rate(step, total_steps, warmup_steps, maximum, minimum):
    if step < warmup_steps:
        return maximum * step / max(warmup_steps, 1)
    progress = min(
        max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
        1.0)
    return minimum + 0.5 * (maximum - minimum) * (
        1.0 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optimizer, epoch, step, config, metrics):
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'epoch': epoch, 'step': step, 'config': config,
        'validation': metrics,
    }, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--preflight', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.preflight:
        config = copy.deepcopy(config)
        config['exp_name'] = f'{config["exp_name"]}_preflight_update'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    seed = int(config.get('seed', 1234))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')

    train_loader = make_loader(make_dataset(config, 'train'), config, 'train')
    val_loader = make_loader(make_dataset(config, 'val'), config, 'val')
    model = CorrespondenceEncoder(config['model']).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.backbone.parameters(),
         'lr_scale': float(config['optim'].get('backbone_lr_scale', 0.5))},
        {'params': model.encoder.transformer.parameters(),
         'lr_scale': float(config['optim'].get('transformer_lr_scale', 1.0))},
        {'params': model.coarse.parameters(),
         'lr_scale': float(config['optim'].get('coarse_lr_scale', 5.0))},
    ], lr=float(config['optim']['lr_max']),
        weight_decay=float(config['optim'].get('weight_decay', 1e-4)),
        betas=tuple(config['optim'].get('betas', (0.9, 0.999))))
    start_epoch = 0
    step = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint['epoch'])
        step = int(checkpoint['step'])

    record_dir = Path(config.get('record_root', 'record')) / config['exp_name']
    checkpoint_dir = (
        Path(config.get('checkpoint_root', 'ckpt')) / config['exp_name'])
    record_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = yaml.safe_dump(
        config, sort_keys=False, allow_unicode=True)
    (record_dir / 'model.yaml').write_text(model_yaml)
    (checkpoint_dir / 'model.yaml').write_text(model_yaml)
    writer = SummaryWriter(str(record_dir / 'tensorboard'))
    metrics_path = record_dir / 'metrics.jsonl'

    baseline = evaluate(model, val_loader, config, device)
    print('[baseline]', json.dumps(baseline, ensure_ascii=False))
    with metrics_path.open('a') as handle:
        handle.write(json.dumps({
            'epoch': start_epoch, 'step': step, 'baseline': True,
            'validation': baseline}, ensure_ascii=False) + '\n')
    if args.preflight:
        config = dict(config)
        config['train'] = dict(config['train'])
        config['optim'] = dict(config['optim'])
        config['train']['epochs'] = 1
        config['train']['steps_per_epoch'] = 10
        config['optim']['warmup_steps'] = 0

    train_config = config['train']
    epochs = int(train_config['epochs'])
    steps_per_epoch = int(train_config['steps_per_epoch'])
    accumulation = int(train_config.get('grad_accum_steps', 1))
    total_steps = epochs * steps_per_epoch
    warmup = int(config['optim'].get('warmup_steps', 0))
    train_iterator = iter(train_loader)
    best_score = float('inf')
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        for iteration in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            latest_results = None
            for _ in range(accumulation):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                with torch.autocast(
                        'cuda', dtype=torch.bfloat16, enabled=True):
                    loss, latest_results = compute_losses(
                        model, batch, config, device, random_queries=True)
                (loss / accumulation).backward()
                loss_value += float(loss.detach()) / accumulation
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config['optim'].get('grad_clip', 1.0)))
            lr = learning_rate(
                step, total_steps, warmup,
                float(config['optim']['lr_max']),
                float(config['optim']['lr_min']))
            for group in optimizer.param_groups:
                group['lr'] = lr * group.get('lr_scale', 1.0)
            optimizer.step()
            step += 1

            if step % int(config['monitor']['log_every_steps']) == 0:
                writer.add_scalar('train/loss', loss_value, step)
                writer.add_scalar('train/lr', lr, step)
                for scale, (_, stats) in latest_results.items():
                    values = finalize_stats(stats)
                    for name, value in values.items():
                        writer.add_scalar(
                            f'train_{scale}/{name}', value, step)
            if iteration % int(config['monitor']['print_every_steps']) == 0:
                text = []
                for scale, (_, stats) in latest_results.items():
                    values = finalize_stats(stats)
                    text.append(
                        f'{scale}:acc={values["top1_accuracy"]:.2%} '
                        f'epe={values["matching_epe_px"]:.2f}px '
                        f'rank={values["positive_rank"]:.1f} '
                        f'ent={values["normalized_entropy"]:.3f}')
                print(
                    f'[corr] epoch:{epoch} {iteration}/{steps_per_epoch} '
                    f'loss:{loss_value:.4f} lr:{lr:.2e} ' + ' | '.join(text),
                    flush=True)

        run_eval = (
            epoch % int(config['monitor']['eval_every_epochs']) == 0
            or epoch == epochs)
        if run_eval:
            validation = evaluate(model, val_loader, config, device)
            # Fine correspondence is primary; coarse EPE still participates
            # so checkpoint selection cannot silently discard large-motion
            # improvements at 1/16.
            current_score = (
                validation['1/8']['matching_epe_px']
                + 0.25 * validation['1/16']['matching_epe_px'])
            row = {
                'epoch': epoch, 'step': step,
                'selection_score': current_score,
                'validation': validation}
            print('[validation]', json.dumps(row, ensure_ascii=False))
            with metrics_path.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            for scale, values in validation.items():
                for name, value in values.items():
                    writer.add_scalar(f'val_{scale}/{name}', value, epoch)
            if current_score < best_score:
                best_score = current_score
                save_checkpoint(
                    checkpoint_dir / 'best.pkl', model, optimizer, epoch, step,
                    config, validation)
        if (epoch % int(config['monitor']['save_every_epochs']) == 0
                or epoch == epochs):
            save_checkpoint(
                checkpoint_dir / f'epoch_{epoch}.pkl',
                model, optimizer, epoch, step,
                config, validation if run_eval else None)
        writer.flush()
        print(
            f'[epoch] {epoch} finished in '
            f'{time.perf_counter() - epoch_started:.1f}s '
            f'peak_alloc={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GiB '
            f'peak_reserved={torch.cuda.max_memory_reserved() / 2 ** 30:.2f}GiB',
            flush=True)
        if args.preflight:
            break
    writer.close()


if __name__ == '__main__':
    main()
