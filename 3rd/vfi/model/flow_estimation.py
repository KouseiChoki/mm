import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from torch.utils.checkpoint import checkpoint

from .warplayer import warp
from .refine import *
from .gmflow_pretrained import GMFlowFeatureEncoder
from .correspondence import CorrespondencePyramid
from .pqmax import PQMaxMotionSynthesizer
from .single_match import ConfidenceGatedMatchRefiner

# 局部精化结构固定在代码中，避免 YAML 改动导致 checkpoint 结构不兼容。
LOCAL_CFG = ((2, 4, 1.0, 8), (1, 4, 1.0, 8))

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class LocalCorrelationVolume(nn.Module):
    """Flow-aligned local correlation used inside a pyramid flow head.

    This is deliberately different from the old post-hoc sparse matcher.  At
    every coarse-to-fine stage, the current bilateral flow first warps the two
    endpoint features onto the invisible target grid.  A local cost volume is
    then built between the aligned features and consumed directly by the flow
    predictor.  Consequently correspondence evidence participates in every
    residual flow update instead of being an optional correction after the
    complete motion stack.
    """

    def __init__(self, feature_channels, radius, output_channels=16,
                 temperature=0.07):
        super().__init__()
        self.radius = int(radius)
        self.temperature = float(temperature)
        if self.radius < 1:
            raise ValueError(f'correlation radius must be >=1, got {radius}')
        if self.temperature <= 0.0:
            raise ValueError(
                f'correlation temperature must be >0, got {temperature}')
        output_channels = max(int(output_channels), 4)
        volume_channels = (2 * self.radius + 1) ** 2
        self.log_volume_channels = math.log(volume_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(volume_channels, output_channels, 1, 1, 0),
            nn.PReLU(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
            nn.PReLU(output_channels),
        )
        self.output_channels = output_channels
        self.last_peak_probability = None
        self.last_normalized_entropy = None
        self.last_feature_abs = None

    @staticmethod
    def _flow_on_feature_grid(flow, feature):
        if flow is None:
            return None
        image_height, image_width = flow.shape[-2:]
        feature_height, feature_width = feature.shape[-2:]
        scale_x = image_width / feature_width
        scale_y = image_height / feature_height
        flow_low = F.interpolate(
            flow, size=(feature_height, feature_width), mode='bilinear',
            align_corners=False).clone()
        flow_low[:, 0::2] /= scale_x
        flow_low[:, 1::2] /= scale_y
        return flow_low

    def forward(self, feature0, feature1, flow):
        if feature0.shape != feature1.shape:
            raise ValueError(
                'correlation endpoint features must have identical shapes, '
                f'got {feature0.shape} and {feature1.shape}')
        flow_low = self._flow_on_feature_grid(flow, feature0)
        if flow_low is not None:
            feature0 = warp(feature0, flow_low[:, :2])
            feature1 = warp(feature1, flow_low[:, 2:4])

        feature0 = F.normalize(feature0, dim=1, eps=1e-6)
        feature1 = F.normalize(feature1, dim=1, eps=1e-6)
        batch, channels, height, width = feature0.shape
        kernel = 2 * self.radius + 1
        patches = F.unfold(
            feature1, kernel_size=kernel, padding=self.radius)
        patches = patches.view(
            batch, channels, kernel * kernel, height, width)
        volume = (
            feature0[:, :, None] * patches).sum(dim=1)
        volume = volume / self.temperature

        with torch.no_grad():
            probability = torch.softmax(volume.float(), dim=1)
            self.last_peak_probability = probability.max(dim=1).values.mean()
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(
                dim=1)
            self.last_normalized_entropy = (
                entropy / max(self.log_volume_channels, 1e-6)
            ).mean()

        encoded = self.encoder(volume)
        self.last_feature_abs = encoded.detach().abs().mean()
        return encoded


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17, zero_init=False,
                 compact_feature=False, correlation_radius=None,
                 correlation_channels=16, correlation_temperature=0.07,
                 external_correlation_radius=None,
                 external_correlation_temperature=0.07,
                 external_correlation_init_scale=0.01):
        super(Head, self).__init__()
        self.scale = scale
        feature_channels = in_planes * 2 // (4 * 4)
        if compact_feature:
            # 1/4 feature head不先PixelShuffle到全分辨率。用1x1投影压通道,
            # 在1/4网格完成匹配/残差预测, 再上采样flow, 避免全分辨率192c卷积。
            self.feature_transform = nn.Sequential(
                nn.Conv2d(in_planes * 2, feature_channels, 1, 1, 0),
                nn.PReLU(feature_channels),
            )
            self.work_scale = scale
        else:
            self.feature_transform = nn.Sequential(
                nn.PixelShuffle(2), nn.PixelShuffle(2))
            self.work_scale = scale // 4
        self.correlation = None
        correlation_input_channels = 0
        if correlation_radius is not None:
            self.correlation = LocalCorrelationVolume(
                feature_channels=in_planes,
                radius=correlation_radius,
                output_channels=correlation_channels,
                temperature=correlation_temperature)
            correlation_input_channels = self.correlation.output_channels
        # The pretrained correspondence cost volume is injected after the
        # historical first convolution.  It therefore does not change that
        # convolution's input shape and every 0729 flow-head weight remains
        # directly reusable.
        self.external_correlation = None
        self.external_correlation_scale = None
        if external_correlation_radius is not None:
            self.external_correlation = LocalCorrelationVolume(
                feature_channels=128,
                radius=external_correlation_radius,
                output_channels=c,
                temperature=external_correlation_temperature)
            self.external_correlation_scale = nn.Parameter(torch.tensor(
                float(external_correlation_init_scale)))
        self.conv = nn.Sequential(
                                  conv(feature_channels + in_else
                                       + correlation_input_channels, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  
        if zero_init:
            nn.init.zeros_(self.conv[-1][0].weight)
            nn.init.zeros_(self.conv[-1][0].bias)

    def forward(self, motion_feature, x, flow, external_features=None):
        correlation = None
        if self.correlation is not None:
            feature0, feature1 = motion_feature.chunk(2, dim=1)
            correlation = self.correlation(feature0, feature1, flow)
        motion_feature = self.feature_transform(motion_feature)
        if correlation is not None:
            correlation = F.interpolate(
                correlation, size=motion_feature.shape[-2:],
                mode='bilinear', align_corners=False)
        if self.work_scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.work_scale,
                              mode="bilinear", align_corners=False)
        if flow is not None:
            if self.work_scale != 1:
                flow = F.interpolate(flow, scale_factor=1. / self.work_scale,
                                     mode="bilinear", align_corners=False)
                flow = flow * (1. / self.work_scale)
            x = torch.cat((x, flow), 1)
        predictor_inputs = [motion_feature, x]
        if correlation is not None:
            predictor_inputs.append(correlation)
        hidden = self.conv[0](torch.cat(predictor_inputs, 1))
        if self.external_correlation is not None:
            if external_features is None:
                raise ValueError(
                    'external correspondence features are required for this '
                    'flow head')
            external0, external1 = external_features
            external = self.external_correlation(
                external0, external1, flow)
            external = F.interpolate(
                external, size=hidden.shape[-2:], mode='bilinear',
                align_corners=False)
            hidden = hidden + self.external_correlation_scale * external
        elif external_features is not None:
            raise ValueError(
                'external correspondence features were passed to a flow '
                'head without an external cost-volume adapter')
        x = self.conv[1](hidden)
        x = self.conv[2](x)
        if self.work_scale != 1:
            x = F.interpolate(x, scale_factor=self.work_scale,
                              mode="bilinear", align_corners=False)
        flow = x[:, :4] * self.work_scale
        mask = x[:, 4:5]
        return flow, mask


class ContentAwareFlowUpsampler(nn.Module):
    """Feature-guided residual-kernel flow upsampling.

    The learned kernel only changes the four flow channels.  The caller keeps
    the historical bilinear path for mask logits, avoiding the flow/mask
    coupling that made the teacher-flow repair regress in real footage.

    The residual-kernel predictor's final layer is initialized to zero, so an
    old checkpoint is bit-identical to bilinear upsampling while gradients can
    train that layer from the first step.  Kernel weights are forced to sum to
    zero over each 3x3 neighborhood, preserving constant flow fields and
    limiting the branch to content-aware boundary corrections.
    """

    def __init__(self, guidance_channels, factor, hidden_channels=32,
                 residual_scale=0.25):
        super().__init__()
        if factor <= 1:
            raise ValueError(f'factor must be > 1, got {factor}')
        self.factor = int(factor)
        self.residual_scale = float(residual_scale)
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError(
                f'residual_scale must be in (0,1], got {residual_scale}')
        hidden_channels = max(int(hidden_channels), 8)
        self.kernel = nn.Sequential(
            conv(guidance_channels + 4, hidden_channels),
            conv(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 9 * self.factor * self.factor, 1),
        )
        # Exact no-op initialization with an immediately trainable output
        # projection.  Earlier layers start receiving gradients after the
        # first optimizer update.
        nn.init.zeros_(self.kernel[-1].weight)
        nn.init.zeros_(self.kernel[-1].bias)
        self.last_residual_abs = None

    def _residual_upsample(self, low_flow, weights):
        batch, channels, height, width = low_flow.shape
        factor = self.factor
        weights = weights.view(
            batch, 1, 9, factor, factor, height, width)
        weights = torch.tanh(weights)
        weights = weights - weights.mean(dim=2, keepdim=True)
        padded_flow = F.pad(low_flow, (1, 1, 1, 1), mode='replicate')
        patches = F.unfold(padded_flow, kernel_size=3)
        patches = patches.view(
            batch, channels, 9, 1, 1, height, width)
        upsampled = torch.sum(weights * patches, dim=2)
        return upsampled.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, channels, height * factor, width * factor)

    def forward(self, low_flow, guidance):
        factor = self.factor
        target_size = (
            low_flow.shape[-2] * factor, low_flow.shape[-1] * factor)
        bilinear = F.interpolate(
            low_flow, size=target_size, mode='bilinear',
            align_corners=False) * factor
        guidance_low = F.interpolate(
            guidance, size=low_flow.shape[-2:], mode='bilinear',
            align_corners=False)
        weights = self.kernel(torch.cat((guidance_low, low_flow), dim=1))
        residual = self._residual_upsample(low_flow, weights)
        residual = residual * (factor * self.residual_scale)
        self.last_residual_abs = residual.detach().abs().mean()
        return bilinear + residual


class SparseGlobalMatcher(nn.Module):
    """Sparse all-pairs correspondence compensation on a low-res feature map.

    This is a lightweight SGM-VFI-inspired ablation, not a reproduction of
    SGM-VFI's GMFlow branch.  It keeps the converged local flow as the primary
    estimate, selects only the highest-error target-grid positions, matches
    them globally between the two endpoint feature maps, rejects ambiguous
    matches with a top-1 margin and a backward consistency check, then learns
    a small spatial adapter that merges the sparse proposal into the flow.

    The adapter's output layer is zero initialized.  Enabling this module on
    an old checkpoint is therefore an exact inference no-op, while the output
    layer receives gradients from the first training step.
    """

    def __init__(self, feature_channels, feature_scale=8, hidden_channels=32,
                 topk_ratio=0.02, min_points=16, max_points=128,
                 confidence_scale=0.05, mutual_sigma=1.5,
                 max_displacement=96.0, residual_scale=0.5,
                 propagation_radius=2, photometric_weight=0.5):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.feature_scale = int(feature_scale)
        self.topk_ratio = float(topk_ratio)
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.confidence_scale = float(confidence_scale)
        self.mutual_sigma = float(mutual_sigma)
        self.max_displacement = float(max_displacement)
        self.residual_scale = float(residual_scale)
        self.propagation_radius = int(propagation_radius)
        self.photometric_weight = float(photometric_weight)

        if self.feature_scale <= 1:
            raise ValueError('feature_scale must be > 1')
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError('topk_ratio must be in (0, 1]')
        if self.min_points < 1 or self.max_points < self.min_points:
            raise ValueError('require 1 <= min_points <= max_points')
        if self.confidence_scale <= 0.0 or self.mutual_sigma <= 0.0:
            raise ValueError('confidence_scale and mutual_sigma must be > 0')
        if self.max_displacement <= 0.0:
            raise ValueError('max_displacement must be > 0')
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError('residual_scale must be in (0, 1]')
        if self.propagation_radius < 0:
            raise ValueError('propagation_radius must be >= 0')

        # flow(4), sparse proposal(4), support confidence(1), feature
        # disagreement(1), photometric disagreement(1).
        adapter_in_channels = 11
        hidden_channels = max(int(hidden_channels), 8)
        self.adapter = nn.Sequential(
            conv(adapter_in_channels, hidden_channels),
            conv(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 4, 3, 1, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.last_selected_ratio = None
        self.last_confidence = None
        self.last_margin = None
        self.last_mutual_error = None
        self.last_mutual_ratio = None
        self.last_valid_ratio = None
        self.last_similarity_gain = None
        self.last_similarity_improved_ratio = None
        self.last_proposal_abs = None
        self.last_residual_abs = None
        self.last_residual_selected_abs = None

    @staticmethod
    def _coordinate_grid(batch, height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij')
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(
            batch, -1, -1, -1)

    @staticmethod
    def _gather_flat(values, indices):
        """Gather [B,C,N] values with [B,K] indices -> [B,C,K]."""
        return torch.gather(
            values, 2, indices[:, None].expand(-1, values.shape[1], -1))

    def _point_count(self, height, width):
        points = round(height * width * self.topk_ratio)
        return min(max(points, self.min_points), self.max_points,
                   height * width)

    def _sparse_proposal(self, query_feature0, endpoint_feature0, feature1,
                         flow_low, score, timestep_low,
                         current_feature1=None, effective_feature_scale=None):
        """Return sparse midpoint-preserving flow correction and support.

        All coordinates and flow values in this method use feature-grid
        pixels.  Matching a frame-0 endpoint to a shifted frame-1 endpoint by
        ``delta`` is converted to ``[-t*delta, (1-t)*delta]``.  This preserves
        the target point because (1-t)*d0 + t*d1 remains zero.
        """
        batch, _, height, width = query_feature0.shape
        num_positions = height * width
        num_points = self._point_count(height, width)

        query0 = F.normalize(query_feature0, dim=1, eps=1e-6)
        endpoint0_features = F.normalize(
            endpoint_feature0, dim=1, eps=1e-6)
        norm1 = F.normalize(feature1, dim=1, eps=1e-6)
        flat_query0 = query0.flatten(2)
        flat_endpoint0 = endpoint0_features.flatten(2)
        flat1 = norm1.flatten(2)
        flat_score = score.flatten(1)
        point_indices = flat_score.topk(
            num_points, dim=1, largest=True, sorted=False).indices

        queries = self._gather_flat(
            flat_query0, point_indices).transpose(1, 2)
        correlations = torch.bmm(queries, flat1)
        top_values, top_indices = correlations.topk(
            k=min(2, num_positions), dim=2)
        match_indices = top_indices[:, :, 0]
        if num_positions > 1:
            margin = top_values[:, :, 0] - top_values[:, :, 1]
        else:
            margin = torch.ones_like(top_values[:, :, 0])
        margin_confidence = (
            margin / self.confidence_scale).clamp(0.0, 1.0)
        if current_feature1 is not None:
            current_similarity = F.cosine_similarity(
                query_feature0, current_feature1, dim=1, eps=1e-6)
            current_similarity = torch.gather(
                current_similarity.flatten(1), 1, point_indices)
            similarity_gain = top_values[:, :, 0] - current_similarity
            self.last_similarity_gain = similarity_gain.detach().mean()
            self.last_similarity_improved_ratio = (
                similarity_gain.detach() > 0).to(margin.dtype).mean()
        else:
            self.last_similarity_gain = margin.new_zeros(())
            self.last_similarity_improved_ratio = margin.new_zeros(())

        grid = self._coordinate_grid(
            batch, height, width, query_feature0.device, query_feature0.dtype)
        flat_grid = grid.flatten(2)
        target_points = self._gather_flat(flat_grid, point_indices)
        flow0_points = self._gather_flat(
            flow_low[:, :2].flatten(2), point_indices)
        flow1_points = self._gather_flat(
            flow_low[:, 2:4].flatten(2), point_indices)
        endpoint0 = target_points + flow0_points
        endpoint1_current = target_points + flow1_points

        match_x = (match_indices % width).to(query_feature0.dtype)
        match_y = torch.div(
            match_indices, width, rounding_mode='floor').to(
                query_feature0.dtype)
        endpoint1_match = torch.stack((match_x, match_y), dim=1)
        delta = endpoint1_match - endpoint1_current

        # A reverse match is cheap because it is evaluated only for the same
        # sparse endpoints.  It strongly suppresses repeated-texture aliases.
        reverse_queries = self._gather_flat(
            flat1, match_indices).transpose(1, 2)
        reverse_indices = torch.bmm(
            reverse_queries, flat_endpoint0).argmax(dim=2)
        reverse_x = (reverse_indices % width).to(query_feature0.dtype)
        reverse_y = torch.div(
            reverse_indices, width, rounding_mode='floor').to(
                query_feature0.dtype)
        reverse_points = torch.stack((reverse_x, reverse_y), dim=1)
        mutual_error = torch.linalg.vector_norm(
            reverse_points - endpoint0, dim=1)
        mutual_confidence = torch.exp(
            -0.5 * (mutual_error / self.mutual_sigma).square())

        coordinate_valid = (
            (endpoint0[:, 0] >= 0)
            & (endpoint0[:, 0] <= width - 1)
            & (endpoint0[:, 1] >= 0)
            & (endpoint0[:, 1] <= height - 1)
            & (endpoint1_current[:, 0] >= 0)
            & (endpoint1_current[:, 0] <= width - 1)
            & (endpoint1_current[:, 1] >= 0)
            & (endpoint1_current[:, 1] <= height - 1))
        confidence = (
            margin_confidence * mutual_confidence
            * coordinate_valid.to(margin_confidence.dtype))
        self.last_margin = margin.detach().mean()
        self.last_mutual_error = mutual_error.detach().mean()
        self.last_mutual_ratio = (
            mutual_error.detach() <= self.mutual_sigma).to(
                margin_confidence.dtype).mean()
        self.last_valid_ratio = (
            confidence.detach() >= 0.25).to(confidence.dtype).mean()

        # Bound a single wrong global match before the learned adapter sees it.
        feature_scale = float(
            effective_feature_scale or self.feature_scale)
        max_displacement_low = self.max_displacement / feature_scale
        delta_norm = torch.linalg.vector_norm(
            delta, dim=1, keepdim=True).clamp(min=1e-6)
        delta = delta * torch.clamp(
            max_displacement_low / delta_norm, max=1.0)

        timestep_points = self._gather_flat(
            timestep_low.flatten(2), point_indices)
        correction0 = -timestep_points * delta
        correction1 = (1.0 - timestep_points) * delta
        point_proposal = torch.cat((correction0, correction1), dim=1)
        point_proposal = point_proposal * confidence[:, None]

        proposal = flow_low.new_zeros(batch, 4, num_positions)
        proposal.scatter_(
            2, point_indices[:, None].expand(-1, 4, -1), point_proposal)
        support = flow_low.new_zeros(batch, 1, num_positions)
        support.scatter_(2, point_indices[:, None], confidence[:, None])
        proposal = proposal.view(batch, 4, height, width)
        support = support.view(batch, 1, height, width)

        self.last_selected_ratio = flow_low.new_tensor(
            num_points / max(num_positions, 1))
        self.last_confidence = confidence.detach().mean()
        confident = confidence[:, None].sum().clamp(min=1e-6)
        self.last_proposal_abs = (
            point_proposal.detach().abs().sum() / (4.0 * confident)
            * feature_scale)
        return proposal, support

    def forward(self, feature0, feature1, img0, img1, flow, timestep):
        feature_size = feature0.shape[-2:]
        height, width = feature_size
        image_height, image_width = flow.shape[-2:]
        scale_x = image_width / width
        scale_y = image_height / height

        flow_low = F.interpolate(
            flow, size=feature_size, mode='bilinear', align_corners=False)
        flow_low = flow_low.clone()
        flow_low[:, 0::2] /= scale_x
        flow_low[:, 1::2] /= scale_y

        warped_feature0 = warp(feature0, flow_low[:, :2])
        warped_feature1 = warp(feature1, flow_low[:, 2:4])
        feature_score = 1.0 - F.cosine_similarity(
            warped_feature0, warped_feature1, dim=1, eps=1e-6)

        image0_low = F.interpolate(
            img0, size=feature_size, mode='bilinear', align_corners=False)
        image1_low = F.interpolate(
            img1, size=feature_size, mode='bilinear', align_corners=False)
        warped_image0 = warp(image0_low, flow_low[:, :2])
        warped_image1 = warp(image1_low, flow_low[:, 2:4])
        photo_score = (warped_image0 - warped_image1).abs().mean(dim=1)
        score = feature_score + self.photometric_weight * photo_score

        if torch.is_tensor(timestep):
            timestep_low = F.interpolate(
                timestep, size=feature_size, mode='nearest')
        else:
            timestep_low = flow_low[:, :1].new_full(
                (flow_low.shape[0], 1, height, width), float(timestep))

        proposal, support = self._sparse_proposal(
            warped_feature0, feature0, feature1, flow_low, score,
            timestep_low, current_feature1=warped_feature1,
            effective_feature_scale=0.5 * (scale_x + scale_y))
        adapter_input = torch.cat((
            flow_low, proposal, support, feature_score[:, None],
            photo_score[:, None]), dim=1)
        raw_residual = self.adapter(adapter_input)
        max_residual_low = (
            self.max_displacement / self.feature_scale
            * self.residual_scale)
        residual_low = torch.tanh(raw_residual) * max_residual_low

        if self.propagation_radius > 0:
            kernel = 2 * self.propagation_radius + 1
            support_context = F.max_pool2d(
                support, kernel_size=kernel, stride=1,
                padding=self.propagation_radius)
            residual_low = residual_low * support_context

        residual = F.interpolate(
            residual_low, size=flow.shape[-2:], mode='bilinear',
            align_corners=False)
        residual = residual.clone()
        residual[:, 0::2] *= scale_x
        residual[:, 1::2] *= scale_y
        self.last_residual_abs = residual.detach().abs().mean()
        support_full = F.interpolate(
            support, size=flow.shape[-2:], mode='nearest')
        selected = (support_full > 0).to(residual.dtype)
        selected_count = selected.sum().clamp(min=1.0)
        self.last_residual_selected_abs = (
            residual.detach().abs().mul(selected).sum()
            / (4.0 * selected_count))
        return flow + residual


class MultiHypothesisBranch(nn.Module):
    """Generate and fuse one alternative bilateral motion hypothesis.

    AMT predicts several fine-grained bilateral flow pairs, backward-warps
    each pair, then combines the candidate images.  This lightweight ablation
    keeps the converged 0729 flow as hypothesis 0 and predicts only one local
    alternative at quarter resolution.  A signed, bounded mixing residual is
    enabled only where the two primary warps disagree.

    The alternative flow head starts with a very small non-zero initialization
    so it can receive direct candidate supervision.  The mixing channel is
    initialized to exactly zero, making the released model output bit-identical
    to the original checkpoint on its first forward pass.
    """

    def __init__(self, hidden_channels=48, work_scale=4,
                 max_flow_delta=4.0, max_mask_delta=2.0, max_mix=0.5,
                 disagreement_threshold=0.03, candidate_init_std=0.005):
        super().__init__()
        self.work_scale = int(work_scale)
        self.max_flow_delta = float(max_flow_delta)
        self.max_mask_delta = float(max_mask_delta)
        self.max_mix = float(max_mix)
        self.disagreement_threshold = float(disagreement_threshold)
        self.candidate_init_std = float(candidate_init_std)
        if self.work_scale < 1:
            raise ValueError('work_scale must be >= 1')
        if self.max_flow_delta <= 0.0 or self.max_mask_delta <= 0.0:
            raise ValueError('flow/mask delta limits must be > 0')
        if not 0.0 < self.max_mix <= 1.0:
            raise ValueError('max_mix must be in (0, 1]')
        if self.disagreement_threshold <= 0.0:
            raise ValueError('disagreement_threshold must be > 0')
        if self.candidate_init_std <= 0.0:
            raise ValueError('candidate_init_std must be > 0')

        hidden_channels = max(int(hidden_channels), 8)
        # img0/img1(6), primary warps(6), flow(4), mask/timestep(2) = 18.
        self.body = nn.Sequential(
            conv(18, hidden_channels),
            conv(hidden_channels, hidden_channels),
            conv(hidden_channels, hidden_channels),
        )
        # alternative flow residual(4), alternative mask residual(1),
        # signed candidate-mixing residual(1).
        self.output = nn.Conv2d(hidden_channels, 6, 3, 1, 1)
        with torch.no_grad():
            nn.init.normal_(
                self.output.weight[:5], std=self.candidate_init_std)
            nn.init.zeros_(self.output.bias[:5])
            nn.init.zeros_(self.output.weight[5:6])
            nn.init.zeros_(self.output.bias[5:6])

        self.last_alternative_merged = None
        self.last_region_gate = None
        self.last_flow_delta_abs = None
        self.last_candidate_delta_abs = None
        self.last_mix_abs = None
        self.last_output_delta_abs = None
        self.last_region_ratio = None

    def _low_resolution_inputs(self, img0, img1, warped_img0, warped_img1,
                               flow, mask_logits, timestep):
        if self.work_scale == 1:
            return torch.cat((
                img0, img1, warped_img0, warped_img1, flow, mask_logits,
                timestep), dim=1)
        size = (
            max(img0.shape[-2] // self.work_scale, 1),
            max(img0.shape[-1] // self.work_scale, 1))
        images = [
            F.interpolate(value, size=size, mode='bilinear',
                          align_corners=False)
            for value in (img0, img1, warped_img0, warped_img1)
        ]
        flow_low = F.interpolate(
            flow, size=size, mode='bilinear', align_corners=False)
        scale_y = img0.shape[-2] / size[0]
        scale_x = img0.shape[-1] / size[1]
        flow_low = flow_low.clone()
        flow_low[:, 0::2] /= scale_x
        flow_low[:, 1::2] /= scale_y
        mask_low = F.interpolate(
            mask_logits, size=size, mode='bilinear', align_corners=False)
        timestep_low = F.interpolate(timestep, size=size, mode='nearest')
        return torch.cat((*images, flow_low, mask_low, timestep_low), dim=1)

    def predict_alternative(self, img0, img1, warped_img0, warped_img1,
                            flow, mask_logits, timestep):
        low_inputs = self._low_resolution_inputs(
            img0, img1, warped_img0, warped_img1, flow, mask_logits,
            timestep)
        raw = self.output(self.body(low_inputs))
        target_size = flow.shape[-2:]

        flow_delta_low = torch.tanh(raw[:, :4]) * (
            self.max_flow_delta / self.work_scale)
        flow_delta = F.interpolate(
            flow_delta_low, size=target_size, mode='bilinear',
            align_corners=False) * self.work_scale
        mask_delta = F.interpolate(
            torch.tanh(raw[:, 4:5]) * self.max_mask_delta,
            size=target_size, mode='bilinear', align_corners=False)
        mix_logits = F.interpolate(
            raw[:, 5:6], size=target_size, mode='bilinear',
            align_corners=False)
        return flow + flow_delta, mask_logits + mask_delta, mix_logits

    def combine(self, primary_merged, alternative_merged, warped_img0,
                warped_img1, mix_logits, flow, alternative_flow):
        disagreement = (
            warped_img0 - warped_img1).abs().mean(dim=1, keepdim=True)
        region_gate = (
            disagreement / self.disagreement_threshold).clamp(0.0, 1.0)
        # The primary branch cannot game the gate while it is frozen during
        # the stage-1 ablation.  Detaching also keeps the branch's semantics
        # explicit if a later experiment unfreezes the full network.
        region_gate = region_gate.detach()
        mix = self.max_mix * torch.tanh(mix_logits) * region_gate
        combined = primary_merged + mix * (
            alternative_merged - primary_merged)

        output_delta = combined - primary_merged
        self.last_alternative_merged = alternative_merged
        self.last_region_gate = region_gate
        self.last_flow_delta_abs = (
            alternative_flow - flow).detach().abs().mean()
        self.last_candidate_delta_abs = (
            alternative_merged - primary_merged).detach().abs().mean()
        self.last_mix_abs = mix.detach().abs().mean()
        self.last_output_delta_abs = output_delta.detach().abs().mean()
        self.last_region_ratio = (
            region_gate.detach() > 0.5).to(region_gate.dtype).mean()
        return combined

class IFBlock(nn.Module):
    """局部精化块。

    scale : 输入预下采样倍率 (2=在1/2分辨率图像上工作, 1=全分辨率图像)
    down  : conv主体的内部下采样倍率:
              4 = 原版行为 (conv0两次stride2, 卷积主体在 输入/4 分辨率)
              2 = 浅下采样 (卷积主体在 输入/2 分辨率, 小物体细节保留更多)
              1 = 零下采样 (卷积主体在 输入原分辨率, 计算/显存开销最大)
    blocks: convblock 层数 (down=1 时建议减少以控制全分辨率下的计算量)

    卷积主体的实际工作分辨率 = 全分辨率 / (scale * down)。
    """

    def __init__(self, in_planes, c, scale, down=4, blocks=8, zero_init=False,
                 content_aware_upsampling=False,
                 content_aware_hidden_channels=32,
                 content_aware_residual_scale=0.25):
        super(IFBlock, self).__init__()
        assert down in (1, 2, 4), f'down must be 1/2/4, got {down}'
        self.scale = scale
        self.down = down

        if down == 4:
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 2, 1),
                conv(c//2, c, 3, 2, 1),
                )
        elif down == 2:
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 2, 1),
                conv(c//2, c, 3, 1, 1),
                )
        else:                                   # down == 1: 全分辨率卷积主体
            self.conv0 = nn.Sequential(
                conv(in_planes, c//2, 3, 1, 1),
                conv(c//2, c, 3, 1, 1),
                )

        self.convblock = nn.Sequential(*[conv(c, c) for _ in range(blocks)])

        if down == 1:
            self.lastconv = nn.Conv2d(c, 5, 3, 1, 1)          # 已在目标分辨率, 无需转置上采样
        else:
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)  # ×2

        if zero_init:
            nn.init.zeros_(self.lastconv.weight)
            nn.init.zeros_(self.lastconv.bias)

        # 预测网格 → 全分辨率 的上采样倍率 (同时也是flow数值的还原倍率):
        #   down>=2: 预测网格 = 全分辨率/(scale*down/2)
        #   down==1: 预测网格 = 全分辨率/scale
        self.up_factor = scale * down // 2 if down > 1 else scale
        self.content_upsampler = None
        if content_aware_upsampling:
            if self.up_factor <= 1:
                raise ValueError(
                    'content-aware upsampling requires up_factor > 1')
            self.content_upsampler = ContentAwareFlowUpsampler(
                # IFBlock.in_planes includes the four flow channels appended
                # in forward; guidance is the image/warp/mask/time part only.
                guidance_channels=in_planes - 4,
                factor=self.up_factor,
                hidden_channels=content_aware_hidden_channels,
                residual_scale=content_aware_residual_scale)

    def forward(self, x, flow):
        scale = self.scale
        if scale != 1:
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        guidance = x
        x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        if self.up_factor != 1:
            if self.content_upsampler is not None:
                flow = self.content_upsampler(tmp[:, :4], guidance)
                # Deliberately preserve historical mask interpolation.
                mask = F.interpolate(
                    tmp[:, 4:5], scale_factor=self.up_factor,
                    mode="bilinear", align_corners=False)
                return flow, mask
            tmp = F.interpolate(
                tmp, scale_factor=self.up_factor,
                mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * self.up_factor
        mask = tmp[:, 4:5]
        return flow, mask
    
class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone
        zero_init_residual_heads = kargs.get('zero_init_residual_heads', False)
        compact_quarter_head = kargs.get('compact_quarter_head', False)
        pyramid_correlation = bool(
            kargs.get('pyramid_correlation', False))
        correlation_radii = list(kargs.get(
            'pyramid_correlation_radii', (6, 4, 3)))
        if pyramid_correlation and len(correlation_radii) < self.flow_num_stage:
            raise ValueError(
                'pyramid_correlation_radii must provide one radius per flow '
                f'stage, got {correlation_radii} for {self.flow_num_stage}')
        correlation_channels = int(
            kargs.get('pyramid_correlation_channels', 16))
        correlation_temperature = float(
            kargs.get('pyramid_correlation_temperature', 0.07))

        self.correspondence_pyramid = None
        self.correspondence_frozen = bool(
            kargs.get('pretrained_correspondence_frozen', True))
        pretrained_correspondence = bool(
            kargs.get('pretrained_correspondence', False))
        raw_external_stages = list(kargs.get(
            'pretrained_correspondence_stages', (0, 1)))
        stage_aliases = {'1/16': 0, '1/8': 1}
        try:
            self.correspondence_stage_indices = {
                stage_aliases.get(str(stage), int(stage))
                if not isinstance(stage, str) or stage not in stage_aliases
                else stage_aliases[stage]
                for stage in raw_external_stages
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'pretrained_correspondence_stages only supports 0/1 or '
                '1/16,1/8') from exc
        if pretrained_correspondence:
            if not self.correspondence_stage_indices:
                raise ValueError(
                    'pretrained_correspondence_stages cannot be empty')
            invalid_stages = sorted(
                stage for stage in self.correspondence_stage_indices
                if stage not in (0, 1) or stage >= self.flow_num_stage)
            if invalid_stages:
                raise ValueError(
                    'pretrained correspondence only supports available '
                    f'flow stages 0/1, got {invalid_stages}')
        external_radii = list(kargs.get(
            'pretrained_correspondence_radii', (6, 4)))
        if pretrained_correspondence:
            required_radius_count = max(
                self.correspondence_stage_indices) + 1
            if len(external_radii) < required_radius_count:
                raise ValueError(
                    'pretrained_correspondence_radii must provide radii for '
                    f'enabled stages {sorted(self.correspondence_stage_indices)}, got '
                    f'{external_radii}')
            self.correspondence_pyramid = CorrespondencePyramid(
                checkpoint_path=kargs.get(
                    'pretrained_correspondence_path'),
                feature_channels=128,
                transformer_layers=int(kargs.get(
                    'pretrained_correspondence_transformer_layers', 6)),
                max_feature_tokens=int(kargs.get(
                    'pretrained_correspondence_max_feature_tokens', 2880)),
                checkpoint_required=bool(kargs.get(
                    'pretrained_correspondence_required', True)))
            if self.correspondence_frozen:
                self.correspondence_pyramid.requires_grad_(False)

        external_temperature = float(kargs.get(
            'pretrained_correspondence_temperature', 0.07))
        external_init_scale = float(kargs.get(
            'pretrained_correspondence_init_scale', 0.01))
        self.block = nn.ModuleList([
            Head(
                kargs['embed_dims'][-1-i],
                kargs['scales'][-1-i],
                kargs['hidden_dims'][-1-i],
                7 if i == 0 else 18,
                zero_init=zero_init_residual_heads and i > 0,
                compact_feature=(compact_quarter_head
                                 and kargs['scales'][-1-i] == 4),
                correlation_radius=(
                    correlation_radii[i] if pyramid_correlation else None),
                correlation_channels=correlation_channels,
                correlation_temperature=correlation_temperature,
                external_correlation_radius=(
                    external_radii[i]
                    if (pretrained_correspondence
                        and i in self.correspondence_stage_indices) else None),
                external_correlation_temperature=external_temperature,
                external_correlation_init_scale=external_init_scale)
            for i in range(self.flow_num_stage)
        ])

        # 每级为 [scale, down, 通道倍率, blocks]，工作分辨率分别为 1/8、1/4。
        base_c = kargs['local_hidden_dims']
        local_zero_init = kargs.get('local_zero_init', False)
        content_aware_upsampling = bool(
            kargs.get('content_aware_upsampling', False))
        content_aware_hidden_channels = int(
            kargs.get('content_aware_hidden_channels', 32))
        content_aware_residual_scale = float(
            kargs.get('content_aware_residual_scale', 0.25))
        self.local_block = nn.ModuleList([
            IFBlock(18, c=max(int(base_c * cr) // 2 * 2, 16), scale=s, down=d,
                    blocks=b, zero_init=local_zero_init,
                    content_aware_upsampling=(
                        content_aware_upsampling
                        and index == len(LOCAL_CFG) - 1),
                    content_aware_hidden_channels=(
                        content_aware_hidden_channels),
                    content_aware_residual_scale=(
                        content_aware_residual_scale))
            for index, (s, d, cr, b) in enumerate(LOCAL_CFG)
        ])

        self.sparse_matcher = None
        self.sparse_matching_feature_encoder = None
        if bool(kargs.get('sparse_matching', False)):
            self.sparse_matching_feature_source = str(kargs.get(
                'sparse_matching_feature_source', 'mamba')).lower()
            if self.sparse_matching_feature_source not in ('mamba', 'gmflow'):
                raise ValueError(
                    'sparse_matching_feature_source must be mamba or gmflow, '
                    f'got {self.sparse_matching_feature_source!r}')
            if self.sparse_matching_feature_source == 'gmflow':
                feature_channels = 128
                feature_scale = 8
                self.sparse_matching_feature_encoder = GMFlowFeatureEncoder(
                    checkpoint_path=kargs.get(
                        'sparse_matching_pretrained_path'),
                    checkpoint_required=bool(kargs.get(
                        'sparse_matching_pretrained_required', True)),
                    feature_channels=feature_channels,
                    num_transformer_layers=6,
                    max_feature_tokens=int(kargs.get(
                        'sparse_matching_max_feature_tokens', 4224)))
                for parameter in (
                        self.sparse_matching_feature_encoder.parameters()):
                    parameter.requires_grad_(False)
            else:
                # The penultimate backbone level is 1/8 for the current
                # five-level architecture.
                self.sparse_matching_feature_index = -2
                feature_channels = kargs['embed_dims'][
                    self.sparse_matching_feature_index]
                feature_scale = kargs['scales'][-2]
            self.sparse_matcher = SparseGlobalMatcher(
                feature_channels=feature_channels,
                feature_scale=feature_scale,
                hidden_channels=int(
                    kargs.get('sparse_matching_hidden_channels', 32)),
                topk_ratio=float(
                    kargs.get('sparse_matching_topk_ratio', 0.02)),
                min_points=int(
                    kargs.get('sparse_matching_min_points', 16)),
                max_points=int(
                    kargs.get('sparse_matching_max_points', 128)),
                confidence_scale=float(
                    kargs.get('sparse_matching_confidence_scale', 0.05)),
                mutual_sigma=float(
                    kargs.get('sparse_matching_mutual_sigma', 1.5)),
                max_displacement=float(
                    kargs.get('sparse_matching_max_displacement', 96.0)),
                residual_scale=float(
                    kargs.get('sparse_matching_residual_scale', 0.5)),
                propagation_radius=int(
                    kargs.get('sparse_matching_propagation_radius', 2)),
                photometric_weight=float(
                    kargs.get('sparse_matching_photometric_weight', 0.5)))

        self.multi_hypothesis = None
        if bool(kargs.get('multi_hypothesis', False)):
            self.multi_hypothesis = MultiHypothesisBranch(
                hidden_channels=int(
                    kargs.get('multi_hypothesis_hidden_channels', 48)),
                work_scale=int(
                    kargs.get('multi_hypothesis_work_scale', 4)),
                max_flow_delta=float(
                    kargs.get('multi_hypothesis_max_flow_delta', 4.0)),
                max_mask_delta=float(
                    kargs.get('multi_hypothesis_max_mask_delta', 2.0)),
                max_mix=float(
                    kargs.get('multi_hypothesis_max_mix', 0.5)),
                disagreement_threshold=float(kargs.get(
                    'multi_hypothesis_disagreement_threshold', 0.03)),
                candidate_init_std=float(kargs.get(
                    'multi_hypothesis_candidate_init_std', 0.005)))

        self.pqmax = None
        self.pqmax_gradient_checkpointing = bool(
            kargs.get('pqmax_gradient_checkpointing', False))
        if bool(kargs.get('pqmax_enabled', False)):
            if self.correspondence_pyramid is None:
                raise ValueError(
                    'pqmax_enabled requires pretrained_correspondence=true')
            if 1 not in self.correspondence_stage_indices:
                raise ValueError(
                    'pqmax_enabled requires stage 1/8 in '
                    'pretrained_correspondence_stages')
            if self.correspondence_frozen:
                raise ValueError(
                    'PQMax is a jointly trained motion core; set '
                    'pretrained_correspondence_frozen=false')
            if self.multi_hypothesis is not None:
                raise ValueError(
                    'PQMax replaces the old multi_hypothesis ablation; '
                    'disable multi_hypothesis')
            self.pqmax = PQMaxMotionSynthesizer(
                feature_channels=128,
                num_fields=int(kargs.get('pqmax_num_fields', 4)),
                recurrent_iterations=int(kargs.get(
                    'pqmax_recurrent_iterations', 6)),
                recurrent_hidden_channels=int(kargs.get(
                    'pqmax_recurrent_hidden_channels', 160)),
                correlation_channels=int(kargs.get(
                    'pqmax_correlation_channels', 96)),
                local_radius=int(kargs.get('pqmax_local_radius', 4)),
                subpixel_radius=int(kargs.get('pqmax_subpixel_radius', 1)),
                correlation_temperature=float(kargs.get(
                    'pqmax_correlation_temperature', 0.05)),
                max_global_delta=float(kargs.get(
                    'pqmax_max_global_delta', 32.0)),
                recurrent_delta=float(kargs.get(
                    'pqmax_recurrent_delta', 2.0)),
                boundary_hidden_channels=int(kargs.get(
                    'pqmax_boundary_hidden_channels', 128)),
                boundary_blocks=int(kargs.get(
                    'pqmax_boundary_blocks', 10)),
                boundary_delta=float(kargs.get(
                    'pqmax_boundary_delta', 3.0)),
                fusion_temperature=float(kargs.get(
                    'pqmax_fusion_temperature', 0.35)),
                primary_logit_bias=float(kargs.get(
                    'pqmax_primary_logit_bias', 1.0)),
                detail_hidden_channels=int(kargs.get(
                    'pqmax_detail_hidden_channels', 64)),
                detail_blocks=int(kargs.get('pqmax_detail_blocks', 8)),
                detail_strength=float(kargs.get(
                    'pqmax_detail_strength', 0.75)),
                detail_residual_scale=float(kargs.get(
                    'pqmax_detail_residual_scale', 0.05)),
                gradient_checkpointing=(
                    self.pqmax_gradient_checkpointing))

        self.single_match = None
        if bool(kargs.get('single_match_enabled', False)):
            if self.correspondence_pyramid is None:
                raise ValueError(
                    'single_match_enabled requires '
                    'pretrained_correspondence=true')
            if 1 not in self.correspondence_stage_indices:
                raise ValueError(
                    'single_match_enabled requires stage 1/8 in '
                    'pretrained_correspondence_stages')
            if self.correspondence_frozen:
                raise ValueError(
                    'single_match jointly adapts correspondence; set '
                    'pretrained_correspondence_frozen=false')
            if self.pqmax is not None:
                raise ValueError(
                    'single_match replaces PQMax multi-field synthesis; '
                    'disable pqmax_enabled')
            if self.sparse_matcher is not None or self.multi_hypothesis is not None:
                raise ValueError(
                    'single_match must not be combined with sparse_matching '
                    'or multi_hypothesis')
            self.single_match = ConfidenceGatedMatchRefiner(
                feature_channels=128,
                recurrent_iterations=int(kargs.get(
                    'single_match_recurrent_iterations', 4)),
                recurrent_hidden_channels=int(kargs.get(
                    'single_match_recurrent_hidden_channels', 128)),
                correlation_channels=int(kargs.get(
                    'single_match_correlation_channels', 64)),
                local_radius=int(kargs.get(
                    'single_match_local_radius', 4)),
                subpixel_radius=int(kargs.get(
                    'single_match_subpixel_radius', 1)),
                correlation_temperature=float(kargs.get(
                    'single_match_correlation_temperature', 0.05)),
                max_global_delta=float(kargs.get(
                    'single_match_max_global_delta', 48.0)),
                recurrent_delta=float(kargs.get(
                    'single_match_recurrent_delta', 2.0)),
                mutual_sigma=float(kargs.get(
                    'single_match_mutual_sigma', 1.5)),
                similarity_gain_scale=float(kargs.get(
                    'single_match_similarity_gain_scale', 0.05)),
                gate_hidden_channels=int(kargs.get(
                    'single_match_gate_hidden_channels', 64)),
                gate_initial_probability=float(kargs.get(
                    'single_match_gate_initial_probability', 0.10)))

        self.version = int(kargs['version'])
        # PerVFI-inspired ablation.  This deliberately keeps the existing
        # flow estimator/refiner and only changes the two-warp blending rule,
        # so old checkpoints remain structurally compatible.  In regions
        # where both warps already agree we retain the original soft mask; in
        # disagreement regions the mask is sharpened towards one source to
        # avoid averaging two slightly misaligned textures.
        self.blend_mode = str(kargs.get('blend_mode', 'soft')).lower()
        if self.blend_mode not in ('soft', 'pervfi'):
            raise ValueError(
                f'blend_mode must be soft or pervfi, got {self.blend_mode}')
        self.pervfi_mask_temperature = float(
            kargs.get('pervfi_mask_temperature', 0.5))
        self.pervfi_disagreement_threshold = float(
            kargs.get('pervfi_disagreement_threshold', 0.03))
        self.pervfi_blend_strength = float(
            kargs.get('pervfi_blend_strength', 1.0))
        if not 0.0 < self.pervfi_mask_temperature <= 1.0:
            raise ValueError(
                'pervfi_mask_temperature must be in (0, 1], got '
                f'{self.pervfi_mask_temperature}')
        if self.pervfi_disagreement_threshold <= 0.0:
            raise ValueError(
                'pervfi_disagreement_threshold must be > 0, got '
                f'{self.pervfi_disagreement_threshold}')
        if not 0.0 <= self.pervfi_blend_strength <= 1.0:
            raise ValueError(
                'pervfi_blend_strength must be in [0, 1], got '
                f'{self.pervfi_blend_strength}')
        self.refine_res_scale = float(kargs.get('refine_res_scale', 0.25))
        if not 0.0 <= self.refine_res_scale <= 1.0:
            raise ValueError(
                f'refine_res_scale must be in [0, 1], '
                f'got {self.refine_res_scale}')
        if self.version == 3:
            self.unet = UnetWithResidualAttention(
                kargs['c'] * 2, kargs['M'],
                attn_dim=kargs.get('refine_attn_dim', 512),
                attn_heads=kargs.get('refine_attn_heads', 8),
                kv_pool=kargs.get('refine_kv_pool', 4))
        elif self.version == 2:
            self.unet = UnetWithAttention(kargs['c'] * 2, kargs['M'])
        else:
            self.unet = Unet(kargs['c'] * 2, kargs['M'])

    def train(self, mode=True):
        super().train(mode)
        # The external correspondence representation is a fixed prior.  In
        # particular, keep its BatchNorm statistics fixed during VFI training.
        if self.sparse_matching_feature_encoder is not None:
            self.sparse_matching_feature_encoder.eval()
        if (self.correspondence_pyramid is not None
                and self.correspondence_frozen):
            self.correspondence_pyramid.eval()
        return self

    def _correspondence_features(self, img0, img1):
        """Return stage-aligned 1/16, 1/8 pretrained descriptors."""
        if self.correspondence_pyramid is None:
            return [None] * self.flow_num_stage
        if self.correspondence_frozen:
            with torch.no_grad():
                pyramid = self.correspondence_pyramid(img0, img1)
        else:
            pyramid = self.correspondence_pyramid(img0, img1)
        features = [
            pyramid['1/16'] if 0 in self.correspondence_stage_indices else None,
            pyramid['1/8'] if 1 in self.correspondence_stage_indices else None,
        ]
        if self.flow_num_stage > 2:
            features.extend([None] * (self.flow_num_stage - 2))
        return features[:self.flow_num_stage]

    def _compose_prediction(self, merged, refine_output):
        if self.version == 3:
            res = refine_output * self.refine_res_scale
            pred = merged + res
            # 训练时保留越界像素梯度；评估/推理输出仍限制到合法图像范围。
            if not self.training:
                pred = torch.clamp(pred, 0, 1)
            return res, pred
        res = refine_output[:, :3] * 2 - 1
        return res, torch.clamp(merged + res, 0, 1)

    def _blend_warps(self, warped_img0, warped_img1, mask_logits):
        """Blend two warped images and return the actual probability mask.

        ``soft`` is the historical sigmoid blend and is kept bit-for-bit for
        control experiments. ``pervfi`` is a lightweight adaptation of
        PerVFI's quasi-binary asymmetric blending: only pixels where the two
        candidate warps disagree are progressively sharpened.  The
        disagreement gate is detached so the flow network cannot reduce this
        signal by making both warps artificially similar.
        """
        soft_mask = torch.sigmoid(mask_logits)
        if self.blend_mode == 'soft':
            return soft_mask, (
                warped_img0 * soft_mask
                + warped_img1 * (1.0 - soft_mask))

        sharp_mask = torch.sigmoid(
            mask_logits / self.pervfi_mask_temperature)
        disagreement = (
            warped_img0 - warped_img1).abs().mean(dim=1, keepdim=True).detach()
        disagreement_gate = (
            disagreement / self.pervfi_disagreement_threshold).clamp(0.0, 1.0)
        blend_amount = self.pervfi_blend_strength * disagreement_gate
        mask = soft_mask + blend_amount * (sharp_mask - soft_mask)

        # Written in primary/secondary form to make the asymmetric selection
        # explicit. It is numerically equivalent to mask-weighted blending,
        # while the sharpened mask suppresses double-image averaging.
        prefer_img0 = mask >= 0.5
        primary = torch.where(prefer_img0, warped_img0, warped_img1)
        secondary = torch.where(prefer_img0, warped_img1, warped_img0)
        primary_weight = torch.where(prefer_img0, mask, 1.0 - mask)
        merged = primary * primary_weight + secondary * (1.0 - primary_weight)
        return mask, merged

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        return y0, y1

    def _apply_sparse_matching(self, af, img0, img1, flow, timestep):
        if self.sparse_matcher is None:
            return flow
        if self.sparse_matching_feature_encoder is not None:
            with torch.no_grad():
                feature0, feature1 = (
                    self.sparse_matching_feature_encoder(img0, img1))
        else:
            feature = af[self.sparse_matching_feature_index]
            batch = img0.shape[0]
            feature0, feature1 = feature[:batch], feature[batch:]
        return self.sparse_matcher(
            feature0, feature1, img0, img1, flow, timestep)

    def _apply_single_match(self, img0, img1, flow, timestep,
                            correspondence_features):
        if self.single_match is None:
            return flow
        return self.single_match(
            img0, img1, flow, timestep, correspondence_features[1])

    def _apply_multi_hypothesis(self, img0, img1, warped_img0, warped_img1,
                                flow, mask, timestep, primary_merged):
        if self.multi_hypothesis is None:
            return primary_merged
        alternative_flow, alternative_mask, mix_logits = (
            self.multi_hypothesis.predict_alternative(
                img0, img1, warped_img0, warped_img1, flow, mask,
                timestep))
        alternative_warp0 = warp(img0, alternative_flow[:, :2])
        alternative_warp1 = warp(img1, alternative_flow[:, 2:4])
        _, alternative_merged = self._blend_warps(
            alternative_warp0, alternative_warp1, alternative_mask)
        return self.multi_hypothesis.combine(
            primary_merged, alternative_merged, warped_img0, warped_img1,
            mix_logits, flow, alternative_flow)

    def calculate_flow(self, imgs, timestep, local=False, af=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        if af is None:
            af = self.feature_bone(img0, img1)
        correspondence_features = self._correspondence_features(img0, img1)
        timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        for i in range(self.flow_num_stage):
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1),
                    flow,
                    external_features=correspondence_features[i]
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat([af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, timestep), 1),
                    None,
                    external_features=correspondence_features[i]
                    )

        flow = self._apply_sparse_matching(
            af, img0, img1, flow, timestep)
        flow = self._apply_single_match(
            img0, img1, flow, timestep, correspondence_features)

        if local:
            for block in self.local_block:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

                flow_d, mask_d = block(
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d

        if self.pqmax is not None:
            pq_output = self.pqmax(
                img0, img1, flow, mask, timestep,
                correspondence_features[1])
            flow = pq_output['flow']
            mask = pq_output['mask_logits']

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        if (self.pqmax_gradient_checkpointing and self.training
                and torch.is_grad_enabled()):
            tmp = checkpoint(
                self.unet, img0, img1, warped_img0, warped_img1,
                mask, flow, c0, c1, use_reentrant=False)
        else:
            tmp = self.unet(
                img0, img1, warped_img0, warped_img1,
                mask, flow, c0, c1)
        mask_, merged = self._blend_warps(warped_img0, warped_img1, mask)
        res, pred = self._compose_prediction(merged, tmp)
        return pred,warped_img0,warped_img1,mask_


    def forward(self, x, local=False, timestep=0.5, scale=0,
                return_all_merges=False):
        if scale > 0: 
            x_o = x
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        af = self.feature_bone(img0, img1)
        correspondence_features = self._correspondence_features(img0, img1)
        timestep = (x[:, :1].clone() * 0 + 1) * (timestep.float() if type(timestep) is not float else timestep)
        for i in range(self.flow_num_stage):
            if flow != None:
                flow_d, mask_d = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow,
                                                external_features=correspondence_features[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i]( torch.cat([af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1, timestep), 1), None,
                                            external_features=correspondence_features[i])
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            mask_prob, stage_merged = self._blend_warps(
                warped_img0, warped_img1, mask)
            mask_list.append(mask_prob)
            merged.append(stage_merged)

        flow = self._apply_sparse_matching(
            af, img0, img1, flow, timestep)
        flow = self._apply_single_match(
            img0, img1, flow, timestep, correspondence_features)
        # Recompute the inputs consumed by the first local block.  With the
        # optional matching branches disabled these are bit-identical to the
        # historical path.
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        if self.sparse_matcher is not None:
            mask_prob, stage_merged = self._blend_warps(
                warped_img0, warped_img1, mask)
            mask_list[-1] = mask_prob
            merged[-1] = stage_merged
        if self.single_match is not None:
            # Unlike the failed post-hoc sparse adapter, the corrected single
            # field is an explicit supervised motion stage.  It is then fed
            # into both existing local IFBlocks for boundary refinement.
            flow_list.append(flow)
            mask_prob, stage_merged = self._blend_warps(
                warped_img0, warped_img1, mask)
            mask_list.append(mask_prob)
            merged.append(stage_merged)

        # LC loss需要监督全部learned-feature与local阶段；普通loss仍只消费
        # 原有返回的merged，保证既有实验语义不变。
        all_merged = list(merged)
        
        if scale>0:
            img0, img1 = x_o[:, :3], x_o[:, 3:6]
            af1 = self.feature_bone(img0, img1)
            scale = img0.shape[3] / flow.shape[3]
            flow = F.interpolate(flow, scale_factor = scale, mode="bilinear", align_corners=False) * scale
            mask = F.interpolate(mask, scale_factor = scale, mode="bilinear", align_corners=False)
            # timestep 常数图与图像尺寸保持同步 (scale>0 + local 时必需)
            timestep = F.interpolate(timestep, scale_factor = scale, mode="bilinear", align_corners=False)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            mask_, stage_merged = self._blend_warps(
                warped_img0, warped_img1, mask)
            merged.append(stage_merged)
            all_merged.append(merged[-1])

        if local:
            # flow_list 保留前面 learned-feature heads, 用于所有阶段的flow监督。
            # mask/merged 仍只返回local阶段, 保持原有图像重建loss量级。
            merged = []
            mask_list = []
            
            for block in self.local_block:
                flow_d, mask_d = block(
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d

                flow_list.append(flow)
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                mask_prob, stage_merged = self._blend_warps(
                    warped_img0, warped_img1, mask)
                mask_list.append(mask_prob)
                merged.append(stage_merged)
                all_merged.append(merged[-1])

        pq_output = None
        if self.pqmax is not None:
            pq_output = self.pqmax(
                img0, img1, flow, mask, timestep,
                correspondence_features[1])
            flow = pq_output['flow']
            mask = pq_output['mask_logits']
            warped_img0 = pq_output['warp0']
            warped_img1 = pq_output['warp1']
            flow_list.append(flow)
            mask_list.append(pq_output['mask_probability'])
            merged.append(pq_output['merged'])
            all_merged.append(merged[-1])

        merged[-1] = self._apply_multi_hypothesis(
            img0, img1, warped_img0, warped_img1, flow, mask, timestep,
            merged[-1])
        # Keep the number and weighting of LC warp stages unchanged.  Only the
        # final warp candidate is replaced by the multi-hypothesis result.
        all_merged[-1] = merged[-1]
        
        if scale: 
            c0, c1 = self.warp_features(af1, flow)
        else:
            c0, c1 = self.warp_features(af, flow)
        if (self.pqmax_gradient_checkpointing and self.training
                and torch.is_grad_enabled()):
            tmp = checkpoint(
                self.unet, img0, img1, warped_img0, warped_img1,
                mask, flow, c0, c1, use_reentrant=False)
        else:
            tmp = self.unet(
                img0, img1, warped_img0, warped_img1,
                mask, flow, c0, c1)
        res, pred = self._compose_prediction(merged[-1], tmp)
        if pq_output is not None:
            pred = self.pqmax.restore_detail(
                img0, img1, merged[-1], pred, warped_img0, warped_img1,
                pq_output['fusion_entropy'])
            # The public ``res`` tensor always describes the complete final
            # correction relative to the last synthesized warp candidate.
            res = pred - merged[-1]
        outputs = (
            flow_list, mask_list, res, warped_img0, warped_img1, merged, pred)
        if return_all_merges:
            return outputs + (all_merged,)
        return outputs
