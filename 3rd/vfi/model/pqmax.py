"""High-capacity motion/synthesis core for the PQMax VFI experiment.

The module is intentionally optional so historical checkpoints keep their
original graph.  Unlike the earlier post-hoc matching ablations, this core
keeps several dense motion fields alive through all-pairs matching, recurrent
refinement, quarter-resolution boundary refinement and final image fusion.
Every field is exposed to the trainer for oracle/selection supervision.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from .warplayer import warp


def _conv(in_channels, out_channels, kernel_size=3, dilation=1):
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding,
            dilation=dilation),
        nn.PReLU(out_channels),
    )


def _coordinate_grid(batch, height, width, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing='ij')
    return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(
        batch, -1, -1, -1)


def _resize_flow(flow, size):
    """Resize a pixel-space flow while preserving x/y units."""
    old_height, old_width = flow.shape[-2:]
    new_height, new_width = size
    resized = F.interpolate(
        flow, size=size, mode='bilinear', align_corners=False).clone()
    resized[:, 0::2] *= new_width / old_width
    resized[:, 1::2] *= new_height / old_height
    return resized


def _project_bilateral_update(update, timestep):
    """Remove the update component that translates the target location.

    This projection is applied only to PQMax residuals.  The base Mamba flow
    remains free to represent non-linear motion learned by the old model.
    """
    update0, update1 = update[:, :2], update[:, 2:4]
    shift = (1.0 - timestep) * update0 + timestep * update1
    return torch.cat((update0 - shift, update1 - shift), dim=1)


class ConvGRU(nn.Module):
    def __init__(self, hidden_channels, input_channels):
        super().__init__()
        total = hidden_channels + input_channels
        self.update_gate = nn.Conv2d(total, hidden_channels, 3, 1, 1)
        self.reset_gate = nn.Conv2d(total, hidden_channels, 3, 1, 1)
        self.candidate = nn.Conv2d(total, hidden_channels, 3, 1, 1)

    def forward(self, hidden, inputs):
        joint = torch.cat((hidden, inputs), dim=1)
        update = torch.sigmoid(self.update_gate(joint))
        reset = torch.sigmoid(self.reset_gate(joint))
        candidate = torch.tanh(self.candidate(torch.cat(
            (reset * hidden, inputs), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, 1, dilation,
                dilation=dilation),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.activation = nn.PReLU(channels)

    def forward(self, inputs):
        return self.activation(inputs + self.body(inputs))


class FullResolutionDetailRestorer(nn.Module):
    """Restore trustworthy high frequencies after the legacy refiner.

    The source detail comes from the multi-field warp, not directly from an
    endpoint.  A learned gate can suppress it at occlusions.  The final
    residual layer starts at zero while a modest detail-copy gate is active,
    preventing the new model from beginning as another fully blurred output.
    """

    def __init__(self, hidden_channels=64, blocks=8, detail_strength=0.75,
                 residual_scale=0.05, gradient_checkpointing=False):
        super().__init__()
        hidden_channels = max(int(hidden_channels), 16)
        blocks = max(int(blocks), 1)
        self.detail_strength = float(detail_strength)
        self.residual_scale = float(residual_scale)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        # endpoints(6), fused/pred(6), selected warps(6), source/pred
        # high-pass(6), fusion entropy(1) = 25.
        self.entry = _conv(25, hidden_channels)
        self.body = nn.Sequential(*[
            ResidualBlock(hidden_channels, dilation=(1, 2, 3, 2)[i % 4])
            for i in range(blocks)
        ])
        self.output = nn.Conv2d(hidden_channels, 4, 3, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # sigmoid(-1.1) ~= 0.25: enough to preserve detail but not blindly
        # copy all warp artifacts before the gate learns occlusion semantics.
        with torch.no_grad():
            self.output.bias[3] = -1.1
        self.last_gate_mean = None
        self.last_detail_abs = None
        self.last_residual_abs = None

    @staticmethod
    def highpass(image):
        return image - F.avg_pool2d(
            image, kernel_size=3, stride=1, padding=1,
            count_include_pad=False)

    def forward(self, img0, img1, fused, pred, warp0, warp1,
                fusion_entropy):
        source_detail = self.highpass(fused)
        pred_detail = self.highpass(pred)
        features = self.entry(torch.cat((
            img0, img1, fused, pred, warp0, warp1,
            source_detail, pred_detail, fusion_entropy), dim=1))
        if (self.gradient_checkpointing and self.training
                and torch.is_grad_enabled()):
            features = checkpoint_sequential(
                self.body, len(self.body), features,
                use_reentrant=False)
        else:
            features = self.body(features)
        raw = self.output(features)
        learned_residual = self.residual_scale * torch.tanh(raw[:, :3])
        detail_gate = torch.sigmoid(raw[:, 3:4])
        detail_delta = (
            self.detail_strength * detail_gate
            * (source_detail - pred_detail))
        restored = pred + detail_delta + learned_residual
        self.last_gate_mean = detail_gate.detach().mean()
        self.last_detail_abs = detail_delta.detach().abs().mean()
        self.last_residual_abs = learned_residual.detach().abs().mean()
        return restored


class PQMaxMotionSynthesizer(nn.Module):
    """All-pairs, recurrent, multi-field motion and detail synthesis.

    The 1/8 correspondence descriptors may be initialized by the separately
    trained FlyingThings checkpoint, but are jointly optimized with VFI.  The
    full all-pairs matrix supplies several distinct global modes.  A shared
    ConvGRU then refines every mode with fresh bidirectional local cost volumes
    at each iteration.  A deep 1/4 head predicts full-resolution boundary
    corrections before sparse (low-temperature) field fusion.
    """

    def __init__(self, feature_channels=128, num_fields=4,
                 recurrent_iterations=6, recurrent_hidden_channels=160,
                 correlation_channels=96, local_radius=4,
                 subpixel_radius=1, correlation_temperature=0.05,
                 max_global_delta=32.0, recurrent_delta=2.0,
                 boundary_hidden_channels=128, boundary_blocks=10,
                 boundary_delta=3.0, fusion_temperature=0.35,
                 primary_logit_bias=1.0, detail_hidden_channels=64,
                 detail_blocks=8, detail_strength=0.75,
                 detail_residual_scale=0.05,
                 gradient_checkpointing=False):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.num_fields = int(num_fields)
        self.recurrent_iterations = int(recurrent_iterations)
        self.local_radius = int(local_radius)
        self.subpixel_radius = int(subpixel_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.max_global_delta = float(max_global_delta)
        self.recurrent_delta = float(recurrent_delta)
        self.boundary_delta = float(boundary_delta)
        self.fusion_temperature = float(fusion_temperature)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.num_fields < 2:
            raise ValueError('pqmax_num_fields must be >= 2')
        if self.recurrent_iterations < 1:
            raise ValueError('pqmax_recurrent_iterations must be >= 1')
        if self.local_radius < 1 or self.subpixel_radius < 0:
            raise ValueError('PQMax correlation radii are invalid')
        if self.correlation_temperature <= 0.0:
            raise ValueError('pqmax_correlation_temperature must be > 0')
        if self.fusion_temperature <= 0.0:
            raise ValueError('pqmax_fusion_temperature must be > 0')

        hidden = max(int(recurrent_hidden_channels), 32)
        correlation_channels = max(int(correlation_channels), 16)
        local_channels = 2 * (2 * self.local_radius + 1) ** 2
        self.correlation_encoder = nn.Sequential(
            _conv(local_channels, correlation_channels, 1),
            _conv(correlation_channels, correlation_channels),
        )
        # f0/f1(2C), encoded corr, field/base flow(8), warped RGB(6),
        # mask/t/confidence(3) = 2C + corr + 17.
        observation_channels = (
            2 * self.feature_channels + correlation_channels + 17)
        self.observation_encoder = nn.Sequential(
            _conv(observation_channels, hidden),
            _conv(hidden, hidden),
        )
        # f0/f1(2C), endpoint RGB(6), base flow(4), timestep(1).
        self.hidden_initializer = nn.Sequential(
            _conv(2 * self.feature_channels + 11, hidden),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.Tanh(),
        )
        self.gru = ConvGRU(hidden, hidden)
        self.recurrent_head = nn.Sequential(
            _conv(hidden, hidden),
            nn.Conv2d(hidden, 6, 3, 1, 1),
        )

        boundary_hidden = max(int(boundary_hidden_channels), 32)
        # endpoints(6), current warps(6), flow(4), mask/fusion/conf/t(4),
        # endpoint high-pass magnitudes(2) = 22.
        self.boundary_entry = _conv(22, boundary_hidden)
        self.boundary_body = nn.Sequential(*[
            ResidualBlock(
                boundary_hidden, dilation=(1, 2, 3, 2, 1)[i % 5])
            for i in range(max(int(boundary_blocks), 1))
        ])
        # PixelShuffle(4): flow(4), mask(1), fusion logit(1).
        self.boundary_output = nn.Conv2d(
            boundary_hidden, 6 * 16, 3, 1, 1)

        prior = torch.zeros(self.num_fields)
        prior[0] = float(primary_logit_bias)
        self.fusion_prior = nn.Parameter(prior)
        self.detail_restorer = FullResolutionDetailRestorer(
            hidden_channels=detail_hidden_channels,
            blocks=detail_blocks, detail_strength=detail_strength,
            residual_scale=detail_residual_scale,
            gradient_checkpointing=self.gradient_checkpointing)

        self.last_candidate_merges = None
        self.last_candidate_flows = None
        self.last_fusion_weights = None
        self.last_global_confidence = None
        self.last_fusion_entropy = None
        self.last_flow_spread = None

    def _subpixel_modes(self, correlation, height, width, count):
        """Return locally soft-argmaxed coordinates for distinct top modes."""
        batch, positions, _ = correlation.shape
        top_values, top_indices = correlation.topk(
            count, dim=2, largest=True, sorted=True)
        radius = self.subpixel_radius
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(
                -radius, radius + 1, device=correlation.device),
            torch.arange(
                -radius, radius + 1, device=correlation.device),
            indexing='ij')
        offsets_x = offsets_x.reshape(1, 1, 1, -1)
        offsets_y = offsets_y.reshape(1, 1, 1, -1)
        center_x = (top_indices % width).unsqueeze(-1)
        center_y = torch.div(
            top_indices, width, rounding_mode='floor').unsqueeze(-1)
        neighbor_x = (center_x + offsets_x).clamp(0, width - 1)
        neighbor_y = (center_y + offsets_y).clamp(0, height - 1)
        neighbor_indices = neighbor_y * width + neighbor_x
        neighbor_values = torch.gather(
            correlation, 2, neighbor_indices.reshape(batch, positions, -1)
        ).reshape_as(neighbor_indices)
        weights = torch.softmax(
            neighbor_values / self.correlation_temperature, dim=-1)
        coordinate_x = (
            weights * neighbor_x.to(correlation.dtype)).sum(dim=-1)
        coordinate_y = (
            weights * neighbor_y.to(correlation.dtype)).sum(dim=-1)
        coordinates = torch.stack((coordinate_x, coordinate_y), dim=2)
        coordinates = coordinates.permute(0, 1, 3, 2).reshape(
            batch, positions, 2 * count).transpose(1, 2).reshape(
                batch, 2 * count, height, width)

        if correlation.shape[2] > count:
            next_value = correlation.topk(
                count + 1, dim=2, largest=True, sorted=True).values[..., -1:]
            margins = top_values - next_value
        elif count > 1:
            margins = top_values - top_values[..., -1:]
        else:
            margins = torch.ones_like(top_values)
        confidence = torch.sigmoid(
            margins / self.correlation_temperature).transpose(1, 2).reshape(
                batch, count, height, width)
        return coordinates, confidence

    def _global_candidates(self, feature0, feature1, base_flow, timestep):
        batch, channels, height, width = feature0.shape
        norm0 = F.normalize(feature0, dim=1, eps=1e-6).flatten(2)
        norm1 = F.normalize(feature1, dim=1, eps=1e-6).flatten(2)
        correlation = torch.bmm(norm0.transpose(1, 2), norm1)
        mode_count = self.num_fields - 1
        forward_coordinates, forward_confidence = self._subpixel_modes(
            correlation, height, width, mode_count)
        reverse_coordinates, reverse_confidence = self._subpixel_modes(
            correlation.transpose(1, 2), height, width, mode_count)

        grid = _coordinate_grid(
            batch, height, width, feature0.device, feature0.dtype)
        endpoint0 = grid + base_flow[:, :2]
        endpoint1 = grid + base_flow[:, 2:4]
        candidates = [base_flow]
        confidences = [base_flow.new_ones((batch, 1, height, width))]
        for index in range(mode_count):
            forward_map = forward_coordinates[:, 2 * index:2 * index + 2]
            reverse_map = reverse_coordinates[:, 2 * index:2 * index + 2]
            desired1 = warp(forward_map, base_flow[:, :2])
            desired0 = warp(reverse_map, base_flow[:, 2:4])
            raw_update = torch.cat((
                desired0 - endpoint0, desired1 - endpoint1), dim=1)
            update = _project_bilateral_update(raw_update, timestep)
            magnitude = torch.maximum(
                torch.linalg.vector_norm(update[:, :2], dim=1, keepdim=True),
                torch.linalg.vector_norm(update[:, 2:4], dim=1, keepdim=True))
            update = update * torch.clamp(
                self.max_global_delta / magnitude.clamp(min=1e-6), max=1.0)
            candidates.append(base_flow + update)
            confidence0 = warp(
                reverse_confidence[:, index:index + 1], base_flow[:, 2:4])
            confidence1 = warp(
                forward_confidence[:, index:index + 1], base_flow[:, :2])
            confidences.append(torch.sqrt(
                (confidence0 * confidence1).clamp(min=1e-6)))
        return torch.stack(candidates, dim=1), torch.stack(confidences, dim=1)

    def _local_correlation(self, feature0, feature1):
        feature0 = F.normalize(feature0, dim=1, eps=1e-6)
        feature1 = F.normalize(feature1, dim=1, eps=1e-6)
        batch, channels, height, width = feature0.shape
        kernel = 2 * self.local_radius + 1
        patches1 = F.unfold(
            feature1, kernel_size=kernel, padding=self.local_radius).view(
                batch, channels, kernel * kernel, height, width)
        patches0 = F.unfold(
            feature0, kernel_size=kernel, padding=self.local_radius).view(
                batch, channels, kernel * kernel, height, width)
        forward = (feature0[:, :, None] * patches1).sum(dim=1)
        reverse = (feature1[:, :, None] * patches0).sum(dim=1)
        return torch.cat((forward, reverse), dim=1)

    @staticmethod
    def _repeat_fields(value, fields):
        return value[:, None].expand(
            -1, fields, -1, -1, -1).reshape(
                value.shape[0] * fields, *value.shape[1:])

    def _recurrent_step(
            self, hidden, flow_fields, mask_fields, fusion_logits,
            feature0_fields, feature1_fields, img0_fields, img1_fields,
            base_fields, timestep_fields, confidence_fields):
        warped_feature0 = warp(feature0_fields, flow_fields[:, :2])
        warped_feature1 = warp(feature1_fields, flow_fields[:, 2:4])
        correlation = self.correlation_encoder(
            self._local_correlation(warped_feature0, warped_feature1))
        warped_img0 = warp(img0_fields, flow_fields[:, :2])
        warped_img1 = warp(img1_fields, flow_fields[:, 2:4])
        observation = self.observation_encoder(torch.cat((
            warped_feature0, warped_feature1, correlation,
            flow_fields, base_fields, warped_img0, warped_img1,
            mask_fields, timestep_fields, confidence_fields), dim=1))
        hidden = self.gru(hidden, observation)
        raw = self.recurrent_head(hidden)
        update = _project_bilateral_update(
            torch.tanh(raw[:, :4]) * self.recurrent_delta,
            timestep_fields)
        return (
            hidden, flow_fields + update,
            mask_fields + raw[:, 4:5],
            fusion_logits + raw[:, 5:6])

    def _recurrent_refine(self, feature0, feature1, img0, img1,
                          fields, base_flow, base_mask, timestep,
                          confidence):
        batch, field_count, _, height, width = fields.shape
        feature0_fields = self._repeat_fields(feature0, field_count)
        feature1_fields = self._repeat_fields(feature1, field_count)
        img0_fields = self._repeat_fields(img0, field_count)
        img1_fields = self._repeat_fields(img1, field_count)
        base_fields = self._repeat_fields(base_flow, field_count)
        timestep_fields = self._repeat_fields(timestep, field_count)
        confidence_fields = confidence.reshape(
            batch * field_count, 1, height, width)
        flow_fields = fields.reshape(batch * field_count, 4, height, width)
        mask_fields = self._repeat_fields(base_mask, field_count)
        fusion_logits = torch.zeros_like(mask_fields)
        hidden = self.hidden_initializer(torch.cat((
            feature0_fields, feature1_fields, img0_fields, img1_fields,
            base_fields, timestep_fields), dim=1))

        recurrent_inputs = (
            feature0_fields, feature1_fields, img0_fields, img1_fields,
            base_fields, timestep_fields, confidence_fields)
        for _ in range(self.recurrent_iterations):
            step_inputs = (
                hidden, flow_fields, mask_fields, fusion_logits,
                *recurrent_inputs)
            if (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled()):
                hidden, flow_fields, mask_fields, fusion_logits = checkpoint(
                    self._recurrent_step, *step_inputs,
                    use_reentrant=False)
            else:
                hidden, flow_fields, mask_fields, fusion_logits = (
                    self._recurrent_step(*step_inputs))

        return (
            flow_fields.reshape(batch, field_count, 4, height, width),
            mask_fields.reshape(batch, field_count, 1, height, width),
            fusion_logits.reshape(batch, field_count, 1, height, width))

    @staticmethod
    def _edge_magnitude(image):
        gray = image.mean(dim=1, keepdim=True)
        return (gray - F.avg_pool2d(
            gray, 3, 1, 1, count_include_pad=False)).abs()

    def _boundary_refine(self, img0, img1, fields, masks, fusion_logits,
                         timestep, confidence):
        batch, field_count, _, full_height, full_width = fields.shape
        quarter_size = (full_height // 4, full_width // 4)
        flat_fields = fields.reshape(batch * field_count, 4,
                                     full_height, full_width)
        flow_quarter = _resize_flow(flat_fields, quarter_size)
        img0_quarter = F.interpolate(
            img0, size=quarter_size, mode='bilinear', align_corners=False,
            antialias=True)
        img1_quarter = F.interpolate(
            img1, size=quarter_size, mode='bilinear', align_corners=False,
            antialias=True)
        img0_fields = self._repeat_fields(img0_quarter, field_count)
        img1_fields = self._repeat_fields(img1_quarter, field_count)
        warped0 = warp(img0_fields, flow_quarter[:, :2])
        warped1 = warp(img1_fields, flow_quarter[:, 2:4])
        mask_quarter = F.interpolate(
            masks.reshape(batch * field_count, 1, *masks.shape[-2:]),
            size=quarter_size, mode='bilinear', align_corners=False)
        fusion_quarter = F.interpolate(
            fusion_logits.reshape(
                batch * field_count, 1, *fusion_logits.shape[-2:]),
            size=quarter_size, mode='bilinear', align_corners=False)
        confidence_quarter = F.interpolate(
            confidence.reshape(
                batch * field_count, 1, *confidence.shape[-2:]),
            size=quarter_size, mode='bilinear', align_corners=False)
        timestep_quarter = F.interpolate(
            timestep, size=quarter_size, mode='nearest')
        timestep_fields = self._repeat_fields(
            timestep_quarter, field_count)
        edge0 = self._repeat_fields(
            self._edge_magnitude(img0_quarter), field_count)
        edge1 = self._repeat_fields(
            self._edge_magnitude(img1_quarter), field_count)
        boundary = self.boundary_entry(torch.cat((
            img0_fields, img1_fields, warped0, warped1, flow_quarter,
            mask_quarter, fusion_quarter, confidence_quarter,
            timestep_fields, edge0, edge1), dim=1))
        if (self.gradient_checkpointing and self.training
                and torch.is_grad_enabled()):
            boundary = checkpoint_sequential(
                self.boundary_body, len(self.boundary_body), boundary,
                use_reentrant=False)
        else:
            boundary = self.boundary_body(boundary)
        raw = F.pixel_shuffle(
            self.boundary_output(boundary), 4)
        if raw.shape[-2:] != (full_height, full_width):
            raw = F.interpolate(
                raw, size=(full_height, full_width), mode='bilinear',
                align_corners=False)
        timestep_full = F.interpolate(
            timestep, size=(full_height, full_width), mode='nearest')
        timestep_full = self._repeat_fields(timestep_full, field_count)
        flow_update = _project_bilateral_update(
            torch.tanh(raw[:, :4]) * self.boundary_delta, timestep_full)
        fields = flat_fields + flow_update
        masks = F.interpolate(
            masks.reshape(batch * field_count, 1, *masks.shape[-2:]),
            size=(full_height, full_width), mode='bilinear',
            align_corners=False) + raw[:, 4:5]
        fusion_logits = F.interpolate(
            fusion_logits.reshape(
                batch * field_count, 1, *fusion_logits.shape[-2:]),
            size=(full_height, full_width), mode='bilinear',
            align_corners=False) + raw[:, 5:6]
        return (
            fields.reshape(batch, field_count, 4, full_height, full_width),
            masks.reshape(batch, field_count, 1, full_height, full_width),
            fusion_logits.reshape(
                batch, field_count, 1, full_height, full_width))

    def forward(self, img0, img1, base_flow, base_mask, timestep,
                correspondence_features):
        if correspondence_features is None:
            raise ValueError(
                'PQMax requires trainable 1/8 correspondence features')
        feature0, feature1 = correspondence_features
        feature_size = feature0.shape[-2:]
        base_flow_low = _resize_flow(base_flow, feature_size)
        base_mask_low = F.interpolate(
            base_mask, size=feature_size, mode='bilinear',
            align_corners=False)
        timestep_low = F.interpolate(
            timestep, size=feature_size, mode='nearest')
        img0_low = F.interpolate(
            img0, size=feature_size, mode='bilinear', align_corners=False,
            antialias=True)
        img1_low = F.interpolate(
            img1, size=feature_size, mode='bilinear', align_corners=False,
            antialias=True)

        fields, confidence = self._global_candidates(
            feature0, feature1, base_flow_low, timestep_low)
        fields, masks, fusion_logits = self._recurrent_refine(
            feature0, feature1, img0_low, img1_low, fields,
            base_flow_low, base_mask_low, timestep_low, confidence)
        full_size = base_flow.shape[-2:]
        fields = _resize_flow(
            fields.reshape(-1, 4, *feature_size), full_size).reshape(
                base_flow.shape[0], self.num_fields, 4, *full_size)
        fields, masks, fusion_logits = self._boundary_refine(
            img0, img1, fields, masks, fusion_logits, timestep, confidence)

        flat_fields = fields.reshape(-1, 4, *full_size)
        img0_fields = self._repeat_fields(img0, self.num_fields)
        img1_fields = self._repeat_fields(img1, self.num_fields)
        warped0 = warp(img0_fields, flat_fields[:, :2]).reshape(
            base_flow.shape[0], self.num_fields, 3, *full_size)
        warped1 = warp(img1_fields, flat_fields[:, 2:4]).reshape(
            base_flow.shape[0], self.num_fields, 3, *full_size)
        mask_probabilities = torch.sigmoid(masks)
        candidates = (
            warped0 * mask_probabilities
            + warped1 * (1.0 - mask_probabilities))
        prior = self.fusion_prior.view(1, self.num_fields, 1, 1, 1)
        weights = torch.softmax(
            (fusion_logits + prior) / self.fusion_temperature, dim=1)
        fused = (weights * candidates).sum(dim=1)
        selected_flow = (weights * fields).sum(dim=1)
        selected_mask = (weights * mask_probabilities).sum(dim=1)
        selected_mask_logits = torch.logit(
            selected_mask.clamp(1e-4, 1.0 - 1e-4))
        selected_warp0 = warp(img0, selected_flow[:, :2])
        selected_warp1 = warp(img1, selected_flow[:, 2:4])

        entropy = -(
            weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(self.num_fields)
        self.last_candidate_merges = candidates
        self.last_candidate_flows = fields
        self.last_fusion_weights = weights
        self.last_global_confidence = confidence.detach().mean()
        self.last_fusion_entropy = entropy.detach().mean()
        self.last_flow_spread = (
            fields[:, 1:] - fields[:, :1]).detach().abs().mean()
        return {
            'flow': selected_flow,
            'mask_logits': selected_mask_logits,
            'mask_probability': selected_mask,
            'warp0': selected_warp0,
            'warp1': selected_warp1,
            'merged': fused,
            'fusion_entropy': entropy,
        }

    def restore_detail(self, img0, img1, fused, pred, warp0, warp1,
                       fusion_entropy):
        restored = self.detail_restorer(
            img0, img1, fused, pred, warp0, warp1, fusion_entropy)
        if not self.training:
            restored = restored.clamp(0.0, 1.0)
        return restored
