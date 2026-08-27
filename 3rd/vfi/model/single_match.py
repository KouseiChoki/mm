"""Confidence-gated explicit matching for one bilateral flow field.

The module deliberately does not synthesize or select image candidates.  It
uses the trainable 1/8 correspondence representation to propose one globally
matched bilateral flow correction, filters that proposal with uniqueness,
forward/backward cycle consistency and similarity gain, and then performs a
small recurrent local refinement.  The corrected flow is returned to the
existing IFBlocks, which remain responsible for full-resolution boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pqmax import (
    ConvGRU, _conv, _coordinate_grid, _project_bilateral_update,
    _resize_flow,
)
from .warplayer import warp


class ConfidenceGatedMatchRefiner(nn.Module):
    """Inject explicit endpoint correspondence into a single VFI flow.

    All matching is performed on endpoint descriptors.  The resulting
    endpoint-to-endpoint map is sampled at the two locations currently reached
    by the bilateral base flow.  This converts an endpoint match into a
    correction on the invisible target-time grid without introducing a second
    image candidate or a selector.
    """

    def __init__(self, feature_channels=128, recurrent_iterations=4,
                 recurrent_hidden_channels=128, correlation_channels=64,
                 local_radius=4, subpixel_radius=1,
                 correlation_temperature=0.05, max_global_delta=48.0,
                 recurrent_delta=2.0, mutual_sigma=1.5,
                 similarity_gain_scale=0.05, gate_hidden_channels=64,
                 gate_initial_probability=0.10):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.recurrent_iterations = int(recurrent_iterations)
        self.local_radius = int(local_radius)
        self.subpixel_radius = int(subpixel_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.max_global_delta = float(max_global_delta)
        self.recurrent_delta = float(recurrent_delta)
        self.mutual_sigma = float(mutual_sigma)
        self.similarity_gain_scale = float(similarity_gain_scale)
        if self.recurrent_iterations < 0:
            raise ValueError('single_match_recurrent_iterations must be >= 0')
        if self.local_radius < 1 or self.subpixel_radius < 0:
            raise ValueError('single_match correlation radii are invalid')
        if self.correlation_temperature <= 0.0:
            raise ValueError(
                'single_match_correlation_temperature must be > 0')
        if self.max_global_delta <= 0.0 or self.recurrent_delta < 0.0:
            raise ValueError('single_match flow limits are invalid')
        if self.mutual_sigma <= 0.0 or self.similarity_gain_scale <= 0.0:
            raise ValueError('single_match confidence scales must be > 0')
        if not 0.0 < gate_initial_probability < 1.0:
            raise ValueError(
                'single_match_gate_initial_probability must be in (0,1)')

        gate_hidden = max(int(gate_hidden_channels), 16)
        # warped descriptors(2C), base flow(4), time/confidence/gain/cycle(4)
        gate_inputs = 2 * self.feature_channels + 8
        self.gate = nn.Sequential(
            _conv(gate_inputs, gate_hidden),
            _conv(gate_hidden, gate_hidden),
            nn.Conv2d(gate_hidden, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        gate_bias = torch.logit(torch.tensor(float(gate_initial_probability)))
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))

        hidden = max(int(recurrent_hidden_channels), 32)
        local_channels = 2 * (2 * self.local_radius + 1) ** 2
        correlation_channels = max(int(correlation_channels), 16)
        self.correlation_encoder = nn.Sequential(
            _conv(local_channels, correlation_channels, 1),
            _conv(correlation_channels, correlation_channels),
        )
        # descriptors(2C), local corr, flow/base(8), warped RGB(6),
        # time/confidence/gain/cycle/gate(5).
        observation_channels = (
            2 * self.feature_channels + correlation_channels + 19)
        self.observation_encoder = nn.Sequential(
            _conv(observation_channels, hidden),
            _conv(hidden, hidden),
        )
        # descriptors(2C), base flow(4), timestep/confidence(2).
        self.hidden_initializer = nn.Sequential(
            _conv(2 * self.feature_channels + 6, hidden),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.Tanh(),
        )
        self.gru = ConvGRU(hidden, hidden)
        self.recurrent_head = nn.Sequential(
            _conv(hidden, hidden),
            nn.Conv2d(hidden, 4, 3, 1, 1),
        )
        # The explicit global proposal is active at a conservative level;
        # learned recurrent changes begin as an exact zero residual.
        nn.init.zeros_(self.recurrent_head[-1].weight)
        nn.init.zeros_(self.recurrent_head[-1].bias)

        self.last_confidence = None
        self.last_gate = None
        self.last_active_ratio = None
        self.last_proposal_abs = None
        self.last_applied_abs = None
        self.last_recurrent_abs = None
        self.last_mutual_error = None
        self.last_similarity_gain = None
        self.last_similarity_improved_ratio = None
        self.last_margin = None

    def _subpixel_top1(self, correlation, height, width):
        """Top-1 coordinate with differentiable local soft-argmax."""
        batch, positions, targets = correlation.shape
        top_count = min(2, targets)
        top_values, top_indices = correlation.topk(
            top_count, dim=2, largest=True, sorted=True)
        center = top_indices[..., :1]
        radius = self.subpixel_radius
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=correlation.device),
            torch.arange(-radius, radius + 1, device=correlation.device),
            indexing='ij')
        offsets_x = offsets_x.reshape(1, 1, -1)
        offsets_y = offsets_y.reshape(1, 1, -1)
        center_x = center % width
        center_y = torch.div(center, width, rounding_mode='floor')
        neighbor_x = (center_x + offsets_x).clamp(0, width - 1)
        neighbor_y = (center_y + offsets_y).clamp(0, height - 1)
        neighbor_indices = neighbor_y * width + neighbor_x
        neighbor_values = torch.gather(
            correlation, 2, neighbor_indices)
        weights = torch.softmax(
            neighbor_values / self.correlation_temperature, dim=2)
        coordinate_x = (
            weights * neighbor_x.to(correlation.dtype)).sum(dim=2)
        coordinate_y = (
            weights * neighbor_y.to(correlation.dtype)).sum(dim=2)
        coordinates = torch.stack((coordinate_x, coordinate_y), dim=1)
        coordinates = coordinates.reshape(batch, 2, height, width)

        if top_count == 2:
            margin = top_values[..., 0] - top_values[..., 1]
            # Maps a tie to 0 and a decisive match towards 1.
            margin_confidence = (
                2.0 * torch.sigmoid(
                    margin / self.correlation_temperature) - 1.0)
        else:
            margin = torch.ones_like(top_values[..., 0])
            margin_confidence = torch.ones_like(margin)
        return (
            coordinates,
            margin.reshape(batch, 1, height, width),
            margin_confidence.reshape(batch, 1, height, width),
            top_values[..., 0].reshape(batch, 1, height, width),
        )

    def _global_proposal(self, feature0, feature1, base_flow, timestep,
                         full_size):
        batch, _, height, width = feature0.shape
        norm0 = F.normalize(feature0, dim=1, eps=1e-6)
        norm1 = F.normalize(feature1, dim=1, eps=1e-6)
        flat0 = norm0.flatten(2)
        flat1 = norm1.flatten(2)
        correlation = torch.bmm(flat0.transpose(1, 2), flat1)
        forward_map, forward_margin, forward_conf, forward_peak = (
            self._subpixel_top1(correlation, height, width))
        reverse_map, reverse_margin, reverse_conf, reverse_peak = (
            self._subpixel_top1(
                correlation.transpose(1, 2), height, width))

        grid = _coordinate_grid(
            batch, height, width, feature0.device, feature0.dtype)
        endpoint0 = grid + base_flow[:, :2]
        endpoint1 = grid + base_flow[:, 2:4]
        desired1 = warp(forward_map, base_flow[:, :2])
        desired0 = warp(reverse_map, base_flow[:, 2:4])

        # A true bidirectional match should return to the original endpoint.
        cycle0 = warp(reverse_map, desired1 - grid)
        cycle1 = warp(forward_map, desired0 - grid)
        mutual_error = 0.5 * (
            torch.linalg.vector_norm(cycle0 - endpoint0, dim=1, keepdim=True)
            + torch.linalg.vector_norm(
                cycle1 - endpoint1, dim=1, keepdim=True))
        mutual_confidence = torch.exp(-mutual_error / self.mutual_sigma)

        warped0 = warp(norm0, base_flow[:, :2])
        warped1 = warp(norm1, base_flow[:, 2:4])
        current_similarity = (warped0 * warped1).sum(dim=1, keepdim=True)
        matched_similarity = 0.5 * (
            warp(forward_peak, base_flow[:, :2])
            + warp(reverse_peak, base_flow[:, 2:4]))
        similarity_gain = matched_similarity - current_similarity
        gain_confidence = torch.sigmoid(
            similarity_gain / self.similarity_gain_scale)
        margin = 0.5 * (
            warp(forward_margin, base_flow[:, :2])
            + warp(reverse_margin, base_flow[:, 2:4]))
        margin_confidence = torch.sqrt((
            warp(forward_conf, base_flow[:, :2])
            * warp(reverse_conf, base_flow[:, 2:4])
        ).clamp_min(1e-8))
        confidence = (
            margin_confidence * mutual_confidence * gain_confidence
        ).clamp(0.0, 1.0)

        raw_update = torch.cat((
            desired0 - endpoint0, desired1 - endpoint1), dim=1)
        proposal = _project_bilateral_update(raw_update, timestep)
        # Enforce the configured limit in full-resolution pixel units.
        proposal_full = _resize_flow(proposal, full_size)
        magnitude = torch.maximum(
            torch.linalg.vector_norm(
                proposal_full[:, :2], dim=1, keepdim=True),
            torch.linalg.vector_norm(
                proposal_full[:, 2:4], dim=1, keepdim=True))
        proposal_full = proposal_full * torch.clamp(
            self.max_global_delta / magnitude.clamp_min(1e-6), max=1.0)
        proposal = _resize_flow(proposal_full, (height, width))
        return {
            'proposal': proposal,
            'confidence': confidence,
            'margin': margin,
            'mutual_error': mutual_error,
            'similarity_gain': similarity_gain,
            'warped_feature0': warped0,
            'warped_feature1': warped1,
        }

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

    def forward(self, img0, img1, base_flow, timestep,
                correspondence_features):
        if correspondence_features is None:
            raise ValueError(
                'single_match requires trainable 1/8 correspondence features')
        feature0, feature1 = correspondence_features
        feature_size = feature0.shape[-2:]
        full_size = base_flow.shape[-2:]
        base_flow_low = _resize_flow(base_flow, feature_size)
        timestep_low = F.interpolate(
            timestep, size=feature_size, mode='nearest')
        img0_low = F.interpolate(
            img0, size=feature_size, mode='bilinear', align_corners=False,
            antialias=True)
        img1_low = F.interpolate(
            img1, size=feature_size, mode='bilinear', align_corners=False,
            antialias=True)

        match = self._global_proposal(
            feature0, feature1, base_flow_low, timestep_low, full_size)
        gate_features = torch.cat((
            match['warped_feature0'], match['warped_feature1'],
            base_flow_low, timestep_low, match['confidence'],
            match['similarity_gain'], match['mutual_error']), dim=1)
        learned_gate = torch.sigmoid(self.gate(gate_features))
        gate = learned_gate * match['confidence']
        applied = gate * match['proposal']
        flow = base_flow_low + applied

        hidden = self.hidden_initializer(torch.cat((
            match['warped_feature0'], match['warped_feature1'],
            base_flow_low, timestep_low, match['confidence']), dim=1))
        recurrent_total = torch.zeros_like(flow)
        for _ in range(self.recurrent_iterations):
            warped_feature0 = warp(feature0, flow[:, :2])
            warped_feature1 = warp(feature1, flow[:, 2:4])
            correlation = self.correlation_encoder(
                self._local_correlation(warped_feature0, warped_feature1))
            warped_img0 = warp(img0_low, flow[:, :2])
            warped_img1 = warp(img1_low, flow[:, 2:4])
            observation = self.observation_encoder(torch.cat((
                warped_feature0, warped_feature1, correlation,
                flow, base_flow_low, warped_img0, warped_img1,
                timestep_low, match['confidence'],
                match['similarity_gain'], match['mutual_error'], gate), dim=1))
            hidden = self.gru(hidden, observation)
            update = _project_bilateral_update(
                torch.tanh(self.recurrent_head(hidden))
                * self.recurrent_delta,
                timestep_low)
            # Confident global matches receive the strongest recurrent update;
            # a small floor still lets local evidence repair missed matches.
            update = update * (0.25 + 0.75 * match['confidence'])
            flow = flow + update
            recurrent_total = recurrent_total + update

        corrected_flow = _resize_flow(flow, full_size)
        proposal_full = _resize_flow(match['proposal'], full_size)
        applied_full = _resize_flow(applied, full_size)
        recurrent_full = _resize_flow(recurrent_total, full_size)
        with torch.no_grad():
            self.last_confidence = match['confidence'].mean()
            self.last_gate = gate.mean()
            self.last_active_ratio = (gate > 0.05).to(gate.dtype).mean()
            self.last_proposal_abs = proposal_full.abs().mean()
            self.last_applied_abs = applied_full.abs().mean()
            self.last_recurrent_abs = recurrent_full.abs().mean()
            self.last_mutual_error = match['mutual_error'].mean()
            self.last_similarity_gain = match['similarity_gain'].mean()
            self.last_similarity_improved_ratio = (
                match['similarity_gain'] > 0).to(gate.dtype).mean()
            self.last_margin = match['margin'].mean()
        return corrected_flow
