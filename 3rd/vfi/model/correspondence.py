"""Reusable correspondence pyramid pretrained before VFI integration."""

from pathlib import Path

import torch
import torch.nn as nn

from .gmflow_pretrained import GMFlowFeatureEncoder


class CorrespondencePyramid(nn.Module):
    """GMFlow 1/8 descriptors and the learned 1/16 projection."""

    def __init__(self, checkpoint_path, feature_channels=128,
                 transformer_layers=6, max_feature_tokens=2880,
                 checkpoint_required=True):
        super().__init__()
        channels = int(feature_channels)
        self.encoder = GMFlowFeatureEncoder(
            checkpoint_path=None, checkpoint_required=False,
            feature_channels=channels,
            num_transformer_layers=int(transformer_layers),
            max_feature_tokens=int(max_feature_tokens))
        self.coarse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
        )
        self.loaded_checkpoint = None
        if checkpoint_path:
            path = self.resolve_checkpoint(checkpoint_path)
            if path.is_file():
                self.load_pretrained(path)
            elif checkpoint_required:
                raise FileNotFoundError(
                    f'correspondence checkpoint not found: {path}')

    @staticmethod
    def resolve_checkpoint(checkpoint_path):
        path = Path(checkpoint_path).expanduser()
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    def load_pretrained(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('model', checkpoint)
        own_state = self.state_dict()
        missing = [
            name for name, value in own_state.items()
            if name not in state or state[name].shape != value.shape]
        if missing:
            raise RuntimeError(
                'correspondence checkpoint incompatible; '
                f'missing/mismatched={missing[:8]}')
        self.load_state_dict(
            {name: state[name] for name in own_state}, strict=True)
        self.loaded_checkpoint = str(checkpoint_path.resolve())

    def forward(self, image0, image1):
        feature0_8, feature1_8 = self.encoder(image0, image1)
        return {
            '1/8': (feature0_8, feature1_8),
            '1/16': (self.coarse(feature0_8), self.coarse(feature1_8)),
        }
