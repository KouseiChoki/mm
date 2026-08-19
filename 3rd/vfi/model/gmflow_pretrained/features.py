"""Frozen GMFlow correspondence features for sparse VFI matching.

The backbone and transformer modules are vendored from the official GMFlow
repository.  This wrapper intentionally exposes only the 1/8 correspondence
features used by SGM-VFI; it does not run GMFlow's dense flow decoder.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .backbone import CNNEncoder
from .position import PositionEmbeddingSine
from .transformer import FeatureTransformer


class GMFlowFeatureEncoder(nn.Module):
    """Return pretrained, cross-frame-enhanced 1/8 feature maps.

    VFI tensors are RGB in [0, 1].  Official GMFlow receives [0, 255] and
    applies ImageNet normalization internally, so the equivalent operation
    here is ``(image - mean) / std``.
    """

    def __init__(self, checkpoint_path=None, checkpoint_required=True,
                 feature_channels=128, num_transformer_layers=6):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.backbone = CNNEncoder(
            output_dim=self.feature_channels, num_output_scales=1)
        self.transformer = FeatureTransformer(
            num_layers=int(num_transformer_layers),
            d_model=self.feature_channels,
            nhead=1,
            attention_type='swin',
            ffn_dim_expansion=4)
        self.register_buffer(
            'image_mean',
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'image_std',
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False)
        self.loaded_checkpoint = None
        if checkpoint_path:
            resolved = self.resolve_checkpoint(checkpoint_path)
            if resolved.is_file():
                self.load_pretrained(resolved)
            elif checkpoint_required:
                raise FileNotFoundError(
                    f'GMFlow checkpoint not found: {resolved}')

    @staticmethod
    def resolve_checkpoint(checkpoint_path):
        path = Path(checkpoint_path).expanduser()
        if path.is_file() or path.is_absolute():
            return path
        # model/gmflow_pretrained/features.py -> vfi/
        return Path(__file__).resolve().parents[2] / path

    def load_pretrained(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint.get('model', checkpoint)
        own_state = self.state_dict()
        compatible = {
            name: value for name, value in state.items()
            if name in own_state and own_state[name].shape == value.shape
        }
        required = [
            name for name in own_state
            if name.startswith(('backbone.', 'transformer.'))
        ]
        missing = [name for name in required if name not in compatible]
        if missing:
            raise RuntimeError(
                'GMFlow checkpoint is incompatible; missing '
                f'{len(missing)} feature keys, first={missing[:4]}')
        self.load_state_dict(compatible, strict=False)
        self.loaded_checkpoint = str(checkpoint_path.resolve())

    def forward(self, img0, img1):
        if img0.shape != img1.shape or img0.shape[1] != 3:
            raise ValueError(
                'GMFlow features require equal-shape RGB endpoint tensors')
        dtype = self.backbone.conv1.weight.dtype
        normalized0 = (
            img0.to(dtype=dtype) - self.image_mean.to(dtype=dtype)
        ) / self.image_std.to(dtype=dtype)
        normalized1 = (
            img1.to(dtype=dtype) - self.image_mean.to(dtype=dtype)
        ) / self.image_std.to(dtype=dtype)
        features = self.backbone(torch.cat((normalized0, normalized1), dim=0))
        feature = features[-1]
        feature0, feature1 = torch.chunk(feature, 2, dim=0)
        position = PositionEmbeddingSine(
            num_pos_feats=self.feature_channels // 2)(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position
        return self.transformer(
            feature0, feature1, attn_num_splits=1)
