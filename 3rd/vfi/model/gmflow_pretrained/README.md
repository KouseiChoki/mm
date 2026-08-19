# GMFlow correspondence feature subset

This directory vendors the minimal GMFlow backbone and transformer needed by
the explicit-matching experiment.  It deliberately excludes GMFlow's dense
flow decoder and upsampler.

- Upstream: <https://github.com/haofeixu/gmflow>
- Upstream commit: `b5123431164d01ec14526a1c3d22218aecb62024`
- License: Apache-2.0; see `LICENSE`
- Checkpoint: official `gmflow_sintel-0c07dcb3.pth`
- Checkpoint SHA256:
  `0c07dcb35770464f38a5ff4de18c04177b242dc5de8cd2068adf46f3d4fe193a`

The wrapper in `features.py` accepts the repository's RGB `[0, 1]` tensors
and applies the normalization equivalent to official GMFlow's `[0, 255]`
input path.
