from collections import OrderedDict
from threading import RLock

import torch


_MAX_GRID_CACHE_ENTRIES = 16
_MAX_GRID_CACHE_BYTES = 256 * 1024 * 1024
backwarp_tenGrid = OrderedDict()
_grid_cache_bytes = 0
_grid_cache_lock = RLock()


def clear_warp_cache():
    """Release all live base-grid references held by the warp cache."""
    global _grid_cache_bytes
    with _grid_cache_lock:
        backwarp_tenGrid.clear()
        _grid_cache_bytes = 0


def _grid_nbytes(grid):
    return grid.numel() * grid.element_size()


def _base_grid(tenFlow):
    global _grid_cache_bytes
    height, width = tenFlow.shape[-2:]
    grid_dtype = torch.get_default_dtype()
    device = tenFlow.device
    key = (device.type, device.index, grid_dtype, height, width)

    with _grid_cache_lock:
        grid = backwarp_tenGrid.get(key)
        if grid is not None:
            backwarp_tenGrid.move_to_end(key)
            return grid

        # Preserve the historical default-dtype coordinate grid. Construct a
        # regular no-grad tensor even if its first use is during inference,
        # because this module is shared by inference and training processes.
        with torch.inference_mode(False), torch.no_grad():
            horizontal = torch.linspace(
                -1.0, 1.0, width, device=device, dtype=grid_dtype
            ).view(1, 1, 1, width).expand(1, -1, height, -1)
            vertical = torch.linspace(
                -1.0, 1.0, height, device=device, dtype=grid_dtype
            ).view(1, 1, height, 1).expand(1, -1, -1, width)
            grid = torch.cat([horizontal, vertical], dim=1)

        grid_bytes = _grid_nbytes(grid)
        if grid_bytes > _MAX_GRID_CACHE_BYTES:
            return grid

        while backwarp_tenGrid and (
            len(backwarp_tenGrid) >= _MAX_GRID_CACHE_ENTRIES
            or _grid_cache_bytes + grid_bytes > _MAX_GRID_CACHE_BYTES
        ):
            _, evicted = backwarp_tenGrid.popitem(last=False)
            _grid_cache_bytes -= _grid_nbytes(evicted)

        backwarp_tenGrid[key] = grid
        _grid_cache_bytes += grid_bytes
        return grid


def warp(tenInput, tenFlow):
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (_base_grid(tenFlow) + tenFlow).permute(0, 2, 3, 1)

    if tenInput.device.type == 'mps':
        # MPS不支持border模式；clamp到[-1,1]后用zeros，数值上等价于border
        g = g.clamp(-1.0, 1.0)
        return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear',
                                               padding_mode='zeros', align_corners=True)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear',
                                           padding_mode='border', align_corners=True)
