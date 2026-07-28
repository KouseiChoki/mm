import torch
import torch.nn as nn
import torch.nn.functional as F
# from .matching import forward_warp

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x, kernel):
    cc = torch.cat([x, torch.zeros_like(x)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros_like(cc)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4 * kernel)

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down, kernel)
        # stride slicing对奇数尺寸向上取整，重建会多出一行/列。
        # 裁回当前层大小，使任意crop尺寸都可使用5层Lap loss。
        up = up[..., :current.shape[-2], :current.shape[-1]]
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.register_buffer(
            'gauss_kernel', gauss_kernel(channels=channels),
            persistent=False)
        
    def pyramid(self, image):
        return laplacian_pyramid(
            img=image, kernel=self.gauss_kernel,
            max_levels=self.max_levels)

    @staticmethod
    def compare_pyramids(input_pyramid, target_pyramid):
        return sum(
            torch.nn.functional.l1_loss(input_level, target_level)
            for input_level, target_level
            in zip(input_pyramid, target_pyramid))

    def forward_with_target_pyramid(self, input, target_pyramid):
        return self.compare_pyramids(
            self.pyramid(input), target_pyramid)

    def forward(self, input, target):
        return self.compare_pyramids(
            self.pyramid(input), self.pyramid(target))


class CharbonnierLoss(torch.nn.Module):
    """LC-Mamba image reconstruction Charbonnier penalty."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, input, target):
        return torch.sqrt((input - target).square() + self.eps ** 2).mean()


class CensusLoss(torch.nn.Module):
    """Differentiable 7x7 census/soft-Hamming loss used by LC-Mamba.

    The transform follows the common VFI ternary census formulation: RGB is
    converted to luminance, every local sample is compared with the center
    pixel, and the normalized descriptors are compared with a soft Hamming
    distance. Border pixels without a complete patch are excluded.
    """

    def __init__(self, patch_size=7, transform_eps=0.81,
                 hamming_eps=0.1):
        super().__init__()
        patch_size = int(patch_size)
        if patch_size < 1 or patch_size % 2 != 1:
            raise ValueError(
                f'patch_size must be a positive odd integer, got {patch_size}')
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.transform_eps = float(transform_eps)
        self.hamming_eps = float(hamming_eps)

        channels = patch_size * patch_size
        kernels = torch.eye(channels).reshape(
            channels, 1, patch_size, patch_size)
        self.register_buffer('kernels', kernels, persistent=False)

    @staticmethod
    def _rgb_to_gray(image):
        if image.shape[1] != 3:
            raise ValueError(
                f'CensusLoss expects 3-channel RGB input, got {image.shape}')
        return (
            0.2989 * image[:, 0:1]
            + 0.5870 * image[:, 1:2]
            + 0.1140 * image[:, 2:3]
        )

    def _transform(self, image):
        gray = self._rgb_to_gray(image)
        patches = F.conv2d(
            gray, self.kernels.to(dtype=gray.dtype),
            padding=self.radius)
        difference = patches - gray
        return difference / torch.sqrt(
            self.transform_eps + difference.square())

    def forward(self, input, target):
        input_census = self._transform(input)
        target_census = self._transform(target)
        distance = (input_census - target_census).square()
        distance = distance / (self.hamming_eps + distance)
        distance = distance.mean(dim=1, keepdim=True)

        radius = self.radius
        if radius and (
                distance.shape[-2] > 2 * radius
                and distance.shape[-1] > 2 * radius):
            distance = distance[
                ..., radius:-radius, radius:-radius]
        return distance.mean()
