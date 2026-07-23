import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from timm.layers import trunc_normal_
except ImportError:
    # timm<0.9 only exposes this helper through timm.models.layers.
    from timm.models.layers import trunc_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
   
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Unet(nn.Module):
    def __init__(self, c, M=False, out=3):
        super(Unet, self).__init__()
        self.down0 = Conv2(17+c, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c if not M else 16*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
        self.M = M
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow,c0[0], c1[0]), 1))
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1))
        s2 = self.down2(torch.cat((s1, c0[2], c1[2]), 1))
        s3 = self.down3(torch.cat((s2, c0[3], c1[3]), 1))
        x = self.up0(torch.cat((s3, c0[4], c1[4]), 1) if not self.M else s3)
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        x = self.conv(x)
        return torch.sigmoid(x)

class UnetWithAttention(nn.Module):
    def __init__(self, c, M=False, out=3):
        super().__init__()
        # 在 bottleneck 加入自注意力
        self.down0 = Conv2(17 + c, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        
        # Bottleneck 注意力模块
        self.bottleneck_attn = nn.MultiheadAttention(
            embed_dim=32 * c, num_heads=8, batch_first=True
        )
        
        self.up0 = deconv(32 * c if not M else 16 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, out, 3, 1, 1)
        self.M = M

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow, c0[0], c1[0]), 1))
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1))
        s2 = self.down2(torch.cat((s1, c0[2], c1[2]), 1))
        s3 = self.down3(torch.cat((s2, c0[3], c1[3]), 1))
        
        # Bottleneck attention
        bottleneck = torch.cat((s3, c0[4], c1[4]), 1) if not self.M else s3
        B, C, H, W = bottleneck.shape
        feat = bottleneck.flatten(2).permute(0, 2, 1)   # (B, H*W, C)
        feat, _ = self.bottleneck_attn(feat, feat, feat)
        bottleneck = feat.permute(0, 2, 1).view(B, C, H, W)
        
        x = self.up0(bottleneck)
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        return torch.sigmoid(self.conv(x))


class BottleneckResidualAttention(nn.Module):
    """低维 pooled-KV attention，避免直接在超宽 UNet bottleneck 上做全局 MHA。

    query 保留完整 1/16 网格，key/value 按 ``kv_pool`` 下采样。depthwise
    positional conv 提供局部位置信息，零初始化 residual gain 使新模块在
    finetune 起点严格退化为 identity。
    """

    def __init__(self, channels, attn_dim=512, num_heads=8, kv_pool=4):
        super().__init__()
        attn_dim = int(attn_dim)
        num_heads = int(num_heads)
        if attn_dim < num_heads or attn_dim % num_heads != 0:
            raise ValueError(
                f'refine_attn_dim({attn_dim}) must be >= and divisible by '
                f'refine_attn_heads({num_heads})')
        self.kv_pool = max(int(kv_pool), 1)
        self.in_proj = nn.Conv2d(channels, attn_dim, 1)
        self.pos_conv = nn.Conv2d(
            attn_dim, attn_dim, 3, 1, 1, groups=attn_dim)
        self.query_norm = nn.LayerNorm(attn_dim)
        self.kv_norm = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Conv2d(attn_dim, channels, 1)
        self.residual_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        feat = self.in_proj(x)
        feat = feat + self.pos_conv(feat)
        query = feat.flatten(2).transpose(1, 2)

        if self.kv_pool > 1 and min(feat.shape[-2:]) >= self.kv_pool:
            kv_feat = F.avg_pool2d(
                feat, kernel_size=self.kv_pool, stride=self.kv_pool,
                ceil_mode=True)
        else:
            kv_feat = feat
        key_value = kv_feat.flatten(2).transpose(1, 2)

        query = self.query_norm(query)
        key_value = self.kv_norm(key_value)
        attn_out, _ = self.attn(
            query, key_value, key_value, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape_as(feat)
        return x + self.residual_gain * self.out_proj(attn_out)


class UnetWithResidualAttention(Unet):
    """version=3 refine UNet。

    复用 version=1 的卷积/反卷积参数命名，便于从旧 checkpoint finetune；
    bottleneck attention 为 identity 起步，输出直接是 [-1, 1] 的残差方向。
    最终残差幅度由 MultiScaleFlow.refine_res_scale 控制。
    """

    def __init__(self, c, M=False, out=3, attn_dim=512,
                 attn_heads=8, kv_pool=4):
        super().__init__(c, M=M, out=out)
        bottleneck_channels = 32 * c if not M else 16 * c
        self.bottleneck_attention = BottleneckResidualAttention(
            bottleneck_channels, attn_dim=attn_dim,
            num_heads=attn_heads, kv_pool=kv_pool)

    def forward(self, img0, img1, warped_img0, warped_img1,
                mask, flow, c0, c1):
        s0 = self.down0(torch.cat(
            (img0, img1, warped_img0, warped_img1, mask, flow,
             c0[0], c1[0]), 1))
        s1 = self.down1(torch.cat((s0, c0[1], c1[1]), 1))
        s2 = self.down2(torch.cat((s1, c0[2], c1[2]), 1))
        s3 = self.down3(torch.cat((s2, c0[3], c1[3]), 1))
        bottleneck = (
            torch.cat((s3, c0[4], c1[4]), 1) if not self.M else s3)
        bottleneck = self.bottleneck_attention(bottleneck)
        x = self.up0(bottleneck)
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        return torch.tanh(self.conv(x))
