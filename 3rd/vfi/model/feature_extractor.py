import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    # timm<0.9 only exposes these helpers through timm.models.layers.
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from typing import Optional, Callable
from functools import partial

if torch.backends.mps.is_available():
    from model.selective_scan_interface import selective_scan_fn, selective_scan_ref
    _SCAN_BACKEND = 'mps_chunked'
else:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
        _SCAN_BACKEND = 'mamba_ssm_cuda'
    except ImportError:
        from model.selective_scan_interface import selective_scan_fn, selective_scan_ref
        _SCAN_BACKEND = 'fallback_chunked'
print(f'[selective_scan] backend = {_SCAN_BACKEND}')

try:
    from model.ssd_scan import ssd_scan                    # Mamba2 SSD (ssm_version=2)
except ImportError:
    from ssd_scan import ssd_scan                          # 单文件测试回退




class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    
    def merge_x(self, x): 
        B, C, H, W = x.shape
        L = 2 * H * W
        x = x.view(B, -1, L//2).transpose(1, 2)
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, L, C)
        return x.transpose(1, 2).contiguous()

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 4
        B = B // 2
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(torch.transpose(x, dim0=2, dim1=3).contiguous())], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        # print(y.shape)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B//2, H*W, 2, int(self.expand*C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B, H, W, int(self.expand*C))#.view(B//2, 2*H, 2*W, -1)
        

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2Dv2(nn.Module):
    """Mamba2 SSD 版四方向Cross-Scan块 (ssm_version=2)。

    与 SS2D (S6) 的外部行为一致: 输入(B,H,W,C) NHWC, 双帧沿batch拼接,
    merge_x 逐token交错两帧后做四方向扫描 (Mixed-SSM结构完全保留)。
    参数化差异 (SSD):
        Δ逐head:  x_proj 直接输出 (nheads + 2*d_state)
        A逐head标量: A_log (K*nheads,)
        D逐head:  (K*nheads,)
    删除 S6 的 dt_rank 低秩投影 / dt_projs 逐通道展开 / 对角A。
    """

    def __init__(
            self,
            d_model,
            d_state=64,
            d_conv=3,
            expand=2.,
            headdim=64,
            ssd_chunk=256,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % headdim == 0, \
            f'd_inner({self.d_inner}) 须被 headdim({headdim}) 整除'
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ssd_chunk = ssd_chunk
        K = 4

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, bias=conv_bias, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 每方向一个投影: d_inner → (dt_head + B + C)
        proj_out = self.nheads + 2 * self.d_state
        self.x_proj_weight = nn.Parameter(
            torch.stack([nn.Linear(self.d_inner, proj_out, bias=False,
                                   **factory_kwargs).weight for _ in range(K)], dim=0))

        # dt_bias: Mamba2式初始化 (softplus逆映射到[dt_min, dt_max])
        dt = torch.exp(torch.rand(K * self.nheads, **factory_kwargs)
                       * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A: 逐head标量, A=-exp(A_log), 初始化 uniform[1,16]
        A = torch.empty(K * self.nheads, **factory_kwargs).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(K * self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def merge_x(self, x):
        B, C, H, W = x.shape
        L = 2 * H * W
        x = x.view(B, -1, L // 2).transpose(1, 2)
        x = torch.cat([x[:B // 2], x[B // 2:]], dim=-1).reshape(B // 2, L, C)
        return x.transpose(1, 2).contiguous()

    def forward_core(self, x: torch.Tensor):
        Bfull, Cdim, H, W = x.shape
        L = 2 * H * W
        K = 4
        Bh = Bfull // 2
        # 四方向序列构造 (与SS2D完全一致: hw / wh × 正反)
        x_hwwh = torch.stack([
            self.merge_x(x),
            self.merge_x(torch.transpose(x, dim0=2, dim1=3).contiguous())
        ], dim=1).view(Bh, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)   # (Bh,K,D,L)

        # 投影出 dt/B/C
        x_dbl = torch.einsum('b k d l, k c d -> b k c l',
                             xs.view(Bh, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.nheads, self.d_state, self.d_state], dim=2)

        # 折叠视图: (Bh, K, L, ...) 逐方向调用SSD (各方向独立的A/D/dt_bias参数)
        xk = rearrange(xs, 'b k (h p) l -> b k l h p',
                       h=self.nheads, p=self.headdim).float()
        dtk = rearrange(dts, 'b k h l -> b k l h').float()
        Bk = rearrange(Bs, 'b k n l -> b k l n').float()
        Ck = rearrange(Cs, 'b k n l -> b k l n').float()
        A_log = rearrange(self.A_log, '(k h) -> k h', k=K)
        D = rearrange(self.D, '(k h) -> k h', k=K)
        dt_bias = rearrange(self.dt_bias, '(k h) -> k h', k=K)
        ys = []
        for k in range(K):
            yk = ssd_scan(xk[:, k], dtk[:, k], A_log[k], Bk[:, k], Ck[:, k],
                          D[k], dt_bias[k], chunk_size=self.ssd_chunk)
            ys.append(rearrange(yk, 'b l h p -> b (h p) l'))
        out_y = torch.stack(ys, dim=1)                                    # (Bh,K,D,L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(Bh, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(Bh, -1, W, H), dim0=2, dim1=3
                               ).contiguous().view(Bh, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(Bh, -1, W, H), dim0=2, dim1=3
                                  ).contiguous().view(Bh, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(
            B // 2, H * W, 2, int(self.expand * C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B, H, W, int(self.expand * C))
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        ssm_version = kwargs.pop('ssm_version', 1)
        if ssm_version == 2:
            self.self_attention = SS2Dv2(d_model=hidden_dim,
                                         d_state=kwargs.pop('ssd_dstate', 64),
                                         headdim=kwargs.pop('ssd_headdim', 64),
                                         ssd_chunk=kwargs.pop('ssd_chunk', 256),
                                         expand=mlp_ratio, dropout=attn_drop_rate, **kwargs)
        else:
            self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)

        x = input * self.skip_scale + self.self_attention(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()


class BiMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 **ssm_kwargs):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                norm_layer=nn.LayerNorm,
                d_state=16,
                **ssm_kwargs,
            ))


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(in_chans)

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

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class MambaFeature(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dims=[16, 32, 64, 128, 256],
                 depths=(2, 2, 2, 2, 2),
                 conv_stages=2,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super(MambaFeature, self).__init__()

        self.num_stages = len(embed_dims)

        self.conv_stages = conv_stages

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    block = BiMambaBlock(embed_dims[i], depths[i],
                                         ssm_version=kwargs.get('ssm_version', 1),
                                         ssd_dstate=kwargs.get('ssd_dstate', 64),
                                         ssd_headdim=kwargs.get('ssd_headdim', 64),
                                         ssd_chunk=kwargs.get('ssd_chunk', 256))
                    patch_embed = OverlapPatchEmbed(patch_size=3,
                                                    stride=2,
                                                    in_chans=embed_dims[i - 1],
                                                    embed_dim=embed_dims[i])
                setattr(self, f"patch_embed{i}", patch_embed)
            setattr(self, f"block{i}", block)

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

    def forward(self, x1, x2):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)

        features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i}", None)
            block = getattr(self, f"block{i}", None)

            if i > 0:
                x = patch_embed(x)
            
            x = block(x)

            features.append(x)

        return features


def feature_extractor(**kargs):
    model = MambaFeature(**kargs)
    return model
