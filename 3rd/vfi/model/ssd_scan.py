"""
model/ssd_scan.py — Mamba2 SSD (State Space Duality) 扫描后端
================================================================
为 SS2Dv2 (ssm_version=2) 提供 SSD 语义的选择性扫描, 双后端分发:

    CUDA + mamba_ssm≥2.x  →  mamba_chunk_scan_combined (官方triton kernel)
    MPS / CPU / 无kernel   →  ssd_chunked (纯PyTorch分块实现, 数学等价)

SSD 与 S6 的差异 (决定本文件接口):
    S6 : Δ逐通道 (B,D,L), A对角 (D,N)          → selective_scan_interface.py
    SSD: Δ逐head (B,L,H), A逐head标量 (H,)     → 本文件
SSD 的计算天然分块 (chunk内矩阵乘 + chunk间一阶递推), 纯PyTorch实现即规避
MPS INT_MAX 索引限制, 且串行深度仅为 L/chunk (S6 chunked为 log2(chunk)×L/chunk)。

公开接口:
    ssd_scan(x, dt, A_log, B, C, D, dt_bias, chunk_size, dt_softplus)
        x  : (batch, L, H, P)   H=nheads, P=headdim
        dt : (batch, L, H)      raw Δ (内部 softplus(dt + dt_bias))
        A_log: (H,)             A = -exp(A_log) 逐head标量
        B,C: (batch, L, N)      ngroups=1 (全head共享)
        D  : (H,)               skip
        返回 (batch, L, H, P)
参考: Dao & Gu, "Transformers are SSMs" (Mamba2), ssd_minimal.py
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# ── 后端探测 ─────────────────────────────────────────────────────────────────
_SSD_BACKEND = 'pytorch_chunked'
_mamba_chunk_scan_combined = None
if torch.cuda.is_available():
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined \
            as _mamba_chunk_scan_combined
        _SSD_BACKEND = 'mamba_ssm_triton'
    except Exception:
        _SSD_BACKEND = 'pytorch_chunked'
print(f'[ssd_scan] backend = {_SSD_BACKEND}')


# ─────────────────────────────────────────────────────────────────────────────
# 纯PyTorch分块SSD (数学参考 ssd_minimal_discrete)
# ─────────────────────────────────────────────────────────────────────────────

def _segsum(x):
    """(..., l) → (..., l, l) 下三角段和: out[i,j] = sum_{k=j+1..i} x[k], 其余 -inf。"""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, float('-inf'))


def ssd_chunked_discrete(X, A, B, C, chunk_size, initial_states=None):
    """
    离散化后的分块SSD。
    X: (b, l, h, p)  已含Δ的输入 (Δ·u)
    A: (b, l, h)     每步log域衰减 (Δ·(-exp(A_log)), ≤0)
    B: (b, l, h, n)  输入投影 (已展开到head)
    C: (b, l, h, n)  输出投影
    返回 Y: (b, l, h, p)
    """
    b, l, h, p = X.shape
    pad = (chunk_size - l % chunk_size) % chunk_size
    if pad:
        # A补0 (衰减=1, 状态直传), X/B/C补0 (无贡献); 输出末尾切掉
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
    Lp = X.shape[1]
    nc = Lp // chunk_size

    X = rearrange(X, 'b (c s) h p -> b c s h p', s=chunk_size)
    B = rearrange(B, 'b (c s) h n -> b c s h n', s=chunk_size)
    C = rearrange(C, 'b (c s) h n -> b c s h n', s=chunk_size)
    A = rearrange(A, 'b (c s) h -> b h c s', s=chunk_size)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. 对角块: chunk内 output
    L_mat = torch.exp(_segsum(A))                                  # (b,h,c,s,s)
    Y_diag = torch.einsum('bcshn,bczhn,bhcsz,bczhp->bcshp', C, B, L_mat, X)

    # 2. 各chunk末状态
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)        # (b,h,c,s)
    states = torch.einsum('bcshn,bhcs,bcshp->bchpn', B, decay_states, X)

    # 3. chunk间递推 (一阶线性, 长度nc的段和矩阵)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)            # (b,nc+1,h,p,n)
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[..., -1], (1, 0))))   # (b,h,nc+1,nc+1)
    new_states = torch.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
    states = new_states[:, :-1]                                    # 各chunk的入口状态

    # 4. 入口状态 → chunk内输出贡献
    state_decay_out = torch.exp(A_cumsum)                          # (b,h,c,s)
    Y_off = torch.einsum('bcshn,bchpn,bhcs->bcshp', C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, 'b c s h p -> b (c s) h p')
    return Y[:, :l]


# ─────────────────────────────────────────────────────────────────────────────
# 高层包装: 离散化 + 后端分发
# ─────────────────────────────────────────────────────────────────────────────

def ssd_scan(x, dt, A_log, B, C, D=None, dt_bias=None,
             chunk_size=256, dt_softplus=True):
    """
    x     : (batch, L, H, P) float
    dt    : (batch, L, H)    raw Δ
    A_log : (H,)             A = -exp(A_log)
    B, C  : (batch, L, N)    ngroups=1
    D     : (H,) or None
    返回  : (batch, L, H, P)
    """
    if _SSD_BACKEND == 'mamba_ssm_triton' and x.is_cuda:
        A = -torch.exp(A_log.float())
        return _mamba_chunk_scan_combined(
            x, dt, A, B.unsqueeze(2), C.unsqueeze(2),     # (b,l,g=1,n)
            chunk_size=chunk_size, D=D,
            dt_bias=dt_bias, dt_softplus=dt_softplus)

    # 纯PyTorch路径 (MPS/CPU/无kernel的CUDA)
    dtype_in = x.dtype
    x = x.float(); dt = dt.float(); B = B.float(); C = C.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)
    A_step = dt * (-torch.exp(A_log.float()))                      # (b,l,h)
    X = x * dt.unsqueeze(-1)                                       # Δ·u
    h = x.shape[2]
    Bh = repeat(B, 'b l n -> b l h n', h=h)                        # g=1 → 逐head展开
    Ch = repeat(C, 'b l n -> b l h n', h=h)
    Y = ssd_chunked_discrete(X, A_step, Bh, Ch, chunk_size)
    if D is not None:
        Y = Y + x * D.float().view(1, 1, -1, 1)
    return Y.to(dtype_in)


# ─────────────────────────────────────────────────────────────────────────────
# 串行参考实现 (仅测试用, 逐步递推)
# ─────────────────────────────────────────────────────────────────────────────

def ssd_scan_ref(x, dt, A_log, B, C, D=None, dt_bias=None, dt_softplus=True):
    """逐步串行SSM, 语义与 ssd_scan 相同, O(L)循环, 仅用于数值验证。"""
    x = x.float(); dt = dt.float(); B = B.float(); C = C.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = F.softplus(dt)
    b, l, h, p = x.shape
    n = B.shape[-1]
    A = -torch.exp(A_log.float())                                  # (h,)
    state = torch.zeros(b, h, p, n, dtype=torch.float32, device=x.device)
    ys = []
    for t in range(l):
        decay = torch.exp(dt[:, t] * A)                            # (b,h)
        state = state * decay[..., None, None] \
            + torch.einsum('bhp,bn,bh->bhpn', x[:, t], B[:, t], dt[:, t])
        ys.append(torch.einsum('bhpn,bn->bhp', state, C[:, t]))
    Y = torch.stack(ys, dim=1)                                     # (b,l,h,p)
    if D is not None:
        Y = Y + x * D.float().view(1, 1, -1, 1)
    return Y
