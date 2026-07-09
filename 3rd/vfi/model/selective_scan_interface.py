"""
mamba_ssm.ops.selective_scan_interface  —  CPU / MPS stub
==========================================================
Three implementations, each a drop-in for the CUDA kernel.

selective_scan_ref      — pure-Python sequential loop (baseline)
selective_scan_fast     — Hillis-Steele parallel prefix scan, O(log L) passes
selective_scan_fn       — alias for whichever variant is fastest on this platform

Hillis-Steele parallel prefix scan
-----------------------------------
The recurrence  x_t = a_t * x_{t-1} + b_t  with x_{-1} = 0  is a first-order
linear recurrence.  The pair (a_t, b_t) forms an associative semigroup under:

    (a_l, b_l) ⊕ (a_r, b_r) = (a_r * a_l, a_r * b_l + b_r)

After a stride-doubling inclusive prefix scan the second component at position t
equals x_t.  This replaces L sequential MPS kernel dispatches (~29 k for 736 p
Stage-3) with ⌈log₂ L⌉ ≈ 15 vectorised passes over the full tensor.

Copyright (c) 2023, Tri Dao, Albert Gu  (original reference logic)
Parallel-scan extension by project maintainers, 2025.
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    Pure-PyTorch sequential selective scan (reference / fallback).

    Shapes
    ------
    u          : (B, D, L)
    delta      : (B, D, L)
    A          : (D, N)
    B          : (B, N, L)  or  (B, G, N, L)   [variable B]
    C          : (B, N, L)  or  (B, G, N, L)   [variable C]
    D          : (D,)            optional skip
    z          : (B, D, L)       optional gate
    delta_bias : (D,)            optional bias added to delta
    Returns    : (B, D, L)  or  ((B, D, L), (B, D, N)) if return_last_state
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
        if is_variable_C:
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
    else:
        B = B.float()
        C = C.float()

    # Precompute deltaA  (B, D, L, N) and deltaB_u  (B, D, L, N)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            # B: (B, N, L)
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            # B: (B, G, N, L) — expand groups to full dim
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if is_variable_C and C.dim() == 4:
        # C: (B, G, N, L) — expand groups to full dim
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    # Sequential scan
    x = A.new_zeros((batch, dim, dstate))
    ys = []

    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (B, D, L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)

    last_state = x
    return out if not return_last_state else (out, last_state)


# ---------------------------------------------------------------------------
# Helper: Hillis-Steele parallel prefix scan
# ---------------------------------------------------------------------------

def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Hillis-Steele inclusive prefix scan over the *last* dimension.

    Computes  x_t = a_t * x_{t-1} + b_t  with  x_{-1} = 0  for all t in
    parallel using ⌈log₂L⌉ stride-doubling passes.

    Parameters
    ----------
    a, b : (..., L)   — element-wise coefficients, scan along last dim.

    Returns
    -------
    x : (..., L)   where  x[..., t] == x_t  in the recurrence above.

    Complexity
    ----------
    Time  : O(L log L) element ops, O(log L) sequential dispatch calls.
    Memory: 2× the size of a (or b) for temporaries per pass.
    """
    L = a.shape[-1]
    stride = 1
    while stride < L:
        # a_prev[..., t] = a[..., t-stride]  (identity 1 for t < stride)
        # b_prev[..., t] = b[..., t-stride]  (identity 0 for t < stride)
        a_prev = F.pad(a[..., :-stride], (stride, 0), value=1.0)
        b_prev = F.pad(b[..., :-stride], (stride, 0), value=0.0)
        # Combine (a_prev, b_prev) ⊕ (a, b):  (a*a_prev,  a*b_prev + b)
        b = a * b_prev + b
        a = a * a_prev
        stride *= 2
    return b  # == x_t for all t


def _parallel_scan_pair(a: torch.Tensor, b: torch.Tensor):
    """
    Hillis-Steele inclusive prefix scan — returns BOTH semigroup components.

    Same stride-doubling loop as _parallel_scan but returns (a, b) where on
    exit:
      a[..., t] = prod(a_orig[..., 0:t+1])   — cumulative A prefix products
      b[..., t] = h_t  with  h_{-1} = 0      — scan from zero initial state

    Used by selective_scan_chunked for carry propagation:
      h_out[..., t] = a[..., t] * h_carry + b[..., t]
    """
    L = a.shape[-1]
    stride = 1
    while stride < L:
        a_prev = F.pad(a[..., :-stride], (stride, 0), value=1.0)
        b_prev = F.pad(b[..., :-stride], (stride, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        stride *= 2
    return a, b  # (A_prefix_products, h_from_zero_init)


# ---------------------------------------------------------------------------
# Fast implementation: parallel prefix scan
# ---------------------------------------------------------------------------

def selective_scan_fast(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    Fast selective scan via Hillis-Steele parallel prefix scan.

    Drop-in replacement for selective_scan_ref.
    Reduces O(L) sequential MPS/CPU kernel dispatches to O(log₂L).
    Falls back to selective_scan_ref for complex-valued A.

    Shapes — identical to selective_scan_ref.
    """
    # Fall back gracefully for complex A (rare in VFIMamba)
    if A.is_complex():
        return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias,
                                  delta_softplus, return_last_state)

    dtype_in = u.dtype
    if dtype_in != torch.float32:
        u = u.float()
        delta = delta.float()

    if delta_bias is not None:
        delta = delta + (delta_bias[..., None].float() if delta_bias.dtype != torch.float32
                         else delta_bias[..., None])
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B_ = B if B.dtype == torch.float32 else B.float()
    C_ = C if C.dtype == torch.float32 else C.float()

    # ---- precompute deltaA [batch, dim, L, dstate] ----
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))   # [B, D, L, N]

    # ---- precompute deltaB_u [batch, dim, L, dstate] ----
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B_, u)
    elif B_.dim() == 3:
        # B: (B, N, L)
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B_, u)
    else:
        # B: (B, G, N, L) — expand groups to full dim
        B_ = repeat(B_, "B G N L -> B (G H) N L", H=dim // B_.shape[1])
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B_, u)

    if is_variable_C and C_.dim() == 4:
        C_ = repeat(C_, "B G N L -> B (G H) N L", H=dim // C_.shape[1])

    # ---- Parallel prefix scan: x_t = deltaA_t * x_{t-1} + deltaB_u_t ----
    # Transpose L to last dim for _parallel_scan  →  [B, D, N, L]
    a = deltaA.permute(0, 1, 3, 2).contiguous()    # [B, D, N, L]
    b = deltaB_u.permute(0, 1, 3, 2).contiguous()  # [B, D, N, L]

    x_all = _parallel_scan(a, b)   # [B, D, N, L]  — all hidden states

    # ---- Compute y_t = sum_n  x_{t,n} * C_{t,n} ----
    if not is_variable_C:
        # C: [D, N]
        y = torch.einsum("bdnl,dn->bdl", x_all, C_)
    elif C_.dim() == 3:
        # C: [B, N, L]
        y = torch.einsum("bdnl,bnl->bdl", x_all, C_)
    else:
        # C: [B, D, N, L]  (after group expansion above)
        y = (x_all * C_).sum(dim=2)   # [B, D, L]

    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)

    last_state = x_all[..., -1]   # [B, D, N]
    return out if not return_last_state else (out, last_state)


# ---------------------------------------------------------------------------
# Chunked implementation: associative scan with carry propagation
# ---------------------------------------------------------------------------

# Chunk length for the inner Hillis-Steele scan.
# Must be a power of 2.  For L = 29 440 (736p Stage-3):
#   Full scan: ⌈log₂ 29440⌉ = 15 passes over the full tensor
#   Chunk 2048: 11 passes per chunk × 15 chunks → ~27% fewer element ops,
#               and 14× smaller temporaries → much better cache reuse on MPS.
_SCAN_CHUNK = 2048


# def selective_scan_chunked(
#     u,
#     delta,
#     A,
#     B,
#     C,
#     D=None,
#     z=None,
#     delta_bias=None,
#     delta_softplus=False,
#     return_last_state=False,
# ):
#     """
#     Chunked associative scan — mathematically equivalent to selective_scan_fast.

#     Splits the sequence dimension L into chunks of _SCAN_CHUNK and runs
#     _parallel_scan_pair per chunk, propagating the carry hidden state:

#         h_out[t] = A_pfx[t] * h_carry + h0[t]

#     where A_pfx[t] and h0[t] come from the within-chunk Hillis-Steele scan
#     started from zero, and h_carry is the final state of the previous chunk.

#     Savings vs selective_scan_fast (L=29440, CHUNK=2048, Stage-3 BDN=1×256×16):
#       Hillis-Steele passes:  15 → 11  per chunk  (~27% fewer element ops)
#       Temporary tensor size: 14× smaller per pass (better L1/L2 cache reuse)

#     Falls back to selective_scan_ref for complex-valued A.
#     """
#     if A.is_complex():
#         return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias,
#                                   delta_softplus, return_last_state)

#     dtype_in = u.dtype
#     if dtype_in != torch.float32:
#         u = u.float()
#         delta = delta.float()

#     if delta_bias is not None:
#         delta = delta + (delta_bias[..., None].float() if delta_bias.dtype != torch.float32
#                          else delta_bias[..., None])
#     if delta_softplus:
#         delta = F.softplus(delta)

#     batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
#     L = u.shape[2]
#     is_variable_B = B.dim() >= 3
#     is_variable_C = C.dim() >= 3

#     B_ = B if B.dtype == torch.float32 else B.float()
#     C_ = C if C.dtype == torch.float32 else C.float()

#     # ---- precompute deltaA [B, D, L, N] and deltaB_u [B, D, L, N] ----
#     deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

#     if not is_variable_B:
#         deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B_, u)
#     elif B_.dim() == 3:
#         deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B_, u)
#     else:
#         B_ = repeat(B_, "B G N L -> B (G H) N L", H=dim // B_.shape[1])
#         deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B_, u)

#     if is_variable_C and C_.dim() == 4:
#         C_ = repeat(C_, "B G N L -> B (G H) N L", H=dim // C_.shape[1])

#     # Transpose to [B, D, N, L] for scan along last dim
#     a = deltaA.permute(0, 1, 3, 2).contiguous()    # [B, D, N, L]
#     b = deltaB_u.permute(0, 1, 3, 2).contiguous()  # [B, D, N, L]

#     # ---- Chunked associative scan with carry propagation ----
#     h_carry = torch.zeros(batch, dim, dstate, device=a.device, dtype=a.dtype)
#     ys = []

#     for start in range(0, L, _SCAN_CHUNK):
#         end = min(start + _SCAN_CHUNK, L)
#         a_c = a[..., start:end]  # [B, D, N, chunk]
#         b_c = b[..., start:end]  # [B, D, N, chunk]

#         # Within-chunk Hillis-Steele from zero initial state
#         A_pfx, h0 = _parallel_scan_pair(a_c, b_c)  # each [B, D, N, chunk]

#         # Carry correction: h_out[t] = A_pfx[t] * h_carry + h0[t]
#         h_out = A_pfx * h_carry.unsqueeze(-1) + h0  # [B, D, N, chunk]

#         # Update carry for next chunk
#         h_carry = h_out[..., -1]

#         # Compute y chunk via C-projection
#         if not is_variable_C:
#             # C_: [D, N]
#             y_c = torch.einsum("bdnl,dn->bdl", h_out, C_)
#         elif C_.dim() == 3:
#             # C_: [B, N, L]
#             y_c = torch.einsum("bdnl,bnl->bdl", h_out, C_[..., start:end])
#         else:
#             # C_: [B, D, N, L]
#             y_c = (h_out * C_[..., start:end]).sum(dim=2)  # [B, D, chunk]

#         ys.append(y_c)

#     y = torch.cat(ys, dim=-1)  # [B, D, L]

#     out = y if D is None else y + u * rearrange(D, "d -> d 1")
#     if z is not None:
#         out = out * F.silu(z)
#     out = out.to(dtype=dtype_in)

#     last_state = h_carry  # [B, D, N] — final hidden state
#     return out if not return_last_state else (out, last_state)

def selective_scan_chunked(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    if A.is_complex():
        return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias,
                                  delta_softplus, return_last_state)

    dtype_in = u.dtype
    if dtype_in != torch.float32:
        u = u.float()
        delta = delta.float()

    if delta_bias is not None:
        delta = delta + (delta_bias[..., None].float() if delta_bias.dtype != torch.float32
                         else delta_bias[..., None])
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    L = u.shape[2]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B_ = B if B.dtype == torch.float32 else B.float()
    C_ = C if C.dtype == torch.float32 else C.float()

    h_carry = torch.zeros(batch, dim, dstate, device=u.device, dtype=torch.float32)
    ys = []

    for start in range(0, L, _SCAN_CHUNK):
        end = min(start + _SCAN_CHUNK, L)

        # 只在当前chunk范围内切片原始输入
        u_c = u[..., start:end]
        delta_c = delta[..., start:end]

        if not is_variable_B:
            B_c = B_
        elif B_.dim() == 3:
            B_c = B_[..., start:end]
        else:
            B_c = B_[..., start:end]  # (B, G, N, chunk)

        if not is_variable_C:
            C_c = C_
        elif C_.dim() == 3:
            C_c = C_[..., start:end]
        else:
            C_c = C_[..., start:end]

        # ---- 预计算也只在chunk范围内做，避免整段L展开 ----
        deltaA_c = torch.exp(torch.einsum("bdl,dn->bdln", delta_c, A))   # [B,D,chunk,N]

        if not is_variable_B:
            deltaB_u_c = torch.einsum("bdl,dn,bdl->bdln", delta_c, B_c, u_c)
        elif B_c.dim() == 3:
            deltaB_u_c = torch.einsum("bdl,bnl,bdl->bdln", delta_c, B_c, u_c)
        else:
            B_c_exp = repeat(B_c, "B G N L -> B (G H) N L", H=dim // B_c.shape[1])
            deltaB_u_c = torch.einsum("bdl,bdnl,bdl->bdln", delta_c, B_c_exp, u_c)

        if is_variable_C and C_c.dim() == 4:
            C_c = repeat(C_c, "B G N L -> B (G H) N L", H=dim // C_c.shape[1])

        a_c = deltaA_c.permute(0, 1, 3, 2).contiguous()    # [B, D, N, chunk]
        b_c = deltaB_u_c.permute(0, 1, 3, 2).contiguous()  # [B, D, N, chunk]

        # ---- 块内Hillis-Steele scan（从零初始态） ----
        A_pfx, h0 = _parallel_scan_pair(a_c, b_c)

        # ---- carry修正 ----
        h_out = A_pfx * h_carry.unsqueeze(-1) + h0
        h_carry = h_out[..., -1]

        # ---- C投影得到y ----
        if not is_variable_C:
            y_c = torch.einsum("bdnl,dn->bdl", h_out, C_c)
        elif C_c.dim() == 3:
            y_c = torch.einsum("bdnl,bnl->bdl", h_out, C_c)
        else:
            y_c = (h_out * C_c).sum(dim=2)

        ys.append(y_c)

    y = torch.cat(ys, dim=-1)  # [B, D, L]

    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)

    last_state = h_carry
    return out if not return_last_state else (out, last_state)

# ---------------------------------------------------------------------------
# Alias: use the fast parallel scan as the default on non-CUDA systems
# ---------------------------------------------------------------------------

# torch.compile wrapper — compiled lazily on first call per input shape.
# Wraps selective_scan_chunked (Step D) for additional kernel-fusion speedup.
_compiled_chunked = None


def selective_scan_compiled(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    torch.compile'd chunked associative scan (default for MPS/CPU).

    First call triggers JIT compilation; subsequent calls use the cached kernel.
    """
    global _compiled_chunked
    if _compiled_chunked is None:
        try:
            _compiled_chunked = torch.compile(selective_scan_chunked, dynamic=False)
        except Exception:
            # torch.compile not available in this build — fall back gracefully
            _compiled_chunked = selective_scan_chunked
    return _compiled_chunked(u, delta, A, B, C, D, z, delta_bias,
                             delta_softplus, return_last_state)


# Public alias used by feature_extractor.py (imported as selective_scan_fn)
# selective_scan_fn = selective_scan_compiled
selective_scan_fn = selective_scan_chunked
