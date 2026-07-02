"""
nsa_fused_kernel.py — FUSE compression + window into the kernel path, and combine
all three NSA branches (compression + selection + window) with the gate.

Kernel rung, step 3 (FUSION). Does NOT wire into nsa_model.py (separate final step).
Purely additive: rung-1/2a files, the forward kernel (nsa_selection_kernel.py) and
the backward kernel (nsa_selection_backward.py) are untouched — this module IMPORTS
them and composes.

------------------------------------------------------------------------------
Design: separate-but-composable, not a mega-kernel
------------------------------------------------------------------------------
The three branches are each softmax-normalized INDEPENDENTLY and then linearly
blended by the gate — it is NOT one attention over concatenated keys. So instead of
one mega-kernel carrying three online-softmax states, we compose:

  * SELECTION : reuse the step-1/step-2 kernel (gathered top-k blocks). UNCHANGED.
  * COMPRESSION + WINDOW : ONE generic "range attention" kernel (fwd+bwd) — both are
    "query t attends a contiguous causal key range [lo(t), hi(t)]":
        - WINDOW   : raw keys,    lo = t-W+1,          hi = t,                 Lk = T
        - COMPRESS : pooled keys, lo = 0,              hi = (t+1)//Bsz - 1,    Lk = n_blk
    A runtime-bounded tile loop keeps window at O(T·W) (not O(T·T)).
  * POOLING (compression summary tokens): plain torch block-mean — a cheap O(T·D)
    reduction, differentiable, kept out of the kernel.
  * GATE COMBINE : torch elementwise.

Because each branch attention is its own autograd.Function (Triton fwd + Triton
bwd), PyTorch autograd chains the whole fused forward — no monolithic backward.

Traps preserved:
  * trap-1: block_idx (top-k selection) is a non-differentiable INPUT; the selection
    Function returns None for it. Compression/window don't use it.
  * each branch has its OWN mask; all-masked rows use finite NEG (=-1e9) + zeroing so
    empty rows give 0, never NaN (compression rows for t < Bsz have no valid block).
  * same online-softmax / gather(range)-not-scan structure; GP-padded tl.dot;
    dQ stored per-query, dK/dV scattered via fp32 atomic_add (+ autotune reset_to_zero).
"""

import torch
import triton
import triton.language as tl

# reuse selection forward+backward unchanged (composability)
from nsa_selection_kernel import selection_forward_reference
from nsa_selection_backward import selection_attn_triton

MODE_WINDOW = 0
MODE_COMPRESS = 1


def _cfgs():
    # trimmed vs the fwd/bwd kernels: these two-pass range kernels compile many
    # variants (2 MODEs × fwd/bwd × shapes), so keep the autotune space small.
    return [triton.Config({}, num_warps=w, num_stages=st)
            for w in (2, 4) for st in (1, 2)]


# ---------------------------------------------------------------------------
# generic range-attention FORWARD: query i attends contiguous causal [lo,hi]
# ---------------------------------------------------------------------------
@triton.autotune(configs=_cfgs(), key=["G", "D", "BLOCK_N", "MODE"])
@triton.jit
def _range_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sob, soh, sot,
    T, Lk, scale, W, BS,
    NUM_KV: tl.constexpr, MODE: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(0)
    i = tl.program_id(1)
    b = bh // NUM_KV
    hkv = bh % NUM_KV
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    g_ok = g < G
    h0 = hkv * G

    q = tl.load(q_ptr + b * sqb + i * sqt + (h0 + g)[:, None] * sqh + d[None, :],
                mask=g_ok[:, None], other=0.0).to(tl.float32)

    # per-query causal key range [lo, hi]  (MODE 0 = window, 1 = compression)
    if MODE == 0:
        hi = i
        lo = i - W + 1
    else:
        hi = (i + 1) // BS - 1
        lo = 0
    lo = tl.maximum(lo, 0)

    NEG = -1.0e9
    m = tl.full((GP,), NEG, tl.float32)
    l = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)

    lo0 = (lo // BLOCK_N) * BLOCK_N
    n_iter = tl.where(hi < lo, 0, (hi - lo0) // BLOCK_N + 1)   # 0 => empty row -> o=0
    for it in range(n_iter):
        j = lo0 + it * BLOCK_N + n
        ok = (j >= lo) & (j <= hi) & (j < Lk)
        safe = tl.where(ok, j, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + b * svb + hkv * svh + safe[:, None] * svt + d[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(ok[None, :], sc, NEG)
        m_cur = tl.max(sc, axis=1)
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)
        p = tl.where(ok[None, :], tl.exp(sc - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, vb, input_precision="ieee")
        m = m_new

    o = tl.where(l[:, None] == 0.0, 0.0, acc / l[:, None])
    tl.store(o_ptr + b * sob + i * sot + (h0 + g)[:, None] * soh + d[None, :],
             o.to(o_ptr.dtype.element_ty), mask=g_ok[:, None])


# ---------------------------------------------------------------------------
# generic range-attention BACKWARD (2-pass recompute; dQ store, dK/dV atomic)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_cfgs(), key=["G", "D", "BLOCK_N", "MODE"],
                 reset_to_zero=["dk_ptr", "dv_ptr"])
@triton.jit
def _range_attn_bwd_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, delta_ptr, dq_ptr, dk_ptr, dv_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt,
    sdob, sdoh, sdot, sdlb, sdlh, sdlt,
    sdqb, sdqh, sdqt, sdkb, sdkh, sdkt, sdvb, sdvh, sdvt,
    T, Lk, scale, W, BS,
    NUM_KV: tl.constexpr, MODE: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(0)
    i = tl.program_id(1)
    b = bh // NUM_KV
    hkv = bh % NUM_KV
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    g_ok = g < G
    h0 = hkv * G

    q = tl.load(q_ptr + b * sqb + i * sqt + (h0 + g)[:, None] * sqh + d[None, :],
                mask=g_ok[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + b * sdob + i * sdot + (h0 + g)[:, None] * sdoh + d[None, :],
                 mask=g_ok[:, None], other=0.0).to(tl.float32)
    delta = tl.load(delta_ptr + b * sdlb + i * sdlt + (h0 + g) * sdlh,
                    mask=g_ok, other=0.0).to(tl.float32)

    if MODE == 0:                                # 0 = window, 1 = compression
        hi = i
        lo = i - W + 1
    else:
        hi = (i + 1) // BS - 1
        lo = 0
    lo = tl.maximum(lo, 0)

    NEG = -1.0e9
    lo0 = (lo // BLOCK_N) * BLOCK_N
    n_iter = tl.where(hi < lo, 0, (hi - lo0) // BLOCK_N + 1)

    # pass 1: recompute (m, l)
    m = tl.full((GP,), NEG, tl.float32)
    l = tl.zeros((GP,), tl.float32)
    for it in range(n_iter):
        j = lo0 + it * BLOCK_N + n
        ok = (j >= lo) & (j <= hi) & (j < Lk)
        safe = tl.where(ok, j, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(ok[None, :], sc, NEG)
        m_cur = tl.max(sc, axis=1)
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)
        p = tl.where(ok[None, :], tl.exp(sc - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new
    l_safe = tl.where(l == 0.0, 1.0, l)

    # pass 2: grads
    dq = tl.zeros((GP, D), tl.float32)
    for it in range(n_iter):
        j = lo0 + it * BLOCK_N + n
        ok = (j >= lo) & (j <= hi) & (j < Lk)
        safe = tl.where(ok, j, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + b * svb + hkv * svh + safe[:, None] * svt + d[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(ok[None, :], sc, NEG)
        p = tl.where(ok[None, :], tl.exp(sc - m[:, None]), 0.0) / l_safe[:, None]
        dp = tl.dot(do, tl.trans(vb), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, kb, input_precision="ieee") * scale
        dk_c = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
        dv_c = tl.dot(tl.trans(p), do, input_precision="ieee")
        tl.atomic_add(dk_ptr + b * sdkb + hkv * sdkh + safe[:, None] * sdkt + d[None, :],
                      dk_c, mask=ok[:, None])
        tl.atomic_add(dv_ptr + b * sdvb + hkv * sdvh + safe[:, None] * sdvt + d[None, :],
                      dv_c, mask=ok[:, None])

    tl.store(dq_ptr + b * sdqb + i * sdqt + (h0 + g)[:, None] * sdqh + d[None, :],
             dq, mask=g_ok[:, None])


# ---------------------------------------------------------------------------
# python wrappers + autograd Function for the range attention
# ---------------------------------------------------------------------------
_BLOCK_N = 64


def _range_attn_forward(q, k, v, mode, window, block_size):
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    Lk = k.shape[2]
    o = torch.empty_like(q)
    grid = (B * Hkv, T)
    _range_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        T, Lk, float(D ** -0.5), window, block_size,
        NUM_KV=Hkv, MODE=mode, G=G, GP=16, D=D, BLOCK_N=_BLOCK_N,
    )
    return o


def _range_attn_backward(q, k, v, o, do, mode, window, block_size):
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    Lk = k.shape[2]
    do = do.contiguous()
    delta = (do.float() * o.float()).sum(-1).contiguous()          # [B,Hq,T]
    dq = torch.zeros(B, Hq, T, D, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, Hkv, Lk, D, device=q.device, dtype=torch.float32)
    dv = torch.zeros(B, Hkv, Lk, D, device=q.device, dtype=torch.float32)
    grid = (B * Hkv, T)
    _range_attn_bwd_kernel[grid](
        q, k, v, do, delta, dq, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        T, Lk, float(D ** -0.5), window, block_size,
        NUM_KV=Hkv, MODE=mode, G=G, GP=16, D=D, BLOCK_N=_BLOCK_N,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class _RangeAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mode, window, block_size):
        o = _range_attn_forward(q, k, v, mode, window, block_size)
        ctx.save_for_backward(q, k, v, o)
        ctx.mode, ctx.window, ctx.block_size = mode, window, block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o = ctx.saved_tensors
        dq, dk, dv = _range_attn_backward(q, k, v, o, do,
                                          ctx.mode, ctx.window, ctx.block_size)
        return dq, dk, dv, None, None, None


def window_attn_triton(q, k, v, window):
    return _RangeAttn.apply(q, k, v, MODE_WINDOW, window, 0)


def compression_attn_triton(q, k_cmp, v_cmp, block_size):
    # k_cmp/v_cmp are the POOLED summary tokens [B,Hkv,n_blk,D]
    return _RangeAttn.apply(q, k_cmp, v_cmp, MODE_COMPRESS, 0, block_size)


# ---------------------------------------------------------------------------
# pooling (torch, differentiable) — compression summary tokens (block mean)
# ---------------------------------------------------------------------------
def block_mean_pool(x, block_size):
    """[B,Hkv,T,D] -> [B,Hkv,n_blk,D] by averaging non-overlapping blocks."""
    B, H, T, D = x.shape
    n_blk = T // block_size
    x = x[:, :, :n_blk * block_size].reshape(B, H, n_blk, block_size, D)
    return x.mean(dim=3)


# ---------------------------------------------------------------------------
# FUSED forward (kernel path) and the three-branch REFERENCE (pure torch)
# ---------------------------------------------------------------------------
def nsa_fused_forward(q, kc, vc, ks, vs, kw, vw, gate_logits, block_idx,
                      block_size, window):
    """Full NSA attention output (pre out_proj) via the kernel path.

    q [B,Hq,T,D]; kc/vc/ks/vs/kw/vw [B,Hkv,T,D]; gate_logits [B,Hq,T,3];
    block_idx [B,Hkv,T,S] (non-differentiable selection input).
    """
    k_cmp = block_mean_pool(kc, block_size)
    v_cmp = block_mean_pool(vc, block_size)
    o_cmp = compression_attn_triton(q, k_cmp, v_cmp, block_size)
    o_slc = selection_attn_triton(q, ks, vs, block_idx, block_size)
    o_win = window_attn_triton(q, kw, vw, window)
    g = torch.sigmoid(gate_logits)                                  # [B,Hq,T,3]
    return g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win


def _range_attn_reference(q, k, v, mode, window, block_size):
    """Pure-torch dense reference for one range branch (window or compression)."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    Lk = k.shape[2]
    scale = D ** -0.5
    kexp = k.float().repeat_interleave(G, dim=1)
    vexp = v.float().repeat_interleave(G, dim=1)
    s = torch.matmul(q.float(), kexp.transpose(-1, -2)) * scale     # [B,Hq,T,Lk]
    tt = torch.arange(T, device=q.device)[:, None]
    jj = torch.arange(Lk, device=q.device)[None, :]
    if mode == MODE_WINDOW:
        keep = (jj <= tt) & (tt - jj < window)                     # [T,Lk]
    else:  # compression: summary j valid if (j+1)*Bsz-1 <= t
        blk_end = (jj + 1) * block_size - 1
        keep = blk_end <= tt
    keep = keep[None, None].expand(B, Hq, T, Lk)
    s = s.masked_fill(~keep, torch.finfo(s.dtype).min)
    row_has = keep.any(dim=-1, keepdim=True)
    p = torch.softmax(s, dim=-1) * row_has
    return torch.matmul(p, vexp).to(q.dtype)


def nsa_fused_reference(q, kc, vc, ks, vs, kw, vw, gate_logits, block_idx,
                        block_size, window):
    """Pure-torch three-branch reference (== rung-1 NSAAttention math, pre out_proj)."""
    k_cmp = block_mean_pool(kc, block_size)
    v_cmp = block_mean_pool(vc, block_size)
    o_cmp = _range_attn_reference(q, k_cmp, v_cmp, MODE_COMPRESS, 0, block_size)
    o_slc = selection_forward_reference(q, ks, vs, block_idx, block_size)
    o_win = _range_attn_reference(q, kw, vw, MODE_WINDOW, window, 0)
    g = torch.sigmoid(gate_logits)
    return g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win
