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
# Q-TILED forward: raise tl.dot M by processing BLOCK_Q query positions per
# program (perf rung). One program handles BLOCK_Q consecutive queries x G heads
# = M = BLOCK_Q*G real rows (NO per-position pad to 16). With BLOCK_Q=16, M=16*G
# is always a multiple of 16, and the real model (G=4) lands on M=64 -> full
# tensor-core tiles instead of the M=16 pad. Consecutive queries also share
# overlapping causal key ranges, so each K/V block is loaded once for all
# BLOCK_Q rows (~BLOCK_Q x less K/V traffic on the window branch).
#
# Correctness is identical to _range_attn_fwd_kernel: same online softmax, same
# per-(row,key) causal mask, same finite-NEG empty-row -> 0 handling. Purely
# additive; the original per-query kernel above is untouched and stays the
# fallback (USE_QTILE / unfriendly G).
# ---------------------------------------------------------------------------
@triton.autotune(configs=[triton.Config({}, num_warps=w, num_stages=st)
                          for w in (4, 8) for st in (1, 2)],
                 key=["G", "D", "BLOCK_N", "BLOCK_Q", "MODE"])
@triton.jit
def _range_attn_fwd_qtiled_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sob, soh, sot,
    T, Lk, scale, W, BS,
    NUM_KV: tl.constexpr, MODE: tl.constexpr,
    G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_Q: tl.constexpr,
    M_PAD: tl.constexpr,
):
    bh = tl.program_id(0)
    qb = tl.program_id(1)
    b = bh // NUM_KV
    hkv = bh % NUM_KV
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)

    # M_PAD (power of 2) >= BLOCK_Q*G real rows; the last (M_PAD - BLOCK_Q*G) rows
    # are padding lanes, masked out. Real model G=4 -> M_PAD=64 exact (no pad).
    ROWS: tl.constexpr = BLOCK_Q * G
    r = tl.arange(0, M_PAD)
    qi = r // G                       # which query in the tile
    gh = r % G                        # which head within the GQA group
    i0 = qb * BLOCK_Q
    qpos = i0 + qi                    # [M_PAD] absolute query position
    head = hkv * G + gh              # [M_PAD] absolute query head
    rmask = (r < ROWS) & (qpos < T)  # drop pad lanes + tail (T % BLOCK_Q != 0)

    q = tl.load(q_ptr + b * sqb + qpos[:, None] * sqt + head[:, None] * sqh + d[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)

    # per-row causal key range [lo_r, hi_r]  (MODE 0 = window, 1 = compression)
    if MODE == 0:
        hi_r = qpos
        lo_r = qpos - W + 1
    else:
        hi_r = (qpos + 1) // BS - 1
        lo_r = qpos * 0
    lo_r = tl.maximum(lo_r, 0)

    # scalar union range over the whole tile -> key-loop bounds
    i_last = tl.minimum(i0 + BLOCK_Q - 1, T - 1)
    if MODE == 0:
        lo_min = tl.maximum(i0 - W + 1, 0)
        hi_max = i_last
    else:
        lo_min = 0
        hi_max = (i_last + 1) // BS - 1

    NEG = -1.0e9
    m = tl.full((M_PAD,), NEG, tl.float32)
    l = tl.zeros((M_PAD,), tl.float32)
    acc = tl.zeros((M_PAD, D), tl.float32)

    lo0 = (lo_min // BLOCK_N) * BLOCK_N
    n_iter = tl.where(hi_max < lo_min, 0, (hi_max - lo0) // BLOCK_N + 1)
    for it in range(n_iter):
        j = lo0 + it * BLOCK_N + n                       # [BLOCK_N] key positions
        okj = (j >= 0) & (j < Lk)
        safe = tl.where(okj, j, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=okj[:, None], other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + b * svb + hkv * svh + safe[:, None] * svt + d[None, :],
                     mask=okj[:, None], other=0.0).to(tl.float32)
        # per-(row,key) causal validity
        ok = (j[None, :] >= lo_r[:, None]) & (j[None, :] <= hi_r[:, None]) \
            & okj[None, :] & rmask[:, None]
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(ok, sc, NEG)
        m_cur = tl.max(sc, axis=1)
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)
        p = tl.where(ok, tl.exp(sc - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, vb, input_precision="ieee")
        m = m_new

    o = tl.where(l[:, None] == 0.0, 0.0, acc / l[:, None])
    tl.store(o_ptr + b * sob + qpos[:, None] * sot + head[:, None] * soh + d[None, :],
             o.to(o_ptr.dtype.element_ty), mask=rmask[:, None])


# ---------------------------------------------------------------------------
# Q-TILED backward (matches _range_attn_bwd_kernel; BLOCK_Q queries per program).
# Same 2-pass recompute + per-(row,key) causal mask as the tiled forward. dK/dV
# for a key block are summed over ALL BLOCK_Q*G rows via tl.dot(trans(ds), q)
# before a SINGLE atomic_add per (program, key block) -> both wider MMA tiles AND
# fewer atomics than the per-query kernel. Pad rows (r >= BLOCK_Q*G) carry p=0/q=0
# so they contribute nothing to dq/dk/dv.
# ---------------------------------------------------------------------------
# NOTE num_stages=1 only: the 2-pass backward keeps many tiles live (dq, q, do,
# kb, vb, p, ds); at D=128 / M_PAD=64 / BLOCK_N=64, num_stages=2 double-buffers to
# ~144KB shared > the 3090's ~99KB (OutOfResources). Single-stage fits.
@triton.autotune(configs=[triton.Config({}, num_warps=w, num_stages=1)
                          for w in (4, 8)],
                 key=["G", "D", "BLOCK_N", "BLOCK_Q", "MODE"],
                 reset_to_zero=["dk_ptr", "dv_ptr"])
@triton.jit
def _range_attn_bwd_qtiled_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, delta_ptr, dq_ptr, dk_ptr, dv_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt,
    sdob, sdoh, sdot, sdlb, sdlh, sdlt,
    sdqb, sdqh, sdqt, sdkb, sdkh, sdkt, sdvb, sdvh, sdvt,
    T, Lk, scale, W, BS,
    NUM_KV: tl.constexpr, MODE: tl.constexpr,
    G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_Q: tl.constexpr,
    M_PAD: tl.constexpr,
):
    bh = tl.program_id(0)
    qb = tl.program_id(1)
    b = bh // NUM_KV
    hkv = bh % NUM_KV
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)

    ROWS: tl.constexpr = BLOCK_Q * G
    r = tl.arange(0, M_PAD)
    qi = r // G
    gh = r % G
    i0 = qb * BLOCK_Q
    qpos = i0 + qi
    head = hkv * G + gh
    rmask = (r < ROWS) & (qpos < T)

    q = tl.load(q_ptr + b * sqb + qpos[:, None] * sqt + head[:, None] * sqh + d[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + b * sdob + qpos[:, None] * sdot + head[:, None] * sdoh + d[None, :],
                 mask=rmask[:, None], other=0.0).to(tl.float32)
    delta = tl.load(delta_ptr + b * sdlb + qpos * sdlt + head * sdlh,
                    mask=rmask, other=0.0).to(tl.float32)

    if MODE == 0:
        hi_r = qpos
        lo_r = qpos - W + 1
    else:
        hi_r = (qpos + 1) // BS - 1
        lo_r = qpos * 0
    lo_r = tl.maximum(lo_r, 0)

    i_last = tl.minimum(i0 + BLOCK_Q - 1, T - 1)
    if MODE == 0:
        lo_min = tl.maximum(i0 - W + 1, 0)
        hi_max = i_last
    else:
        lo_min = 0
        hi_max = (i_last + 1) // BS - 1

    NEG = -1.0e9
    lo0 = (lo_min // BLOCK_N) * BLOCK_N
    n_iter = tl.where(hi_max < lo_min, 0, (hi_max - lo0) // BLOCK_N + 1)

    # pass 1: recompute (m, l)
    m = tl.full((M_PAD,), NEG, tl.float32)
    l = tl.zeros((M_PAD,), tl.float32)
    for it in range(n_iter):
        j = lo0 + it * BLOCK_N + n
        okj = (j >= 0) & (j < Lk)
        safe = tl.where(okj, j, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=okj[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        ok = (j[None, :] >= lo_r[:, None]) & (j[None, :] <= hi_r[:, None]) \
            & okj[None, :] & rmask[:, None]
        sc = tl.where(ok, sc, NEG)
        m_cur = tl.max(sc, axis=1)
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)
        p = tl.where(ok, tl.exp(sc - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new
    l_safe = tl.where(l == 0.0, 1.0, l)

    # pass 2: grads
    dq = tl.zeros((M_PAD, D), tl.float32)
    for it in range(n_iter):
        j = lo0 + it * BLOCK_N + n
        okj = (j >= 0) & (j < Lk)
        safe = tl.where(okj, j, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=okj[:, None], other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + b * svb + hkv * svh + safe[:, None] * svt + d[None, :],
                     mask=okj[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        ok = (j[None, :] >= lo_r[:, None]) & (j[None, :] <= hi_r[:, None]) \
            & okj[None, :] & rmask[:, None]
        sc = tl.where(ok, sc, NEG)
        p = tl.where(ok, tl.exp(sc - m[:, None]), 0.0) / l_safe[:, None]
        dp = tl.dot(do, tl.trans(vb), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, kb, input_precision="ieee") * scale
        dk_c = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
        dv_c = tl.dot(tl.trans(p), do, input_precision="ieee")
        tl.atomic_add(dk_ptr + b * sdkb + hkv * sdkh + safe[:, None] * sdkt + d[None, :],
                      dk_c, mask=okj[:, None])
        tl.atomic_add(dv_ptr + b * sdvb + hkv * sdvh + safe[:, None] * sdvt + d[None, :],
                      dv_c, mask=okj[:, None])

    tl.store(dq_ptr + b * sdqb + qpos[:, None] * sdqt + head[:, None] * sdqh + d[None, :],
             dq, mask=rmask[:, None])


# ---------------------------------------------------------------------------
# python wrappers + autograd Function for the range attention
# ---------------------------------------------------------------------------
_BLOCK_N = 64
# The 2-pass q-tiled backward keeps q/do/kb/vb tiles live in fp32 shared at once;
# kb/vb scale with BLOCK_N, so at D=128 / M_PAD=64 a key tile of 64 overflows the
# 3090's ~99KB shared. 32 keeps it under the limit. Forward has fewer live tiles
# and stays at 64.
_BLOCK_N_BWD = 32
_BLOCK_Q = 16          # queries per program in the q-tiled path (M = 16*G)
USE_QTILE = True       # perf rung: use the q-tiled forward (fall back if False)


def _range_attn_forward(q, k, v, mode, window, block_size):
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    Lk = k.shape[2]
    o = torch.empty_like(q)
    if USE_QTILE:
        grid = (B * Hkv, triton.cdiv(T, _BLOCK_Q))
        _range_attn_fwd_qtiled_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            T, Lk, float(D ** -0.5), window, block_size,
            NUM_KV=Hkv, MODE=mode, G=G, D=D, BLOCK_N=_BLOCK_N, BLOCK_Q=_BLOCK_Q,
            M_PAD=triton.next_power_of_2(_BLOCK_Q * G),
        )
        return o
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
    if USE_QTILE:
        # q/do/dq tiles are [M_PAD, D] (BLOCK_N-independent); at D>=128 they alone
        # crowd the 3090's ~99KB shared, so shrink the key tile further there.
        bn_bwd = 16 if D >= 128 else _BLOCK_N_BWD
        grid = (B * Hkv, triton.cdiv(T, _BLOCK_Q))
        _range_attn_bwd_qtiled_kernel[grid](
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
            NUM_KV=Hkv, MODE=mode, G=G, D=D, BLOCK_N=bn_bwd, BLOCK_Q=_BLOCK_Q,
            M_PAD=triton.next_power_of_2(_BLOCK_Q * G),
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
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
