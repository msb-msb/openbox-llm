"""
nsa_selection_backward.py — BACKWARD kernel for the NSA selection branch (Triton).

Companion to nsa_selection_kernel.py (forward). This step is BACKWARD ONLY. Fusion
(compression+window) and nsa_model.py wiring are separate later steps — untouched.
Purely additive: does not modify rung-1/2a files or the forward kernel/tests; it
IMPORTS the forward kernel (to produce O) and the forward reference (for testing).

------------------------------------------------------------------------------
What backward computes
------------------------------------------------------------------------------
Forward:  s = (q · kᵀ)·scale ; p = softmax(s over the SELECTED, causal keys) ;
          o = p · v.
Given upstream dO, the analytical grads are:
    delta_i = Σ_d dO_i·O_i                 (row scalar; == Σ_j p_ij·(dO_i·v_j))
    dP_ij   = dO_i · v_j
    dS_ij   = p_ij · (dP_ij − delta_i)
    dQ_i    = scale · Σ_j dS_ij · k_j
    dK_j    = scale · Σ_i dS_ij · q_i
    dV_j    = Σ_i p_ij · dO_i

TRAP-1 STAYS INTACT: `block_idx` (the top-k selection) is a non-differentiable
INPUT. No gradient flows through block selection — only dQ/dK/dV over the gathered
blocks. The autograd.Function returns None for block_idx / block_size.

------------------------------------------------------------------------------
Softmax-stat handling: RECOMPUTE (FlashAttention-2 style)
------------------------------------------------------------------------------
The forward returns only O (we're not allowed to change it to also emit the
log-sum-exp). So the backward is self-contained:
  * pass 1 re-derives per-row (m_i, l_i) with the SAME online-softmax loop as
    forward (finite NEG = -1e9 mask, so all-masked rows never produce NaN),
  * pass 2 recomputes p_ij = exp(s_ij − m_i)/l_i and the gradients.
Only `delta_i` is precomputed in torch from the saved O (cheap, O(T·D)).
Recompute (vs storing p) keeps memory O(T) — the whole point — since p is
[T × selected_keys] and would blow that up.

dQ is accumulated per query program (each query owns its row — no conflict).
dK/dV are scattered: many queries select the same key block, so they use fp32
atomic_add into fp32 accumulators (cast to the input dtype at the end).
"""

import torch
import triton
import triton.language as tl

from nsa_selection_kernel import selection_forward_triton  # produce O (+ LSE)
from nsa_fsa import build_block_to_query_csr               # FSA prereq (Chunk 1)

# Perf rung (FSA, Chunk 2): block-outer, atomic-free dK/dV backward. Flag-gated;
# the atomic query-outer kernel below stays the fallback. dQ is unchanged (already
# conflict-free) — the FSA path just splits it into its own dQ-only launch.
USE_FSA_BWD = True


def _autotune_configs():
    return [triton.Config({}, num_warps=w, num_stages=st)
            for w in (1, 2, 4, 8) for st in (1, 2, 3)]


# reset_to_zero: dK/dV are accumulated with atomic_add, so autotune's repeated
# benchmark launches would pile onto the same buffers. Zero them before each trial.
@triton.autotune(configs=_autotune_configs(), key=["G", "D", "BLOCK_N", "S"],
                 reset_to_zero=["dk_ptr", "dv_ptr"])
@triton.jit
def _selection_bwd_kernel(
    q_ptr, k_ptr, v_ptr, idx_ptr, do_ptr, delta_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    sqb, sqh, sqt,               # q strides (head dim contiguous)
    skb, skh, skt,               # k strides
    svb, svh, svt,               # v strides
    sib, sih, sit, sis,          # block_idx strides
    sdob, sdoh, sdot,            # dO strides ([B,Hq,T,D])
    sdlb, sdlh, sdlt,            # delta strides ([B,Hq,T])
    sdqb, sdqh, sdqt,            # dQ strides ([B,Hq,T,D])
    sdkb, sdkh, sdkt,            # dK strides ([B,Hkv,T,D])
    sdvb, sdvh, sdvt,            # dV strides ([B,Hkv,T,D])
    T, scale,
    NUM_KV: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, S: tl.constexpr,
):
    bh = tl.program_id(0)
    i = tl.program_id(1)
    b = bh // NUM_KV
    hkv = bh % NUM_KV

    g = tl.arange(0, GP)                 # group lanes, padded to 16 for tl.dot
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    g_ok = g < G
    h0 = hkv * G

    # --- load per-query tensors for the G group heads (junk lanes -> 0) ---
    q = tl.load(q_ptr + b * sqb + i * sqt + (h0 + g)[:, None] * sqh + d[None, :],
                mask=g_ok[:, None], other=0.0).to(tl.float32)          # [GP,D]
    do = tl.load(do_ptr + b * sdob + i * sdot + (h0 + g)[:, None] * sdoh + d[None, :],
                 mask=g_ok[:, None], other=0.0).to(tl.float32)         # [GP,D]
    delta = tl.load(delta_ptr + b * sdlb + i * sdlt + (h0 + g) * sdlh,
                    mask=g_ok, other=0.0).to(tl.float32)               # [GP]

    NEG = -1.0e9

    # --- pass 1: recompute online-softmax stats (m_i, l_i) ---
    m = tl.full((GP,), NEG, tl.float32)
    l = tl.zeros((GP,), tl.float32)
    for s in range(S):
        blk = tl.load(idx_ptr + b * sib + hkv * sih + i * sit + s * sis)
        pos = blk * BLOCK_N + n
        kv_ok = (blk >= 0) & (pos <= i) & (pos < T)
        safe = tl.where(kv_ok, pos, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=kv_ok[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(kv_ok[None, :], sc, NEG)
        m_cur = tl.max(sc, axis=1)
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)
        p_row = tl.where(kv_ok[None, :], tl.exp(sc - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p_row, axis=1)
        m = m_new
    l_safe = tl.where(l == 0.0, 1.0, l)          # empty rows -> p==0 anyway

    # --- pass 2: recompute p and accumulate grads ---
    dq = tl.zeros((GP, D), tl.float32)
    for s in range(S):
        blk = tl.load(idx_ptr + b * sib + hkv * sih + i * sit + s * sis)
        pos = blk * BLOCK_N + n
        kv_ok = (blk >= 0) & (pos <= i) & (pos < T)
        safe = tl.where(kv_ok, pos, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=kv_ok[:, None], other=0.0).to(tl.float32)     # [N,D]
        vb = tl.load(v_ptr + b * svb + hkv * svh + safe[:, None] * svt + d[None, :],
                     mask=kv_ok[:, None], other=0.0).to(tl.float32)     # [N,D]

        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale     # [GP,N]
        sc = tl.where(kv_ok[None, :], sc, NEG)
        p = tl.where(kv_ok[None, :], tl.exp(sc - m[:, None]), 0.0) / l_safe[:, None]

        dp = tl.dot(do, tl.trans(vb), input_precision="ieee")            # [GP,N]
        ds = p * (dp - delta[:, None])                                   # [GP,N]

        dq += tl.dot(ds, kb, input_precision="ieee") * scale             # [GP,D]

        dk_c = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale   # [N,D]
        dv_c = tl.dot(tl.trans(p), do, input_precision="ieee")           # [N,D]
        dk_ptrs = dk_ptr + b * sdkb + hkv * sdkh + safe[:, None] * sdkt + d[None, :]
        dv_ptrs = dv_ptr + b * sdvb + hkv * sdvh + safe[:, None] * sdvt + d[None, :]
        tl.atomic_add(dk_ptrs, dk_c, mask=kv_ok[:, None])
        tl.atomic_add(dv_ptrs, dv_c, mask=kv_ok[:, None])

    dq_ptrs = dq_ptr + b * sdqb + i * sdqt + (h0 + g)[:, None] * sdqh + d[None, :]
    tl.store(dq_ptrs, dq, mask=g_ok[:, None])


def selection_backward_triton(q, k, v, block_idx, block_size, do, o):
    """Return dq, dk, dv (same shapes/dtype as q, k, v) for the selection branch."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    S = block_idx.shape[-1]
    assert D in (16, 32, 64, 128)
    assert block_size >= 16
    idx = block_idx.to(torch.int32).contiguous()
    do = do.contiguous()

    # delta_i = sum_d dO_i * O_i  (fp32, from the saved forward output)
    delta = (do.float() * o.float()).sum(-1).contiguous()      # [B,Hq,T]

    dq = torch.zeros(B, Hq, T, D, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, Hkv, T, D, device=q.device, dtype=torch.float32)
    dv = torch.zeros(B, Hkv, T, D, device=q.device, dtype=torch.float32)

    grid = (B * Hkv, T)
    _selection_bwd_kernel[grid](
        q, k, v, idx, do, delta, dq, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        T, float(D ** -0.5),
        NUM_KV=Hkv, G=G, GP=16, D=D, BLOCK_N=block_size, S=S,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


# ===========================================================================
# FSA backward (Chunk 2): dQ-only (query-outer, unchanged) + block-outer dK/dV.
# ===========================================================================

# dQ-only: the original backward MINUS the dK/dV atomics. dQ is owned per query
# program (no conflict), so it is untouched; we only stop it also scattering dK/dV.
@triton.autotune(configs=_autotune_configs(), key=["G", "D", "BLOCK_N", "S"])
@triton.jit
def _selection_dq_kernel(
    q_ptr, k_ptr, v_ptr, idx_ptr, do_ptr, delta_ptr, dq_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt,
    sib, sih, sit, sis, sdob, sdoh, sdot, sdlb, sdlh, sdlt, sdqb, sdqh, sdqt,
    T, scale,
    NUM_KV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, S: tl.constexpr,
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
    NEG = -1.0e9

    m = tl.full((GP,), NEG, tl.float32)
    l = tl.zeros((GP,), tl.float32)
    for s in range(S):
        blk = tl.load(idx_ptr + b * sib + hkv * sih + i * sit + s * sis)
        pos = blk * BLOCK_N + n
        kv_ok = (blk >= 0) & (pos <= i) & (pos < T)
        safe = tl.where(kv_ok, pos, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=kv_ok[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(kv_ok[None, :], sc, NEG)
        m_cur = tl.max(sc, axis=1)
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)
        p_row = tl.where(kv_ok[None, :], tl.exp(sc - m_new[:, None]), 0.0)
        l = l * alpha + tl.sum(p_row, axis=1)
        m = m_new
    l_safe = tl.where(l == 0.0, 1.0, l)

    dq = tl.zeros((GP, D), tl.float32)
    for s in range(S):
        blk = tl.load(idx_ptr + b * sib + hkv * sih + i * sit + s * sis)
        pos = blk * BLOCK_N + n
        kv_ok = (blk >= 0) & (pos <= i) & (pos < T)
        safe = tl.where(kv_ok, pos, 0)
        kb = tl.load(k_ptr + b * skb + hkv * skh + safe[:, None] * skt + d[None, :],
                     mask=kv_ok[:, None], other=0.0).to(tl.float32)
        vb = tl.load(v_ptr + b * svb + hkv * svh + safe[:, None] * svt + d[None, :],
                     mask=kv_ok[:, None], other=0.0).to(tl.float32)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(kv_ok[None, :], sc, NEG)
        p = tl.where(kv_ok[None, :], tl.exp(sc - m[:, None]), 0.0) / l_safe[:, None]
        dp = tl.dot(do, tl.trans(vb), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, kb, input_precision="ieee") * scale
    tl.store(dq_ptr + b * sdqb + i * sdqt + (h0 + g)[:, None] * sdqh + d[None, :],
             dq, mask=g_ok[:, None])


# block-outer dK/dV: one program owns KV block j. Loads block j's K/V once, loops
# its attending queries (from the CSR index) so REAL queries×heads fill the MMA
# M-dim (no GP-16 pad), consumes stored LSE (no pass-1 recompute), and writes a
# DISJOINT slice dK[j]/dV[j] once — no atomics.
def _dkdv_cfgs():
    return [triton.Config({}, num_warps=w, num_stages=1) for w in (2, 4, 8)]


@triton.autotune(configs=_dkdv_cfgs(), key=["G", "D", "BLOCK_N", "BLOCK_Q"])
@triton.jit
def _selection_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, delta_ptr, lse_ptr, qob_ptr, off_ptr, dk_ptr, dv_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt, sdob, sdoh, sdot,
    sdlb, sdlh, sdlt, slb, slh, slt, soffb, soffh, soffj,
    sdkb, sdkh, sdkt, sdvb, sdvh, sdvt,
    T, scale,
    NUM_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_Q: tl.constexpr, M_PAD: tl.constexpr,
):
    bh = tl.program_id(0)
    j = tl.program_id(1)                       # KV block this program owns
    b = bh // NUM_KV
    hkv = bh % NUM_KV
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)

    base = tl.load(off_ptr + b * soffb + hkv * soffh + j * soffj)
    end = tl.load(off_ptr + b * soffb + hkv * soffh + (j + 1) * soffj)
    n_q = end - base
    if n_q == 0:                               # early-return: dK/dV stay zero-init
        return

    # block j's K/V, loaded ONCE (positions j*BLOCK_N..+BLOCK_N-1, all < T)
    pos = j * BLOCK_N + n
    kb = tl.load(k_ptr + b * skb + hkv * skh + pos[:, None] * skt + d[None, :]).to(tl.float32)
    vb = tl.load(v_ptr + b * svb + hkv * svh + pos[:, None] * svt + d[None, :]).to(tl.float32)

    ROWS: tl.constexpr = BLOCK_Q * G
    r = tl.arange(0, M_PAD)
    qi = r // G
    gh = r % G
    head = hkv * G + gh
    dk = tl.zeros((BLOCK_N, D), tl.float32)
    dv = tl.zeros((BLOCK_N, D), tl.float32)

    n_iter = (n_q + BLOCK_Q - 1) // BLOCK_Q
    for it in range(n_iter):
        slot = it * BLOCK_Q + qi                            # index into this block's query list
        row_ok = (r < ROWS) & (slot < n_q)
        qsafe = tl.where(row_ok, base + slot, base)
        qpos = tl.load(qob_ptr + qsafe, mask=row_ok, other=0)   # [M_PAD] query positions
        qm = row_ok[:, None]
        q_ = tl.load(q_ptr + b * sqb + qpos[:, None] * sqt + head[:, None] * sqh + d[None, :],
                     mask=qm, other=0.0).to(tl.float32)
        do_ = tl.load(do_ptr + b * sdob + qpos[:, None] * sdot + head[:, None] * sdoh + d[None, :],
                      mask=qm, other=0.0).to(tl.float32)
        lse_ = tl.load(lse_ptr + b * slb + head * slh + qpos * slt, mask=row_ok, other=0.0)
        delta_ = tl.load(delta_ptr + b * sdlb + head * sdlh + qpos * sdlt, mask=row_ok, other=0.0)

        sc = tl.dot(q_, tl.trans(kb), input_precision="ieee") * scale   # [M_PAD, BLOCK_N]
        kv_ok = (pos[None, :] <= qpos[:, None]) & row_ok[:, None]       # per-token causal + valid row
        p = tl.where(kv_ok, tl.exp(sc - lse_[:, None]), 0.0)           # normalized via stored LSE
        dp = tl.dot(do_, tl.trans(vb), input_precision="ieee")         # [M_PAD, BLOCK_N]
        ds = p * (dp - delta_[:, None])
        dv += tl.dot(tl.trans(p), do_, input_precision="ieee")         # [BLOCK_N, D]
        dk += tl.dot(tl.trans(ds), q_, input_precision="ieee") * scale

    # disjoint per-block write -> no atomics
    tl.store(dk_ptr + b * sdkb + hkv * sdkh + pos[:, None] * sdkt + d[None, :], dk)
    tl.store(dv_ptr + b * sdvb + hkv * sdvh + pos[:, None] * sdvt + d[None, :], dv)


def selection_backward_fsa(q, k, v, block_idx, block_size, do, o, lse):
    """FSA backward: dQ-only (query-outer) + block-outer atomic-free dK/dV.
    Returns dq, dk, dv (same shapes/dtype as q, k, v)."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    S = block_idx.shape[-1]
    n_blk = T // block_size
    idx = block_idx.to(torch.int32).contiguous()
    do = do.contiguous()
    delta = (do.float() * o.float()).sum(-1).contiguous()      # [B,Hq,T]
    dq = torch.zeros(B, Hq, T, D, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, Hkv, T, D, device=q.device, dtype=torch.float32)
    dv = torch.zeros(B, Hkv, T, D, device=q.device, dtype=torch.float32)
    scale = float(D ** -0.5)

    # dQ — query-outer, unchanged
    _selection_dq_kernel[(B * Hkv, T)](
        q, k, v, idx, do, delta, dq,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        T, scale, NUM_KV=Hkv, G=G, GP=16, D=D, BLOCK_N=block_size, S=S,
    )

    # dK/dV — block-outer, atomic-free (Chunk-1 CSR + stored LSE)
    q_of_block, block_offsets = build_block_to_query_csr(idx, block_size)
    # cap M_PAD so kb/vb [block_size,D] + q_/do_ [M_PAD,D] fit the 3090's ~99KB shared;
    # the block key tile (block_size*D) is the fixed cost, so shrink M_PAD when it's big.
    if D >= 128 and block_size >= 64:
        bq = 4
    elif D >= 128:
        bq = 8
    else:
        bq = 16
    M_PAD = max(16, triton.next_power_of_2(bq * G))   # tl.dot needs M >= 16
    _selection_dkdv_kernel[(B * Hkv, n_blk)](
        q, k, v, do, delta, lse, q_of_block, block_offsets, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        lse.stride(0), lse.stride(1), lse.stride(2),
        block_offsets.stride(0), block_offsets.stride(1), block_offsets.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        T, scale, NUM_KV=Hkv, G=G, D=D, BLOCK_N=block_size, BLOCK_Q=bq, M_PAD=M_PAD,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class SelectionAttnTriton(torch.autograd.Function):
    """Autograd wrapper: Triton forward (from the forward module) + Triton backward.

    block_idx / block_size are non-differentiable inputs (trap-1): backward returns
    None for them. The FSA backward's CSR index is derived internally from block_idx
    (non-diff) and never becomes an input — trap-1 preserved.
    """

    @staticmethod
    def forward(ctx, q, k, v, block_idx, block_size):
        o, lse = selection_forward_triton(q, k, v, block_idx, block_size, return_lse=True)
        ctx.save_for_backward(q, k, v, block_idx, o, lse)
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, block_idx, o, lse = ctx.saved_tensors
        T, D = q.shape[2], q.shape[3]
        # Shape-gate: the FSA path has ~0.5ms fixed cost (CSR build + extra launch)
        # that only pays off at real shapes. Below D>=64 & T>=512 (toy configs) the
        # atomic kernel is faster, so fall back — FSA is never slower than atomic.
        if USE_FSA_BWD and D >= 64 and T >= 512:
            dq, dk, dv = selection_backward_fsa(q, k, v, block_idx,
                                                ctx.block_size, do, o, lse)
        else:
            dq, dk, dv = selection_backward_triton(q, k, v, block_idx,
                                                   ctx.block_size, do, o)
        return dq, dk, dv, None, None


def selection_attn_triton(q, k, v, block_idx, block_size):
    return SelectionAttnTriton.apply(q, k, v, block_idx, block_size)
