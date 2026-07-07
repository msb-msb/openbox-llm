"""
nsa_selection_kernel.py — NSA SELECTION-branch forward: pure-torch reference +
a Triton (FlashAttention-style) kernel. FORWARD ONLY (backward is the next step).

Kernel rung, step 1. Only the SELECTION branch is ported to Triton — it's the
branch whose cost NSA actually cuts (attend to only the top-k selected blocks
instead of the whole sequence). Compression + sliding-window stay pure PyTorch.

------------------------------------------------------------------------------
The operation (matches rung-1 nsa_model.py's selection branch)
------------------------------------------------------------------------------
Inputs (GQA layout):
    q         [B, Hq,  T, D]   query, one row per query head
    k, v      [B, Hkv, T, D]   key/value, one row per kv head (Hq = Hkv * G)
    block_idx [B, Hkv, T, S]   for each (batch, kv-group, query i), the S selected
                               BLOCK indices (block_size tokens each), SHARED across
                               the G heads of a group — exactly as NSA decides
                               selection per group. Slots may be -1 (padding: query i
                               near the start has fewer than S causally-valid blocks).

Output:
    o         [B, Hq,  T, D]

Semantics per query i and query head h (kv head = h // G):
    attend over the tokens of the selected blocks, causally masked (key j <= i),
    softmax, weighted sum of v. A query with no valid selected block outputs 0
    (mirrors rung-1's masked_softmax zeroing fully-masked rows).

The block choice is passed IN as an index list (the compression branch produces it
in the full model; here it's an input so the kernel can be tested in isolation).
The Triton kernel's inner loop iterates that GATHERED index list — gather, not a
contiguous scan over KV — which is the whole point of the speedup.
"""

import torch
import triton
import triton.language as tl

from nsa_fsa import build_block_to_query_csr   # FSA prereq (Chunk 1); no cycle

# Perf rung (FSA fwd): block-outer, real-M (no GP-16 pad) selection FORWARD. Flag-
# gated; the query-outer padded kernel below stays the fallback. The CSR build +
# split-K combine has a fixed cost that the query-outer forward (no atomics, already
# cheap) only loses to at big D. Measured: block-outer wins at D>=128 (1.5x-2.1x) but
# LOSES at D=64/T=512 (~0.2-0.4x — baseline is only ~0.2ms there), so gate D>=128 &
# T>=512. Stricter than the FSA-backward gate (D>=64): the atomic backward it replaced
# was far slower, so its crossover sat lower. Below the gate -> fall back, never slower.
USE_FSA_FWD = True


# ---------------------------------------------------------------------------
# pure-torch reference (ground truth for the correctness gate)
# ---------------------------------------------------------------------------

def selection_forward_reference(q, k, v, block_idx, block_size):
    """Full-score + keep-mask selection attention, identical to rung-1's branch.

    Deliberately the slow O(T^2) path (build the score matrix, mask, softmax) — it
    is the trusted reference and it is what OOMs at long seq_len in the bench.
    """
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    S = block_idx.shape[-1]
    n_blk = T // block_size
    dev = q.device
    scale = D ** -0.5

    qf = q.float()
    kexp = k.float().repeat_interleave(G, dim=1)          # [B,Hq,T,D]
    vexp = v.float().repeat_interleave(G, dim=1)
    s = torch.matmul(qf, kexp.transpose(-1, -2)) * scale  # [B,Hq,T,T]

    # chosen[b,hkv,i,blk] = is block `blk` selected for query i? (skip -1 padding)
    valid = block_idx >= 0                                 # [B,Hkv,T,S]
    safe = block_idx.clamp(min=0).long()
    oh = torch.zeros(B, Hkv, T, S, n_blk, dtype=torch.bool, device=dev)
    oh.scatter_(-1, safe.unsqueeze(-1), valid.unsqueeze(-1))
    chosen = oh.any(dim=3)                                 # [B,Hkv,T,n_blk]

    blk_of_tok = (torch.arange(T, device=dev) // block_size).clamp(max=n_blk - 1)
    keep = chosen[..., blk_of_tok]                         # [B,Hkv,T,T]
    causal = torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None]
    keep = keep & causal[None, None]                      # causal within selected blocks
    keep = keep.repeat_interleave(G, dim=1)               # [B,Hq,T,T]

    # finite mask value (like rung-1's masked_softmax): a fully-masked row then
    # softmaxes to uniform instead of NaN, and row_has zeroes it cleanly.
    s = s.masked_fill(~keep, torch.finfo(s.dtype).min)
    row_has = keep.any(dim=-1, keepdim=True)
    p = torch.softmax(s, dim=-1) * row_has                # zero fully-masked rows
    return torch.matmul(p, vexp).to(q.dtype)              # [B,Hq,T,D]


# ---------------------------------------------------------------------------
# Triton kernel — one program per (batch*kv-head, query position).
# ---------------------------------------------------------------------------

def _autotune_configs():
    return [triton.Config({}, num_warps=w, num_stages=st)
            for w in (1, 2, 4, 8) for st in (1, 2, 3)]


@triton.autotune(configs=_autotune_configs(), key=["G", "D", "BLOCK_N", "S"])
@triton.jit
def _selection_fwd_kernel(
    q_ptr, k_ptr, v_ptr, idx_ptr, o_ptr, lse_ptr,
    sqb, sqh, sqt,               # q strides (head dim contiguous: sqd == 1)
    skb, skh, skt,               # k strides
    svb, svh, svt,               # v strides
    sib, sih, sit, sis,          # block_idx strides
    sob, soh, sot,               # o strides
    slb, slh, slt,               # lse strides ([B,Hq,T])
    T, scale,
    NUM_KV: tl.constexpr,        # Hkv, to split the flattened (batch,kv) grid dim
    G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    BLOCK_N: tl.constexpr, S: tl.constexpr,
):
    bh = tl.program_id(0)                 # flattened index == b * NUM_KV + hkv
    i = tl.program_id(1)                  # query position
    b = bh // NUM_KV
    hkv = bh % NUM_KV

    g = tl.arange(0, GP)                  # group lanes, padded to GP(=16) for tl.dot
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    g_ok = g < G
    h0 = hkv * G                          # first query head of this group

    # --- load Q for the G heads of this group: [GP, D], reused for every block ---
    q_ptrs = q_ptr + b * sqb + i * sqt + (h0 + g)[:, None] * sqh + d[None, :]
    q = tl.load(q_ptrs, mask=g_ok[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e9                          # finite mask value; padded keys are also
                                         # zeroed via kv_ok, so empty rows stay 0.
    m = tl.full((GP,), NEG, tl.float32)   # running max (per head)
    l = tl.zeros((GP,), tl.float32)       # running denom
    acc = tl.zeros((GP, D), tl.float32)   # running weighted sum

    # --- iterate the GATHERED selected-block list (gather, not scan) ---
    for s in range(S):
        blk = tl.load(idx_ptr + b * sib + hkv * sih + i * sit + s * sis)
        pos = blk * BLOCK_N + n                   # [BLOCK_N] absolute key positions
        kv_ok = (blk >= 0) & (pos <= i) & (pos < T)   # causal + in-range + not padding
        safe_pos = tl.where(kv_ok, pos, 0)

        k_ptrs = k_ptr + b * skb + hkv * skh + safe_pos[:, None] * skt + d[None, :]
        v_ptrs = v_ptr + b * svb + hkv * svh + safe_pos[:, None] * svt + d[None, :]
        kb = tl.load(k_ptrs, mask=kv_ok[:, None], other=0.0).to(tl.float32)   # [N,D]
        vb = tl.load(v_ptrs, mask=kv_ok[:, None], other=0.0).to(tl.float32)

        # scores [GP, N] = q @ kbᵀ  (ieee fp32 to match the reference)
        sc = tl.dot(q, tl.trans(kb), input_precision="ieee") * scale
        sc = tl.where(kv_ok[None, :], sc, NEG)

        m_cur = tl.max(sc, axis=1)               # [GP]
        m_new = tl.maximum(m, m_cur)
        alpha = tl.exp(m - m_new)                # [GP], finite
        p = tl.where(kv_ok[None, :], tl.exp(sc - m_new[:, None]), 0.0)   # [GP,N]

        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, vb, input_precision="ieee")  # [GP,D]
        m = m_new

    o = tl.where(l[:, None] == 0.0, 0.0, acc / l[:, None])       # empty row -> 0
    o_ptrs = o_ptr + b * sob + i * sot + (h0 + g)[:, None] * soh + d[None, :]
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=g_ok[:, None])

    # log-sum-exp per (query, head), finite sentinel for empty rows (FSA prereq 0).
    # Additive: consumed by the FSA backward; the ref path ignores it.
    lse = tl.where(l == 0.0, NEG, m + tl.log(l))
    tl.store(lse_ptr + b * slb + (h0 + g) * slh + i * slt, lse, mask=g_ok)


def selection_forward_triton(q, k, v, block_idx, block_size, return_lse=False):
    """Launch wrapper. q [B,Hq,T,D], k/v [B,Hkv,T,D], block_idx [B,Hkv,T,S].

    return_lse=False (default) returns o only — unchanged for every existing caller.
    return_lse=True returns (o, lse) where lse [B,Hq,T] = m + log(l) per (query,head),
    with a finite sentinel (-1e9) for fully-masked (empty) rows.
    """
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    S = block_idx.shape[-1]
    assert block_idx.shape[:3] == (B, Hkv, T)
    assert D in (16, 32, 64, 128), "head dim must be a power of two <= 128"
    assert block_size >= 16, "block_size must be >= 16 (tl.dot inner dim)"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    # FSA fwd: block-outer, real queries fill the MMA. Shape-gated (below D>=128 &
    # T>=512 the CSR build + combine overhead loses to the query-outer kernel).
    if USE_FSA_FWD and D >= 128 and T >= 512:
        o, lse = selection_forward_fsa(q, k, v, block_idx, block_size)
        return (o, lse) if return_lse else o

    GP = 16          # pad the GQA group to 16 lanes so q@kᵀ / p@v use tl.dot
    idx = block_idx.to(torch.int32).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(B, Hq, T, device=q.device, dtype=torch.float32)

    grid = (B * Hkv, T)
    _selection_fwd_kernel[grid](
        q, k, v, idx, o, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        lse.stride(0), lse.stride(1), lse.stride(2),
        T, float(D ** -0.5),
        NUM_KV=Hkv, G=G, GP=GP, D=D, BLOCK_N=block_size, S=S,
    )
    return (o, lse) if return_lse else o


# ===========================================================================
# FSA forward (block-outer, split-K): grid over KV blocks. One program owns block
# j, loads its K/V ONCE, loops its attending queries from the Chunk-1 CSR so REAL
# queries×heads fill the MMA M-dim (no GP-16 pad), and emits per-(query,block)
# softmax PARTIALS (local max m, denom l, weighted-v acc). A torch log-sum-exp
# combine then reduces each query's blocks into o + lse — mathematically one
# softmax over the union of its selected tokens (split-K / flash-decode).
#
# TRAP-1 intact: the CSR is built from block_idx only (non-diff, no_grad); it is
# never a differentiable input. block_idx stays a plain int index.
# ===========================================================================

def _fwd_bo_cfgs():
    return [triton.Config({}, num_warps=w, num_stages=1) for w in (2, 4, 8)]


@triton.autotune(configs=_fwd_bo_cfgs(), key=["G", "D", "BLOCK_N", "BLOCK_Q"])
@triton.jit
def _selection_fwd_blockouter_kernel(
    q_ptr, k_ptr, v_ptr, qob_ptr, off_ptr,
    pm_ptr, pl_ptr, pacc_ptr,
    sqb, sqh, sqt, skb, skh, skt, svb, svh, svt,
    soffb, soffh, soffj,
    spmp, spmg, splp, splg, spap, spag,
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
    if n_q == 0:                               # no query selects block j
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
    NEG = -1.0e9

    n_iter = (n_q + BLOCK_Q - 1) // BLOCK_Q
    for it in range(n_iter):
        slot = it * BLOCK_Q + qi                            # index into block j's query list
        row_ok = (r < ROWS) & (slot < n_q)
        qsafe = tl.where(row_ok, base + slot, base)
        qpos = tl.load(qob_ptr + qsafe, mask=row_ok, other=0)   # [M_PAD] query positions
        q_ = tl.load(q_ptr + b * sqb + qpos[:, None] * sqt + head[:, None] * sqh + d[None, :],
                     mask=row_ok[:, None], other=0.0).to(tl.float32)

        sc = tl.dot(q_, tl.trans(kb), input_precision="ieee") * scale   # [M_PAD, BLOCK_N]
        kv_ok = (pos[None, :] <= qpos[:, None]) & row_ok[:, None]       # per-token causal + valid row
        sc = tl.where(kv_ok, sc, NEG)
        m_local = tl.max(sc, axis=1)                                    # [M_PAD] (>=1 valid token/pair)
        p = tl.where(kv_ok, tl.exp(sc - m_local[:, None]), 0.0)         # [M_PAD, BLOCK_N]
        l_local = tl.sum(p, axis=1)                                     # [M_PAD]
        acc = tl.dot(p, vb, input_precision="ieee")                     # [M_PAD, D]

        # per-pair partials -> buffers, reduced by the torch combine below
        pair = base + it * BLOCK_Q + qi                                 # [M_PAD] pair index
        tl.store(pm_ptr + pair * spmp + gh * spmg, m_local, mask=row_ok)
        tl.store(pl_ptr + pair * splp + gh * splg, l_local, mask=row_ok)
        tl.store(pacc_ptr + pair[:, None] * spap + gh[:, None] * spag + d[None, :],
                 acc, mask=row_ok[:, None])


def selection_forward_fsa(q, k, v, block_idx, block_size):
    """FSA forward: block-outer split-K kernel emits per-(query,block) softmax
    partials; a torch log-sum-exp combine reduces each query's blocks into o + lse.
    Returns (o, lse) with o same shape/dtype as q, lse [B,Hq,T] fp32 (NEG sentinel
    for fully-masked/empty query rows)."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    n_blk = T // block_size
    idx = block_idx.to(torch.int32).contiguous()
    scale = float(D ** -0.5)
    NEG = -1.0e9

    # inverted CSR (Chunk 1). pair_bh gives the (b,hkv) group of each pair for the combine.
    q_of_block, block_offsets, pair_bh = build_block_to_query_csr(idx, block_size)
    P = q_of_block.numel()

    pm = torch.full((P, G), NEG, device=q.device, dtype=torch.float32)   # per-pair local max
    pl = torch.zeros((P, G), device=q.device, dtype=torch.float32)       # per-pair local denom
    pacc = torch.zeros((P, G, D), device=q.device, dtype=torch.float32)  # per-pair p@v

    # same shared-mem budget as the FSA backward (kb/vb [block,D] + q_ [M_PAD,D];
    # forward loads one fewer tile than backward, so this cap is safe).
    if D >= 128 and block_size >= 64:
        bq = 4
    elif D >= 128:
        bq = 8
    else:
        bq = 16
    M_PAD = max(16, triton.next_power_of_2(bq * G))   # tl.dot needs M >= 16
    if P > 0:
        _selection_fwd_blockouter_kernel[(B * Hkv, n_blk)](
            q, k, v, q_of_block, block_offsets, pm, pl, pacc,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            block_offsets.stride(0), block_offsets.stride(1), block_offsets.stride(2),
            pm.stride(0), pm.stride(1), pl.stride(0), pl.stride(1),
            pacc.stride(0), pacc.stride(1),
            T, scale, NUM_KV=Hkv, G=G, D=D, BLOCK_N=block_size, BLOCK_Q=bq, M_PAD=M_PAD,
        )

    # --- combine: log-sum-exp merge of each query's block-partials (split-K reduce)
    # o_row = (Σ_j w_j·acc_j) / (Σ_j w_j·l_j),  w_j = exp(m_j − M_row),  M_row = max_j m_j
    # exactly one softmax over the union of the query's selected, causal tokens.
    o = torch.zeros(B, Hq, T, D, device=q.device, dtype=torch.float32)
    lse = torch.full((B, Hq, T), NEG, device=q.device, dtype=torch.float32)
    if P > 0:
        pb = pair_bh.long()
        b_ = pb // Hkv
        hkv_ = pb % Hkv
        qpos = q_of_block.long()
        g_ar = torch.arange(G, device=q.device)
        # flat row id [P,G] into [B*Hq*T] (full query head = hkv*G + g)
        flat = ((b_[:, None] * Hq + hkv_[:, None] * G + g_ar[None, :]) * T
                + qpos[:, None])                                    # [P,G]
        flat_f = flat.reshape(-1)                                   # [P*G]

        Mrow = torch.full((B * Hq * T,), NEG, device=q.device, dtype=torch.float32)
        Mrow.scatter_reduce_(0, flat_f, pm.reshape(-1), reduce="amax", include_self=True)
        w = torch.exp(pm - Mrow[flat])                             # [P,G]

        lrow = torch.zeros(B * Hq * T, device=q.device, dtype=torch.float32)
        lrow.scatter_add_(0, flat_f, (pl * w).reshape(-1))
        of = o.view(B * Hq * T, D)
        of.index_add_(0, flat_f, (pacc * w[:, :, None]).reshape(-1, D))

        lrow_safe = torch.where(lrow == 0.0, torch.ones_like(lrow), lrow)
        of.div_(lrow_safe[:, None])                                # empty rows: acc==0 -> 0
        lse = torch.where(lrow == 0.0, torch.full_like(lrow, NEG),
                          Mrow + torch.log(lrow_safe)).view(B, Hq, T)
    return o.to(q.dtype), lse
