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
    q_ptr, k_ptr, v_ptr, idx_ptr, o_ptr,
    sqb, sqh, sqt,               # q strides (head dim contiguous: sqd == 1)
    skb, skh, skt,               # k strides
    svb, svh, svt,               # v strides
    sib, sih, sit, sis,          # block_idx strides
    sob, soh, sot,               # o strides
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


def selection_forward_triton(q, k, v, block_idx, block_size):
    """Launch wrapper. q [B,Hq,T,D], k/v [B,Hkv,T,D], block_idx [B,Hkv,T,S]."""
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    S = block_idx.shape[-1]
    assert block_idx.shape[:3] == (B, Hkv, T)
    assert D in (16, 32, 64, 128), "head dim must be a power of two <= 128"
    assert block_size >= 16, "block_size must be >= 16 (tl.dot inner dim)"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    GP = 16          # pad the GQA group to 16 lanes so q@kᵀ / p@v use tl.dot
    idx = block_idx.to(torch.int32).contiguous()
    o = torch.empty_like(q)

    grid = (B * Hkv, T)
    _selection_fwd_kernel[grid](
        q, k, v, idx, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        T, float(D ** -0.5),
        NUM_KV=Hkv, G=G, GP=GP, D=D, BLOCK_N=block_size, S=S,
    )
    return o
