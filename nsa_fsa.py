"""nsa_fsa.py — FSA-rework prerequisites (Chunk 1). Additive; nothing here changes
the existing selection path. Provides the inverted block->query CSR index that the
FSA-style block-outer backward (later chunk) will consume.

TRAP-1: the index is derived ONLY from block_idx — the top-k block choice, which is
already non-differentiable and produced upstream under no_grad. It carries NO
gradient; it is a plain int input, and the autograd Function must return None for it
(exactly like block_idx today). Building it must never touch a differentiable tensor.
"""

import torch


@torch.no_grad()
def build_block_to_query_csr(block_idx, block_size):
    """Invert block_idx [B,Hkv,T,S] (query -> its S selected blocks) into a CSR
    mapping block -> attending queries, with CAUSAL validity baked in.

    A (query i, block j) pair is included iff
        j >= 0                    # real selection, not -1 padding
        j * block_size <= i       # block's first token is causal for query i
    which is exactly the condition under which the forward kernel's per-token
    kv_ok = (blk>=0) & (pos<=i) & (pos<T), pos=j*block_size+n, has any true lane
    (min pos = j*block_size; j < n_blk => pos < T always).

    Returns (int32, on block_idx.device):
      q_of_block    [P]               query ids, grouped by (b, hkv, block) then query;
                                      P = total causal (query, block) pairs
      block_offsets [B,Hkv,n_blk+1]   CSR row pointers (GLOBAL offsets into q_of_block):
                                      block (b,hkv,j)'s queries are
                                      q_of_block[block_offsets[b,hkv,j] : block_offsets[b,hkv,j+1]]
    """
    B, Hkv, T, S = block_idx.shape
    n_blk = T // block_size
    dev = block_idx.device
    idx = block_idx.to(torch.int64)

    qpos = torch.arange(T, device=dev).view(1, 1, T, 1)             # [1,1,T,1]
    valid = (idx >= 0) & (idx * block_size <= qpos)                 # [B,Hkv,T,S]
    jb = idx.clamp(min=0)                                           # safe block id

    G = B * Hkv
    # per (b,hkv,block) counts: scatter_add valid flags into the block dim
    counts = torch.zeros(G, n_blk, dtype=torch.int64, device=dev)
    counts.scatter_add_(1, jb.reshape(G, T * S),
                        valid.reshape(G, T * S).to(torch.int64))
    # global CSR offsets over (b,hkv,block) row-major
    off = torch.zeros(G * n_blk + 1, dtype=torch.int64, device=dev)
    off[1:] = counts.reshape(-1).cumsum(0)
    rows = (torch.arange(G, device=dev).view(G, 1) * n_blk
            + torch.arange(n_blk + 1, device=dev).view(1, n_blk + 1))
    block_offsets = off[rows].reshape(B, Hkv, n_blk + 1).to(torch.int32)

    # query ids ordered by (b,hkv,block) then query — aligned with `off`
    bh, ti, si = valid.reshape(G, T, S).nonzero(as_tuple=True)
    jj = jb.reshape(G, T, S)[bh, ti, si]
    key = bh * n_blk + jj                                          # global block key
    order = torch.argsort(key, stable=True)                        # stable => query ascending
    q_of_block = ti[order].to(torch.int32).contiguous()

    return q_of_block, block_offsets
