"""
test_kernel.py — CORRECTNESS GATE for the NSA selection Triton kernel.

Runs BEFORE trusting any speed number: torch.allclose(kernel_fwd, reference_fwd)
across a spread of shapes (seq_len, #heads, GQA group size incl. non-power-of-two,
block_size, n_selected_blocks). bf16 tolerance. Asserts on mismatch.

Fast-but-wrong fails the gate.
"""

import sys
import torch

from nsa_selection_kernel import (selection_forward_reference,
                                  selection_forward_triton)

# VRAM guard: hard-cap ~70% of the 24GB card, leave the desktop/browser room.
torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)

DEV = "cuda"

# Vary: batch, query heads, kv heads (=> GQA group G=Hq/Hkv, incl. G=1 and G=3),
# head dim, seq_len, block_size, n_selected_blocks. T divisible by block_size.
CONFIGS = [
    dict(B=2, Hq=4,  Hkv=2, D=32,  T=256,  block_size=16, S=4),   # G=2
    dict(B=1, Hq=8,  Hkv=2, D=64,  T=512,  block_size=32, S=8),   # G=4
    dict(B=2, Hq=12, Hkv=4, D=64,  T=512,  block_size=16, S=8),   # G=3 (non-pow2)
    dict(B=1, Hq=16, Hkv=4, D=128, T=1024, block_size=64, S=16),  # big D, long-ish
    dict(B=1, Hq=4,  Hkv=4, D=64,  T=384,  block_size=16, S=6),   # G=1 (MHA-like)
    dict(B=2, Hq=6,  Hkv=3, D=32,  T=256,  block_size=32, S=3),   # G=2, few blocks
    dict(B=1, Hq=8,  Hkv=2, D=64,  T=512,  block_size=16, S=32),  # S > some rows' n_valid
]

# bf16 tolerances (accumulation-order + rounding differences between the two paths)
ATOL, RTOL = 2e-2, 2e-2


def make_block_idx(B, Hkv, T, block_size, S, device, seed=0):
    """Random, causally-valid selected-block lists with -1 padding.

    For query i, only FULLY-PAST blocks are eligible (block's last token <= i),
    matching rung-1's selection validity. Fewer than S valid -> pad slots with -1.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_blk = T // block_size
    blk_ids = torch.arange(n_blk)
    q_pos = torch.arange(T)
    valid = (blk_ids[None, :] + 1) * block_size <= (q_pos[:, None] + 1)   # [T,n_blk]
    score = torch.rand(B, Hkv, T, n_blk, generator=g)
    score = score.masked_fill(~valid[None, None], -1.0)                   # invalid last
    k = min(S, n_blk)
    topv, topi = score.topk(k, dim=-1)                                    # [B,Hkv,T,k]
    sel = torch.where(topv >= 0, topi.to(torch.int32),
                      torch.full_like(topi, -1, dtype=torch.int32))
    idx = torch.full((B, Hkv, T, S), -1, dtype=torch.int32)
    idx[..., :k] = sel
    return idx.to(device)


def run_one(cfg, seed=0):
    torch.manual_seed(seed)
    B, Hq, Hkv, D, T = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"], cfg["T"]
    bs, S = cfg["block_size"], cfg["S"]
    q = torch.randn(B, Hq, T, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device=DEV, dtype=torch.bfloat16)
    idx = make_block_idx(B, Hkv, T, bs, S, DEV, seed=seed)

    ref = selection_forward_reference(q, k, v, idx, bs)
    ker = selection_forward_triton(q, k, v, idx, bs)

    a, b = ref.float(), ker.float()
    max_abs = (a - b).abs().max().item()
    denom = b.abs().max().item() + 1e-6
    ok = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    return ok, max_abs, max_abs / denom


def main():
    print(f"correctness gate: allclose(kernel, reference) | atol={ATOL} rtol={RTOL}\n")
    header = f"{'config':52s} {'max_abs':>10s} {'max_rel':>10s}  verdict"
    print(header); print("-" * len(header))
    all_ok = True
    for cfg in CONFIGS:
        G = cfg["Hq"] // cfg["Hkv"]
        tag = (f"B{cfg['B']} Hq{cfg['Hq']} Hkv{cfg['Hkv']}(G{G}) D{cfg['D']} "
               f"T{cfg['T']} bs{cfg['block_size']} S{cfg['S']}")
        ok, mabs, mrel = run_one(cfg)
        all_ok &= ok
        print(f"{tag:52s} {mabs:10.2e} {mrel:10.2e}  {'PASS' if ok else 'FAIL'}")

    print()
    if all_ok:
        print("GATE GREEN — kernel matches reference across all configs.")
        sys.exit(0)
    else:
        print("GATE RED — kernel disagrees with reference. Fast-but-wrong fails.")
        sys.exit(1)


if __name__ == "__main__":
    main()
