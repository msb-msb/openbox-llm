"""
bench_kernel.py — throughput + VRAM: Triton selection kernel vs pure-torch reference.

Sweeps seq_len {512,1k,2k,4k,8k,16k,32k}, logs a CSV, and writes two plots:
    plots/vram_vs_seqlen.png  — kernel vs reference peak VRAM; marks the reference
                                OOM ceiling (recorded as a real result, not a crash).
    plots/toks_vs_seqlen.png  — kernel vs reference tokens/sec; shows the crossover
                                seq_len where the sparse kernel starts to win.

The point: the reference is O(T^2) (materializes the full score matrix, OOMs at
long context); the kernel touches only S*block_size keys per query (O(T)), so it
scales far past where the reference dies.

VRAM guard: per-process cap at 70% of the card. The reference WILL OOM at high
seq_len — that's expected data. We catch torch.cuda.OutOfMemoryError, record the
ceiling, empty_cache, and continue.
"""

import csv
import os
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nsa_selection_kernel import (selection_forward_reference,
                                  selection_forward_triton)

torch.cuda.set_per_process_memory_fraction(0.70)   # hard-cap ~17GB, leave desktop room
torch.set_num_threads(4)
DEV = "cuda"

# Fixed model shape; only seq_len varies. S*block_size keys/query is CONSTANT in T
# (that's NSA's point), so kernel work is O(T) while the reference is O(T^2).
B, Hq, Hkv, D = 1, 16, 4, 64
BLOCK_SIZE, S = 64, 16
SEQLENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def make_block_idx_recent(B, Hkv, T, block_size, S, device):
    """Each query selects its S most-recent causally-valid blocks (-1 padded).

    Vectorized (O(T*S)) so it's cheap even at 32k. Any valid index layout gives the
    same kernel cost; the recent pattern is a realistic, deterministic choice.
    """
    n_valid = (torch.arange(T, device=device) + 1) // block_size      # [T]
    offs = torch.arange(S, device=device)
    blk = (n_valid[:, None] - 1) - offs[None, :]                      # [T,S]
    blk = torch.where(blk >= 0, blk, torch.full_like(blk, -1))
    return blk[None, None].expand(B, Hkv, T, S).contiguous().to(torch.int32)


def timed(fn, warmup=2, iters=3):
    """Median wall time (s) and peak VRAM (GB) for fn(); raises OOM to caller."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return ts[len(ts) // 2], peak


def bench_seqlen(T):
    q = torch.randn(B, Hq, T, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device=DEV, dtype=torch.bfloat16)
    idx = make_block_idx_recent(B, Hkv, T, BLOCK_SIZE, S, DEV)
    row = {"seq_len": T}

    # --- kernel (shouldn't OOM) ---
    try:
        dt, peak = timed(lambda: selection_forward_triton(q, k, v, idx, BLOCK_SIZE))
        row["kernel_ms"] = dt * 1e3
        row["kernel_tok_s"] = B * T / dt
        row["kernel_vram_gb"] = peak
        row["kernel_oom"] = 0
    except torch.cuda.OutOfMemoryError:
        row.update(kernel_ms=float("nan"), kernel_tok_s=float("nan"),
                   kernel_vram_gb=float("nan"), kernel_oom=1)
        torch.cuda.empty_cache()

    # --- reference (expected to OOM at high seq_len — that's the data) ---
    try:
        dt, peak = timed(lambda: selection_forward_reference(q, k, v, idx, BLOCK_SIZE),
                         warmup=1, iters=2)
        row["ref_ms"] = dt * 1e3
        row["ref_tok_s"] = B * T / dt
        row["ref_vram_gb"] = peak
        row["ref_oom"] = 0
    except torch.cuda.OutOfMemoryError:
        row.update(ref_ms=float("nan"), ref_tok_s=float("nan"),
                   ref_vram_gb=float("nan"), ref_oom=1)
        torch.cuda.empty_cache()

    del q, k, v, idx
    torch.cuda.empty_cache()
    return row


def main():
    os.makedirs("plots", exist_ok=True)
    rows = []
    print(f"bench: B{B} Hq{Hq} Hkv{Hkv} D{D} block_size{BLOCK_SIZE} S{S} "
          f"(selected keys/query = {S*BLOCK_SIZE})\n")
    hdr = f"{'seq_len':>8} {'ker_ms':>9} {'ker_tok/s':>11} {'ker_GB':>7} " \
          f"{'ref_ms':>9} {'ref_tok/s':>11} {'ref_GB':>7}  note"
    print(hdr); print("-" * len(hdr))
    for T in SEQLENS:
        r = bench_seqlen(T)
        note = "ref OOM" if r["ref_oom"] else ""
        def f(x, w, p=1):
            return (f"{x:>{w}.{p}f}" if x == x else f"{'OOM':>{w}}")  # nan -> OOM
        print(f"{T:>8} {f(r['kernel_ms'],9,2)} {f(r['kernel_tok_s'],11,0)} "
              f"{f(r['kernel_vram_gb'],7,3)} {f(r['ref_ms'],9,2)} "
              f"{f(r['ref_tok_s'],11,0)} {f(r['ref_vram_gb'],7,3)}  {note}")
        rows.append(r)

    # --- CSV ---
    fields = ["seq_len", "kernel_ms", "kernel_tok_s", "kernel_vram_gb", "kernel_oom",
              "ref_ms", "ref_tok_s", "ref_vram_gb", "ref_oom"]
    with open("plots/bench_kernel.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("\nwrote plots/bench_kernel.csv")

    xs = [r["seq_len"] for r in rows]
    ref_oom_T = next((r["seq_len"] for r in rows if r["ref_oom"]), None)

    # --- plot 1: VRAM vs seq_len ---
    plt.figure(figsize=(8, 5))
    kv = [r["kernel_vram_gb"] for r in rows]
    rv = [r["ref_vram_gb"] for r in rows]
    plt.plot(xs, kv, "o-", color="tab:red", lw=2, label="kernel (Triton, sparse)")
    plt.plot(xs, rv, "s-", color="tab:blue", lw=2, label="reference (dense O(T²))")
    if ref_oom_T:
        plt.axvline(ref_oom_T, color="gray", ls="--", alpha=0.7)
        plt.annotate(f"reference OOM ≥ {ref_oom_T}\n(70% VRAM cap)",
                     xy=(ref_oom_T, plt.ylim()[1] * 0.5), ha="right", fontsize=9,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("seq_len"); plt.ylabel("peak VRAM (GB)")
    plt.title("NSA selection forward — peak VRAM vs seq_len")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/vram_vs_seqlen.png", dpi=130)
    print("wrote plots/vram_vs_seqlen.png")

    # --- plot 2: tokens/sec vs seq_len (crossover) ---
    plt.figure(figsize=(8, 5))
    kt = [r["kernel_tok_s"] for r in rows]
    rt = [r["ref_tok_s"] for r in rows]
    plt.plot(xs, kt, "o-", color="tab:red", lw=2, label="kernel (Triton, sparse)")
    plt.plot(xs, rt, "s-", color="tab:blue", lw=2, label="reference (dense O(T²))")
    # crossover: first seq_len where kernel tok/s >= reference tok/s (both present)
    cross = None
    for r in rows:
        if r["kernel_tok_s"] == r["kernel_tok_s"] and r["ref_tok_s"] == r["ref_tok_s"]:
            if r["kernel_tok_s"] >= r["ref_tok_s"]:
                cross = r["seq_len"]; break
    if cross:
        plt.axvline(cross, color="green", ls="--", alpha=0.7)
        plt.annotate(f"kernel wins ≥ {cross}", xy=(cross, plt.ylim()[1] * 0.6),
                     ha="left", fontsize=9,
                     bbox=dict(boxstyle="round", fc="honeydew", alpha=0.8))
    if ref_oom_T:
        plt.axvline(ref_oom_T, color="gray", ls="--", alpha=0.5)
        plt.annotate(f"ref OOM ≥ {ref_oom_T}", xy=(ref_oom_T, plt.ylim()[1] * 0.25),
                     ha="right", fontsize=9,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.xscale("log", base=2)
    plt.xlabel("seq_len"); plt.ylabel("throughput (tokens/sec)")
    plt.title("NSA selection forward — throughput vs seq_len")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/toks_vs_seqlen.png", dpi=130)
    print("wrote plots/toks_vs_seqlen.png")

    if ref_oom_T:
        print(f"\nreference OOM ceiling: seq_len >= {ref_oom_T} (recorded result)")
    if cross:
        print(f"kernel throughput crossover: seq_len >= {cross}")


if __name__ == "__main__":
    main()
