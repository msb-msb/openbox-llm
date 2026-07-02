"""
bench_kernel_backward.py — throughput + VRAM for the selection BACKWARD:
Triton backward kernel vs autograd through the pure-torch reference.

Same sweep as the forward bench (seq_len {512..32768}), CSV + two plots:
    plots/vram_vs_seqlen_bwd.png   — kernel vs reference peak VRAM (+ ref OOM ceiling)
    plots/toks_vs_seqlen_bwd.png   — kernel vs reference tokens/sec (+ crossover)

Confirms the flat-VRAM O(T) scaling holds for BACKWARD too: the reference backward
must retain the O(T^2) forward graph (the [T,T] softmax probs), so it OOMs at long
context; the Triton backward recomputes softmax over the gathered blocks and only
allocates dQ/dK/dV (O(T)).

VRAM guard: per-process 70% cap; reference OOM is caught, recorded as data, and the
sweep continues (empty_cache).
"""

import csv
import os
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nsa_selection_kernel import (selection_forward_triton,
                                  selection_forward_reference)
from nsa_selection_backward import selection_backward_triton

torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)
DEV = "cuda"

B, Hq, Hkv, D = 1, 16, 4, 64
BLOCK_SIZE, S = 64, 16
SEQLENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def make_block_idx_recent(B, Hkv, T, block_size, S, device):
    n_valid = (torch.arange(T, device=device) + 1) // block_size
    offs = torch.arange(S, device=device)
    blk = (n_valid[:, None] - 1) - offs[None, :]
    blk = torch.where(blk >= 0, blk, torch.full_like(blk, -1))
    return blk[None, None].expand(B, Hkv, T, S).contiguous().to(torch.int32)


def _median(ts):
    ts.sort(); return ts[len(ts) // 2]


def time_kernel_bwd(q, k, v, idx, do, iters=3):
    o = selection_forward_triton(q, k, v, idx, BLOCK_SIZE)     # not timed
    selection_backward_triton(q, k, v, idx, BLOCK_SIZE, do, o)  # warmup (autotune)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        selection_backward_triton(q, k, v, idx, BLOCK_SIZE, do, o)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return _median(ts), torch.cuda.max_memory_allocated() / 1e9


def time_reference_bwd(q, k, v, idx, do, iters=2):
    qr, kr, vr = (t.clone().detach().requires_grad_(True) for t in (q, k, v))
    torch.cuda.reset_peak_memory_stats()
    o = selection_forward_reference(qr, kr, vr, idx, BLOCK_SIZE)  # graph retained
    torch.autograd.grad(o, [qr, kr, vr], do, retain_graph=True)   # warmup (cuBLAS)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        torch.autograd.grad(o, [qr, kr, vr], do, retain_graph=True)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9   # includes retained fwd graph
    return _median(ts), peak


def bench_seqlen(T):
    q = torch.randn(B, Hq, T, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, D, device=DEV, dtype=torch.bfloat16)
    do = torch.randn(B, Hq, T, D, device=DEV, dtype=torch.bfloat16)
    idx = make_block_idx_recent(B, Hkv, T, BLOCK_SIZE, S, DEV)
    row = {"seq_len": T}

    try:
        dt, peak = time_kernel_bwd(q, k, v, idx, do)
        row.update(kernel_ms=dt * 1e3, kernel_tok_s=B * T / dt,
                   kernel_vram_gb=peak, kernel_oom=0)
    except torch.cuda.OutOfMemoryError:
        row.update(kernel_ms=float("nan"), kernel_tok_s=float("nan"),
                   kernel_vram_gb=float("nan"), kernel_oom=1)
        torch.cuda.empty_cache()

    try:
        dt, peak = time_reference_bwd(q, k, v, idx, do)
        row.update(ref_ms=dt * 1e3, ref_tok_s=B * T / dt,
                   ref_vram_gb=peak, ref_oom=0)
    except torch.cuda.OutOfMemoryError:
        row.update(ref_ms=float("nan"), ref_tok_s=float("nan"),
                   ref_vram_gb=float("nan"), ref_oom=1)
        torch.cuda.empty_cache()

    del q, k, v, do, idx
    torch.cuda.empty_cache()
    return row


def main():
    os.makedirs("plots", exist_ok=True)
    # warm up cuBLAS so the first reference timing isn't polluted by context init
    _w = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    (_w @ _w).sum().backward()
    torch.cuda.synchronize()
    rows = []
    print(f"backward bench: B{B} Hq{Hq} Hkv{Hkv} D{D} block_size{BLOCK_SIZE} S{S} "
          f"(selected keys/query = {S*BLOCK_SIZE})\n")
    hdr = (f"{'seq_len':>8} {'ker_ms':>9} {'ker_tok/s':>11} {'ker_GB':>7} "
           f"{'ref_ms':>9} {'ref_tok/s':>11} {'ref_GB':>7}  note")
    print(hdr); print("-" * len(hdr))
    for T in SEQLENS:
        r = bench_seqlen(T)
        note = "ref OOM" if r.get("ref_oom") else ""

        def f(x, w, p=1):
            return f"{x:>{w}.{p}f}" if x == x else f"{'OOM':>{w}}"
        print(f"{T:>8} {f(r['kernel_ms'],9,2)} {f(r['kernel_tok_s'],11,0)} "
              f"{f(r['kernel_vram_gb'],7,3)} {f(r['ref_ms'],9,2)} "
              f"{f(r['ref_tok_s'],11,0)} {f(r['ref_vram_gb'],7,3)}  {note}")
        rows.append(r)

    fields = ["seq_len", "kernel_ms", "kernel_tok_s", "kernel_vram_gb", "kernel_oom",
              "ref_ms", "ref_tok_s", "ref_vram_gb", "ref_oom"]
    with open("plots/bench_kernel_backward.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("\nwrote plots/bench_kernel_backward.csv")

    xs = [r["seq_len"] for r in rows]
    ref_oom_T = next((r["seq_len"] for r in rows if r.get("ref_oom")), None)

    # --- VRAM ---
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["kernel_vram_gb"] for r in rows], "o-", color="tab:red", lw=2,
             label="kernel backward (Triton, sparse)")
    plt.plot(xs, [r["ref_vram_gb"] for r in rows], "s-", color="tab:blue", lw=2,
             label="reference backward (dense, retains O(T²) graph)")
    if ref_oom_T:
        plt.axvline(ref_oom_T, color="gray", ls="--", alpha=0.7)
        plt.annotate(f"reference OOM ≥ {ref_oom_T}\n(70% VRAM cap)",
                     xy=(ref_oom_T, plt.ylim()[1] * 0.5), ha="right", fontsize=9,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("seq_len"); plt.ylabel("peak VRAM (GB)")
    plt.title("NSA selection BACKWARD — peak VRAM vs seq_len")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/vram_vs_seqlen_bwd.png", dpi=130)
    print("wrote plots/vram_vs_seqlen_bwd.png")

    # --- throughput ---
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["kernel_tok_s"] for r in rows], "o-", color="tab:red", lw=2,
             label="kernel backward (Triton, sparse)")
    plt.plot(xs, [r["ref_tok_s"] for r in rows], "s-", color="tab:blue", lw=2,
             label="reference backward (dense O(T²))")
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
    plt.title("NSA selection BACKWARD — throughput vs seq_len")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/toks_vs_seqlen_bwd.png", dpi=130)
    print("wrote plots/toks_vs_seqlen_bwd.png")

    if ref_oom_T:
        print(f"\nreference backward OOM ceiling: seq_len >= {ref_oom_T}")
    if cross:
        print(f"kernel backward throughput crossover: seq_len >= {cross}")


if __name__ == "__main__":
    main()
