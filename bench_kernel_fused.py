"""
bench_kernel_fused.py — throughput + VRAM for the FUSED three-branch NSA forward:
kernel path (compression⊕selection⊕window + gate) vs the pure-torch three-branch
reference.

Same sweep as the other benches (seq_len {512..32768}), CSV + two plots:
    plots/vram_vs_seqlen_fused.png   — kernel vs reference peak VRAM (+ ref OOM ceiling)
    plots/toks_vs_seqlen_fused.png   — kernel vs reference tokens/sec (+ crossover)

Confirms flat-VRAM scaling for the whole fused path: the reference materializes
[T,T] score matrices for the window and selection branches (O(T^2)) and OOMs at
long context; the kernel path never materializes them (selection O(T·S·Bsz),
window O(T·W), compression O(T·n_blk)) and keeps VRAM ~flat.

VRAM guard: per-process 70% cap; reference OOM is caught, recorded, and the sweep
continues.
"""

import csv
import os
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nsa_fused_kernel import nsa_fused_forward, nsa_fused_reference

torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)
DEV = "cuda"

B, Hq, Hkv, D = 1, 16, 4, 64
BLOCK_SIZE, S, WINDOW = 64, 16, 512
SEQLENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def make_block_idx_recent(B, Hkv, T, block_size, S, device):
    n_valid = (torch.arange(T, device=device) + 1) // block_size
    offs = torch.arange(S, device=device)
    blk = (n_valid[:, None] - 1) - offs[None, :]
    blk = torch.where(blk >= 0, blk, torch.full_like(blk, -1))
    return blk[None, None].expand(B, Hkv, T, S).contiguous().to(torch.int32)


def _median(ts):
    ts.sort(); return ts[len(ts) // 2]


def timed(fn, warmup=2, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return _median(ts), torch.cuda.max_memory_allocated() / 1e9


def bench_seqlen(T):
    mk = lambda h: torch.randn(B, h, T, D, device=DEV, dtype=torch.bfloat16)
    q = mk(Hq)
    kc, vc, ks, vs, kw, vw = (mk(Hkv) for _ in range(6))
    gate = torch.randn(B, Hq, T, 3, device=DEV, dtype=torch.bfloat16)
    idx = make_block_idx_recent(B, Hkv, T, BLOCK_SIZE, S, DEV)
    args = (q, kc, vc, ks, vs, kw, vw, gate, idx, BLOCK_SIZE, WINDOW)
    row = {"seq_len": T}

    try:
        dt, peak = timed(lambda: nsa_fused_forward(*args))
        row.update(kernel_ms=dt * 1e3, kernel_tok_s=B * T / dt,
                   kernel_vram_gb=peak, kernel_oom=0)
    except torch.cuda.OutOfMemoryError:
        row.update(kernel_ms=float("nan"), kernel_tok_s=float("nan"),
                   kernel_vram_gb=float("nan"), kernel_oom=1)
        torch.cuda.empty_cache()

    try:
        dt, peak = timed(lambda: nsa_fused_reference(*args), warmup=1, iters=2)
        row.update(ref_ms=dt * 1e3, ref_tok_s=B * T / dt,
                   ref_vram_gb=peak, ref_oom=0)
    except torch.cuda.OutOfMemoryError:
        row.update(ref_ms=float("nan"), ref_tok_s=float("nan"),
                   ref_vram_gb=float("nan"), ref_oom=1)
        torch.cuda.empty_cache()

    del q, kc, vc, ks, vs, kw, vw, gate, idx
    torch.cuda.empty_cache()
    return row


def main():
    os.makedirs("plots", exist_ok=True)
    rows = []
    print(f"fused bench: B{B} Hq{Hq} Hkv{Hkv} D{D} block_size{BLOCK_SIZE} S{S} "
          f"window{WINDOW}\n")
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
    with open("plots/bench_kernel_fused.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("\nwrote plots/bench_kernel_fused.csv")

    xs = [r["seq_len"] for r in rows]
    ref_oom_T = next((r["seq_len"] for r in rows if r.get("ref_oom")), None)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["kernel_vram_gb"] for r in rows], "o-", color="tab:red", lw=2,
             label="fused kernel path (sparse)")
    plt.plot(xs, [r["ref_vram_gb"] for r in rows], "s-", color="tab:blue", lw=2,
             label="three-branch reference (dense O(T²))")
    if ref_oom_T:
        plt.axvline(ref_oom_T, color="gray", ls="--", alpha=0.7)
        plt.annotate(f"reference OOM ≥ {ref_oom_T}\n(70% VRAM cap)",
                     xy=(ref_oom_T, plt.ylim()[1] * 0.5), ha="right", fontsize=9,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("seq_len"); plt.ylabel("peak VRAM (GB)")
    plt.title("NSA FUSED forward (3 branches + gate) — peak VRAM vs seq_len")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/vram_vs_seqlen_fused.png", dpi=130)
    print("wrote plots/vram_vs_seqlen_fused.png")

    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r["kernel_tok_s"] for r in rows], "o-", color="tab:red", lw=2,
             label="fused kernel path (sparse)")
    plt.plot(xs, [r["ref_tok_s"] for r in rows], "s-", color="tab:blue", lw=2,
             label="three-branch reference (dense O(T²))")
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
    plt.title("NSA FUSED forward — throughput vs seq_len")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/toks_vs_seqlen_fused.png", dpi=130)
    print("wrote plots/toks_vs_seqlen_fused.png")

    if ref_oom_T:
        print(f"\nreference OOM ceiling: seq_len >= {ref_oom_T}")
    if cross:
        print(f"fused kernel throughput crossover: seq_len >= {cross}")


if __name__ == "__main__":
    main()
