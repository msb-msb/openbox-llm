"""
smoke_integration.py — Gate 3: a short REAL training run at the 150M rung-2a config
with --kernel fused. Confirms: (a) it completes and loss drops, (b) the fused path's
VRAM stays flat vs ref at the model level (not just the kernel microbench), and the
gap widens with seq_len until the dense ref OOMs.

Uses the cached FineWeb-Edu tokens from rung 2a (data/train.bin). Resource caps:
mem fraction 0.70, threads 4; run under nice -n 10. bf16 autocast (as rung 2a).
"""

import json
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nsa_model import NSATransformer

torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)
DEV = "cuda"

# rung-2a 150M config
CFG = dict(d_model=768, n_layers=16, n_q_heads=12, n_kv_heads=4, ffn_mult=4,
           block_size=16, n_selected_blocks=8, window=64)
SEQ_LEN, BATCH, STEPS, SEED = 512, 8, 80, 1337


def load_data():
    meta = json.load(open("data/meta.json"))
    data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    return meta["vocab_size"], data


def make_starts(n_tokens, seq_len, n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tokens - seq_len - 1, size=n, dtype=np.int64)


def get_batch(data, starts, step, seq_len, batch):
    idx = starts[step * batch:(step + 1) * batch]
    xb = np.stack([data[s:s + seq_len].astype(np.int64) for s in idx])
    yb = np.stack([data[s + 1:s + 1 + seq_len].astype(np.int64) for s in idx])
    return (torch.from_numpy(xb).to(DEV), torch.from_numpy(yb).to(DEV))


def build(impl, vocab):
    torch.manual_seed(SEED)
    return NSATransformer(vocab, max_seq_len=SEQ_LEN, attn_type="nsa",
                          attn_impl=impl, **CFG).to(DEV)


def train(impl, vocab, data, starts):
    model = build(impl, vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], time.time()
    for step in range(STEPS):
        x, y = get_batch(data, starts, step, SEQ_LEN, BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 20 == 0:
            print(f"  [{impl}] step {step+1:3d}/{STEPS}  loss {loss.item():.4f}  "
                  f"({(time.time()-t0):.1f}s)")
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return losses, peak


def vram_at_seqlen(impl, vocab, data, T, batch=4):
    """Peak VRAM for one fwd+bwd training step at seq_len T (model level)."""
    torch.manual_seed(SEED)
    model = NSATransformer(vocab, max_seq_len=T, attn_type="nsa",
                           attn_impl=impl, **CFG).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    starts = make_starts(len(data), T, batch, seed=7)
    x, y = get_batch(data, starts, 0, T, batch)
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        oom = False
    except torch.cuda.OutOfMemoryError:
        peak, oom = float("nan"), True
    del model, opt
    torch.cuda.empty_cache()
    return peak, oom


def main():
    os.makedirs("plots", exist_ok=True)
    vocab, data = load_data()
    print(f"150M rung-2a config | seq_len {SEQ_LEN} batch {BATCH} steps {STEPS} | "
          f"vocab {vocab}\n")
    starts = make_starts(len(data), SEQ_LEN, STEPS * BATCH, seed=SEED)

    n_params = sum(p.numel() for p in build("ref", vocab).parameters()) / 1e6
    print(f"params: {n_params:.1f}M\n")

    print("training FUSED:")
    lf, peak_f = train("fused", vocab, data, starts)
    print("training REF:")
    lr, peak_r = train("ref", vocab, data, starts)

    drop_f = lf[0] - lf[-1]
    print(f"\nloss drop  fused: {lf[0]:.3f} -> {lf[-1]:.3f}  (Δ {drop_f:+.3f})")
    print(f"loss drop  ref  : {lr[0]:.3f} -> {lr[-1]:.3f}  (Δ {lr[0]-lr[-1]:+.3f})")
    print(f"final loss delta (fused-ref): {lf[-1]-lr[-1]:+.4f}")
    print(f"peak VRAM @ seq{SEQ_LEN} b{BATCH}:  fused {peak_f:.2f}GB   ref {peak_r:.2f}GB")

    # ---- model-level VRAM vs seq_len (the memory win) ----
    print("\nmodel-level peak VRAM vs seq_len (one fwd+bwd step, batch 4):")
    seqs = [512, 1024, 2048, 4096, 8192]
    rows = []
    for T in seqs:
        pf, of = vram_at_seqlen("fused", vocab, data, T)
        pr, orr = vram_at_seqlen("ref", vocab, data, T)
        rows.append((T, pf, of, pr, orr))
        fs = "OOM" if of else f"{pf:.2f}GB"
        rs = "OOM" if orr else f"{pr:.2f}GB"
        print(f"  seq {T:5d}:  fused {fs:>8}   ref {rs:>8}")

    # ---- plots ----
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, STEPS + 1), lr, color="tab:blue", lw=2, label="ref")
    plt.plot(range(1, STEPS + 1), lf, color="tab:red", lw=2, ls="--", label="fused")
    plt.xlabel("step"); plt.ylabel("train loss")
    plt.title(f"Integration smoke — 150M NSA, ref vs fused (seq {SEQ_LEN})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/integration_loss.png", dpi=130)
    print("\nwrote plots/integration_loss.png")

    plt.figure(figsize=(8, 5))
    xs = [r[0] for r in rows]
    fused_v = [r[1] for r in rows]
    ref_v = [r[3] for r in rows]
    plt.plot(xs, fused_v, "o-", color="tab:red", lw=2, label="fused (kernel path)")
    plt.plot(xs, ref_v, "s-", color="tab:blue", lw=2, label="ref (dense O(T²))")
    ref_oom = next((r[0] for r in rows if r[4]), None)
    if ref_oom:
        plt.axvline(ref_oom, color="gray", ls="--", alpha=0.7)
        plt.annotate(f"ref OOM ≥ {ref_oom}", xy=(ref_oom, 1.0), ha="right", fontsize=9,
                     bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("seq_len"); plt.ylabel("peak VRAM (GB), fwd+bwd, batch 4")
    plt.title("Integration smoke — model-level VRAM vs seq_len (150M NSA)")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("plots/integration_vram.png", dpi=130)
    print("wrote plots/integration_vram.png")

    ok_drop = drop_f > 0.3
    print(f"\nloss dropped (fused): {ok_drop}")
    print("SMOKE GREEN" if ok_drop else "SMOKE: loss did not drop enough")


if __name__ == "__main__":
    main()
