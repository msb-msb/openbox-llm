#!/usr/bin/env python3
"""
bench_e2e_sweep.py — MEASURED end-to-end fwd+bwd speedup vs sequence length.

Maps the curve: FSA selection-backward off vs on, at seq {512,1024,2048,4096},
on the fused kernel path at the 1.5B shape. This tells us whether the paper's
1.63x is a seq-1024 point on a rising curve (thesis-consistent) or an outlier.

Also backs out selection-backward's implied share of the step at each seq via
Amdahl: ratio = 1/(1 - s*(1 - 1/K)), K=8.4 -> s = (1 - 1/ratio)/(1 - 1/K).

Run:  nice -n 10 python bench_e2e_sweep.py     (GPU must be clear)
"""
import torch
import nsa_selection_backward as nsb
from nsa_model import NSATransformer

torch.set_num_threads(4)
DEV = "cuda"
assert torch.cuda.is_available()

SEQS = [512, 1024, 2048, 4096]
BATCH = 1
STEPS, WARMUP = 15, 6
K_BRANCH = 8.4                      # measured selection-backward isolation speedup

# 1.5B shape (smoke_rung2b CFG); max_seq_len set to the longest sweep point.
CFG = dict(vocab_size=50257, d_model=2048, n_layers=30, n_q_heads=16,
           n_kv_heads=4, max_seq_len=max(SEQS), ffn_mult=4,
           block_size=16, n_selected_blocks=8, window=64)


def build():
    torch.manual_seed(0)
    m = NSATransformer(attn_type="nsa", attn_impl="fused", **CFG).to(DEV)
    m.train()
    return m


def time_at(model, seq, fsa_on):
    nsb.USE_FSA_BWD = fsa_on
    idx = torch.randint(0, CFG["vocab_size"], (BATCH, seq), device=DEV)
    tgt = torch.randint(0, CFG["vocab_size"], (BATCH, seq), device=DEV)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(idx, tgt)
            loss = out[1] if isinstance(out, (tuple, list)) else out
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(STEPS):
        step()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / STEPS
    gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return ms, gb


def implied_share(ratio):
    # s such that Amdahl with K_BRANCH reproduces the measured ratio
    return (1 - 1 / ratio) / (1 - 1 / K_BRANCH)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  1.5B fused  batch={BATCH}  "
          f"selection-bwd isolation={K_BRANCH}x\n")
    model = build()
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e9:.3f}B\n")
    print(f"{'seq':>6}{'off ms':>9}{'on ms':>9}{'ratio':>8}"
          f"{'sel-bwd share':>15}{'peakGB':>9}")
    print("-" * 56)
    for seq in SEQS:
        try:
            off_ms, _   = time_at(model, seq, fsa_on=False)
            on_ms,  gb  = time_at(model, seq, fsa_on=True)
            r = off_ms / on_ms
            print(f"{seq:>6}{off_ms:>9.1f}{on_ms:>9.1f}{r:>8.3f}"
                  f"{implied_share(r)*100:>13.1f}%{gb:>9.2f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{seq:>6}    OOM (batch {BATCH}) — bare step won't fit at this seq")
            break
    print("\nfwd+bwd only, no optimizer. Real AdamW-in-loop speedup is <= these,")
    print("since opt.step() over 1.5B dense params dilutes selection's share.")


if __name__ == "__main__":
    main()
