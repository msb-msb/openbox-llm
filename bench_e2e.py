#!/usr/bin/env python3
"""
bench_e2e.py — MEASURED end-to-end fwd+bwd step speedup, FSA selection-backward
off vs on, on the fused kernel path. This produces the real ratio to replace the
Amdahl-projected 1.63x in the paper.

Method: build the 1.5B model on attn_impl="fused", time a full forward+backward
step (no optimizer, grad-ckpt not needed at seq 512/batch 1 on the flat-VRAM fused
path). Run twice: USE_FSA_BWD True vs False. Report ms/step, tok/s, and the ratio.

USE_FSA_BWD is a module-level constant in nsa_selection_backward.py, read at kernel
dispatch (line 359), so we flip it at runtime before each timing loop.

Run:  nice -n 10 python bench_e2e.py
Prereq: GPU clear (ollama stopped) — this builds a 1.5B model.
"""
import time
import torch
import nsa_selection_backward as nsb          # so we can flip USE_FSA_BWD
from nsa_model import NSATransformer

torch.set_num_threads(4)
DEV = "cuda"
assert torch.cuda.is_available()

# 1.5B config — must match the paper's shape (smoke_rung2b.py CFG).
CFG = dict(vocab_size=50257, d_model=2048, n_layers=30, n_q_heads=16,
           n_kv_heads=4, max_seq_len=1024, ffn_mult=4,
           block_size=16, n_selected_blocks=8, window=64)
SEQ, BATCH = 1024, 2
STEPS, WARMUP = 20, 8


def build():
    torch.manual_seed(0)
    m = NSATransformer(attn_type="nsa", attn_impl="fused", **CFG).to(DEV)
    m.train()
    return m


def time_step(model, fsa_on):
    nsb.USE_FSA_BWD = fsa_on                     # flip the kernel dispatch
    idx = torch.randint(0, CFG["vocab_size"], (BATCH, SEQ), device=DEV)
    tgt = torch.randint(0, CFG["vocab_size"], (BATCH, SEQ), device=DEV)

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
    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return ms, peak


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  1.5B fused path  "
          f"seq={SEQ} batch={BATCH} tok/step={SEQ*BATCH}")
    model = build()
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e9:.3f}B\n")

    # FSA backward OFF (reference selection-backward, still fused fwd path)
    off_ms, off_gb = time_step(model, fsa_on=False)
    # FSA backward ON (the block-outer kernel)
    on_ms,  on_gb  = time_step(model, fsa_on=True)

    tok = SEQ * BATCH
    print(f"{'':16}{'ms/step':>10}{'tok/s':>10}{'peakGB':>9}")
    print(f"{'FSA bwd OFF':16}{off_ms:>10.2f}{tok/(off_ms/1e3):>10.0f}{off_gb:>9.2f}")
    print(f"{'FSA bwd ON':16}{on_ms:>10.2f}{tok/(on_ms/1e3):>10.0f}{on_gb:>9.2f}")
    print(f"\nend-to-end fwd+bwd speedup (OFF/ON): {off_ms/on_ms:.3f}x")
    print("NOTE: fwd+bwd only, no optimizer. A full training step adds opt.step()")
    print("over 1.5B params (dense), which shrinks selection's share further, so the")
    print("real AdamW-in-loop speedup is <= this figure. Report as 'fwd+bwd step'.")


if __name__ == "__main__":
    main()
