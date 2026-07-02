"""
smoke_rung2b.py — RUNG 2b SMOKE TEST (throughput + peak VRAM only, NO full run).

Goal: measure real tokens/sec and peak VRAM for the ~1.5B config on this 3090 so
we can pick a token budget. Runs ~60s of real training steps (fwd+bwd+opt) on the
NSA arm (the heavier one; if it fits, full fits). HARD CAP: peak VRAM < 18GB
(Xorg already uses ~3GB, so ~21GB of 24GB total).

The 1.5B config (scaled from rung 2a, same-hidden-dims framing):
    d_model=2048, n_layers=30, GQA 16 q-heads / 4 kv-heads, ffn_mult=4, ctx 512.
    full ≈ 1.426B params, nsa ≈ 1.556B (delta = NSA's per-branch kv + gate + compressor).

Why the fallback ladder matters: a 1.5B model under plain fp32 AdamW needs
~4× params of memory just for weights+grads+Adam moments (~25GB) — that FIXED cost
alone busts a 24GB card regardless of batch size or activation checkpointing.
So this script runs the ladder the brief asked for and reports which rung works:
    1. reduce batch (keep ctx 512)
    2. gradient checkpointing
    3. 8-bit Adam (bitsandbytes)
"""

import argparse
import time

import torch

from nsa_model import NSATransformer

# The 1.5B config — fixed across the whole ladder.
CFG = dict(vocab_size=50257, d_model=2048, n_layers=30, n_q_heads=16,
           n_kv_heads=4, max_seq_len=512, ffn_mult=4,
           block_size=16, n_selected_blocks=8, window=64)


def build_optim(model, kind, lr=3e-4):
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    if kind == "adam8bit":
        import bitsandbytes as bnb
        return bnb.optim.Adam8bit(model.parameters(), lr=lr, betas=(0.9, 0.95))
    raise ValueError(kind)


def run(attn, batch, ctx, optim_kind, grad_ckpt, seconds, device, weight_dtype):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = NSATransformer(attn_type=attn, **{**CFG, "max_seq_len": ctx}).to(device)
    if weight_dtype == "bf16":
        # bf16 master weights: halves the fixed params+grads cost (12.4GB->6.2GB
        # at 1.5B). Departs from fp32-master mixed precision, but that purity is
        # already gone once we use 8-bit Adam; the realistic lever for 1.5B/24GB.
        model = model.to(torch.bfloat16)
    model.grad_checkpoint = grad_ckpt
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optim(model, optim_kind)

    x = torch.randint(0, CFG["vocab_size"], (batch, ctx), device=device)
    y = torch.randint(0, CFG["vocab_size"], (batch, ctx), device=device)

    # warmup (also where OOM usually first bites: grads + optimizer moments)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # timed window
    steps = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
    torch.cuda.synchronize()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    toks = steps * batch * ctx
    return dict(params=n_params, steps=steps, dt=dt, tok_s=toks / dt,
                peak_gb=peak, loss=loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="nsa", choices=["nsa", "full"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "adam8bit"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--weight_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--cap_gb", type=float, default=18.0)
    args = ap.parse_args()

    torch.set_num_threads(4)  # leave cores for the desktop
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    tag = (f"attn={args.attn} batch={args.batch} ctx={args.ctx} "
           f"optim={args.optim} grad_ckpt={args.grad_ckpt} wdtype={args.weight_dtype}")
    print(f">>> {tag}")
    try:
        r = run(args.attn, args.batch, args.ctx, args.optim,
                args.grad_ckpt, args.seconds, device, args.weight_dtype)
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"    RESULT: OOM  (peak before OOM ~{peak:.2f}GB)")
        print(f"    -> {str(e)[:110]}")
        return
    verdict = "OK  " if r["peak_gb"] < args.cap_gb else "OVER-CAP"
    print(f"    params={r['params']/1e9:.3f}B  steps={r['steps']}  "
          f"tok/s={r['tok_s']:.0f}  peak_vram={r['peak_gb']:.2f}GB  "
          f"[{verdict} cap {args.cap_gb}GB]")


if __name__ == "__main__":
    main()
