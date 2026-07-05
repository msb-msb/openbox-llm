"""
train.py — cloud training entrypoint for openbox-llm (container ENTRYPOINT).

Built for ephemeral rented GPU pods (H100 / sm_90):
  * fully configurable via ENV VARS (or CLI args) — see CONFIG below,
  * periodic CHECKPOINT (model + optimizer + step + config) to $CKPT_DIR, written
    atomically, so a killed pod RESUMES instead of restarting,
  * RESUME-from-checkpoint on startup (automatic if $CKPT_DIR has one),
  * data prepared once into $DATA_DIR (point it at a network volume so it persists),
  * VERIFY_KERNELS=1 runs the correctness gates and exits — use it on the FIRST cloud
    boot to validate the Triton NSA kernels on sm_90 before spending on a training run.

Does not change the sm_86 path or any rung/kernel file — purely additive. Defaults are
sized for a small ~200M proof run on a $50-100 / ~25-50 H100-hr budget.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from nsa_model import NSATransformer


# ---------------------------------------------------------------------------
# config: every knob reads an env var (cloud-friendly) with a sensible default.
# CLI args override env (e.g. `python train.py --batch_size 32`).
# ---------------------------------------------------------------------------
def _env(key, default, cast):
    return cast(os.environ.get(key, default))


# Model-size presets (env-driven, additive). PRESET picks the DEFAULTS for the
# model-dim knobs below; any explicit D_MODEL/N_LAYERS/... env var still overrides.
# Default preset "198m" reproduces the original ~200M defaults byte-for-byte.
PRESETS = {
    "198m": dict(d_model=1024, n_layers=12, n_q_heads=16, n_kv_heads=4,
                 ffn_mult=4, seq_len=1024),   # ~200M proof default
    "500m": dict(d_model=1536, n_layers=16, n_q_heads=12, n_kv_heads=3,
                 ffn_mult=4, seq_len=1024),   # GQA 4:1, head dim 128
    "1.5b": dict(d_model=2048, n_layers=29, n_q_heads=16, n_kv_heads=4,
                 ffn_mult=4, seq_len=1024),   # GQA 4:1, head dim 128, ~1.50B params
}


def get_config(argv):
    preset = os.environ.get("PRESET", "198m").lower()
    assert preset in PRESETS, f"PRESET must be one of {list(PRESETS)}, got {preset!r}"
    P = PRESETS[preset]
    C = dict(
        # --- model (PRESET sets these defaults; explicit env vars override) ---
        preset=preset,
        d_model=_env("D_MODEL", P["d_model"], int),
        n_layers=_env("N_LAYERS", P["n_layers"], int),
        n_q_heads=_env("N_Q_HEADS", P["n_q_heads"], int),
        n_kv_heads=_env("N_KV_HEADS", P["n_kv_heads"], int),
        ffn_mult=_env("FFN_MULT", P["ffn_mult"], int),
        seq_len=_env("SEQ_LEN", P["seq_len"], int),
        block_size=_env("BLOCK_SIZE", 32, int),
        n_selected_blocks=_env("N_SELECTED_BLOCKS", 8, int),
        window=_env("WINDOW", 256, int),
        attn_type=_env("ATTN_TYPE", "nsa", str),      # nsa | full
        attn_impl=_env("ATTN_IMPL", "fused", str),    # fused (Triton) | ref (torch)
        # --- optimization ---
        batch_size=_env("BATCH_SIZE", 24, int),
        token_budget=_env("TOKEN_BUDGET", 2_000_000_000, int),  # ~2B tokens
        lr=_env("LR", 3e-4, float),
        min_lr=_env("MIN_LR", 3e-5, float),
        warmup=_env("WARMUP", 500, int),
        weight_decay=_env("WEIGHT_DECAY", 0.1, float),
        grad_clip=_env("GRAD_CLIP", 1.0, float),
        optim=_env("OPTIM", "adamw", str),            # adamw | adam8bit
        seed=_env("SEED", 1337, int),
        # --- checkpoint / resume (ephemeral-pod survival) ---
        ckpt_dir=_env("CKPT_DIR", "checkpoints", str),
        ckpt_interval=_env("CKPT_INTERVAL", 1000, int),
        # --- eval / logging ---
        eval_interval=_env("EVAL_INTERVAL", 1000, int),
        eval_iters=_env("EVAL_ITERS", 50, int),
        log_interval=_env("LOG_INTERVAL", 20, int),
        # --- data ---
        data_dir=_env("DATA_DIR", "data", str),
        data_train_tokens=_env("DATA_TRAIN_TOKENS", 500_000_000, int),
        data_val_tokens=_env("DATA_VAL_TOKENS", 5_000_000, int),
        # --- runtime ---
        num_threads=_env("TORCH_NUM_THREADS", 8, int),
    )
    # minimal CLI override: --key value
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            k = argv[i][2:]
            if k in C:
                v = argv[i + 1]
                C[k] = type(C[k])(v)
                i += 2
                continue
        i += 1
    return C


# ---------------------------------------------------------------------------
# data: prepare once into $DATA_DIR (persists on a network volume), then sample.
# ---------------------------------------------------------------------------
def ensure_data(C):
    d = C["data_dir"]
    need = [os.path.join(d, f) for f in ("train.bin", "val.bin", "meta.json")]
    if all(os.path.exists(p) for p in need):
        meta = json.load(open(need[2]))
        print(f"[data] cache present in {d}: "
              f"{meta['train_tokens']/1e6:.0f}M train / {meta['val_tokens']/1e6:.0f}M val")
        return meta
    print(f"[data] preparing into {d} "
          f"({C['data_train_tokens']/1e6:.0f}M train tokens) — one-time stream...")
    subprocess.run([sys.executable, "data_prep.py", "--out_dir", d,
                    "--train_tokens", str(C["data_train_tokens"]),
                    "--val_tokens", str(C["data_val_tokens"])], check=True)
    return json.load(open(need[2]))


def get_batch(data, seq_len, batch, seed, step, device):
    # data order is a pure function of (seed, step) -> fully resumable, no loader state.
    rng = np.random.default_rng([seed, step])
    ix = rng.integers(0, len(data) - seq_len - 1, size=batch, dtype=np.int64)
    x = np.stack([data[i:i + seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + seq_len].astype(np.int64) for i in ix])
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


def cosine_lr(step, warmup, total, lr, min_lr):
    import math
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, prog)))


def build_optimizer(model, C):
    if C["optim"] == "adam8bit":
        import bitsandbytes as bnb
        return bnb.optim.Adam8bit(model.parameters(), lr=C["lr"],
                                  betas=(0.9, 0.95), weight_decay=C["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=C["lr"], betas=(0.9, 0.95),
                             weight_decay=C["weight_decay"])


# ---------------------------------------------------------------------------
# checkpointing: atomic save (tmp + os.replace) so a mid-save kill can't corrupt.
# ---------------------------------------------------------------------------
def ckpt_path(C):
    return os.path.join(C["ckpt_dir"], "ckpt.pt")


def save_checkpoint(C, model, opt, step):
    os.makedirs(C["ckpt_dir"], exist_ok=True)
    path = ckpt_path(C)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                "step": step, "config": C}, tmp)
    os.replace(tmp, path)   # atomic on POSIX


def maybe_resume(C, model, opt):
    path = ckpt_path(C)
    if not os.path.exists(path):
        print("[resume] no checkpoint found — starting from step 0")
        return 0
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    step = ck["step"]
    print(f"[resume] resumed from {path} at step {step}")
    return step


# ---------------------------------------------------------------------------
# first-boot kernel validation on sm_90 (the real cloud risk).
# ---------------------------------------------------------------------------
def verify_kernels():
    gates = ["test_kernel.py", "test_kernel_backward.py",
             "test_kernel_fused.py", "test_integration.py"]
    print("=== VERIFY_KERNELS: running correctness gates on this GPU ===")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
    for g in gates:
        print(f"\n----- {g} -----", flush=True)
        r = subprocess.run([sys.executable, g])
        if r.returncode != 0:
            print(f"GATE FAILED: {g} (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\n=== ALL KERNEL GATES GREEN on this GPU ===")
    sys.exit(0)


@torch.no_grad()
def evaluate(model, data, C, device):
    model.eval()
    losses = []
    for i in range(C["eval_iters"]):
        x, y = get_batch(data, C["seq_len"], C["batch_size"],
                         C["seed"] + 999983, i, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    C = get_config(sys.argv[1:])
    torch.set_num_threads(C["num_threads"])
    torch.manual_seed(C["seed"])
    np.random.seed(C["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")

    if _env("VERIFY_KERNELS", "0", str) == "1":
        verify_kernels()   # runs gates and exits

    print(f"[cfg] {json.dumps(C, indent=0)}")
    meta = ensure_data(C)
    vocab = meta["vocab_size"]
    train = np.memmap(os.path.join(C["data_dir"], "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(C["data_dir"], "val.bin"), dtype=np.uint16, mode="r")

    model = NSATransformer(
        vocab, d_model=C["d_model"], n_layers=C["n_layers"],
        n_q_heads=C["n_q_heads"], n_kv_heads=C["n_kv_heads"],
        max_seq_len=C["seq_len"], ffn_mult=C["ffn_mult"],
        block_size=C["block_size"], n_selected_blocks=C["n_selected_blocks"],
        window=C["window"], attn_type=C["attn_type"], attn_impl=C["attn_impl"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    total_steps = C["token_budget"] // (C["batch_size"] * C["seq_len"])
    print(f"[model] {n_params:.1f}M params | attn={C['attn_type']}/{C['attn_impl']} | "
          f"seq {C['seq_len']} batch {C['batch_size']} | "
          f"{total_steps} steps for {C['token_budget']/1e9:.2f}B tokens | dev {device}")

    opt = build_optimizer(model, C)
    start_step = maybe_resume(C, model, opt)
    os.makedirs(C["ckpt_dir"], exist_ok=True)
    json.dump(C, open(os.path.join(C["ckpt_dir"], "config.json"), "w"), indent=2)

    model.train()
    t0 = time.time()
    tok_seen = start_step * C["batch_size"] * C["seq_len"]
    for step in range(start_step, total_steps):
        lr = cosine_lr(step, C["warmup"], total_steps, C["lr"], C["min_lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, C["seq_len"], C["batch_size"], C["seed"], step, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C["grad_clip"])
        opt.step()
        tok_seen += C["batch_size"] * C["seq_len"]

        if step % C["log_interval"] == 0:
            dt = time.time() - t0
            tps = (tok_seen - start_step * C["batch_size"] * C["seq_len"]) / max(dt, 1e-6)
            print(f"step {step:6d}/{total_steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {tps/1e3:.1f}k tok/s | {tok_seen/1e9:.3f}B seen",
                  flush=True)

        if step > start_step and step % C["eval_interval"] == 0:
            vl = evaluate(model, val, C, device)
            print(f"  [eval] step {step} val_loss {vl:.4f}", flush=True)

        if step > start_step and step % C["ckpt_interval"] == 0:
            save_checkpoint(C, model, opt, step)
            print(f"  [ckpt] saved at step {step} -> {ckpt_path(C)}", flush=True)

    save_checkpoint(C, model, opt, total_steps)
    print(f"[done] {total_steps} steps, {tok_seen/1e9:.2f}B tokens -> {ckpt_path(C)}")


if __name__ == "__main__":
    main()
