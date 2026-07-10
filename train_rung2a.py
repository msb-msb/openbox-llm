"""
train_rung2a.py — matched-baseline harness: NSA vs full attention.

RUNG 2a question:
    Does NSA track full attention on a GENERALIZATION task (held-out val loss),
    and is our matched-baseline harness sound?

THE HARNESS IS THE DELIVERABLE. There is ONE codebase; the attention module is
swapped by a single flag:  --attn {nsa,full}. Everything else is held identical:
same transformer, same tokenized data + same data ORDER (precomputed from the
seed, so both arms see byte-identical batches), same seed, same optimizer,
schedule, steps, and eval set. Only the attention math differs.

Outputs (per run, under --out_dir):
    {attn}_config.json   full config + param counts (both arms, for the delta)
    {attn}_metrics.csv   step,train_loss,val_loss at every eval
    {attn}_ckpt.pt       final checkpoint (model+opt+config+step)

Resource discipline (Miu is a daily-use workstation):
    torch.set_num_threads(4); dataloader num_workers=2, pin_memory=True.
    bf16 autocast + a batch/context sized to stay well under 21GB free VRAM.
    Run it nice:  nice -n 10 python train_rung2a.py ...   (see README)
"""

# expandable_segments reduces fragmentation OOMs; must be set before torch inits CUDA.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nsa_model import NSATransformer


# ---------------------------------------------------------------------------
# data: fixed windows over the cached uint16 token stream.
# The set + ORDER of windows is precomputed from the seed, so both arms train on
# byte-identical batches in the identical order regardless of dataloader workers.
# ---------------------------------------------------------------------------

class TokenWindows(Dataset):
    def __init__(self, bin_path, starts, seq_len):
        self.bin_path = bin_path
        self.starts = starts
        self.seq_len = seq_len
        self._data = None  # opened lazily per worker (memmap isn't fork-safe)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        s = int(self.starts[i])
        chunk = self._data[s:s + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def make_starts(n_tokens, seq_len, n_windows, seed):
    """Deterministic random window start positions (reproducible across runs)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tokens - seq_len - 1, size=n_windows, dtype=np.int64)


def cosine_lr(step, warmup, total, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    import math
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_iters):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_iters:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["nsa", "full"], required=True,
                    help="THE swap: which attention module to use")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs")
    # training
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    # model (identical for both arms)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--n_q_heads", type=int, default=12)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--ffn_mult", type=int, default=4)
    # nsa-only knobs (ignored by full)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_selected_blocks", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--profile", action="store_true",
                    help="profile ~N steps and exit; no training")
    ap.add_argument("--profile_steps", type=int, default=20)
    ap.add_argument("--profile_warmup", type=int, default=8)

    args = ap.parse_args()

    # ---- resource caps (leave the desktop usable) ------------------------
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    amp_dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    vocab_size = meta["vocab_size"]
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    n_train = np.memmap(train_bin, dtype=np.uint16, mode="r").shape[0]
    n_val = np.memmap(val_bin, dtype=np.uint16, mode="r").shape[0]

    # ---- deterministic, run-independent window schedule ------------------
    train_starts = make_starts(n_train, args.seq_len,
                               args.steps * args.batch_size, seed=args.seed)
    # val windows: same for every eval and every run (own seed offset)
    val_starts = make_starts(n_val, args.seq_len,
                             args.eval_iters * args.batch_size, seed=args.seed + 1)

    train_ds = TokenWindows(train_bin, train_starts, args.seq_len)
    val_ds = TokenWindows(val_bin, val_starts, args.seq_len)
    common = dict(batch_size=args.batch_size, shuffle=False,   # order fixed by starts
                  num_workers=2, pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_ds, **common)
    val_loader = DataLoader(val_ds, **common)

    # ---- model (only attn differs between arms) --------------------------
    model = NSATransformer(
        vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_q_heads=args.n_q_heads, n_kv_heads=args.n_kv_heads,
        max_seq_len=args.seq_len, ffn_mult=args.ffn_mult,
        block_size=args.block_size, n_selected_blocks=args.n_selected_blocks,
        window=args.window, attn_type=args.attn,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.attn}] params: {n_params/1e6:.2f}M | vocab {vocab_size} | "
          f"train {n_train/1e6:.1f}M val {n_val/1e6:.1f}M tokens | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # ---- log config to disk (reproducibility) ----------------------------
    cfg = vars(args).copy()
    cfg.update(param_count=n_params, param_count_M=round(n_params / 1e6, 3),
               train_tokens=int(n_train), val_tokens=int(n_val),
               data_meta=meta, torch=torch.__version__)
    json.dump(cfg, open(os.path.join(args.out_dir, f"{args.attn}_config.json"), "w"),
              indent=2)

    csv_path = os.path.join(args.out_dir, f"{args.attn}_metrics.csv")
    cw = csv.writer(open(csv_path, "w", newline=""))
    cw.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    # ---- train loop ------------------------------------------------------
    model.train()
    t0 = time.time()
    train_iter = iter(train_loader)
    running = []
    if args.profile:
        from profile_step import profile_step
        seen, n_params = set(), 0                 # dedup tied embeddings
        for p in model.parameters():
            if id(p) in seen: continue
            seen.add(id(p)); n_params += p.numel()
        pit = iter(train_loader)
        xb, yb = next(pit)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        def step_fn():
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        profile_step(step_fn, n_params=n_params,
                     tokens_per_step=args.batch_size * args.seq_len,
                     warmup=args.profile_warmup, active=args.profile_steps,
                     trace=True, label=args.attn)
        raise SystemExit


    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        running.append(loss.item())

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            train_loss = float(np.mean(running)); running = []
            val_loss = evaluate(model, val_loader, device, amp_dtype, args.eval_iters)
            el = time.time() - t0
            cw.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}",
                         f"{lr:.2e}", f"{el:.1f}"])
            print(f"[{args.attn}] step {step:5d}/{args.steps} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"lr {lr:.2e} | {el/60:.1f}min")

    # ---- checkpoint ------------------------------------------------------
    ckpt_path = os.path.join(args.out_dir, f"{args.attn}_ckpt.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "config": cfg, "step": args.steps}, ckpt_path)
    print(f"[{args.attn}] done in {(time.time()-t0)/60:.1f}min -> {ckpt_path}")


if __name__ == "__main__":
    main()
