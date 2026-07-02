"""
train_rung1.py — overfit a tiny char-level corpus with the NSA transformer.

RUNG 1 success criterion:
    The model drives loss on a few KB of text to near-zero, AND every NSA branch
    (compression, selection, window, gate, shared q-proj) shows non-None,
    non-zero gradient every step.

If loss collapses toward 0 and the grad table shows all branches lit up green,
backward flows correctly through the gate and all three branches. That is the
whole point of this rung.

Pure PyTorch (torch + numpy only). Runs on a single GPU (e.g. RTX 3090) or CPU.
"""

import argparse
import numpy as np
import torch

from nsa_model import (NSATransformer, param_groups, branch_of,
                       REQUIRED_BRANCHES)

# ---------------------------------------------------------------------------
# tiny corpus (embedded so the script is self-contained; a few KB of text).
# Overfitting THIS to near-zero loss is the rung-1 pass condition.
# ---------------------------------------------------------------------------
CORPUS = """\
The open-box principle: publish the whole box, not just the weights.
Native Sparse Attention blends three branches through a learned gate.
Branch one compresses token blocks into coarse summary tokens.
Branch two selects the top-k fine-grained blocks, scored from compression.
Branch three slides a window over the most recent local tokens.
Grouped-query attention lets many query heads share few key-value heads.
The top-k pick is not differentiable, so we never backprop through the index.
Gradient reaches selection through the compression scores and through the gate.
Rung one asks a single question: does backward flow through every branch?
Success is a tiny corpus overfit to near-zero loss with all branches lit.
Retrieval all the way down: split compute from weights, keep the box open.
Correct first, fast later; pure PyTorch now, Triton kernels much later.
A small decoder-only transformer learns these lines until it can recite them.
When the loss falls toward zero the sparse attention has learned to copy.
Compression is the coarse global view; the window is the sharp local view.
Selection bridges them by promoting the blocks that compression ranks highest.
"""


def get_batch(data, seq_len, batch_size, device):
    """Random contiguous windows from the encoded corpus (next-char targets)."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])
    return x.to(device), y.to(device)


def grad_norm_table(model):
    """Per-branch grad L2 norm + parameter count. Returns (rows, per_branch_norm)."""
    groups = param_groups(model)
    rows, norms = [], {}
    for label, items in groups.items():
        sq, n_params, n_with_grad = 0.0, 0, 0
        for _, p in items:
            n_params += p.numel()
            if p.grad is not None:
                sq += p.grad.detach().float().pow(2).sum().item()
                n_with_grad += 1
        norm = sq ** 0.5
        norms[label] = norm
        rows.append((label, norm, n_with_grad, len(items)))
    return rows, norms


def assert_all_branches_flow(norms):
    """Rung-1 gate: every required branch must have live, non-zero gradient."""
    failures = []
    for b in REQUIRED_BRANCHES:
        if b not in norms:
            failures.append(f"{b}: MISSING (no params matched)")
        elif not (norms[b] > 0.0):
            failures.append(f"{b}: grad norm == {norms[b]} (no gradient!)")
    if failures:
        raise AssertionError("Backward did NOT reach every branch:\n  "
                             + "\n  ".join(failures))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_q_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_selected_blocks", type=int, default=4)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ---- char-level tokenizer (no deps) -----------------------------------
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    print(f"corpus: {len(CORPUS)} chars, vocab: {vocab_size}, device: {device}")

    model = NSATransformer(
        vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_q_heads=args.n_q_heads, n_kv_heads=args.n_kv_heads,
        max_seq_len=args.seq_len, block_size=args.block_size,
        n_selected_blocks=args.n_selected_blocks, window=args.window,
    ).to(device)
    print(f"model params: {model.num_params()/1e6:.2f}M "
          f"(d_model={args.d_model}, layers={args.n_layers}, "
          f"GQA {args.n_q_heads}q/{args.n_kv_heads}kv)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    seq_len = min(args.seq_len, len(data) - 2)
    checked_once = False
    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, seq_len, args.batch_size, device)
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        rows, norms = grad_norm_table(model)
        # HARD assertion on step 1: prove backward reached every branch.
        assert_all_branches_flow(norms)
        checked_once = True

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"\nstep {step:4d} | loss {loss.item():.4f}")
            print("  per-branch grad norms:")
            for label, norm, ng, nt in sorted(rows):
                req = " *required" if label in REQUIRED_BRANCHES else ""
                print(f"    {label:16s} grad_norm={norm:10.4e} "
                      f"params_with_grad={ng}/{nt}{req}")

    assert checked_once
    print("\n" + "=" * 60)
    print(f"final loss: {loss.item():.4f}")
    print("all required branches had live gradient every step:",
          ", ".join(REQUIRED_BRANCHES))

    # ---- quick sanity: sample greedily from a seed prompt -----------------
    model.eval()
    prompt = "Native Sparse Attention"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(120):
            logits, _ = model(idx[:, -seq_len:])
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
    print("\nsample after overfit (should echo the corpus):")
    print("  " + "".join(itos[i] for i in idx[0].tolist()).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
