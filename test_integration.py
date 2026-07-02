"""
test_integration.py — PARITY gates for the --kernel {ref,fused} flag in nsa_model.py.

Gate 1 (LOSS PARITY): same seed / weights / data / config, a few training steps run
both ways — forward loss must track between attn_impl='ref' and 'fused'.
Gate 2 (GRAD PARITY): one backward step, both paths — every parameter's grad must
match within tolerance (this is what proves the flag doesn't silently change
training dynamics).

fp32 = tight, bf16 = loose. trap-1 preserved (block_idx derived under no_grad, same
as ref, fed to the selection kernel as a non-differentiable index input).
"""

import copy
import sys
import torch

from nsa_model import NSATransformer

torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)
DEV = "cuda"

CFG = dict(vocab_size=256, d_model=128, n_layers=2, n_q_heads=4, n_kv_heads=2,
           max_seq_len=128, ffn_mult=2, block_size=16, n_selected_blocks=4, window=32)
B, T, STEPS = 4, 128, 8

# fp32 tight, bf16 loose
LOSS_TOL = {torch.float32: 5e-4, torch.bfloat16: 5e-2}
GRAD_TOL = {torch.float32: 2e-3, torch.bfloat16: 8e-2}


def make_batches(n, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, CFG["vocab_size"], (B, T), generator=g)
        y = torch.randint(0, CFG["vocab_size"], (B, T), generator=g)
        out.append((x.to(DEV), y.to(DEV)))
    return out


def build(impl, snapshot):
    m = NSATransformer(attn_type="nsa", attn_impl=impl, **CFG).to(DEV)
    m.load_state_dict(snapshot)
    return m


def train_losses(impl, snapshot, batches, dtype):
    torch.manual_seed(0)
    m = build(impl, snapshot)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    use_amp = dtype == torch.bfloat16
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def grad_parity(snapshot, batch, dtype):
    x, y = batch
    use_amp = dtype == torch.bfloat16

    def grads(impl):
        torch.manual_seed(0)
        m = build(impl, snapshot)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = m(x, y)
        m.zero_grad(set_to_none=True)
        loss.backward()
        return {n: p.grad.detach().float().clone()
                for n, p in m.named_parameters() if p.grad is not None}

    gr, gf = grads("ref"), grads("fused")
    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    for n in gr:
        a, b = gr[n], gf[n]
        ad = (a - b).abs().max().item()
        rd = ((a - b).abs() / (a.abs() + 1e-6)).max().item()
        if ad > worst_abs:
            worst_abs, worst_name = ad, n
        worst_rel = max(worst_rel, rd)
    return worst_abs, worst_rel, worst_name, len(gr)


def main():
    torch.manual_seed(0)
    ref0 = NSATransformer(attn_type="nsa", attn_impl="ref", **CFG).to(DEV)
    snapshot = copy.deepcopy(ref0.state_dict())
    all_ok = True

    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).split(".")[-1]
        batches = make_batches(STEPS, seed=1)

        # ---- Gate 1: loss parity over a few steps ----
        lr = train_losses("ref", snapshot, batches, dtype)
        lf = train_losses("fused", snapshot, batches, dtype)
        tol = LOSS_TOL[dtype]
        print(f"\n=== Gate 1  LOSS PARITY | {name} | tol={tol} ===")
        print(f"{'step':>4} {'ref loss':>12} {'fused loss':>12} {'|delta|':>12}  ok")
        g1 = True
        for i, (a, b) in enumerate(zip(lr, lf)):
            d = abs(a - b); ok = d <= tol; g1 &= ok
            print(f"{i:>4} {a:12.6f} {b:12.6f} {d:12.2e}  {'ok' if ok else 'FAIL'}")
        all_ok &= g1

        # ---- Gate 2: grad parity (one backward) ----
        wa, wr, wn, npar = grad_parity(snapshot, batches[0], dtype)
        tol = GRAD_TOL[dtype]
        g2 = wa <= tol
        all_ok &= g2
        print(f"\n=== Gate 2  GRAD PARITY | {name} | tol={tol} ===")
        print(f"  params compared : {npar}")
        print(f"  worst |dgrad|   : {wa:.2e}  (at {wn})")
        print(f"  worst rel dgrad : {wr:.2e}")
        print(f"  verdict         : {'PASS' if g2 else 'FAIL'}")

    print()
    if all_ok:
        print("INTEGRATION GATES GREEN — ref and fused match on loss + grads.")
        sys.exit(0)
    print("INTEGRATION GATES RED.")
    sys.exit(1)


if __name__ == "__main__":
    main()
