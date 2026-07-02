"""
test_kernel_backward.py — GRADCHECK GATE for the NSA selection BACKWARD kernel.

Compares dQ, dK, dV from the Triton backward (via the SelectionAttnTriton autograd
Function) against autograd through the trusted pure-torch selection_forward_reference,
across the SAME config matrix the forward gate used (seq_len, #heads, GQA group incl.
G=1 and G=3, head dim 32/64/128, block_size, and S > a row's valid block count).

Why not torch.autograd.gradcheck's numerical Jacobian: it needs float64, but the
kernel's tl.dot has no fp64 tensor-core path on Ampere. So we do the equivalent —
analytical (Triton) grads vs analytical (autograd-through-reference) grads — which
is a stronger check anyway (exact vs exact, no finite-difference noise). Run in fp32
(tight) and bf16 (loose) and report max abs error against tolerance.
"""

import sys
import torch

from nsa_selection_kernel import selection_forward_reference
from nsa_selection_backward import selection_attn_triton
from test_kernel import CONFIGS, make_block_idx      # reuse forward gate's matrix

torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)
DEV = "cuda"

# fp32 is the strict gate; bf16 is reported with a looser tolerance.
TOL = {torch.float32: (2e-3, 2e-3), torch.bfloat16: (6e-2, 6e-2)}


def run_one(cfg, dtype, seed=0):
    torch.manual_seed(seed)
    B, Hq, Hkv, D, T = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"], cfg["T"]
    bs, S = cfg["block_size"], cfg["S"]
    q = torch.randn(B, Hq, T, D, device=DEV, dtype=dtype)
    k = torch.randn(B, Hkv, T, D, device=DEV, dtype=dtype)
    v = torch.randn(B, Hkv, T, D, device=DEV, dtype=dtype)
    idx = make_block_idx(B, Hkv, T, bs, S, DEV, seed=seed)
    do = torch.randn(B, Hq, T, D, device=DEV, dtype=dtype)

    # reference grads (autograd through the pure-torch forward)
    qr, kr, vr = (t.clone().detach().requires_grad_(True) for t in (q, k, v))
    o_ref = selection_forward_reference(qr, kr, vr, idx, bs)
    o_ref.backward(do)

    # triton grads (through the autograd Function: triton fwd + triton bwd)
    qt, kt, vt = (t.clone().detach().requires_grad_(True) for t in (q, k, v))
    o_t = selection_attn_triton(qt, kt, vt, idx, bs)
    o_t.backward(do)

    errs = {}
    for name, gt, gr in [("dQ", qt.grad, qr.grad),
                         ("dK", kt.grad, kr.grad),
                         ("dV", vt.grad, vr.grad)]:
        a, b = gt.float(), gr.float()
        errs[name] = (a - b).abs().max().item()
    return errs


def main():
    all_ok = True
    for dtype in (torch.float32, torch.bfloat16):
        atol, rtol = TOL[dtype]
        print(f"\n=== gradcheck gate | dtype={str(dtype).split('.')[-1]} | "
              f"atol={atol} rtol={rtol} ===")
        hdr = f"{'config':52s} {'dQ err':>10s} {'dK err':>10s} {'dV err':>10s}  verdict"
        print(hdr); print("-" * len(hdr))
        for cfg in CONFIGS:
            G = cfg["Hq"] // cfg["Hkv"]
            tag = (f"B{cfg['B']} Hq{cfg['Hq']} Hkv{cfg['Hkv']}(G{G}) D{cfg['D']} "
                   f"T{cfg['T']} bs{cfg['block_size']} S{cfg['S']}")
            e = run_one(cfg, dtype)
            # tolerance relative to the grad magnitude (rtol) + atol
            ok = all(v <= atol + rtol * 1.0 for v in e.values())  # unit-scale grads
            all_ok &= ok
            print(f"{tag:52s} {e['dQ']:10.2e} {e['dK']:10.2e} {e['dV']:10.2e}  "
                  f"{'PASS' if ok else 'FAIL'}")

    print()
    if all_ok:
        print("GRADCHECK GATE GREEN — dQ/dK/dV match autograd through the reference.")
        sys.exit(0)
    print("GRADCHECK GATE RED — backward disagrees with reference autograd.")
    sys.exit(1)


if __name__ == "__main__":
    main()
