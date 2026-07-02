"""
test_kernel_fused.py — CORRECTNESS + GRADCHECK gate for the FUSED three-branch path.

Gate 1: full three-branch output (compression + selection + window, recombined by
the gate) from the kernel path matches the pure-torch three-branch reference
(== rung-1 NSAAttention math, pre out_proj).
Gate 2: gradcheck (analytical-vs-analytical) of grads w.r.t. every differentiable
input (q, kc, vc, ks, vs, kw, vw, gate_logits) through the fused output. block_idx
is a non-differentiable input (trap-1) — not checked, no grad expected.

Same config matrix as forward/backward: seq_len, #heads, GQA G=1/2/3/4, head dim
32/64/128, block_size, S > a row's valid block count. fp32 (tight) + bf16 (loose).
"""

import sys
import torch

from nsa_fused_kernel import nsa_fused_forward, nsa_fused_reference
from test_kernel import CONFIGS, make_block_idx

torch.cuda.set_per_process_memory_fraction(0.70)
torch.set_num_threads(4)
DEV = "cuda"

FWD_TOL = {torch.float32: 2e-3, torch.bfloat16: 3e-2}
BWD_TOL = {torch.float32: 3e-3, torch.bfloat16: 7e-2}


def window_for(cfg):
    # deliberately NOT a multiple of block_size or the kernel tile (64), <= T
    return min(cfg["T"], 2 * cfg["block_size"] + 5)


def make_inputs(cfg, dtype, seed=0):
    torch.manual_seed(seed)
    B, Hq, Hkv, D, T = cfg["B"], cfg["Hq"], cfg["Hkv"], cfg["D"], cfg["T"]
    mk_q = lambda h: torch.randn(B, h, T, D, device=DEV, dtype=dtype)
    q = mk_q(Hq)
    kc, vc, ks, vs, kw, vw = (mk_q(Hkv) for _ in range(6))
    gate = torch.randn(B, Hq, T, 3, device=DEV, dtype=dtype)
    idx = make_block_idx(B, Hkv, T, cfg["block_size"], cfg["S"], DEV, seed=seed)
    return q, kc, vc, ks, vs, kw, vw, gate, idx


def run_fwd(cfg, dtype):
    q, kc, vc, ks, vs, kw, vw, gate, idx = make_inputs(cfg, dtype)
    bs, w = cfg["block_size"], window_for(cfg)
    ref = nsa_fused_reference(q, kc, vc, ks, vs, kw, vw, gate, idx, bs, w)
    ker = nsa_fused_forward(q, kc, vc, ks, vs, kw, vw, gate, idx, bs, w)
    return (ref.float() - ker.float()).abs().max().item()


def run_bwd(cfg, dtype):
    q, kc, vc, ks, vs, kw, vw, gate, idx = make_inputs(cfg, dtype)
    bs, w = cfg["block_size"], window_for(cfg)
    do = torch.randn_like(q)
    names = ["q", "kc", "vc", "ks", "vs", "kw", "vw", "gate"]

    def grads(fn):
        ins = [t.clone().detach().requires_grad_(True)
               for t in (q, kc, vc, ks, vs, kw, vw, gate)]
        out = fn(ins[0], ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[7],
                 idx, bs, w)
        out.backward(do)
        return [t.grad for t in ins]

    gr = grads(nsa_fused_reference)
    gt = grads(nsa_fused_forward)
    return {nm: (a.float() - b.float()).abs().max().item()
            for nm, a, b in zip(names, gt, gr)}


def main():
    all_ok = True

    for dtype in (torch.float32, torch.bfloat16):
        tol = FWD_TOL[dtype]
        print(f"\n=== FORWARD gate | {str(dtype).split('.')[-1]} | atol={tol} ===")
        hdr = f"{'config':52s} {'max_abs':>10s}  verdict"
        print(hdr); print("-" * len(hdr))
        for cfg in CONFIGS:
            G = cfg["Hq"] // cfg["Hkv"]
            tag = (f"B{cfg['B']} Hq{cfg['Hq']} Hkv{cfg['Hkv']}(G{G}) D{cfg['D']} "
                   f"T{cfg['T']} bs{cfg['block_size']} S{cfg['S']}")
            e = run_fwd(cfg, dtype)
            ok = e <= tol; all_ok &= ok
            print(f"{tag:52s} {e:10.2e}  {'PASS' if ok else 'FAIL'}")

    for dtype in (torch.float32, torch.bfloat16):
        tol = BWD_TOL[dtype]
        print(f"\n=== GRADCHECK gate | {str(dtype).split('.')[-1]} | atol={tol} ===")
        hdr = f"{'config':40s} " + " ".join(f"{n:>8s}" for n in
              ["q", "kc", "vc", "ks", "vs", "kw", "vw", "gate"]) + "  verdict"
        print(hdr); print("-" * len(hdr))
        for cfg in CONFIGS:
            G = cfg["Hq"] // cfg["Hkv"]
            tag = f"B{cfg['B']} Hq{cfg['Hq']} Hkv{cfg['Hkv']}(G{G}) D{cfg['D']} T{cfg['T']}"
            e = run_bwd(cfg, dtype)
            ok = all(x <= tol for x in e.values()); all_ok &= ok
            cells = " ".join(f"{e[n]:8.1e}" for n in
                             ["q", "kc", "vc", "ks", "vs", "kw", "vw", "gate"])
            print(f"{tag:40s} {cells}  {'PASS' if ok else 'FAIL'}")

    print()
    if all_ok:
        print("FUSED GATE GREEN — three-branch forward + all grads match the reference.")
        sys.exit(0)
    print("FUSED GATE RED.")
    sys.exit(1)


if __name__ == "__main__":
    main()
