"""bench_range_micro.py — focused before/after tok/s for the range-attn branches
(window + compression) and the full fused forward, at the REAL training head
config (Hq=16, Hkv=4, D=64, G=4). Not the 32k sweep — a clean single-config timer
so per-change tok/s deltas are readable. Miu (3090) only.

Env knobs:
  RANGE_QTILE=0|1   toggle the q-tiled kernels (default whatever the module default is)
  SEQ=2048          sequence length
  ITERS=20          timed iters
"""
import os, time, torch
import nsa_fused_kernel as F

DEV = "cuda"
B, Hq, Hkv, D = 1, 16, 4, 64
BLOCK_SIZE, S, WINDOW = 64, 16, 512
SEQ = int(os.environ.get("SEQ", 2048))
ITERS = int(os.environ.get("ITERS", 20))

if "RANGE_QTILE" in os.environ:
    F.USE_QTILE = os.environ["RANGE_QTILE"] == "1"


def mk(h, T):
    return torch.randn(B, h, T, D, device=DEV, dtype=torch.bfloat16)


def block_idx_recent(T):
    n_valid = (torch.arange(T, device=DEV) + 1) // BLOCK_SIZE
    offs = torch.arange(S, device=DEV)
    blk = (n_valid[:, None] - 1) - offs[None, :]
    blk = torch.where(blk >= 0, blk, torch.full_like(blk, -1))
    return blk[None, None].expand(B, Hkv, T, S).contiguous().to(torch.int32)


def timed(fn, warmup=5, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    T = SEQ
    q = mk(Hq, T)
    kc, vc, ks, vs, kw, vw = (mk(Hkv, T) for _ in range(6))
    gate = torch.randn(B, Hq, T, 3, device=DEV, dtype=torch.bfloat16)
    idx = block_idx_recent(T)
    k_cmp = F.block_mean_pool(kc, BLOCK_SIZE)
    v_cmp = F.block_mean_pool(vc, BLOCK_SIZE)

    qtile = getattr(F, "USE_QTILE", None)
    print(f"config B{B} Hq{Hq} Hkv{Hkv} D{D} G{Hq//Hkv} seq{T} window{WINDOW} "
          f"block{BLOCK_SIZE} | USE_QTILE={qtile}")

    dt_win = timed(lambda: F.window_attn_triton(q, kw, vw, WINDOW))
    dt_cmp = timed(lambda: F.compression_attn_triton(q, k_cmp, v_cmp, BLOCK_SIZE))
    dt_fwd = timed(lambda: F.nsa_fused_forward(
        q, kc, vc, ks, vs, kw, vw, gate, idx, BLOCK_SIZE, WINDOW))

    def toks(dt):
        return B * T / dt

    print(f"  window branch : {dt_win*1e3:8.3f} ms  {toks(dt_win):12.0f} tok/s")
    print(f"  compress branch:{dt_cmp*1e3:8.3f} ms  {toks(dt_cmp):12.0f} tok/s")
    print(f"  FULL fused fwd : {dt_fwd*1e3:8.3f} ms  {toks(dt_fwd):12.0f} tok/s")


if __name__ == "__main__":
    main()
