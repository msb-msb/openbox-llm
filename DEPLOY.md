# Deploying openbox-llm on rented cloud GPUs (H100 / sm_90)

Validated on a local RTX 3090 (sm_86). This describes the reproducible container, the
Hopper (sm_90) kernel audit, checkpoint/resume for ephemeral pods, and the small
proof-run config. **Purely additive — the sm_86 path is unchanged.**

## Container

**Base image:** `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`

- Official PyTorch image pinned to **exactly** Miu's stack: `torch 2.12.1` + `CUDA 13.0`
  + cuDNN 9. torch/triton/CUDA come integrated and tested — no bespoke torch install.
- **`-devel`, not `-runtime`**: Triton JIT-compiles the NSA kernels at runtime and needs
  `nvcc`/`ptxas` from the CUDA toolkit. On a fresh H100 the kernels re-autotune and
  compile for sm_90 on first use, which requires that toolchain in the image.
- **CUDA 13 is multi-arch**: the *same* image runs on sm_86 (3090) and sm_90 (H100).
  Nothing in the kernels is arch-locked (see the audit below).

`requirements.txt` is fully pinned (every dep `==`), so the cloud image matches Miu.
torch/triton/nvidia-\* are already at these versions in the base; `pip install -r
requirements.txt` reconciles them and adds the extras (tiktoken, datasets, matplotlib,
bitsandbytes).

```bash
docker build -t openbox-llm:latest .
```

## First cloud boot — validate the kernels on sm_90 BEFORE spending on training

This is the real risk: the Triton kernels were `@triton.autotune`'d and validated on
sm_86. Run the committed correctness gates on the H100 first:

```bash
docker run --gpus all -e VERIFY_KERNELS=1 openbox-llm:latest
```

That runs `test_kernel.py`, `test_kernel_backward.py`, `test_kernel_fused.py`, and
`test_integration.py` on the actual GPU and exits green/red. Green = the sparse
attention forward, backward (gradcheck), fused three-branch, and the ref-vs-fused
model integration all match reference **on sm_90**. Only start training after this.

## Kernel audit (sm_90) — static findings

Audited `nsa_selection_kernel.py`, `nsa_selection_backward.py`, `nsa_fused_kernel.py`:

**No arch hardcoding.** No `get_device_capability`/compute-cap checks, no `sm_86`/`sm_90`
literals, no shared-memory-size assumptions, no explicit warp-count (32) logic, and no
Hopper-only features (wgmma/TMA/thread-block clusters, `num_ctas`). The kernels are
plain arch-agnostic Triton.

**Autotune re-tunes per device.** Configs are `num_warps ∈ {1,2,4,8}` (selection
fwd/bwd) or `{2,4}` (fused range kernel), `num_stages ∈ {1,2,3}` — all valid on sm_90.
Triton's autotune cache is keyed per device, so on the H100 it re-benchmarks and picks
Hopper-appropriate configs automatically; nothing is locked to Ampere.

**Shared memory heads the right way.** Hopper has *more* shared memory per SM (~228 KB)
than Ampere (~100 KB). Configs (incl. `num_stages=3`, `BLOCK_N=64`, `D≤128`) that fit
the 3090 fit the H100 with room to spare — no risk of exceeding smem on Hopper.

**Two non-blocking notes to verify on first boot (perf, not correctness):**
1. `tl.dot` uses a group padded to `GP=16` (M-dim = 16). Hopper's `wgmma` tensor cores
   prefer M≥64; at M=16 Triton falls back to a smaller MMA path. **Correct, but leaves
   Hopper tensor-core throughput on the table** — a later optimization (tile the group,
   or batch queries) could add Hopper-specific autotune configs.
2. Autotune will spend ~1–2 min compiling on the *first* fused step per shape on a new
   device. Expected; amortized after that.

**What must be verified on the actual H100 (can't test locally):** that all four gates
above pass (compile + numerical parity under sm_90's tf32/ieee `tl.dot` and fp32
atomics). Everything points to yes; `VERIFY_KERNELS=1` confirms it in ~a few minutes.

## Checkpoint / resume (ephemeral-pod survival)

Pods can be killed at any time. `train.py`:

- **Saves** `{model, optimizer, step, config}` to `$CKPT_DIR/ckpt.pt` every
  `CKPT_INTERVAL` steps (and at the end), written **atomically** (`ckpt.pt.tmp` →
  `os.replace`) so a kill mid-save can't corrupt the file.
- **Resumes automatically** on startup if `$CKPT_DIR/ckpt.pt` exists — restores model +
  optimizer + step and continues. A killed pod **resumes, not restarts**.
- Data order is a pure function of `(seed, step)`, so resume replays the same schedule
  with no data-loader state to persist.

Point `$CKPT_DIR` **and** `$DATA_DIR` at mounted network volumes so weights and the
tokenized corpus survive the pod:

```bash
docker run --gpus all \
  -v /netvol/openbox/ckpt:/workspace/checkpoints \
  -v /netvol/openbox/data:/workspace/data \
  openbox-llm:latest
```

Restarting the same command after a kill picks up where it left off.

## Proof-run config (~$50–100 / ~25–50 H100-hrs)

Defaults are sized for a **small ~200M proof run**, not the full 1.5B. Everything is an
env var (or `--flag`):

| knob | env | default | note |
|------|-----|--------:|------|
| model width | `D_MODEL` | 1024 | ~**198M params** at the defaults |
| depth | `N_LAYERS` | 12 | |
| GQA heads | `N_Q_HEADS` / `N_KV_HEADS` | 16 / 4 | |
| context | `SEQ_LEN` | 1024 | |
| NSA blocks | `BLOCK_SIZE` / `N_SELECTED_BLOCKS` / `WINDOW` | 32 / 8 / 256 | |
| attention | `ATTN_IMPL` | `fused` | Triton path (the thing we're proving); `ref` = torch |
| batch | `BATCH_SIZE` | 24 | raise on an 80 GB H100 |
| token budget | `TOKEN_BUDGET` | 2e9 (~2B) | training tokens; `≈81k` steps at defaults |
| unique data | `DATA_TRAIN_TOKENS` | 5e8 (500M) | streamed once to `$DATA_DIR`; budget cycles it |
| checkpoint | `CKPT_INTERVAL` | 1000 | ~81 checkpoints over the run |
| optimizer | `OPTIM` | `adamw` | `adam8bit` (bitsandbytes) available |

Budget math: a ~200M model in the fused path on one H100 lands (very roughly) in the
tens-of-thousands tok/s range, so ~2B tokens ≈ order-of ~10–20 H100-hrs → well within
$50–100. Tune `TOKEN_BUDGET` / `BATCH_SIZE` once you see real tok/s from the logs.

Example override for a shorter/cheaper first run:

```bash
docker run --gpus all -v /netvol/ckpt:/workspace/checkpoints -v /netvol/data:/workspace/data \
  -e TOKEN_BUDGET=500000000 -e BATCH_SIZE=32 -e ATTN_IMPL=fused \
  openbox-llm:latest
```

Data is streamed from FineWeb-Edu on first boot into `$DATA_DIR` (fast on cloud
network; one-time). Put `$DATA_DIR` on a volume so re-runs skip re-tokenizing.
