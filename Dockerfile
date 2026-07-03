# openbox-llm — reproducible training image for rented cloud GPUs (H100 / sm_90).
#
# BASE: official PyTorch image pinned to EXACTLY our stack — torch 2.12.1 + CUDA 13.0
# + cuDNN 9 (matches Miu's `torch==2.12.1`, cu130). Rationale:
#   * Official & reproducible; torch/triton/CUDA already integrated and tested.
#   * `-devel` (not `-runtime`): ships nvcc + ptxas, which Triton needs to JIT-compile
#     the NSA kernels at runtime. On a fresh H100 the kernels re-autotune and compile
#     for sm_90 on first use — that requires the CUDA toolchain in the image.
#   * CUDA 13 is multi-arch: the SAME image runs on sm_86 (validated on a 3090) and
#     sm_90 (H100). Nothing here is arch-locked — see the kernel audit in the README.
# Verify the tag: docker manifest inspect pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # cloud pods are dedicated (not a shared desktop) — allow more threads than Miu.
    TORCH_NUM_THREADS=8 \
    # default mount points; override with -e / --env-file to point at network volumes.
    CKPT_DIR=/workspace/checkpoints \
    DATA_DIR=/workspace/data

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps: fully pinned (== everywhere) so the cloud image matches Miu exactly.
# torch / triton / nvidia-* are already present in the base at these versions; pip
# reconciles them and installs the extras (tiktoken, datasets, matplotlib, bitsandbytes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# project code (see .dockerignore — no venv/git/data/artifacts copied)
COPY . .

# ephemeral pods: these should be MOUNTED network volumes so weights/data survive a
# killed pod. train.py checkpoints to $CKPT_DIR and resumes from it on startup.
RUN mkdir -p "$CKPT_DIR" "$DATA_DIR"
VOLUME ["/workspace/checkpoints", "/workspace/data"]

# Configurable entirely via env vars (see train.py CONFIG block) or CLI args.
#   first cloud boot — validate the kernels on sm_90 BEFORE spending on training:
#     docker run --gpus all -e VERIFY_KERNELS=1 <img>
#   real proof run (resumes automatically if $CKPT_DIR has a checkpoint):
#     docker run --gpus all -v /vol/ckpt:/workspace/checkpoints \
#                           -v /vol/data:/workspace/data <img>
ENTRYPOINT ["python", "train.py"]
