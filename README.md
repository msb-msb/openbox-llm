# openbox-llm

**An open-box language model. Weights, data, training code, and kernels — all published, all rebuildable.**

*Sovereignty over your tools. Understand the machine from top to bottom.* See [MANIFESTO.md](MANIFESTO.md).

---

## What this is

A from-scratch, reproducible LLM built for people who run models on their own hardware and want to take them apart. Not a frontier-model competitor — a **bench you can build on**.

The design is a small reproducible base + a growable memory store + explicit-selection offload, aimed at running a bigger *effective* model on a small GPU.

**Status:** early. The attention foundation (NSA) is built and validated on a single RTX 3090. Memory layers, offload, and the 1.5B training run are next.

## Results so far

Native Sparse Attention (NSA), implemented clean-room from scratch and gated against a reference implementation at every step.

**Flat VRAM scaling** (forward, RTX 3090) — sparse stays ~constant while dense hits the O(T²) wall:

| seq_len | NSA VRAM | dense VRAM |
|--------:|---------:|-----------:|
| 4,096   | 0.03 GB  | 3.60 GB    |
| 8,192   | 0.05 GB  | 14.24 GB   |
| 16,384  | 0.10 GB  | **OOM**    |
| 32,768  | 0.19 GB  | **OOM**    |

**Matched-baseline quality** (~150M params, FineWeb-Edu): NSA val loss 5.44 vs full-attention 5.42 (+0.5%) — competitive quality, with the scaling win above.

**Model-level memory** (fused kernel wired behind a flag): −45% VRAM vs dense at the same config; dense OOMs where the NSA path still fits. Grad-parity verified across all params — the kernel path is a true drop-in, training dynamics unchanged.

Plots + CSVs in `plots/`. Every result has a green correctness gate behind it (see commit history).

## Architecture

- **Small base** — compact, trainable from scratch on modest hardware, understandable end to end.
- **Growable memory** — trainable memory layers with explicit top-k selection; capacity grows by adding memory, not retraining.
- **Selection offload** — because selection is explicit (a lookup, not a prediction), hot/cold GPU/CPU residency is tractable. Goal: bigger effective model on a small GPU.

## The bets (labeled by confidence)

- **NSA** — *proven here.* Foundation, not a bet.
- **Trainable memory layers** — *published research, our bet to validate at scale.*
- **LARQL** — *most speculative for us.* Chris Hay's [chrishayuk/larql](https://github.com/chrishayuk/larql) — query and edit model weights like a graph database. Our bet: use it as the inspection/patch layer for the memory store.
- **SubQ-style subquadratic** — *claims, not facts.* Testing the direction honestly.
- **Hot/cold offload** — *the mechanism that makes the small-GPU goal real.* Explicit top-k makes residency a lookup, not a forecast (cf. PowerInfer). The right engine may be something else — open question.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# rung 1 — overfit tiny corpus, confirm all three NSA branches learn
python train_rung1.py

# rung 2a — matched baseline, NSA vs full attention
python train_rung2a.py --attn nsa      # or --attn full

# kernel correctness gates
python test_kernel.py                  # forward
python test_kernel_backward.py         # backward (gradcheck)
python test_kernel_fused.py            # fused three-branch

# fused kernel path (opt-in; default is ref)
python test_integration.py             # ref vs fused: loss + grad parity gates
python smoke_integration.py            # 150M NSA on the fused path (+ VRAM win)
```

Requires an NVIDIA GPU with Triton (developed on RTX 3090, sm_86; Hopper/sm_90 support in progress for cloud runs).

## Roadmap

1. ✅ NSA — pure-torch → selection fwd/bwd → fusion → wired behind a flag
2. Trainable memory layers (scale to 10B+ slots for capacity-without-compute)
3. Offload engine — hot/cold slot residency
4. 1.5B base training run (compute-optimal ~30B tokens)
5. Paper + build-log series

## Outputs

Open model (weights + data + code + kernels), a paper (including negative results), and a documented build-log on [InsiderLLM](https://insiderllm.com).

## License

Apache-2.0. See [LICENSE](LICENSE).
