# Open-Box LLM — Rung 1: Native Sparse Attention (pure PyTorch)

A minimal, **pure-PyTorch** implementation of **Native Sparse Attention** (NSA,
DeepSeek — [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)) inside a tiny
~24M-param decoder-only transformer.

This is **rung 1** of the [open-box LLM ladder](./openbox-llm-objectives.md). It
answers exactly one question:

> **Does backward flow correctly through the NSA gate and all three branches?**

Success = the model overfits a tiny char-level corpus to near-zero loss, while a
per-step grad-norm table proves every branch received live gradient.

No Triton, no CUDA, no fused kernels, no optimization passes. Correct, not fast —
tiny matmuls are fine at this scale. Dependencies: `torch` + `numpy` only.

## Files

| file | what |
|------|------|
| `nsa_model.py`  | NSA attention (3 branches + gate, GQA) and the tiny transformer. Heavy comments on the branch logic. |
| `train_rung1.py`| Overfit loop over an embedded few-KB corpus. Prints loss + per-branch grad norms; asserts every branch has gradient. |

## How to run

```bash
# torch + numpy already in .venv (see requirements.txt)
python train_rung1.py                 # defaults: 24.4M params, 800 steps
python train_rung1.py --device cpu    # force CPU
python train_rung1.py --steps 400 --log_every 50
```

Runs on a single GPU (RTX 3090) or CPU. The default config is ~24.4M params
(`d_model=512, n_layers=8`, GQA `8` query heads / `2` kv heads).

## What success looks like

1. **Loss collapses toward zero.** Starts ~3.8 (≈ ln 44, uniform over the
   vocab), ends < 0.05 within 800 steps.
2. **Every branch shows non-zero gradient, every step.** After `loss.backward()`
   the loop asserts `compression`, `selection`, `window`, `gate`, and the shared
   `q_proj` all have non-None, non-zero grads, and prints:

   ```
   step  800 | loss 0.0385
     per-branch grad norms:
       compression      grad_norm=1.8238e-03 params_with_grad=64/64 *required
       gate             grad_norm=4.7893e-02 params_with_grad=16/16 *required
       q_proj (shared)  grad_norm=4.7772e-02 params_with_grad=8/8  *required
       selection        grad_norm=6.2920e-02 params_with_grad=16/16 *required
       window           grad_norm=9.4193e-02 params_with_grad=16/16 *required
       ...
   ```
   If any required branch had zero/None gradient, the run **hard-fails** with an
   `AssertionError` — you cannot silently pass rung 1.
3. **It recites the corpus.** A greedy sample from `"Native Sparse Attention"`
   echoes the training text back, confirming the whole stack learned to copy.

## How NSA works here (and the two traps)

For every query, attention is computed three ways and blended by a **learned,
per-head sigmoid gate**:

- **Compression** — squash non-overlapping blocks of `block_size` tokens into one
  coarse *summary* token each (learnable pooling with an intra-block positional
  embedding). Cheap global view.
- **Selection** — pick the **top-k fine-grained blocks** and attend to them at
  full resolution. Block importance is read **off the compression attention**.
- **Sliding window** — attend to the most recent `window` tokens. Local detail.

`out = g_cmp·o_cmp + g_slc·o_slc + g_win·o_win` (per head). GQA: many query heads
share each key/value head; the selection block-choice is shared per group.

**Trap 1 — the top-k pick is non-differentiable.** We `detach()` the importance
scores before `torch.topk`, so the chosen block *indices carry no gradient*. We
never backprop through the hard index pick. Gradient still reaches selection
honestly: (a) through the **gate** and the real attention over the selected
tokens (trains selection's k/v projections), and (b) through the **compression
scores** that rank the blocks (trains the compression branch). See the long
comment block at the top of `nsa_model.py`.

**Trap 2 — you must be able to SEE the gradient.** Hence the hard assertion +
grad-norm table every step (`train_rung1.py`).

### Rung-1 simplifications (deliberate, documented)

- Compression and selection share **one aligned block grid** (non-overlapping,
  stride == block size). This makes "compression score of block *j*" == "selection
  importance of block *j*" with no mapping formula. NSA's general form allows
  overlapping compression blocks; that's a later rung.
- Each branch has its **own k/v projections** (the query projection is shared) so
  every branch is a cleanly nameable parameter group in the grad table.
- Selection is implemented as full-score attention with a keep-mask rather than
  gathering/packing blocks — identical math, far easier to read and verify.
- Positional scheme is plain learned embeddings. RoPE + YaRN is rung 2+.

## Not in scope for rung 1

Speed/kernels, long context, the trainable memory layer, quantization/offload —
all later rungs. See `openbox-llm-objectives.md`.
