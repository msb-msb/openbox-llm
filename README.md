# Open-Box LLM — rungs

- **Rung 1** — pure-PyTorch NSA, prove backward flows through the gate + 3 branches. ↓ below.
- **Rung 2a** — matched NSA-vs-full baseline on real data; does NSA track full attention on held-out val loss? See [Rung 2a](#rung-2a--matched-baseline-nsa-vs-full-attention).
- **Rung 2b (smoke test)** — throughput + peak VRAM for the ~1.5B config on one 3090, to pick a token budget. See [Rung 2b](#rung-2b--15b-smoke-test-throughput--vram).

---

# Rung 1: Native Sparse Attention (pure PyTorch)

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

---

# Rung 2a — matched baseline: NSA vs full attention

**Question:** does NSA track full attention on a *generalization* task (held-out
val loss), and is our matched-baseline harness sound?

The **harness is the deliverable.** One codebase, the attention module swapped by a
single flag — `--attn {nsa,full}` — with *everything else held identical*: same
transformer, same tokenized data **and data order** (batches are precomputed from
the seed, so both arms see byte-identical inputs in the same order), same seed,
optimizer, schedule, steps, and eval set. The only thing that differs is the
attention math (`FullAttention` vs `NSAAttention` in `nsa_model.py`, selected in
`make_attention()`).

## Files (added this rung)

| file | what |
|------|------|
| `data_prep.py`   | Stream a FineWeb-Edu sample, tokenize with GPT-2 BPE (tiktoken), cache a fixed train/val split to `data/*.bin` (uint16). Tokenized **once**. |
| `train_rung2a.py`| The matched harness. `--attn {nsa,full}`; logs train+val loss to CSV, writes config JSON + checkpoint. |
| `plot_val.py`    | Overlay both val-loss curves → `plots/val_loss.png`. **This plot is the result.** |
| `run_rung2a.sh`  | data → full → nsa → plot, all under `nice -n 10`. |
| `nsa_model.py`   | +`FullAttention` (matched GQA baseline) and `attn_type` flag. |

## How to run

```bash
source .venv/bin/activate
# one command does everything (data is cached, so reruns skip tokenization):
nice -n 10 ./run_rung2a.sh                      # optional: pass extra flags, e.g. --steps 8000

# or step by step:
nice -n 10 python data_prep.py                  # -> data/train.bin, data/val.bin (once)
nice -n 10 python train_rung2a.py --attn full   # baseline
nice -n 10 python train_rung2a.py --attn nsa    # NSA
nice -n 10 python plot_val.py                   # -> plots/val_loss.png
```

Defaults: ~150M params, `d_model=768, n_layers=16`, GQA `12q/4kv`, `seq_len=512`,
`batch=8`, `5000` steps, bf16 autocast. Both arms train in well under an hour each
on an RTX 3090.

## Matched configs & the param delta

Both arms are byte-for-byte identical except attention. NSA is slightly larger —
its three branches each carry their own k/v projections, plus a block compressor
and the gate. Everything else (embeddings ≈38.6M tied, FFN, LayerNorms, shared
query projection, output projection) is the same.

| config | params | where the delta lives |
|--------|--------|-----------------------|
| `full` | ~140.2M | 1× k/v projection per layer |
| `nsa`  | ~153.4M | 3× k/v projections + compressor + gate per layer |
| **Δ**  | **+13.2M (~9%)** | entirely inside attention; hidden dims otherwise identical |

(Run either arm; the exact count for both is printed and saved to
`runs/{attn}_config.json`.)

## What success looks like

- **NSA's val curve tracks full attention within a small margin** — the paper says
  NSA can match or beat dense; for this rung we only need *tracks, no pathology*
  (no divergence, no blow-up, no stuck-high loss). `plot_val.py` prints the final
  `gap(nsa-full)` and annotates it on the plot.
- Both curves descend smoothly on held-out data (train < 1 epoch, so val is a real
  generalization measurement, not memorization).

## Reproducibility

Seeded (`--seed`, default 1337) across torch/numpy/cuda and the data-window
schedule. `requirements.txt` pins the environment; each run's full config +
param counts + data meta are saved to `runs/{attn}_config.json`; checkpoints to
`runs/{attn}_ckpt.pt`.

## Resource discipline (Miu is a daily-use workstation)

- `torch.set_num_threads(4)` — leaves cores for the desktop.
- Dataloader `num_workers=2, pin_memory=True` (not `os.cpu_count()`).
- Run everything `nice -n 10` (baked into `run_rung2a.sh`).
- VRAM: default `batch=8, seq_len=512` peaks ~14GB (NSA) / ~9GB (full) — well under
  21GB free. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to avoid
  fragmentation OOMs. **If you raise batch/context and near OOM, lower `--batch_size`
  — never let it spill to system RAM/swap.**
- Data is streamed as a small subset and cached to `data/` once, not re-tokenized
  per run.

## Rung-2a scope / caveats

Still pure PyTorch, full-score + keep-mask (correct, not fast — block-gather and
Triton are later rungs). `seq_len=512` is short: NSA's efficiency payoff is a
*long-context* story (64k+) and is explicitly **not** what this rung measures —
here we only check that sparse attention generalizes on par with dense at small
scale, and that the comparison harness is sound.

---

# Rung 2b — 1.5B smoke test (throughput + VRAM)

**Smoke test only — no full run, no plot.** Goal: measure real throughput and peak
VRAM for the ~1.5B config on this 3090 so we can pick a token budget. Hard cap:
peak VRAM **< 18GB** (Xorg already uses ~3GB of the 24GB card).

Run it (`smoke_rung2b.py`, ~60s of real fwd+bwd+opt steps):

```bash
nice -n 10 python smoke_rung2b.py --attn nsa --batch 16 \
    --optim adam8bit --weight_dtype bf16 --grad_ckpt --seconds 60
```

## The 1.5B config (scaled from rung 2a, same-hidden-dims framing)

`d_model=2048, n_layers=30, GQA 16 q-heads / 4 kv-heads, ffn_mult=4, ctx 512`.

| arm | params |
|-----|--------|
| `full` | **1.426B** |
| `nsa`  | **1.556B** (delta = NSA's per-branch k/v + gate + compressor, all in attention) |

## The problem: a fixed ~25GB floor

A 1.5B model under **plain fp32 AdamW** needs ~4× params of memory just for
weights + grads + Adam moments (m, v) = ~25GB. That FIXED cost busts the 24GB card
*before any activations* — so batch size and gradient checkpointing (which only
shrink activation memory) can't fix it. Measured:

| step | config | result |
|------|--------|--------|
| baseline | fp32 AdamW, batch 4 | **OOM** (~24.1GB) |
| ladder 1 | fp32 AdamW, batch **1** | **OOM** (~23.9GB) — batch doesn't help |
| ladder 2 | fp32 AdamW, batch 1, **+ckpt** | **OOM** (~24.1GB) — ckpt doesn't help |
| ladder 3 | **8-bit Adam** + ckpt, batch 1 | runs, but **19.1GB > 18GB cap** |

8-bit Adam cuts the moments (12.4GB → 3.1GB) but fp32 params+grads are still a
12.4GB floor → 19.1GB, over cap. The lever that actually gets under 18GB without
shrinking the model is **bf16 master weights** (params+grads 12.4GB → 6.2GB). That
purity was already gone once we quantized the optimizer, so it's the honest move
for 1.5B on a 24GB card.

## Working config (all four levers) + throughput

**bf16 weights + 8-bit Adam + gradient checkpointing**, ctx 512:

| arm | batch | tok/s | peak VRAM | under 18GB? |
|-----|------:|------:|----------:|:-----------:|
| nsa (1.556B) | 8 | 2400 | 12.05GB | ✅ |
| nsa | **16** | **2730** | **14.70GB** | ✅ **(recommended)** |
| nsa | 24 | 2763 | 17.35GB | ✅ (max throughput, tight) |
| nsa | 32 | — | OOM (~20GB) | ❌ |
| full (1.426B) | 16 | 4216 | 13.91GB | ✅ |

Checkpointing is **required** here (without it, NSA's 30 layers of full-score
attention OOM even in bf16). Throughput plateaus past batch 16 (2730→2763), so
**batch 16 is the pick**: 2730 tok/s at 14.7GB with real headroom. NSA runs ~0.65×
the full arm's throughput — expected, since it computes three branches and pays the
checkpointing recompute (and this is the deliberately-slow full-score path, not
kernels).

## Token-budget implication (the point of the smoke test)

At **~2730 NSA tok/s** on one 3090: ~9.8M tok/hour, ~236M tok/day, ~0.47B over a
weekend. A Chinchilla-optimal 1.5B (~30B tokens) would take **~127 days** — off the
table on a single card. So rung 2b proper should target a **modest budget**
(a few hundred M to ~1B tokens, i.e. hours-to-a-few-days), not compute-optimal
scale. Fast kernels (Triton) and/or multi-GPU are what a real 1.5B run needs — a
later rung.

## Caveats

Numbers are for the pure-PyTorch full-score path (correct, not fast). The
bf16-master + 8-bit-Adam recipe trades numerical headroom for fit; a real run
should watch for instability (loss spikes) and can fall back to fewer layers or a
smaller `d_model` if needed. This rung measured *fit and speed only* — not loss.
