# The Open-Box Manifesto

*Sovereignty over your tools. Understand the machine from top to bottom.*

---

## Why

There was a time you could know your computer completely. The RCA 1802 had an instruction set small enough to hold in your head — you could single-step it and watch each operation flick across the LEDs. Forth sat on top with the same bargain: here are the primitives, build the language you need. The machine was *yours*, top to bottom.

Large language models broke that bargain. They arrive as black boxes — weights you can download but not rebuild, trained on data you can't see, with code and kernels held back. "Open weight" is not open. You get to run the machine. You don't get to understand it, and you can't grow it.

The field needs players who put the whole box on the bench. So we're building one.

## Principles

**Open-box, not open-weight.** Weights, training data, training code, and kernels — all published, all permissive. Not "here's a binary," but "here's how to rebuild it yourself."

**Comprehensible.** Small enough to understand. You should be able to trace what happens and why, the way you could single-step an 1802.

**Forkable.** Built to be taken apart, modified, and grown by anyone. A kit, not a product.

**Honest about limits.** This is not a frontier-model killer. We say what works, what doesn't, and what's still a bet.

## Architecture

A small reproducible base + a growable memory store + explicit-selection offload.

- **Small base** — a compact model you can train from scratch on modest hardware and understand end to end.
- **Growable memory** — trainable memory layers with explicit top-k selection, so capacity grows by *adding memory*, not retraining the base. Knowledge accretes over time.
- **Selection offload** — because selection is explicit (a lookup, not a prediction), hot/cold slot residency across GPU/CPU is tractable. The goal: run a bigger *effective* model on a small GPU.

The stretch goal this points toward: usable local-model quality — think punching toward a much larger model's competence while fitting in 24GB. We state that as the target the architecture is *testing*, not a promise we've delivered.

## The Bets

We're testing specific efficiency ideas in the open, and we label each by how proven it is:

- **NSA (Native Sparse Attention)** — *proven.* Implemented from scratch, gated against reference, flat-VRAM scaling demonstrated on a 3090. Our foundation, not a bet.
- **Trainable memory layers** — *published research, our bet to validate.* Can capacity-without-compute actually hold up at our scale?
- **LARQL** — *most speculative for us.* Chris Hay's system for querying and editing model weights like a graph database ([chrishayuk/larql](https://github.com/chrishayuk/larql)): decompile a model to a queryable index, inspect what it knows, patch knowledge without retraining or GPU. Our bet: use it as the inspection-and-patching layer for the growable memory store.
- **SubQ-style subquadratic claims** — *claims, not facts.* We're inspired by the direction and want to test it honestly — not endorse unverified numbers.
- **Hot/cold offload** — *the mechanism that makes the small-GPU goal real, our bet to prove at scale.* Keep hot weights on the GPU, cold weights in CPU/DB, and swap by need. Prior art (PowerInfer, arXiv 2312.12456) *predicts* which weights fire with an extra model. Our angle: memory-layer top-k makes it *explicit* — which weights are cold this token is a lookup, not a forecast. PowerInfer is one approach; the right offload engine may be something else entirely. That's the open question.
- **Knowledge injection as a RAG alternative** — *most speculative, an open question.* Instead of retrieving document chunks into context, inject a document's knowledge directly into memory (via LARQL) so it's *known*, not fetched — and, unlike RAG, inspectable and editable after the fact. The bet isn't the tooling (PDF → text is trivial); it's whether injected knowledge recalls *faithfully* at document scale and beats a RAG baseline. We've felt the hard part firsthand: ingestion is easy, faithful recall is not.

Proving even some of these, in the open, with reproducible code, is the near-term accomplishment.

## Outputs

- **The model** — weights + data + training code + kernels, permissively licensed.
- **A paper** — the results, including negative ones. Honest findings on the bets.
- **A build-log series** — documented start to finish on InsiderLLM, so others can follow and fork.

## Who It's For

Home-brewers and experimenters. People who want to run models on their own hardware, take them apart, and grow them — the same people who once built a computer from a kit because they wanted to know how it worked.

Not a ChatGPT competitor. A bench you can build on.

---

*This is the first instantiation of a mission, not a finished thing. If it grows, it grows in the open.*
