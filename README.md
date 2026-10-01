# segresearch

An autonomous research loop, in the style of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), for one question:

> **Where should a byte-level model start its patches — at the boundaries of independent units (alignment), or where the bytes are hard to predict (allocation)?**

A coding agent (e.g. Claude Code) edits one file, `boundary.py`, runs a fixed experiment, and keeps the change only if the metric improves. Everything else is fixed.

## Background

A pilot on a synthetic byte stream found that:

- **Raw entropy ranks noise above learnable content.** Random ids had the highest entropy; answers to sums had less.
- **Small-model minus strong-model entropy detects learnable bytes almost perfectly** (AUROC 0.998), a Rho-1-style reducible-uncertainty signal.
- **But patching at the learnable bytes made the model worse.** A boundary at an answer separates it from its operands.
- **Patching only at record starts (9% of bytes) beat every 25% rule**, solving all sums, because each computation stayed inside one patch.

In the pilot every dependency fit inside a record. This repo adds **long-range queries** (variables assigned 1–6 records earlier), where the answer must flow through the global model, so alignment and allocation can finally be separated.

## Files

| File | Role | Who edits |
|---|---|---|
| `program.md` | Agent instructions and research context | Human |
| `boundary.py` | The patching rule | **Agent** |
| `notes.md` | The agent's lab notebook: hypotheses, results, mechanisms | **Agent** |
| `prepare.py` | Synthetic corpus (formats A, B, and held-out C), scorer models, cached signals | Fixed |
| `harness.py` | BLT-lite model, budget enforcement, source checks, evaluation | Fixed |
| `baselines.py` | Reference rules: stride, BLT entropy, excess vs. reference, gated trajectory | Fixed |
| `run.py` | One experiment on the dev formats (A, B), validation split | Fixed |
| `test_final.py` | Final evaluation on test splits, including held-out format C | **Human only** |
| `test_causal.py` | Causality test for every rule and model: masks and predictions must not see the byte being predicted | Fixed |

## Quick start

```bash
uv sync
uv run prepare.py        # one-time: builds corpora, trains scorers, caches signals
uv run run.py            # one experiment with the baseline rule
```

Then start Claude Code in this directory and prompt:

> Read program.md and let's set up a new experiment run.

When the loop has run for a while, evaluate the result yourself:

```bash
uv run test_final.py
```

A smoke test of the whole pipeline in under a minute:

```bash
SEGR_SMOKE=1 uv run prepare.py && SEGR_SMOKE=1 uv run run.py
```

## Compute

Everything is written in [MLX](https://github.com/ml-explore/mlx) and runs on the GPU of an Apple Silicon Mac.

- **Apple Silicon (M-series GPU):** the smoke test takes under a minute; a full experiment should take a few minutes instead of the ~12 CPU-minutes the original JAX version needed.
- The port matches the original JAX model numerically (forward passes agree to ~3e-7; same AdamW and warmup-cosine schedule). Random initialization differs, so absolute numbers shift slightly relative to JAX-era runs; compare runs made with the same framework.

`prepare.py` prints each scorer's entropy by role. **The reference model must learn the long-range answers** (well under 0.5 bits); if it warns otherwise, increase `REFERENCE["steps"]` in `prepare.py` before starting the loop.

## Guarding against overfitting the benchmark

- `boundary.py` never sees role labels.
- The harness rejects string/bytes literals, `ord`/`chr`, f-strings and non-numeric imports, so delimiters can't be hardcoded.
- Formats A and B differ in delimiters and keywords; the final test adds format C, which the agent never optimizes on. A rule that secretly relies on surface syntax will fail there.
- Changes smaller than 0.02 bits must replicate across two seeds.

## Caveats

This is a toy: tiny models, a synthetic corpus, a simplified BLT (mean-free sum pooling instead of cross-attention, a single-layer local decoder). It can show which *mechanism* dominates in a controlled setting; it cannot tell you what BLT at scale should do.
