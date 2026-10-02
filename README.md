# segresearch

Where should a byte-level model start its patches? This repo began as an autonomous research loop in the style of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and grew into a set of experiments on that question, run on an Apple Silicon Mac (MLX) and on Meta's trained BLT-1B (PyTorch).

## Main finding

Byte-level models such as [BLT](https://arxiv.org/abs/2412.09871) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high.

**Under tight patch budgets, entropy-triggered patching skips positions whose type is predictable but whose value must be computed** (e.g. the number after `=` in a worked math solution). A patch start there matters for accuracy, and a label-free rule can find those positions.

| Evidence | Result |
|---|---|
| BLT-1B, patches changed at inference, 15% of bytes | computed results exact: entropy 34% vs a boundary after each `= ` 59% vs label-free entropy + dependence 72% (default layout at ~28%: 73%) |
| Small BLT-like models trained at a 10% budget (3 seeds) | computed / final-answer exact: entropy 7% / 10%, label-free dependence 17% / 67%, hand-written result boundaries 25% / 73% |
| Same, 10-20% budgets | hand-written > dependence > entropy at every budget; dependence at 10% beats entropy at 20% on final answers |
| Scratchpad Patching (reimplemented) | its entropy trigger is no better than random for answers (6.0% vs 5.8%); answer-start scratchpads 40% |
| Copies and lookups (code identifiers, proof steps) | small effect (1-7 points): the value is not computed |
| Program traces with operands inside the local window | no effect; with an 8-byte window, suggestive but not confirmed |

What is known (BLT, SpaceByte, H-Net, Scratchpad Patching) and what looks new is in `literature.md`. The full lab log, including negative results and corrections, is `notes.md`. Open work is tracked in the GitHub issues and milestones.

## Layout

| Path | What it is |
|---|---|
| `program.md` | Instructions for the research agent, including the experiment standards learned the hard way |
| `notes.md` | Lab notebook: every experiment, result and correction, in order |
| `literature.md`, `refs.bib` | Verified related work |
| `results/logs/` | Raw logs of every run cited in `notes.md` (index in `results/logs/README.md`) |
| `results/registry/` | One JSON record per result: `backfill.jsonl` (rebuilt from the logs by `build_registry.py`) and `<script>.jsonl` (appended live by every script through `registry.py`) |
| `results/tables.md`, `results/figures/` | Headline tables and figures, rebuilt by `make_tables.py` and `make_figures.py` |
| `prepare.py`, `harness.py`, `baselines.py`, `boundary.py`, `run.py`, `test_final.py` | The original synthetic-corpus loop: the agent edits `boundary.py` only |
| `test_causal.py` | Checks that every rule is causal and never reads labels |
| `realtext.py` | Real-text checks on GSM8K and code with the MLX model |
| `mathexp.py` | GSM8K + MATH training experiments: patch rules, budgets, scratchpads, greedy exact-match accuracy |
| `deptrigger.py` | Fits the label-free boundary-dependence table from the model's own losses |
| `gaintrigger.py`, `learned_chunking.py` | Earlier learned triggers / chunkers (weak; kept for the record) |
| `reasonexp.py` | Logic and program-trace tasks as a training corpus |
| `flops.py` | Compute per byte by patch rate (harness and BLT-1B) |
| `realblt.py`, `realblt_budget.py`, `realblt_code.py`, `realblt_reason.py`, `blt_layouts.py`, `blt_screens/` | BLT-1B experiments (separate PyTorch environment) |

## Reproducing

MLX experiments (Apple Silicon):

```bash
uv sync
uv run prepare.py                     # synthetic corpora for the original loop
uv run mathexp.py prepare             # GSM8K + MATH corpus and its entropy model
export SEGR_D=128 SEGR_GLAYERS=4 SEGR_POOL=xattn SEGR_LOCAL=window SEGR_MATH_NACC=1000
uv run deptrigger.py 16000 4096 marginal          # label-free dependence table
uv run mathexp.py run dep10 0 32000               # one training run: rule, seed, steps
uv run mathexp.py rates                           # patch rate and answer coverage of every rule
uv run flops.py
uv run build_registry.py && uv run make_tables.py && uv run make_figures.py   # tables and figures from the registry
```

Rules for `mathexp.py run` include `entropy`, `jump`, `words+syntax`, `entropyR`, `depR`, `syntax+entropyR` (R = percent of bytes), and the Scratchpad Patching variants `sp16:*` (with `SEGR_SCRATCH=1`). A run takes about an hour at D=128.

BLT-1B experiments need PyTorch and `transformers` with BLT support, and the `itazap/blt-1b-hf` weights (about 15 GB):

```bash
<env>/bin/python realblt_budget.py 300      # tight-budget layouts on GSM8K
<env>/bin/python realblt_code.py 300 300    # Python code
<env>/bin/python realblt_reason.py trace 300 300
```

The original agent loop: start Claude Code here and prompt "Read program.md and let's set up a new experiment run." Run `uv run test_final.py` yourself on a frozen rule.

## Caveats

Models trained here are small (up to 128 dimensions, 4 global layers). Most accuracy is teacher-forced exact match of each target given the true prefix. BLT-1B was tested with inference-time layout changes, not retrained at tight budgets. See `notes.md` and the issues for what is still unconfirmed.
