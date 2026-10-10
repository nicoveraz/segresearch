# Raw run logs

Logs of the runs cited in `notes.md`, collected from the working directory and the session scratchpad on 2026-10-02. Each MLX training log holds `RESULT` lines from `mathexp.py`, `realtext.py` or `reasonexp.py`; BLT-1B logs hold `RESULT` lines per patch layout.

| Directory | Experiment (section in notes.md) |
|---|---|
| `mlx_math_small/` | Specialized math, small model: entropy, jump, syntax+jump, words+syntax (2 seeds) |
| `mlx_math_new_strategies/` | Per-digit, operand-aligned and matched random controls |
| `mlx_math_5seed_confirm/` | Seeds 2-4 for the new strategies (they did not hold) |
| `mlx_math_larger_model/` | D=128, 4 global layers: entropy, jump, words, stride6+syntax, words+syntax |
| `mlx_scratchpad8/`, `mlx_scratchpad16/` | Scratchpad Patching with 8- and 16-byte patches |
| `mlx_realtext_code/`, `mlx_realtext_more/`, `mlx_realtext_wordstart_fix/` | Real-text checks (GSM8K, code), and reruns after the word-start leak fix |
| `mlx_math_tight10/` | Training at a 10% budget: entropy10, jump10, syntax+entropy10 |
| `mlx_math_dep_trigger/` | deptrigger.py fits (v1 "remove", v2 "marginal"), coverage screens, entdep10 / dep10 runs |
| `mlx_math_budget_sweep/` | 15% and 20% budgets, and third seeds at 10% |
| `mlx_reason_trained/`, `mlx_reason_trained_window8/` | Logic and program traces trained here (window 32 and 8) |
| `blt1b/` | BLT-1B: default layout tests, tight-budget layouts, label-free triggers, code, logic and traces; `realblt26_*` are the #26 reruns (problem vs train-fitted thresholds) |
| `blt1b_reason_truefalse_superseded/` | First logic run with True/False answers (unusable format; superseded) |
| `prepare/` | Corpus preparation and scorer training output |
| `reuse_screen/` | Reuse vs novelty screen (#31): LZ-style match signals and self-supervised novelty predictors, coverage at a 10% budget |
| `misc/` | Original loop runs, held-out test runs, gain-trigger fit, learned chunking |

`../results_tsv_snapshot.tsv` is a copy of the agent loop's `results.tsv` (untracked by design).
