# Paper 2, version 2: results, scripts and data

This folder is the record behind version 2 of paper 2 (arXiv:2610.22100), for the results that are not already
elsewhere in segresearch. Version 2 is a corrected and completed record of closed work; it is not aimed at any venue.
It is part of segresearch release v2.1.0 (doi:10.5281/zenodo.24023934).

Most files here were produced between 2026-10-22 and 2026-11-01 in the author's private working repository, which held
segresearch as a submodule pinned to commit 1985984 (the parent of this release). They are copied unchanged, except for
the path edits and redactions listed under [Changes made when copying](#changes-made-when-copying). Files marked "New"
were written for this release; they compute numbers from shipped or public files only, on a CPU, without a model.

Cited in place, not copied (already public in segresearch):
- `results/scale/runs/*/result.json`: every model trained from scratch (1M, 12M, 50M): arm, seed, device, clipping,
  teacher-forced exact match, bits per byte, evaluation patch rate, end-to-end GSM8K and MATH `\boxed{}` accuracy;
  `results/scale/cross/`: two same-seed T4 re-runs of the 12M table runs;
- `results/enrichment.json`: coverage of computed-result and final-answer starts at the 50M thresholds;
- `results/hnet/` with `scale/cloud/hnet_check.py`: the check of the public H-Net checkpoints (Appendix G);
- `paper2/tables/` (version 1's tables), `paper/` (paper 1), `prereg/2026-10-13_scale150_bltft.md` (Study A's
  preregistration, amendments A-1 to A-3), `patches/transformers_pr49188_blt_window.diff`.

**Section labels** follow version 2 of the paper: §3 Setup, §4 Training from scratch, §5 Why entropy skips cued values
(and what the table detects), §6 Budget and objective at inference; Appendix A the label-free detector, B copies, C
outside math and no general rule, D other negative results, E the R13 pilot, F work that was not run (with Study A's
interim look), G the H-Net
check, H where each rule spends its budget (the selection analysis); "Code and data" is the paper's data statement.
Every BLT-1B number is teacher-forced and measured at inference on a model trained under entropy patching.

**Codes** in file names and below: R1 Study A (145M, preregistered; stopped after an interim look); R2 the
hand-written rule at 50M (not run); R3 templated cue-free text and R4 the BLT-1B cue controls (both in
`realblt_cues.py`); R5 compressors as patch rules; R6 the selection analysis; R8 the budget by objective at inference;
R8b the hybrid at inference; R8c the hybrid trained from scratch (not run); D a 12M training budget curve (not run);
R9 the prose rewrite; R10 and R12 the patcher-state detector (marker labels, auto-labels); R11 gain-trained probes
across domains; R13 the detector as a training rule (one-seed pilot only); R14 what the detector reads; R15 "a new word
or number" across domains; R16 training the small patcher; R17 the signal's shape; R18 second-digit starts; R19 the
forced share; R20 a matched share on fresh problems; R21 a second prose writer (not run); R22 the cue ceiling; X1-X4
the six-domain test; O6, O6b and O6c the copy tests (dev, confirmatory, cutoff rival); P1 and P2 the quadrant test's
predictions.

## Layout

```
paper2/v2/
  README.md      this file
  run.py         runner: python paper2/v2/run.py SCRIPT [ARGS]
  scripts/       the scripts (38 copied unchanged, 4 with path edits, the R13 overlay, 6 new) and the public _seg.py
  analysis/2026-10-30_copyfix/bytes/final_rules.py    kept at its original path (realblt_copy_latch.py loads it there)
  results/       result files under their original names (blt_cue_budget.json gzipped); new outputs: v2_*.json,
                 gsm8k_final_answer_copy.json
  data/          the prose rewrites (R9, R20) and the texts of the two copy tests (O6b, O6c)
  rules/         decision rules filed before each run, under their original names
  checks/        three independent check reports (copy tests)
  figures/       the two figures drawn by scripts/fig_*.py
```

Paths in the private repository: `results/` and `data/` keep their paths. In `rules/`, the R2, R8, R8b, R8c, R9, R13
and Study A files come from `paper2v2/`, R10-R22 and `XDOMAIN_RULES.md` from `paper3/`, and the O6, O6b and O6c files
from `followup/`; `QUADRANT_P1_P2_RULES.md` is an extract of `general_rule.md`. In `scripts/`, `intervals.py`,
`fig_*.py`, `prose_rewrite.py` and `pod_patch/` come from `paper2v2/`, `measure_table_copy.py` from
`analysis/2026-10-31_p2copy/table_check/`, and the rest from the repository's root. Each check report names its
source in its header; the figures come from `paper2v2/figures/`.

## How to run

```
python paper2/v2/run.py v2_training_numbers      # from anywhere; the script name without .py
```

`run.py` puts `scripts/` and the segresearch root on `sys.path` and writes no bytecode. Ten scripts find `results/`
from their own location; the runner gives them a `__file__` in this folder (the four that sat in the private
repository's `paper2v2/` folder get one in `paper2/v2/paper2v2/`). Scripts that import `_seg` also run directly.
`realblt_prose.py` reads the rewrites from `$R9_DATA` (set it to `paper2/v2/data/prose_rewrites`).

- **CPU scripts** need Python 3.11 or later with numpy (and matplotlib for the figures). The new scripts were run with
  Python 3.11.5 and numpy 2.4.6. The CPU scripts that can run from this package alone: `v2_training_numbers`,
  `v2_blt1b_numbers`, `v2_outside_math`, `gsm8k_final_answer_copy` (it reads the GSM8K test file), `check_copy_texts`, `r9_cue_split`,
  `copy_latch_paired_odds`, `fig_tradeoff`, `fig_detector_budget`, `plot_profiles`, and the
  `training()` and `layouts()` parts of `intervals.py`. Re-run for this release: `r9_cue_split` rewrote its output
  byte for byte; `copy_latch_paired_odds` printed its shipped log exactly; `intervals.training()` and
  `intervals.layouts()` reproduced every pair in `results/paper2v2_intervals.json` exactly from segresearch's
  `results/scale/runs` and the shipped `blt_cues.json`; the three figure scripts ran.
- **BLT-1B scripts** used the environment that O6b and O6c record: torch 2.14.1 and transformers 5.18.0 with
  `patches/transformers_pr49188_blt_window.diff` applied (it restores BLT-1B's 512-byte sliding windows; an upstream
  fix that is not merged), numpy 2.5.3, bf16 on an Apple laptop (MPS). The model is `itazap/blt-1b-hf`, a community
  Hugging Face conversion of `facebook/blt-1b`, at snapshot 91aa6b8e168046ad91e517d2191e1e974f50cc01 (weights blob
  sha256 fdc3e6f1571c7c81a2d6f1b3c36b0deabd56509c466d4735aef72a9517436f94). `blt_load` refuses an unpatched library
  unless `SEGR_BLT_WINDOW=0`, which loads the conversion as released (used by `blt_copy_gate_w0`, O6b's J7 and O6c).
  No model was run for this release.

## Results

| File | What it is | Paper 2 v2 | Private commit | How to re-run |
|---|---|---|---|---|
| `results/blt_autolabel.json` | R12: the probe trained on auto-labels (a number not seen earlier in the text) | App. A | b2a4aba | BLT-1B: `run.py realblt_autolabel` |
| `results/blt_compress.json` | Compressors and tables as patch rules on BLT-1B (document LZ, PPM-C, LZ plus the table, value uncertainty, the solution and corpus tables) | App. D, §4 | 45d413b | BLT-1B: `run.py realblt_compress`; the corpus-table rows need the training-corpus bundle |
| `results/blt_copy_gate.json` | O6 step 1 (dev, exploratory): starts at and inside the copy cause BLT-1B's failed ticket copies | App. B (design history) | 097ff1d | BLT-1B: `run.py realblt_copy_gate` |
| `results/blt_copy_gate_w0.json` | O6 step 1 under the released conversion (window 0); dev rows used by O6c's predictions | App. B (design history) | 2bf8da0 | BLT-1B: `SEGR_BLT_WINDOW=0 python paper2/v2/run.py realblt_copy_gate` |
| `results/blt_copy_latch_cutoff_blocks.json` (redacted) | O6c C0: the new texts' GSM8K indices, IDs and text hashes | App. B | a075f86 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_cutoff_score.json` | O6c C1: five layouts under the released conversion on 600 test texts | App. B | c5ce090 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_cutoff_selftest.json` | O6c self-test (no model) | App. B | 8c5fdf0 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_cutoff_verdict.json` | O6c verdict: the class, its reading and the rows beside it | App. B | 3d03679 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_devcheck.json` | O6b dev recount with the confirmatory code (before filing) | App. B (design history) | d84d097 | no model, but needs the dev patcher outputs (not shipped); written by `realblt_copy_latch_fresh.py devcheck` at d84d097, before the filed code |
| `results/blt_copy_latch_fresh_blocks.json` (redacted) | O6b J0: the fresh texts' GSM8K indices, IDs and text hashes | App. B | a2d8c95 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j1.json` | O6b J1: the patcher on every fresh text; implementation checks, R1, R2, C4 and K2 | App. B | 717ca5e | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j2.json` | O6b J2: the main cell (F1) at the default rate and at 10% | App. B | 8bd0b96 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j3.json` | O6b J3: costs on the same problems without the ticket (F6) | App. B | ceba4dc | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j4.json` | O6b J4: other wording of the copy line (F2a, F2b) | App. B | 1f30951 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j5.json` | O6b J5: other ID kinds (F3: 6-digit, letters, mixed, doses) | App. B | fe67ebc | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j6.json` | O6b J6: distant copies (F4) | App. B | 609199d | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_j7.json` | O6b J7: the released conversion, window 0 (F5) | App. B | b60f319 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_selftest.json` | O6b self-test: text builders and counters on dev problems (no model) | App. B | 5dd8d76 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_fresh_verdict.json` | O6b verdict: every bar, kill rule, claim and paired verdict, with the readings quoted whole | App. B | af15140 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `results/blt_copy_latch_step1.json` | O6b dev step 1: patcher outputs and checks of L8 on the dev texts | App. B (design history) | 83d48e3 | BLT-1B: `run.py realblt_copy_latch step1` |
| `results/blt_copy_latch_step2.json` | O6b dev step 2: L8 on the 293 dev test texts | App. B (design history) | 4bd2a7a | BLT-1B: `run.py realblt_copy_latch step2` |
| `results/blt_copy_latch_step3.json` | O6b dev step 3 | App. B (design history) | 4bd2a7a | BLT-1B: `run.py realblt_copy_latch step3` |
| `results/blt_copy_veto.json` | O6 step 2 (dev, exploratory): a suffix-match veto | App. B (design history) | 2bf8da0 | BLT-1B: `run.py realblt_copy_veto` |
| `results/blt_cue_budget.json.gz` (gzipped) | R22: the patcher's entropy after type-fixing cues against the budget's threshold, the budget per value, and the detector's costs (915 fresh problems); `gunzip -k` gives the file the run wrote (sha256 1726a63f57eabf2c7c6bf40adbd3455ba2e6e7002996e90377a7848d206d7512, 13.0 MB) | §5, App. A | 3de575a | BLT-1B: `run.py realblt_cue_budget` |
| `results/blt_cues.json` | BLT-1B at 10% and 15% budgets: entropy, the `=` rule, the syntax rule, the novelty table and SpaceByte-style starts on 294 stripped GSM8K test problems; templated cue-free problems | §5 | f470b35 | BLT-1B: `run.py realblt_cues` |
| `results/blt_forced_share.json` | R19: how many starts the detector forces; the fused ranking | App. A | 5b7c0ee | BLT-1B: `run.py realblt_forced_share` |
| `results/blt_hybrid.json` | R8b: entropy plus the table's top q of positions, at inference | §6 | 468cf50 | BLT-1B: `run.py realblt_hybrid` |
| `results/blt_learned.json` | R10: probes on BLT-1B's patcher state (and on Qwen2.5-1.5B's) trained on marker labels | App. A | fd29fc3 | BLT-1B: `run.py realblt_learned` (plus Qwen2.5-1.5B) |
| `results/blt_matched_share.json` | R20: a forced share matched to the share of numbers, on 915 fresh problems | App. A | fe8a7c9 | BLT-1B: `run.py realblt_matched_share` |
| `results/blt_probe_reads.json` | R14: what the label-free probe reads within numbers | App. A | a8f9b40 | BLT-1B: `run.py realblt_probe_reads` |
| `results/blt_profiles.json` | R17: the patcher's entropy and the probe's score around numbers (descriptive) | App. A | 6d94595 | BLT-1B: `run.py realblt_profiles` |
| `results/blt_profiles.png` | R17's figure | App. A | 6d94595 | CPU: `run.py plot_profiles` |
| `results/blt_prose.json` | R9: the same GSM8K problems with the solutions rewritten as prose (221 test problems, 711 computed values) | §5 | 152c2df | BLT-1B: `R9_DATA=paper2/v2/data/prose_rewrites python paper2/v2/run.py realblt_prose` |
| `results/blt_prose_cue_split.json` | R9's rows split by the cue before each value (found after the data) | §5 | 0979b0f | CPU: `run.py r9_cue_split` (byte-identical, checked) |
| `results/blt_quadrant.json` | What a patch start buys, by kind of content, within the same texts (BLT-1B, 10%, 293 problems with a ticket written twice) | §6 (Table 4), App. B | e66006e | BLT-1B: `run.py realblt_quadrant` |
| `results/blt_quadrant_verdicts.json` | P1 and P2 by the rules in rules/QUADRANT_P1_P2_RULES.md | §6 | f470b35 | CPU plus Qwen2.5-1.5B: `run.py quadrant_analysis` |
| `results/blt_second_digit.json` | R18: patch starts at a number's second digit | App. A | d3e071e | BLT-1B: `run.py realblt_second_digit` |
| `results/blt_table_copy_onsets.json` (redacted) | Bytes-only check: does the novelty table start patches at copy onsets (293 dev texts; after the data) | App. B | 5376f76 | needs the dev patcher outputs (not shipped) and the BLT tokenizer: `run.py measure_table_copy` |
| `results/blt_waste.json` | R8: the value of each patch start (A), the patch rate needed for equal answers (B) and bits per byte against rate (C) at inference | §6 | 21211ee | BLT-1B: `run.py realblt_waste` |
| `results/gsm8k_final_answer_copy.json` | New: how often the final answer repeats the last number before '####', with the distance in bytes | §3, §4 | new | CPU: `run.py gsm8k_final_answer_copy` (reads the GSM8K test file) |
| `results/paper2v2_intervals.json` | Intervals for the revised text: seed-level Welch over training runs (1M-50M), paired problem bootstrap of the BLT-1B layouts at 10% and 15% (2,000 resamples), the selection analysis (400 resamples) and extras | §5, §4, App. D, App. H | 24b336f | CPU for `training()` and `layouts()` (reproduced exactly from public files); `mechanism()` and `extras()` need inputs not shipped |
| `results/patcher_budget_by_kind.json` | Where BLT-1B's patcher entropy puts a 10% budget, by kind of position, and its 90th-percentile threshold (3.309 bits) | §5 | a5d0a34 | CPU: `run.py budget_by_kind`, from reducible2's per-byte cache (not shipped) |
| `results/patcher_variants.json` | R16: training variants of the paper's one-layer patcher | App. A | 9d9c349 | needs the 1.9 GB training corpus (not shipped): `run.py patcher_variants` |
| `results/prose_rewrite_cost.json` (redacted) | Cost, counts and validation failures of the prose rewrites | Code and data | 878c963 | record only (re-running spends API credit) |
| `results/r13_pilot/pod.log` | The pilot's pipeline log | App. E | 0b15721 | written by the pilot run |
| `results/r13_pilot/probe.json` | The pilot's probe: normalization and logistic weights, fitted on the training corpus with auto-labels | App. E | 0b15721 | written by the pilot run |
| `results/r13_pilot/result.json` | R13's one-seed pilot: 12M, probetop3+entropy10, trained on MPS (exploratory) | App. E | 0b15721 | not from this package: the 1.9 GB corpus, scale/ with the overlay in scripts/pod_patch/, about 7 h on MPS |
| `results/r13_pilot/rules.json` | The pilot's rule settings (thresholds, rates) | App. E | 0b15721 | written by the pilot run |
| `results/r13_pilot/train.log` | The pilot's training log | App. E | 0b15721 | written by the pilot run |
| `results/reducible_stripped.json` | What entropy's and the table's top 10% of positions select on 1,292 stripped GSM8K test problems: new-content share, BLT-1B surprisal, an independent LZ measure | App. H | b35ced7 | BLT-1B: `run.py reducible2` with the training-corpus bundle |
| `results/v2_outside_math.json` | New: the 12M runs outside math (prose, code, traces; 2 seeds, T4): arm means and SDs, and novelty minus entropy with seed-level Welch intervals (bits per byte, exact match) | App. C | new | CPU: `run.py v2_outside_math` (reads segresearch's `results/generic/`) |
| `results/v2_blt1b_numbers.json` | New: bits per byte of the `=` rule, the table and the hybrid against entropy (whole text, question, solution); where zero-value starts fall; the bracket of R8's rate ratio; LZ plus the table against the table; R9's prose test split by the cue before each value, for entropy and the table (after the data) | §1, §5, §6, App. D | new | CPU: `run.py v2_blt1b_numbers` |
| `results/v2_training_numbers.json` | New: per-run devices and floors; seed-level Welch pairs (same recipe, across recipes, clipping); T4 re-runs; the R13 pilot beside the 12M seeds; checked against paper2v2_intervals.json | §3, §4, App. D, App. E | new | CPU: `run.py v2_training_numbers` |
| `results/xdomain.json` | X1-X4: where an extra patch start pays in six domains, and whether any rule predicts it | App. C | 843dbd6 | CPU from the per-document gains (not shipped): `run.py xdomain_analysis`; the gains come from `run.py realblt_xdomain` on BLT-1B |
| `results/xdomain_newtok.json` | R15: a label-free probe for 'a new word or number starts here', across domains | App. C | df61e2d | CPU from R11's state cache: `run.py xdomain_newtok` |
| `results/xdomain_probe.json` | R11: patcher-state probes trained on measured gains, within and across domains | App. C | b2a4aba | BLT-1B: `run.py xdomain_probe` (builds a 682 MB state cache, then CPU) |

## Logs

Each log was written by the run that produced the listed output.

| File | Log of | Paper 2 v2 | Private commit | How to re-run |
|---|---|---|---|---|
| `results/blt_autolabel.log` | blt_autolabel.json | App. A | b2a4aba | with that run |
| `results/blt_copy_gate.log` (redacted) | blt_copy_gate.json | App. B (design history) | 097ff1d | with that run |
| `results/blt_copy_gate_w0.log` (redacted) | blt_copy_gate_w0.json | App. B (design history) | 2bf8da0 | with that run |
| `results/blt_copy_latch_cutoff_predict.log` | O6c predictions (copy_latch_cutoff_predict.py) | App. B | 8c5fdf0 | with that run |
| `results/blt_copy_latch_cutoff_run.log` (redacted) | O6c jobs C0, C1, verdict | App. B | 1b1e452 | with that run |
| `results/blt_copy_latch_cutoff_verdict_test.log` | O6c branch test (copy_latch_cutoff_test.py) | App. B | 8c5fdf0 | with that run |
| `results/blt_copy_latch_devcheck.log` (redacted) | blt_copy_latch_devcheck.json | App. B (design history) | d84d097 | with that run |
| `results/blt_copy_latch_fresh_paired_odds.log` | O6b paired-verdict odds (copy_latch_paired_odds.py) | App. B | 18dc460 | with that run |
| `results/blt_copy_latch_fresh_recount_check.log` | dev recount (copy_latch_recount_check.py) | App. B | 15ab486 | with that run |
| `results/blt_copy_latch_fresh_run.log` (redacted) | O6b jobs J0-J7, verdict | App. B | b6d1100 | with that run |
| `results/blt_copy_latch_fresh_verdict_test.log` | O6b branch test (copy_latch_verdict_test.py) | App. B | 2e9a1d4 | with that run |
| `results/blt_copy_latch_step1.log` (redacted) | blt_copy_latch_step1.json | App. B (design history) | 83d48e3 | with that run |
| `results/blt_copy_latch_step2.log` (redacted) | blt_copy_latch_step2.json | App. B (design history) | 4bd2a7a | with that run |
| `results/blt_copy_latch_step3.log` (redacted) | blt_copy_latch_step3.json | App. B (design history) | 4bd2a7a | with that run |
| `results/blt_copy_veto.log` (redacted) | blt_copy_veto.json | App. B (design history) | 2bf8da0 | with that run |
| `results/blt_cue_budget.log` | blt_cue_budget.json | §5, App. A | 3de575a | with that run |
| `results/blt_forced_share.log` | blt_forced_share.json | App. A | 5b7c0ee | with that run |
| `results/blt_learned.log` | blt_learned.json | App. A | fd29fc3 | with that run |
| `results/blt_matched_share.log` | blt_matched_share.json | App. A | fe8a7c9 | with that run |
| `results/blt_probe_reads.log` | blt_probe_reads.json | App. A | a8f9b40 | with that run |
| `results/blt_profiles.log` | blt_profiles.json | App. A | 6d94595 | with that run |
| `results/blt_prose.log` | blt_prose.json | §5 | 152c2df | with that run |
| `results/blt_second_digit.log` | blt_second_digit.json | App. A | d3e071e | with that run |
| `results/blt_table_copy_onsets.log` | blt_table_copy_onsets.json | App. B | 5376f76 | with that run |
| `results/log_blt_compress.txt` | blt_compress.json | App. D, §4 | 45d413b | with that run |
| `results/log_reducible_stripped.txt` | reducible_stripped.json | App. H | 45d413b | with that run |
| `results/patcher_variants.log` | patcher_variants.json | App. A | 9d9c349 | with that run |
| `results/prose_rewrite_fresh.log` | R20's rewrite batch (prose_rewrite.py fresh) | Code and data | 878c963 | with that run |
| `results/xdomain_analysis.log` | xdomain.json | App. C | 843dbd6 | with that run |
| `results/xdomain_newtok.log` | xdomain_newtok.json | App. C | df61e2d | with that run |
| `results/xdomain_probe.log` | xdomain_probe.json | App. C | b2a4aba | with that run |
| `results/xdomain_run.log` | xdomain_gains (realblt_xdomain) | App. C | 843dbd6 | with that run |

## Data

| File | What it is | Paper 2 v2 | Private commit | How to re-run |
|---|---|---|---|---|
| `data/copy_latch_texts/o6b_fresh.jsonl` | New: the O6b texts (4,200; groups F1-F4, F6), extracted from the stored patcher outputs | App. B | new | check: `run.py check_copy_texts` |
| `data/copy_latch_texts/o6c_cutoff.jsonl` | New: the O6c texts (900), extracted likewise | App. B | new | check: `run.py check_copy_texts` |
| `data/prose_rewrites/test.jsonl` | R9's prose rewrites, the first 300 GSM8K test problems (275 valid) | §5, App. A | 152c2df | record (see the data card) |
| `data/prose_rewrites/train.jsonl` | R9's prose rewrites, GSM8K train problems (1,382 valid) | §5, App. A | 152c2df | record (see the data card) |
| `data/prose_rewrites_fresh/test.jsonl` | R20's prose rewrites, GSM8K test problems 300-1,318 (915 valid) | §5, App. A | 878c963 | record (see the data card) |

## Figures and checks

| File | What it is | Paper 2 v2 | Private commit | How to re-run |
|---|---|---|---|---|
| `checks/O6b_INDEPENDENT_CHECK.md` | Independent check of O6b's verdict against two blind derivations | App. B | 0d36323 | report |
| `checks/O6c_INDEPENDENT_CHECK.md` | Independent check of O6c's run and verdict; the source of the result's qualifications | App. B | 28e0e85 | report |
| `checks/TABLE_COPY_ONSETS_CHECK.md` | Report of the bytes-only table check | App. B | 5376f76 | report |
| `figures/fig_detector_budget.png` | The detector's gain and final-answer cost against the forced share | App. A | 3885ccd | CPU: `run.py fig_detector_budget` |
| `figures/fig_tradeoff.png` | Answers against bits per byte on BLT-1B, by rule and budget | §6 | 12b2030 | CPU: `run.py fig_tradeoff` |

## Scripts

"Private commit" is the last commit that touched the script in the private repository. "BLT-1B" means the script
loads the model (see [How to run](#how-to-run)).

| File | What it does | Paper 2 v2 | Private commit | Needs |
|---|---|---|---|---|
| `scripts/_seg.py` | New: public path shim (replaces the private one) | all | new | imported by the scripts |
| `scripts/blt_light.py` | Helper: reads BLT-1B weights from the local Hugging Face cache | App. B | 097ff1d | imported |
| `scripts/budget_by_kind.py` | Budget by kind of position | §5 | a5d0a34 | CPU |
| `scripts/check_copy_texts.py` | New: checks data/copy_latch_texts/ against the block files | App. B | new | CPU |
| `scripts/copy_latch_cutoff_predict.py` | O6c's predictions from dev rows and O6b's J7 | App. B | 3875d32 | needs the O6b patcher outputs (not shipped) |
| `scripts/copy_latch_cutoff_test.py` | Branch test of O6c's verdict code | App. B | 3875d32 | BLT environment (imports torch; no model) |
| `scripts/copy_latch_paired_odds.py` | O6b paired verdict's odds, from dev outputs | App. B | 56b2d00 | CPU (output equals the shipped log, checked) |
| `scripts/copy_latch_recount_check.py` | Recount of the dev check with the filed code | App. B | 15ab486 | needs the dev patcher outputs |
| `scripts/copy_latch_verdict_test.py` | Branch test of O6b's verdict code | App. B | 4026bec | needs the O6b smoke outputs (not shipped) |
| `scripts/extract_copy_texts.py` | New: how data/copy_latch_texts/ was made (record) | App. B | new | needs the patcher outputs (not shipped) |
| `scripts/fig_detector_budget.py` | Figure: the detector against the forced share (edited: output path) | App. A | 3885ccd | CPU |
| `scripts/fig_tradeoff.py` | Figure: answers against bits per byte (edited: output path) | §6 | 12b2030 | CPU |
| `scripts/gsm8k_final_answer_copy.py` | New: final answers as copies of the last result | §3, §4 | new | CPU |
| `scripts/intervals.py` | Intervals for the revised text (edited: RUN_DIRS) | §5, §4, App. D, App. H | b59294d | see results/paper2v2_intervals.json |
| `scripts/measure_table_copy.py` | Bytes-only table check at copy onsets (edited: root path) | App. B | 5376f76 | needs the dev patcher outputs; writes next to itself |
| `scripts/patcher_variants.py` | R16 | App. A | 950410d | training corpus |
| `scripts/plot_profiles.py` | R17 figure | App. A | 6d94595 | CPU |
| `scripts/prose_rewrite.py` | R9 and R20 prose rewrites through the Anthropic Message Batches API (prompt and validation inside) | §5, App. A, Code and data | 513e14c | record only (spends API credit) |
| `scripts/quadrant_analysis.py` | P1/P2 verdicts | §6 | e66006e | CPU plus Qwen2.5-1.5B |
| `scripts/r9_cue_split.py` | R9 cue split | §5 | 0979b0f | CPU |
| `scripts/realblt_autolabel.py` | R12 | App. A | 4c99fa1 | BLT-1B |
| `scripts/realblt_compress.py` | R5 compressors as patch rules | App. D | bff90ce | BLT-1B |
| `scripts/realblt_copy_gate.py` | O6 step 1 (dev) | App. B | 097ff1d | BLT-1B |
| `scripts/realblt_copy_latch.py` | L8 and the O6b dev steps | App. B | 83d48e3 | BLT-1B |
| `scripts/realblt_copy_latch_cutoff.py` | O6c confirmatory code, filed version (3875d32) | App. B | 3875d32 | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `scripts/realblt_copy_latch_fresh.py` | O6b confirmatory code, filed version (4026bec) | App. B | 4026bec | record only: the code checks the private repository's git history and refuses to run elsewhere |
| `scripts/realblt_copy_veto.py` | O6 step 2 (dev) | App. B | 097ff1d | BLT-1B |
| `scripts/realblt_cue_budget.py` | R22 | §5, App. A | 2944f63 | BLT-1B |
| `scripts/realblt_cues.py` | R3/R4 cue controls | §5 | e66006e | BLT-1B |
| `scripts/realblt_forced_share.py` | R19 | App. A | 5266327 | BLT-1B |
| `scripts/realblt_hybrid.py` | R8b | §6 | e77e11f | BLT-1B |
| `scripts/realblt_learned.py` | R10 | App. A | 3509b9d | BLT-1B, Qwen2.5-1.5B |
| `scripts/realblt_matched_share.py` | R20 | App. A | 927a230 | BLT-1B |
| `scripts/realblt_probe_reads.py` | R14 | App. A | 6f565fe | BLT-1B |
| `scripts/realblt_profiles.py` | R17 | App. A | 6d94595 | BLT-1B |
| `scripts/realblt_prose.py` | R9 | §5 | 88e03f0 | BLT-1B |
| `scripts/realblt_quadrant.py` | What a patch start buys, by kind (also builds the ticket texts) | §6, App. B | f506c94 | BLT-1B |
| `scripts/realblt_second_digit.py` | R18 | App. A | 5ea79c2 | BLT-1B |
| `scripts/realblt_waste.py` | R8 | §6 | 7bc557a | BLT-1B |
| `scripts/realblt_xdomain.py` | Per-document gains of an extra start | App. C | 6d09bca | BLT-1B |
| `scripts/reducible2.py` | Selection analysis on stripped GSM8K | App. H | b35ced7 | BLT-1B, corpus bundle |
| `scripts/stripped_units.py` | Helper: GSM8K text and unit kinds in the stripped format | App. H | 60bd551 | imported |
| `scripts/v2_blt1b_numbers.py` | New: BLT-1B numbers from shipped results | §1, §5, §6, App. D | new | CPU |
| `scripts/v2_outside_math.py` | New: outside-math numbers from public run files | App. C | new | CPU |
| `scripts/v2_training_numbers.py` | New: training-side numbers from public run files | §3, §4, App. D, App. E | new | CPU |
| `scripts/xdomain_analysis.py` | X1-X4 | App. C | 6d09bca | CPU |
| `scripts/xdomain_data.py` | Cross-domain documents (Wikipedia, MedQA, Python stdlib, MATH algebra, GSM8K, traces) | App. C | 6d09bca | CPU, downloads |
| `scripts/xdomain_newtok.py` | R15 | App. C | f8f7be0 | CPU |
| `scripts/xdomain_probe.py` | R11 | App. C | 4c99fa1 | BLT-1B, then CPU |
| `scripts/pod_patch/scale/prep.py` | R13 overlay on scale/prep.py: the probe and table hybrid rules (record) | App. E, App. F | f7c58d7 | not on sys.path |
| `scripts/pod_patch/scale/rules.py` | R13 overlay on scale/rules.py (record) | App. E, App. F | f7c58d7 | not on sys.path |
| `scripts/pod_patch/scale/train.py` | Divergence guard for the planned rental runs, added after the pilot (record) | App. F | b59294d | not on sys.path |
| `analysis/2026-10-30_copyfix/bytes/final_rules.py` | SIV, PO and V8 byte rules, loaded by realblt_copy_latch.py from this path | App. B | 4b76f51 | imported |

## Rules

Each test's decision rules were committed before its data were scored ("Filed at"); later changes are dated notes or
amendments inside the file. "Private commit" is the file's last commit.

| File | Test | Status | Filed at | Private commit | Paper 2 v2 |
|---|---|---|---|---|---|
| `rules/R8_WASTE_RULES.md` | R8: value of each start; budget for equal answers; bits per byte | run | cef6bcf | cef6bcf | §6 |
| `rules/R8b_HYBRID_D_RULES.md` | R8b: the hybrid at inference (H, run); the 12M training budget curve (D, not run) | H run; D not run | d28ec52 | b78b2ce | §6, App. F |
| `rules/R8c_HYBRID_TRAIN_RULES.md` | R8c: the hybrid trained from scratch at 12M | not run | 8283aa5 | b78b2ce | App. F |
| `rules/R2_DECISION_RULE.md` | R2: the hand-written rule at 50M, clipped, 3 seeds | not run | 42d53fe | b59294d | §4, App. F |
| `rules/R9_PROSE_RULES.md` | R9: the same problems as prose | run | 96816e8 | 152c2df | §5 |
| `rules/R13_PROBE_TRAIN_RULES.md` | R13: the probe hybrid in training (3 seeds); the one-seed local pilot | pilot run; 3 seeds not run | f7c58d7 | b59294d | App. E, App. F |
| `rules/STUDY_A_INTERIM_RULES.md` (redacted) | Study A (145M): amendment A-5 (interim look at seed 0), its outcome, diagnosis, A-6 | stopped after the interim look | 23d20a2 | 006fd6b | §4, App. F |
| `rules/R10_LEARNED_RULES.md` | R10: probes trained on marker labels | run | 3509b9d | 3509b9d | App. A |
| `rules/R11_PROBE_XDOMAIN_RULES.md` | R11: gain-trained probes across domains | run | 4c99fa1 | 4c99fa1 | App. C |
| `rules/R12_AUTOLABEL_RULES.md` | R12: the label-free probe | run | 4c99fa1 | 4c99fa1 | App. A |
| `rules/R14_PROBE_READS_RULES.md` | R14: what the probe reads | run | 1b366ae | a8f9b40 | App. A |
| `rules/R15_NEWTOKEN_XDOMAIN_RULES.md` (redacted) | R15: 'a new word or number' across domains | run | f3d3251 | df61e2d | App. C |
| `rules/R16_PATCHER_TRAINING_RULES.md` | R16: training variants of the small patcher | run | c798a1b | c22f913 | App. A |
| `rules/R17_SIGNAL_SHAPE_NOTE.md` | R17: descriptive note (no verdict) | run | baa0b0e | 6d94595 | App. A |
| `rules/R18_SECOND_DIGIT_RULES.md` | R18: second-digit starts | run | b5ab73a | d3e071e | App. A |
| `rules/R19_FORCED_SHARE_RULES.md` | R19: the forced share | run | 1cb5199 | 5b7c0ee | App. A |
| `rules/R20_MATCHED_SHARE_RULES.md` (redacted) | R20: a matched share on fresh problems | run | 83816e5 | fe8a7c9 | App. A |
| `rules/R22_CUE_BUDGET_RULES.md` (redacted) | R22: the cue ceiling and the detector's costs | run | d9b3b5b | 3de575a | §5, App. A |
| `rules/XDOMAIN_RULES.md` | X1-X4: six domains | run | 6d09bca | 6d09bca | App. C |
| `rules/QUADRANT_P1_P2_RULES.md` | P1-P2 of the quadrant test (extract) | run | 36886d3 | 36886d3 | §6 |
| `rules/O6_COPY_VETO_RULES.md` | O6: the copy hazard gate and a suffix-match veto (dev) | run (exploratory) | 4c7808d | 2bf8da0 | App. B |
| `rules/O6b_COPY_LATCH_RULES.md` (redacted) | O6b: L8 on dev texts, then the confirmatory rule | run | 267de39 (dev part); 7e9c6fa (confirmatory part) | 7e9c6fa | App. B |
| `rules/O6c_CUTOFF_RULES.md` (redacted) | O6c: L8 against a byte rule with a cutoff at token 620 | run | 48618dc | 48618dc | App. B |

`rules/STUDY_A_INTERIM_RULES.md` is the only Study A record in this folder: amendment A-5 (a stop-only interim look at
seed 0, written before any Study A evaluation was opened), its outcome, the diagnosis and amendment A-6. Its text said
seed 0's values would not be reported before all three seeds were in; the study was stopped on 2026-11-01, before
seeds 1 and 2 finished. No Study A output is shipped, nor the diagnostic it cites (`paper2v2/diag_repetition.py`,
`results/studyA_diag_repetition.json`).

## Inputs that are not shipped

| Input | Needed by | How to get it |
|---|---|---|
| GSM8K train and test, `~/.cache/segresearch-gsm8k/{train,test}.jsonl` (sha256 17f347dc51477c50… and 3730d312f6e34405…) | almost every script | `realtext._load` downloads them from openai/grade-school-math (MIT) |
| The GSM8K-fitted novelty table, `~/.cache/segresearch-blt/novtab_gsm.npz` (sha256 62d69c9e3add7422ca651bc1a6dab626e0ed133f8859adc1f2d4256d030f5897) | realblt_cues, realblt_waste, realblt_hybrid, realblt_compress, reducible2, measure_table_copy | `realblt_combo.novelty_table()` fits it on all 7,473 GSM8K train solutions in the stripped format, on a CPU (it imports torch; no model) |
| BLT-1B weights | every BLT-1B script | `itazap/blt-1b-hf` on Hugging Face, at the snapshot above |
| Qwen2.5-1.5B | realblt_learned (L3), quadrant_analysis (P2) | `Qwen/Qwen2.5-1.5B` on Hugging Face; the revision used was not recorded |
| The 1.9 GB training corpus and its bundle (`~/.cache/segresearch-scale/`) | realblt_compress (corpus-table rows), reducible2, patcher_variants, the R13 pilot | `python -m scale.prep` (checks the corpus SHA-256) |
| reducible2's per-byte cache, `~/.cache/segresearch-scale/ncm_math/reducible2_cols.npz` | budget_by_kind, `intervals.mechanism()`, `intervals.extras()` | `run.py reducible2` |
| Cross-domain documents, `~/.cache/segresearch-xdomain/` | realblt_xdomain | `run.py xdomain_data` (English Wikipedia 20231101.en, MedQA-USMLE-4-options, the Python 3.11 standard library, MATH algebra, GSM8K, synthetic traces) |
| Per-document gains, `results/xdomain_gains_{domain}.npz` (7.7 MB; they hold the documents' text) | xdomain_analysis, xdomain_probe, xdomain_newtok | `run.py realblt_xdomain` |
| Patcher-state cache, `~/.cache/segresearch-xdomain/patcher_states.npz` (682 MB) | xdomain_probe, xdomain_newtok | built by `run.py xdomain_probe` |
| Copy-test patcher outputs, `results/blt_copy_latch_fresh_patcher.npz` (16.8 MB, sha256 d6610ef78a20e4d5…) and `results/blt_copy_latch_cutoff_patcher.npz` (3.5 MB, sha256 daa978d1c10ab79e…) | copy_latch_cutoff_predict, extract_copy_texts | written by the filed jobs; their texts are in `data/copy_latch_texts/` |
| Dev patcher outputs, `results/blt_copy_latch_patcher.npz` (7.2 MB, sha256 a2a4d6befd8b9e1b…; it also holds the 660 cross-domain documents) | measure_table_copy, copy_latch_recount_check | `run.py realblt_copy_latch step1` |
| Smoke outputs (`*_smoke.json`, `*_smoke.npz`) | copy_latch_verdict_test, measure_table_copy | code checks on a few dev problems, not results |

## Changes made when copying

1. **`scripts/_seg.py` is new.** The private repository's `_seg.py` (sha256 563fdcaac8927ab2…) set
   `HERE = os.path.dirname(os.path.abspath(__file__))` and put `SEG = os.path.join(HERE, "segresearch")` on `sys.path`.
   The public one sets `HERE` to `paper2/v2` and `SEG` to the segresearch root; `emit` is unchanged.
2. **Path edits in four scripts**, nothing else: `intervals.py` reads training runs from segresearch's
   `results/scale` only (the private copy also read local caches of rental and Kaggle outputs);
   `fig_tradeoff.py` and `fig_detector_budget.py` write to `figures/`; `measure_table_copy.py` takes the package root
   instead of a hard-coded absolute path. It still writes `results_table_copy.json` and `.log` next to itself; the
   shipped copies are those files renamed to `results/blt_table_copy_onsets.{json,log}`.
3. **Redactions**: absolute local paths in logs and in three result files; two Anthropic Message Batches IDs; the
   private repository's name; mentions of a session's temporary folder; the names of a private Kaggle dataset and a
   scheduled task; and the wording, option labels and texts, tool names and times of the private messages in which the PI (the author)
   asked for, chose among or approved options. Those messages are replaced by plain statements of what the PI chose and
   when (the author's decision, 2026-11-02). Identifiers removed from the rules files are marked `[redacted: ...]`;
   elsewhere a local path became a relative one, a batch ID `msgbatch_[redacted]`, and the repository's name "the
   private working repository". Decisions and their dates are kept.
4. **Headers** added to the three check reports, and `rules/QUADRANT_P1_P2_RULES.md` is an extract of a longer file
   (lines 119-136 at 3de575a, the file's last commit; lines 66-83 at 36886d3, where the rules were fixed; the section is
   identical at both), with a header.
5. **Compression.** `results/blt_cue_budget.json`, 13.0 MB as written by its run, ships gzipped
   (`results/blt_cue_budget.json.gz`, `gzip -n -9`); it decompresses to the run's file byte for byte (sha256
   1726a63f57eabf2c7c6bf40adbd3455ba2e6e7002996e90377a7848d206d7512). No other file is compressed.
6. **What the redactions keep.** The PI's decisions and their dates are kept, stated plainly; the wording of the PI's
   messages, the option labels and texts, the tool names and the chat times are not shipped, in the rules files, the
   check reports or the logs. One comment in `scripts/realblt_copy_latch_fresh.py` still names an option label ("Split,
   label honestly (Recommended)"): changing it would break the code sha256 that O6b's outputs record. The copy-test rules
   files and the check reports keep the process roles of the work ("main session", "O6 session", the agents'
   tool-call inputs).

| File | Change | Replacements | Original sha256 | Shipped sha256 |
|---|---|---|---|---|
| `scripts/intervals.py` | read training runs from the public results/scale only (the private copy also read local caches) | 1 | `c7da6253162ff609` | `a505d5abc30c9eef` |
| `scripts/fig_tradeoff.py` | write the figure to paper2/v2/figures/ (the private copy wrote paper2v2/figures/) | 1 | `d2e8e9ed5ce244f7` | `223f35f9810c1bb3` |
| `scripts/fig_detector_budget.py` | write the figure to paper2/v2/figures/ (the private copy wrote paper2v2/figures/) | 1 | `bba1be47edaa7c4e` | `ca225bd499cad333` |
| `scripts/measure_table_copy.py` | package root instead of a hard-coded absolute path | 1 | `75853510158ec433` | `0a80ecf5f60649e5` |
| `checks/O6c_INDEPENDENT_CHECK.md` | a header saying what the file is; mentions of a session's temporary folder; the PI's approvals stated plainly, without their wording or times | 4 | `4e2023deb7bc502c` | `cde0549eb2815275` |
| `checks/O6b_INDEPENDENT_CHECK.md` | a header saying what the file is | 0 | `88c54a303b0785f5` | `d039dc0192c541ab` |
| `checks/TABLE_COPY_ONSETS_CHECK.md` | a header saying what the file is; private repository name | 1 | `ec5e8646dfe84f6c` | `7d34d246b4ff2a6f` |
| `scripts/pod_patch/scale/train.py` | private repository name in a docstring | 1 | `de829bfb02187c21` | `82241afa5aa8f664` |
| `scripts/pod_patch/scale/rules.py` | private repository name in a docstring | 1 | `6e51745233d4745c` | `77a2324445d5013d` |
| `scripts/pod_patch/scale/prep.py` | private repository name in a docstring | 1 | `9de4bfc1679a030b` | `422580757552ef25` |
| `results/blt_copy_gate.log` | absolute local paths -> paths relative to the private repository root | 1 | `f2cbac3b39593d31` | `25affe6cb9c030c1` |
| `results/blt_copy_gate_w0.log` | absolute local paths -> paths relative to the private repository root | 1 | `6478ce8f5f43f711` | `39f52332bed9138d` |
| `results/blt_copy_latch_cutoff_run.log` | absolute local paths -> paths relative to the private repository root; the PI's go-ahead stated plainly | 4 | `81c35f8e8baa0bb9` | `6f0574f6eb1e5efe` |
| `results/blt_copy_latch_devcheck.log` | absolute local paths -> paths relative to the private repository root | 1 | `a8e1e05d38aad89b` | `2a5b278f3ff89d8a` |
| `results/blt_copy_latch_fresh_run.log` | absolute local paths -> paths relative to the private repository root | 9 | `93675a7f0b01a1a2` | `05238ee44c23bb38` |
| `results/blt_copy_latch_step1.log` | absolute local paths -> paths relative to the private repository root | 1 | `1e5d5fe925defcdd` | `9994ddd086aad3f4` |
| `results/blt_copy_latch_step2.log` | absolute local paths -> paths relative to the private repository root | 1 | `58d0beef66897707` | `bd8477c7eb8f4e36` |
| `results/blt_copy_latch_step3.log` | absolute local paths -> paths relative to the private repository root | 1 | `34e26a0ade914e5c` | `8ca3abcd7de463db` |
| `results/blt_copy_veto.log` | absolute local paths -> paths relative to the private repository root | 1 | `c849b6f92db476f9` | `d5cbb81e9cd6e45d` |
| `results/blt_copy_latch_fresh_blocks.json` | absolute local path of the GSM8K train file (pool_file.path) | 1 | `dc2e91881f82f9ff` | `e3d69235ebe5a3c1` |
| `results/blt_copy_latch_cutoff_blocks.json` | absolute local path of the GSM8K train file (pool_file.path) | 1 | `0cdece6f9fd73de2` | `86ef4be8caa5d807` |
| `results/blt_table_copy_onsets.json` | absolute local paths in inputs_sha256 keys (the script that ran was a temporary copy, identical to the committed one) | 6 | `1cef151e3494cc81` | `1f73f78d83f72df7` |
| `results/prose_rewrite_cost.json` | Anthropic Message Batches IDs (account identifiers) | 2 | `37a4d3c158cb9893` | `840838b6fcab758a` |
| `rules/R15_NEWTOKEN_XDOMAIN_RULES.md` | the PI's request stated plainly | 1 | `e20120c6a1b39447` | `9e29671c757f6684` |
| `rules/R20_MATCHED_SHARE_RULES.md` | Anthropic Message Batches ID | 1 | `09537b84755e5be5` | `608c05337eb46918` |
| `rules/R22_CUE_BUDGET_RULES.md` | the PI's approval stated plainly | 1 | `13395ca61a4f3a46` | `0003e335705a95c3` |
| `rules/STUDY_A_INTERIM_RULES.md` | the PI's confirmation stated plainly; names of a private Kaggle dataset and a scheduled task | 3 | `4a5a5455b560d02f` | `87d3c36a248dc062` |
| `rules/O6b_COPY_LATCH_RULES.md` | the PI's choices stated plainly, without the private exchange's wording, option labels and texts, tool name or time | 17 | `e81336680cd44f40` | `f954f56e12cf5ad4` |
| `rules/O6c_CUTOFF_RULES.md` | the PI's requests, choices and approval stated plainly, without the private exchanges' wording, button labels or times | 10 | `558e78342ed66bb7` | `b834ef7499fca58a` |

Every other copied file is byte-identical to its source at the private commit listed. The O6b and O6c outputs record
the sha256 of their code (`code.files_sha256`) and of their rules file (`code.rules_sha256`). The code files they list
are shipped unchanged (segresearch's own modules are those of this release), except `_seg.py`. The rules files are
redacted copies, so they do not match the recorded hashes (e81336680cd44f40… for O6b, 558e78342ed66bb7… for O6c); the
originals are unchanged in the private repository since their filing commits, 7e9c6fa and 48618dc.

## Provenance limits

- The commits listed are of the private working repository, whose history is not public. Statements that a rule was
  written before its data can be checked only against that history; each rules file gives its dates.
- Issue numbers in the scripts and rules (written #N) refer to that repository's tracker, not to segresearch's issues
  and pull requests.
- The segresearch commits cited in `rules/STUDY_A_INTERIM_RULES.md` (962400b, e60510d, aaef0b0, 236d6dc, 504305f) are on
  an unpublished local branch. Amendments A-4 to A-6 of Study A's preregistration are not in this release's `prereg/`
  file, which stops at A-3; A-5 and A-6 are recorded in `rules/STUDY_A_INTERIM_RULES.md`.
- Two scripts changed after the output they produced. `intervals.py` (output 24b336f, script b59294d) gained run-folder
  and label arguments, a guard for non-finite bits per byte and a per-run list of patch rates; its training and layout
  pairs are unchanged (checked above). `prose_rewrite.py` (output 878c963, script 513e14c) changed only how retries
  record API errors. The R13 overlay's `train.py` was added after the pilot (b59294d) as a divergence guard for the
  planned rental runs; the pilot ran segresearch's `scale/train.py` with the `prep.py` and `rules.py` overlays.
- Only O6b and O6c record BLT-1B's snapshot; R22 records the library versions, the window and the device. The other
  BLT-1B outputs record none of these; they loaded the model through `blt_load`, which refuses to run without the window
  fix unless `SEGR_BLT_WINDOW=0`.
- The O6b and O6c confirmatory jobs (`blocks`, `j1`-`j7`, `score`, `verdict`) check their filing commit in the private
  repository's history and refuse to run inside segresearch. Their outputs are the record.

## Data card: prose rewrites (`data/prose_rewrites/`, `data/prose_rewrites_fresh/`)

- **Source.** GSM8K (openai/grade-school-math, MIT): for R9 the first 1,500 train problems and the first 300 test
  problems; for R20 test problems 300-1,318.
- **Writer.** Claude Opus 5.5 (model ID `claude-opus-5-5`, effort "low", JSON output), through Anthropic's Message
  Batches API, run by the author with `scripts/prose_rewrite.py`, which holds the system prompt. R9 on 2026-10-26 (a
  20-problem pilot by standard requests, then a batch, then one retry of the failures by standard requests); R20 on
  2026-10-29 (one batch; its 4 retries failed with an API error). Every record has `served_by: claude-opus-5-5`.
  One writer only: a check with a second writer (R21) was planned and not run.
- **Task.** Rewrite each solution as natural explanatory prose with no `=` and no arithmetic symbol between numbers;
  keep every computed result of the original (the number after `= `) in order, with the same digits, and mark it
  `[[...]]` at its first statement; a `$` before money is allowed; plain ASCII. The markers are removed before
  scoring. Because `$` was allowed, values after `$` keep a type-fixing cue in the prose (R22).
- **What was checked** (automatically, before any BLT-1B run; `rules/R9_PROSE_RULES.md`): the marked values equal the
  original's computed values in order (commas and `$` ignored); no `=`; no operator between two numbers; ASCII; both
  texts (original and prose) at most 999 bytes, so they fit the BLT-1B scripts' 1,000-token limit. A rewrite that failed was retried once, then
  dropped from both formats. Four pilot rewrites were read by hand. Nothing checks that the prose explains the
  solution correctly beyond the kept values.
- **Counts.** R9: 1,800 requested, 1,657 valid (1,382 train, 275 test; 221 of the test problems have computed values,
  711 values in all). R20: 1,019 requested, 915 valid (2,448 computed values). Most failures were too long for the
  token limit (R9's batch: 159 too long, 5 with different marked values, 2 non-ASCII; R20: 100 too long, 4 with
  different marked values). Counts and failures by problem: `results/prose_rewrite_cost.json`.
- **Cost.** R9 $7.57 (pilot $0.14, batch $5.94, retry $1.48; `rules/R9_PROSE_RULES.md` gives $7.56, the sum of the
  rounded parts); R20 $3.46.
- **Fields.** `split`, `idx` (the problem's line in GSM8K's train or test file), `question`, `solution` and `answer`
  (GSM8K's), `values` (the computed results), `orig` and `prose` (the two scored texts: question, solution and a final
  line "The final answer is N"), `orig_spans` and `prose_spans` (character offsets of each value in those texts),
  `final_start_orig` and `final_start_prose` (character offset of the final answer), `marked` (the model's output with
  its markers), `served_by`.
- **Licence.** Released with the repository (MIT); the source problems are GSM8K's (MIT).

## Data card: copy-test texts (`data/copy_latch_texts/`)

- **What.** The texts BLT-1B scored in the two copy tests, built by the filed code (`realblt_copy_latch_fresh.py`,
  `realblt_quadrant.build`) from GSM8K train problems (MIT): a synthetic ticket is opened at the start of the solution
  (`Ticket 5287.`) and closed before the final-answer line (`Ticket 5287 closed.`), so the ID is copied verbatim.
  `o6b_fresh.jsonl`: 4,200 texts from train problems 1,800-6,252, in groups F1 (the main template), F2a and F2b (other
  wording of the copy line), F3 (6-digit, letter, mixed and dose IDs), F4 (distant copies), each with fit and test
  texts, and F6 (F1's problems without a ticket). `o6c_cutoff.jsonl`: 900 F1 texts from train problems 6,253-7,176
  (300 fit, 600 test); one O6c test text repeats an O6b problem with a renamed person.
- **How made.** `scripts/extract_copy_texts.py` read them from the stored patcher outputs (not shipped; sha256 in
  [Inputs](#inputs-that-are-not-shipped)), keeping the stored metadata: `group`, `idx`, `id`, `kind`, `distance`,
  `n_bytes`, `sha256`, `units` (`[start, end)` byte offsets of each kind of unit: computed values, the new ID, its
  copy, copied numbers, new and repeated words) and `text`.
- **Check.** `run.py check_copy_texts`: every text has the byte count and sha256 that the block files
  (`results/blt_copy_latch_fresh_blocks.json`, `results/blt_copy_latch_cutoff_blocks.json`) list for its group and
  index, every listed text is present, and each ID sits at its stored spans (passes).

## Work that was not run

Designed with decision rules and not run (no results exist): Study B, BLT-1B fully fine-tuned under the table at a 10%
budget (`prereg/2026-10-13_scale150_bltft.md`); R2, the hand-written rule at 50M with the table's recipe
(`rules/R2_DECISION_RULE.md`); D, a 12M training budget curve with a same-device A40 anchor for the table
(`rules/R8b_HYBRID_D_RULES.md`, part D); R8c, the label-free hybrid trained from scratch (`rules/R8c_HYBRID_TRAIN_RULES.md`);
R13's three rented seeds (`rules/R13_PROBE_TRAIN_RULES.md`; only the one-seed pilot ran); Study A beyond its interim look
(`prereg/2026-10-13_scale150_bltft.md`, `rules/STUDY_A_INTERIM_RULES.md`). Planned with no decision rules written: R21, the
detector on prose from a second writer; copy checks on BLT-7B and on H-Net at a controlled budget.

## Licences

Code and results: MIT, as the rest of segresearch (`LICENSE`). GSM8K: MIT. The prose rewrites and the copy-test texts
derive from GSM8K. Not shipped, only referenced: BLT-1B's weights (Meta's licence for `facebook/blt-1b`; the
conversion `itazap/blt-1b-hf`), Qwen2.5-1.5B (Apache-2.0), and the cross-domain documents (Wikipedia CC BY-SA 4.0,
MedQA CC BY 4.0, the Python standard library under the PSF licence).
