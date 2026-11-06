# Support files for version 2 of paper 1

Version 2 of *Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy
patching misses* (arXiv:2610.11790v2) reports a few results obtained after version 1, where they bound or correct its
claims. This folder holds every file behind a number that version 2 adds and that is not elsewhere in the repository:
results, the scripts that produced them, and the data they need. `tests/test_paper_numbers.py` recomputes the numbers
from these files and checks that the manuscript states them.

Some files come from the author's private working repository, where the work after 2026-10-22 was done; it ran against
this repository at commit 1985984. For each such file the table gives the commit that last changed it there
(`git log -1 --format=%h -- <file>`). Copied files are byte-identical to the originals except where noted: the
docstrings of the copied scripts were rewritten for this copy, because they referred to files and issue numbers of the
private repository (which do not match this repository's), and one line of `r9_cue_split.py` was changed.

## Files

| File | What it is | Numbers in version 2 | Origin |
|---|---|---|---|
| `scripts/gsm8k_final_copy.py` | How often a GSM8K final answer repeats the last number of its solution, and how far back that number ends. Standard library only. | §3: 94.6% of 1,319 test problems, 95.6% of the 294 BLT-1B problems, median 8 bytes; §6: 94.2% of the 660 evaluation problems of the models trained from scratch | new |
| `results/gsm8k_final_copy.json` | Its output (also the first 660 test problems, the evaluation split of the models trained from scratch) | as above | new |
| `scripts/trained_compare.py` | Welch intervals over seeds for the small models trained from scratch, from `results/registry/` with `stats.py`'s formula, for the comparisons that version 1 stated without one | §6: jump against entropy, each rule from 10% to 20%, bits per byte (the hand-written rule against entropy, jump and dependence; dependence against entropy), word starts plus math syntax against entropy at 25%; §7: dependence against entropy plus dependence; §12: per-digit and operand-aligned patches against their random controls, the learned patcher against the table | new |
| `results/trained_compare.json` | Its output | as above | new |
| `scripts/blt1b_numbers.py` | From `results/items/blt_budget_train_w512.json` and the GSM8K test set: the 796 BLT-1B results split by the cue before them (`= $N` or `= N`; an analysis after the fact), and paired comparisons of dependence and entdep with the default layout | §3 and §5: 181 results written `= $N`, entropy at 10% covers 0% of them and 21% of the 615 others; §11: dependence at 15% against the default, +0.5 (-1.7 to +2.7) and -32.7 (-38.1 to -27.2) | new |
| `results/blt1b_numbers.json` | Its output | as above | new |
| `scripts/scaling_unclipped.py` | Rebuilds Table 3 and Figure 2 with `make_tables.scale_table()` and `make_figures.scaling()`, through a loader that keeps only Table 3's runs (see below) | Table 3, Figure 2 | new |
| `scripts/scale_compare.py` | Welch intervals over seeds for the models trained from scratch, keeping the runs of Table 3 (no gradient clipping) apart from the 50M re-runs with gradient clipping (see below) | §6: final-answer gaps -1.6 (-7.8 to +4.5), +10.1 (+1.4 to +18.7), +19.8 (+16.7 to +23.0); computed results +1.9 (-4.5 to +8.2) at 11.7M and +22.0 (+13.6 to +30.4) at 53.7M; bits per byte -0.019; with clipping +18.3, +14.3 (-5.7 to +34.4), +0.005 | new |
| `results/scale_compare.json` | Its output | as above | new |
| `scripts/r9_numbers.py` | The prose-rewrite numbers from `results/blt_prose.json`, with the bootstrap of `realblt_prose.py` (reproduces its stored verdicts exactly), plus the gap between the entropy layout and the layout with a start at every value, and bytes per value | §5: coverage 13.6% and 57.2%; entropy's layout plus a start at every value against entropy's, +37.0 and +12.9; plus against minus those starts, +44.3 and +28.4; default layout 77.9% and 56.8%; 1.43 times the bytes per value; patch rates (the layout with the added starts runs slightly above the budget) | new |
| `results/r9_numbers.json` | Its output | as above | new |
| `scripts/realblt_prose.py` | The prose-rewrite measurement: BLT-1B at a 10% budget on 221 GSM8K test problems in two writings (entropy, the novelty table of arXiv:2610.22100, a start at every value or at none, the default layout) | §5 (through `r9_numbers.py`) | private repository, 88e03f0; changed: docstring |
| `scripts/realblt_cues.py` | Imported by `realblt_prose.py` for its signals (`signals()`); its own experiment, cue controls for the novelty table of arXiv:2610.22100, is not used in this paper | none directly | private repository, e66006e; changed: docstring |
| `scripts/r9_cue_split.py` | Splits the prose-rewrite rows by the cue before each value (an analysis after the fact) | §5: values after `$` uncovered in both writings (0 of 167, 0 of 163); in prose, entropy's layout plus a start at every value gains 36 points after `$` and 6 after a plain space, where entropy covers 74% | private repository, 0979b0f; changed: docstring, and one line, `HERE`, so that it reads and writes `paper/v2/` |
| `scripts/_seg.py` | Path shim for the three copied scripts above: puts the repository root on `sys.path` and points their results at `paper/v2/results/` | none | new; replaces the private repository's module of the same name, which put its pinned checkout of this repository (1985984) on the path |
| `results/blt_prose.json` | Output of `realblt_prose.py`: per value, layout and writing, coverage and exact match, with the stored verdicts | §5 | private repository, 152c2df |
| `results/blt_prose_cue_split.json` | Output of `r9_cue_split.py` | §5 | private repository, 0979b0f |
| `data/prose_rewrites/train.jsonl`, `data/prose_rewrites/test.jsonl` | The LLM rewrites of GSM8K solutions as prose, with the original texts and value offsets; see the data card `data/prose_rewrites/README.md` | §5 | private repository, 152c2df |
| `data/prose_rewrites/README.md` | Data card: source, writer model, prompt, settings, checks, fields, limits | | new |
| `data/prose_rewrites/LICENSE-GSM8K` | GSM8K's MIT licence (Copyright (c) 2021 OpenAI), which covers the GSM8K questions and solutions in the two files above | | new |
| `scripts/cue_budget_subset.py` | Writes `results/blt_cue_budget_entropy.json` from the full results file of a measurement in the private repository (below) | none | new |
| `results/blt_cue_budget_entropy.json` | BLT-1B's entropy patcher at the 2,448 computed values and the final answers of 915 GSM8K test problems (from 300-1,318), in the original writing and with `is ` in place of each `= `: next-byte entropy, probability of a digit, coverage at budgets of 5-25%, thresholds | §5 (through `cue_budget_numbers.py`) | subset of `results/blt_cue_budget.json` in the private repository, 3de575a |
| `scripts/cue_budget_numbers.py` | Recomputes the §5 numbers from that subset with the bootstrap of the original measurement, and checks the summaries stored with it | §1 and §5: threshold 3.36 bits at 10%; P(digit) medians 0.996 and 0.966 (0.736 with `is `); median entropy 2.76 and 3.00 bits; 99th percentile 3.13 bits; coverage 0% and 18.9% at 10%; 90.1% (88.4 to 91.8); 18.9% to 65.7% with `is `; coverage by budget (28% at 12.5%) | new |
| `results/cue_budget_numbers.json` | Its output | as above | new |

## The measurement behind `blt_cue_budget_entropy.json`

The 915 problems are those among GSM8K test problems 300-1,318 that fit BLT-1B's context and also had a valid prose
rewrite, made for another part of the same measurement. BLT-1B (`itazap/blt-1b-hf`) ran with its 512-byte window
restored (`blt_load.py`, transformers 5.18.0, torch 2.14.1, on Apple MPS), at inference only. Thresholds per budget were
fitted on the original text of the 1,382 training problems of `data/prose_rewrites/train.jsonl`, and refitted on the
`is ` copy for that writing. The script that made the measurement is not released: it also runs experiments that are
not public. It was last changed at commit 2944f63 of the private repository, the run recorded commit ef20796, and the
full results file was committed at 3de575a. `cue_budget_subset.py` keeps, of that file, only the rows of the original
and `is ` writings with the columns the paper uses; `cue_budget_numbers.py` recomputes every number from those rows and
checks that the summaries stored with them come out exactly. The cue split was found after the fact, on the rows of
`blt_prose.json` (`r9_cue_split.py`); these 915 problems were measured afterwards to test it.

## Table 3, Figure 2 and the runs with gradient clipping

Since the merge of arXiv:2610.22100's experiments (commit 7ba188e), `results/scale/runs/` also holds 50M re-runs of
entropy10 and dep10 with gradient clipping (`cfg["clip"] = 1.0`, folders ending in `_clip`) and runs of other rules,
some on a Tesla T4 in fp16. Table 3 and Figure 2 use only the 28 runs without gradient clipping (all on the A40). At
commit 1985984, `scale/analyze.load()` keys runs by size, rule and seed, so it lets a `_clip` run replace the
unclipped run with the same key (which one wins depends on directory order); `make_tables.py` and `make_figures.py`
then mix the two recipes in Table 3 and Figure 2. `scripts/scaling_unclipped.py` rebuilds both from Table 3's runs
alone, with the builders of `make_tables.py` and `make_figures.py` unchanged: the table it writes is the published
one, byte for byte, and the figure renders the same as the published one (checked by rendering both at 100 dpi).
`scale_compare.py` and the tests also load the two recipes separately;
`tests/test_paper_numbers.py::test_table3_and_figure2_rebuild` checks the table and, once `scale/analyze.load()` keeps
the recipes apart, that `make_tables.py` itself rebuilds Table 3.

## Re-running

From the repository root:

    python paper/v2/scripts/gsm8k_final_copy.py --out paper/v2/results/gsm8k_final_copy.json
    python paper/v2/scripts/blt1b_numbers.py
    python paper/v2/scripts/scale_compare.py
    python paper/v2/scripts/trained_compare.py
    python paper/v2/scripts/scaling_unclipped.py      # rewrites Table 3 and Figure 2
    python paper/v2/scripts/r9_numbers.py
    python paper/v2/scripts/r9_cue_split.py
    python paper/v2/scripts/cue_budget_numbers.py
    uv run pytest tests/test_paper_numbers.py

These need numpy (the repository's environment) and, for the first two, the GSM8K test set that `realtext.py`
downloads to `~/.cache/segresearch-gsm8k/`. Re-running the BLT-1B measurement of the prose rewrite needs the BLT
environment of `blt_load.py` (transformers 5.18 with `patches/transformers_pr49188_blt_window.diff` applied):

    R9_DATA=paper/v2/data/prose_rewrites python paper/v2/scripts/realblt_prose.py

`cue_budget_subset.py` needs the private results file and is listed for provenance only.
