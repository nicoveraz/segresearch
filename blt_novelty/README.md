# blt_novelty: novelty patching for Meta's BLT code base

A drop-in implementation of *novelty patching* ([arXiv:2610.22100](https://arxiv.org/abs/2610.22100)) for
[facebookresearch/blt](https://github.com/facebookresearch/blt), so the rule can be tried at larger scale.

**What it is.** Start a patch where the upcoming bytes are unlikely to repeat the last 128 bytes. The rule is a
lookup table over the previous two bytes, fitted in seconds on raw text (no labels, no reference model, no losses).
In BLT-style models trained from scratch on math (3 seeds per rule, ~11% patch rate) it raised computed-result
accuracy from 21% to 57% and final-answer accuracy from 60% to 90% at 50M parameters, with no clear effect at 1M; on
prose it lowered bits per byte and on code it tied entropy. **It is tested up to 50M parameters only**; whether it
helps at BLT's scales is the open question this folder is meant to make easy to answer.

**Why it fits BLT directly.** `bytelatent/data/patcher.py` already contains a lookup-table patcher
(`find_lookup_table_patch_start_ids`: a patch starts at token *i* + 2 when `table[token[i], token[i+1]]` is True),
present since the initial release but not connected to any patching mode and released without a table. This folder
supplies the table and a small diff (22 added lines) that exposes the function as an opt-in mode. The default patcher is unchanged.

## Contents

| File | What it does |
|---|---|
| `novelty.py` | novelty target (forward LZ match), table fit, BLT token mapping (byte + 4; special tokens as byte 0) |
| `fit_table.py` | fit a table on your text and save it for BLT: `python -m blt_novelty.fit_table table.pt --rate 0.10 data.jsonl` |
| `blt_lookup_table_mode.diff` | adds `patching_mode: lookup_table` and `lookup_table_path` to BLT's `PatcherArgs` / `Patcher` |
| `test_novelty.py` | tests (below) |

## Use with BLT

```bash
cd blt                                   # a checkout of facebookresearch/blt
git apply /path/to/segresearch/blt_novelty/blt_lookup_table_mode.diff
python -m blt_novelty.fit_table novelty_table.pt --rate 0.10 your_training_text.jsonl   # PYTHONPATH=/path/to/segresearch
```

Then set, in the patcher config: `patching_mode: lookup_table` and `lookup_table_path: novelty_table.pt`. `--rate`
is the fraction of bytes that start a patch on the fitting text (the paper used 10% of training bytes, about 11% on
held-out text); fit the table on a sample of the same training data (the paper used 4 MB).

## Tests

```bash
python -m pytest blt_novelty -q
```

1. the forward match equals a brute-force search;
2. patches are valid (lengths sum to the sequence length, none empty) and the rate is the requested one;
3. decisions are causal (a token only affects the next two patch-start decisions);
4. inside the segresearch repository: the scores equal the paper's implementation, and on BLT-1B token ids
   (`[BOS] + byte + 4`) the patch starts equal the paper's novelty layout;
5. with Meta's BLT code importable: `find_lookup_table_patch_start_ids` gives the same starts as `start_mask`, and
   (diff applied) the new mode produces those patches through `Patcher.patch`.

Checked on 2026-10-16 against `patcher.py` from facebookresearch/blt `main` (diff applied): all 7 tests pass, and
the static, byte, space and entropy modes give identical output before and after the diff. Refitting on the GSM8K
training solutions reproduces, value for value, the table behind the paper's BLT-1B inference results.

## Licensing

This folder is MIT (as the rest of segresearch) and contains no BLT code; the diff only adds lines to BLT's
`patcher.py`, which is under BLT's own license (CC BY-NC 4.0).
