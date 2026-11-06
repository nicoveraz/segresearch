# R8b and D (issue #29): decision rules, written 2026-10-23 before any run of either

## H. A label-free hybrid at inference (BLT-1B, laptop)

**Question.** The layout that "gets both" (bits per byte like entropy, answers like the cue rules) adds a patch start
before each value with a hand-written `=` rule. Can a label-free version do the same: keep entropy, and force patch
starts only at the novelty table's strongest positions?

**Layouts** (294 stripped GSM8K test problems, the same as R8; thresholds fitted on the same 300 training problems):
`hybrid(q)@R` = a patch start at every position whose table value is in the top q of training bytes, the rest of the
budget R filled by entropy. **Primary: q = 3%, R = 10%** (fixed now). Also reported: q = 1.5% and 5% at R = 10%;
q = 3% at R = 15%. Compared, paired by problem, with R8's entropy, `=` rule and table layouts at the same R
(results/blt_waste.json; identical problems and thresholds).

**Rule** (primary layout against entropy at 10%; 95% bootstrap over problems, 1,000 resamples):
- **HOLDS** ("a label-free hybrid gets both") if the computed-result gain over entropy has a lower bound of at least
  +20 points **and** the whole-text bits-per-byte difference has an upper bound of at most +0.02.
- **FAILS** if the computed-result gain's upper bound is below +10 points, or the bits-per-byte difference's lower
  bound is above +0.05.
- Otherwise INCONCLUSIVE. Reported without a verdict: final answers; the same differences against the `=` rule.

## D. The budget curve in training (12M, GPU; for the rental package)

**Question.** R8's B and C were measured at inference on a model trained under entropy patching. Do they hold when
each rule is trained from scratch under its own layout?

**Runs.** 12M, unclipped (as paper 2's 12M runs), seeds 0-2: `entropy15`, `entropy20`, `nov15` (9 runs). Existing
runs complete the curves: `entropy10` and `nov10` at 12M (3 seeds each). Same corpus, evaluation and code (published
commit 1985984).

**Measures.** Evaluation patch rate; final answers and computed results; bits per byte.
- **k_train (answers)** = the patch rate entropy needs to reach nov10's final-answer accuracy (linear interpolation on
  entropy's 10/15/20% points) divided by nov10's rate; if entropy at 20% does not reach it, k_train is reported as
  greater than (entropy20's rate / nov10's rate). Interval: bootstrap over seeds within each arm (1,000 resamples;
  3 seeds per arm, so the interval is rough and is reported as such). Primary target: final answers (computed results
  are near the floor at 12M).
- **B holds in training** if k_train's lower bound is at least 1.5; **fails** if its upper bound is below 1.2;
  otherwise inconclusive.
- **C in training** (reported, no verdict): the rate the table needs to match entropy10's bits per byte, from nov10 and
  nov15.

Any change to either design after a smoke test is disclosed with the results.

## Note on the code (added 2026-10-24, before any run of this study)

The rental archive overlays `paper2v2/pod_patch/scale/rules.py` and `prep.py` on the published code to add one rule
kind (`novtop{Q}+entropy{R}`, for R8c). On the Mac, every existing rule (entropy10/15/20, nov10/15, novc13, dep10,
syntax+entropy10, jump10) gives byte-identical thresholds and masks with and without the overlay; a toy run trained and
evaluated end to end. The runs of this study therefore use the same computation as the published code.

## Amendment to D (2026-10-29, before any run of D; disclosed with the result)

Written after the project review of 2026-10-29 (`ANALYSIS_2026-10-29.md` section 2; review findings P2-2, TR-4, ISS-1,
TR-8, MISS-3, M7, M4, TR-12, CMP-4, CMP-5) and before the rental; no run of D has started. The question, the target
(final answers at 12M) and the thresholds (1.5 and 1.2) are unchanged.

1. **Same-device anchor.** Paper 2's 12M nov10 runs were trained on a Kaggle T4 in fp16, its 12M entropy10 runs on an
   A40 in bf16 (the `device` field of each result.json). D adds 12M `nov10`, seeds 0-2, on the A40 in the same session
   (`d_budget12.sh`: entropy15, entropy20, nov15 and nov10, 12 runs). k_train's target and C's 10% point are these A40
   runs. k_train with the T4 runs as the target (the rule as first written) is reported beside it. The A40 runs
   against paper 2's T4 runs (the three in `runs/`, and all five with the two same-seed re-runs in
   `results/scale/cross`) are reported as the measured device-and-precision difference at 12M, without a verdict. If
   fewer than 2 A40 nov10 runs succeed, the T4 runs are the target and the result is labelled cross-device. (The
   author's decision, 2026-10-29: the anchor is run; novc13 is not re-run on the A40, and C6's 12M bits-per-byte
   sentence is dropped instead.)
2. **Rates.** Patch rates are measured evaluation patch rates (the `rate` field of result.json: the share of
   evaluation bytes that start a patch), as in R8 (`R8_WASTE_RULES.md`). An arm's evaluation mask depends on neither
   the seed nor the model size, so each arm has one rate (in paper 2's runs: entropy10 0.1120, nov10 0.1195, entropy20
   0.2279). Nominal training rates (10, 15 and 20%) are reported beside.
3. **k_train** = the smallest evaluation rate at which entropy's piecewise-linear curve of mean final answers (entropy10,
   15 and 20 at their measured rates) reaches nov10's mean, divided by nov10's rate. If the curve has not reached it
   by entropy20, k_train is reported as greater than entropy20's rate / nov10's rate (as first written). If the curve
   is not monotone, its first crossing counts and the curve is reported as non-monotone. An entropy arm with fewer than
   2 valid seeds is left out, and the curve then ends at the highest arm left.
4. **The bound that decides is t-based**, not the percentile bootstrap: over 3 seeds a nominal 95% percentile bootstrap
   covers about 75% (review MISS-3). At each rate r of a grid over the curve (steps of 0.00001), d(r) = interpolated
   entropy - nov10 is a linear combination of independent arm means; its 95% interval uses the Welch-Satterthwaite
   degrees of freedom and the t values of `scale.analyze.welch` (df floored). k_train's lower bound is the first rate
   at which d's upper limit reaches 0 (below it, entropy is significantly below nov10), and its upper bound the first
   rate at which d's lower limit passes 0; both are divided by nov10's rate. If d stays significantly below 0 through
   entropy20, the lower bound is greater than entropy20's rate / nov10's rate (1.91 with the rates above); if d never
   passes 0 significantly, the upper bound is open. The committed bootstrap (1,000 resamples over seeds) is reported as
   a description, with the share of resamples whose curve does not reach the target.
5. **B in training** holds if k_train's lower bound is at least 1.5 and fails if its upper bound is below 1.2;
   otherwise it is inconclusive (thresholds as first written). **C in training** (reported, no verdict): the same
   construction on bits per byte, the table's curve (A40 nov10 and nov15) against entropy10's mean, where reaching
   means at or below. **Also reported without a verdict:** nov10 (A40) against entropy interpolated at nov10's
   evaluation rate, final answers and bits per byte: the 12M, equal-rate counterpart of C8's training cost (review
   CMP-5).
6. **Failed runs**: the pilot's rule, as in R2 (bits per byte above its size's median + 1, or not finite; a run the
   divergence guard stops has no result). Failed runs are reported and not replaced; an arm with fewer than 2 valid
   seeds drops out as in points 1 and 3.
7. **Code**: `paper2v2/rental_verdicts.py` (`d_rule`), committed with this amendment, tested
   (`tests/test_rental_verdicts.py`, including the censored, non-monotone and one-seed cases) and dry-run on paper 2's
   runs with stand-ins (`results/rental_verdicts_dryrun.json`). R2's amendment of the same date (point 5) describes
   the divergence guard, the kept weights and the pod check, which apply here too.
8. **Rates measured on the Mac before the rental** (2026-10-29; the archive's code, masks only, no training; entropy scores on MPS, which reproduce paper 2's CUDA evaluation rates for entropy10 0.1120, nov10 0.1195 and entropy20 0.2279 to four decimals): entropy15 0.1754, nov15 0.1536 (its train rate is 0.1439, not 0.15: the
   table's values tie), novtop3+entropy10 0.1163. With these, k_train's censoring line is 0.2279 / 0.1195 = 1.91, and
   R13's pilot rate (0.1303) lies 29% of the way from entropy10 to entropy15. The pod's own rates decide.
