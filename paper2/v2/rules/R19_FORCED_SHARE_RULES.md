# R19: how many starts should the detector force? Decision rules

Written 2026-10-29, before any layout in this test was run.

**Why.** R12's label-free detector (BLT-1B at 10%, its top 3% of positions forced, the rest by entropy) gains
+16.7 points on computed values in prose but lowers final answers (57.5% vs 62.2% under entropy). Starting at every
computed value without displacing entropy does not lower them ("with": 62.9%, R10). Numbers are only about 1.6% of
positions, so roughly half of the 3% forced share falls elsewhere and pushes out entropy starts (R14). Detection is
near its ceiling (computed values covered 97-99%, AUC 0.96; a much larger model's state, Qwen-1.5B, gave no better
layouts, R10). Asked by the PI how to design a better early detector: the open question is how much to force, not
what to read.

**Detector.** Unchanged from R12/R18: logistic probe on BLT-1B's patcher state (last layer, normed, PCA-32,
lambda 10), auto-labels "a number not seen earlier in the text", all positives and a seeded 25% of negatives, fitted
on each format's 1,382 train problems; out-of-fold scores (5 folds by problem) on train, full fit on test.
**Calibrated probability:** p = sigmoid(logit + log 0.25), correcting for the negative subsampling.

**Layouts** (BLT-1B, 10% of bytes on average, thresholds fitted on each format's train problems, no labels):
- **E**: entropy only;
- **D_q** for q in 0.5, 1, 1.5, 2, 3%: the detector's top q% of positions forced (threshold on train out-of-fold
  scores), the rest by entropy. D_3 is R12's layout (check: it should reproduce 38.4 / 57.5 on prose);
- **Dp**: forced where p > 0.5, the rest by entropy (the forced share follows the detector's confidence);
- **F_w** for w in 0.5, 1, 2, 4 bits: one ranking, no forced set: score = entropy + w * p (p = 0 off candidates),
  thresholded to 10%.

**Selection (on train only).** 300 train problems per format (seeded, seed 19), using their out-of-fold detector
scores, are scored under every layout. **L\*** = the layout with the highest exact match on computed values among
those whose final answers on these problems are at most 1 point below E's; if none qualifies, the layout with the
highest final answers. L\* is fixed before the test problems are scored.

**Measures.** Exact match on computed values and on final answers (teacher-forced greedy, as R12), bits per byte of
the solution, patch rate, forced share. 2,000 bootstrap resamples of the 275 test problems, differences paired.

**Verdicts (prose test, L\* chosen on prose dev; the equation test reported alongside with its own L\*).**
- **B1, the gain survives:** L\* - E on computed values. **Holds** if the interval is above 0.
- **B2, the cost is recovered:** L\* - D_3 on final answers. **Holds** if the interval is above 0.
- **B3, no final-answer loss against entropy:** L\* - E on final answers. **Holds** if the lower bound of the
  interval is above -3 points.

**Predictions (written now).** L\* is D_1.5 or a fused layout; B1 holds with at least +12 points; B2 holds; B3 holds.
Exact match on computed values falls slowly as q drops toward the share of numbers (~1.6%), then faster.

**Reported without a verdict.** Every layout on both test formats (exact, final answers, bits per byte, rate, forced
share); L\* - D_3 on computed values (how much gain is given up); the dev table used for the selection.

Script: realblt_forced_share.py -> results/blt_forced_share.json (R19_SMOKE=1: 40 train / 20 test problems, 20 of
them as dev; not used).

## Smoke test (2026-10-29, 40 / 20 problems, 20 as dev; results not used)

The pipeline ran end to end; no change. Patch rates 0.097-0.107 for every layout; forced shares match q (D_3 3.1-3.4%,
D_1.5 1.6-1.8%); Dp forces 0.4-0.9%.

## Results (2026-10-29; results/blt_forced_share.json, .log)

**Selection (300 train problems per format, out-of-fold scores).** Prose: E's final answers 57.7%; every forced
layout (D_q, Dp) fell more than 1 point below it (54.7-56.0%), so only F_0.5, F_1 and F_2 qualified: **L\* = F_2**.
Equation text: every layout qualified (final answers 56-66% vs E 48.7%): **L\* = D_3**.

Prose test, BLT-1B at 10% (275 problems; points; differences with 95% intervals):

| layout | forced share | exact, computed values | final answers | bits/byte | vs E, exact | vs E, final |
|---|---|---|---|---|---|---|
| E entropy | 0 | 21.7 | 62.2 | 0.799 | | |
| D_0.5 | 0.6% | 31.5 | 62.2 | 0.793 | +9.8 | +0.0 |
| D_1 | 1.1% | 34.7 | 63.3 | 0.790 | +13.1 | +1.1 |
| D_1.5 | 1.6% | 37.6 | 62.9 | 0.788 | +15.9 [+12.0, +19.8] | +0.7 [-2.2, +3.6] |
| D_2 | 2.1% | 37.8 | 60.7 | 0.789 | +16.2 | -1.5 [-4.7, +1.8] |
| D_3 (R12) | 3.1% | 38.4 | 57.5 | 0.787 | +16.7 | -4.7 [-8.4, -1.1] |
| Dp (p > 0.5) | 0.4% | 28.0 | 62.5 | 0.795 | +6.3 | +0.4 |
| F_0.5 | | 24.9 | 62.2 | 0.797 | +3.2 | +0.0 |
| F_1 | | 30.5 | 62.9 | 0.790 | +8.9 | +0.7 |
| **F_2 (L\*)** | | 32.9 | 62.9 | 0.787 | **+11.3 [+8.3, +14.4]** | **+0.7 [-1.8, +3.3]** |
| F_4 | | 35.7 | 63.3 | 0.785 | +14.1 | +1.1 |

D_3 reproduces R12 (38.4 / 57.5).
- **B1 HOLDS:** F_2 - E on computed values +11.3 [+8.3, +14.4], below the predicted +12 or more.
- **B2 HOLDS:** F_2 - D_3 on final answers +5.5 [+2.2, +8.7].
- **B3 HOLDS:** F_2 - E on final answers +0.7 [-1.8, +3.3].
- Given up against D_3 on computed values: -5.5 [-8.4, -2.7]. Predictions: L\* fused (right), B1's size (wrong), B2 and
  B3 (right), exact match falling slowly down to ~1.6% forced and faster below (right: 38.4, 37.8, 37.6, then 34.7,
  31.5).

**Reported without a verdict.**
- **D_1.5, not selected, does better on the test problems than L\*:** +15.9 on computed values (-0.8 [-3.1, +1.4]
  against D_3) with no final-answer loss (+0.7 [-2.2, +3.6] against E; +5.5 [+2.2, +8.7] against D_3). The selection
  rejected it on its dev final answers (55.7 vs 57.7, 300 problems). A test-set observation, not a result.
- The final-answer cost appears only once the forced share exceeds the share of numbers: on the train text number
  starts are 1.53% of positions in prose (computed values 0.38%); D_2 -1.5, D_3 -4.7.
- Equation text: L\* = D_3, +54.6 [+50.8, +58.4] on computed values and +14.2 [+9.1, +19.3] on final answers over E;
  forcing fewer loses (D_1.5 - D_3: -12.5 [-15.7, -9.5] exact). Number starts there are 2.68% of positions.
- Bits per byte are below entropy's in every layout (prose 0.785-0.797 vs 0.799).

**Reading.** The final-answer cost of R12's detector comes from the budget, not the detector: it appears when the
forced share exceeds the share of numbers in the text (prose 1.5%; in equation text, 2.7% numbers, 3% costs nothing).
Two ways remove it in prose: a fused ranking (selected; keeps two thirds of the gain) or forcing about as many starts as
there are numbers (D_1.5; keeps nearly all of it, seen on the test problems, not selected). That suggests a label-free
rule, a forced share equal to the share of number starts in the format's own train text, which is untested and needs
its own decision rule on new data. R8c and R13 train and evaluate on equation-style text, where 3% is the best share
here; they stay as committed.
