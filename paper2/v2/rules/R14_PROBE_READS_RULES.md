# R14: what does the patcher-state probe read? "A computed number" or "some number"? Decision rules

Written 2026-10-28, before any probe in this test was fitted (R10 and R12's probes and results exist; nothing below
has been computed).

**Why.** R10-R12: a linear probe on BLT-1B's patcher state, read before the candidate byte, finds computed-value
starts in prose (AUC 0.96-0.97; entropy 0.66) and, used to force patch starts, raises exact match on computed values.
Most candidates the probe rejects precede words, so much of that AUC may be "a number comes next" rather than "a
computed number comes next". This test separates the two.

**Data and probes.** As R12 (data/prose_rewrites, 1,382 train / 275 test problems, both formats; candidates: positions
in the solution whose previous byte is ASCII and not alphanumeric, before the final-answer line; BLT-1B patcher
state, last layer, normed, at each candidate; logistic regression on PCA-32, all positives and a seeded 25% of
negatives, out-of-fold scores on the training format's own train problems). Labels per candidate:
- **computed** (y): a computed value starts here (the rewrite markers; R10's label);
- **new** (ya): a number starts here whose value does not occur earlier in the text (R12's auto-label);
- **number**: a number starts here (any digit);
- among numbers: computed, other new numbers (new but not computed), copies (value seen earlier).
Probes: **marker_prose** (y), **auto_prose** (new), **anynum_prose** (number), all trained on prose; **auto_orig**,
**anynum_orig** trained on equation text.

**Measures** (prose test; the equation test reported alongside; 2,000 bootstrap resamples of test problems):
1. AUC for computed starts among all candidates (as R12).
2. **Within-number AUC**: among candidates where a number starts, AUC for computed vs other numbers.
3. AUC for "a number starts here" among all candidates.
4. Composition of each probe's forced set (its top 3% by the R12 threshold): shares that are computed, other new
   numbers, copies, non-numbers; recall of computed starts.
5. BLT-1B at 10% (as R12): entropy and forced[P]:entropy for P in marker_prose, auto_prose, anynum_prose: exact match
   on computed values and bits per byte of the solution.
6. Descriptive, no verdict: the words before the highest-scoring candidates, and before computed values vs copies.

**Verdicts (prose test).**
- **W1, the probe reads more than "a number comes next":** within-number AUC of auto_prose. **Holds** if the 95%
  interval's lower bound is above 0.70; **fails** if its upper bound is below 0.60; otherwise mixed.
- **W2, the "new number" label matters:** forced[auto_prose] - forced[anynum_prose], exact on computed values.
  **Holds** if the interval is above 0; **reversed** if below 0; otherwise **no difference**.

**Predictions (written now).** W1 holds: computed values follow conclusion phrases ("so", "in total", "left") and
copies follow restatements of the question. W2 holds modestly (+3 to +8 points), because "any number" also forces
starts on copies, which pay little (C17: +1 to +3 points). If numbers make up no more than the 3% budget, both
probes force nearly every number and W2 shows no difference; the forced-set composition (measure 4) shows whether
that happened.

Script: realblt_probe_reads.py -> results/blt_probe_reads.json (R14_SMOKE=1: 40 train / 20 test problems, results
not used).

## Smoke test (2026-10-28, 40 train / 20 test problems; results not used)

The pipeline ran end to end; no change.

## Results (2026-10-28; results/blt_probe_reads.json, .log)

Prose test, 275 problems, 26,610 candidates, of which 3,085 start a number (711 computed values, 418 other new
numbers, 1,956 copies).
- **W1 HOLDS.** Within-number AUC (computed vs other numbers): auto_prose **0.768 [0.738, 0.795]**; marker_prose
  0.848; anynum_prose 0.589; equation-trained auto_orig 0.601. AUC for "a number starts here": 0.89-0.94 for every
  probe.
- **W2 HOLDS, below the predicted size.** forced[auto_prose] - forced[anynum_prose], exact on computed values:
  **+2.8 [+0.4, +5.3]** points (predicted +3 to +8). Against entropy: marker +14.9, auto +16.7, anynum +13.9.
  Reported without a verdict: on the equation test the order reverses (auto - anynum -3.5 [-5.5, -1.6]); bits per
  byte of the solution vs entropy: anynum -0.019, auto -0.012, marker -0.012 (prose).
- **The budget condition happened:** the 3% forced budget (~5,800 starts on the prose test) exceeds the number of
  number starts (3,085), so every probe forces nearly all numbers plus ~2,600 non-number positions (54-60% of each
  forced set); recall of computed values 97% (auto), 94% (anynum). The label's effect is therefore bounded by
  ranking within the 3% of the probes' disagreement.
- **Words before the number (descriptive):** computed values follow "comes to", "total of", "to $", "gives",
  "out to" (mean auto_prose score +1.6 to +2.0); copies follow "multiplying by", "so multiplying", "away from",
  "from the" (-3.8 to -2.9). The 100 highest-scoring candidates: 79 other new numbers, 20 computed values, 1 copy.

**Reading.** The patcher's state holds more than "a number comes next": it separates a number that concludes a step
from an operand being restated, from the phrase before it. The label-free probe reads part of that (0.77; the
marker-trained probe 0.85). At BLT-1B's 10% rate with a 3% forced share, though, "a number comes next" already
captures most of the benefit, because there are fewer numbers than forced starts.
