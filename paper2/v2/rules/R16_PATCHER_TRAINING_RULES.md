# R16: can the patcher's training make the computed-value signal stronger? Decision rules

Written 2026-10-28, before any patcher variant was trained.

**Why.** The patcher learns "a new number is coming" only as a by-product of next-byte prediction (R10-R14). Asked by
the PI whether its training can be changed to strengthen that signal and give others. BLT-1B's patcher cannot be
retrained here; paper 2's own training-time patcher can (EntropyLM: 1 layer, d = 32, 2 heads, 128-byte context;
4,000 updates of 32 random windows of paper 2's 1.9 GB corpus, as prep.py trains it). This is a feasibility check on
held-out text, not a training-time test.

**Variants** (same corpus, sampler seed, updates, batch, learning-rate schedule and optimizer as paper 2's patcher):
- **V0, standard:** next-byte prediction (paper 2's patcher, retrained here; the bundle's entropy.pt is reported
  alongside);
- **V1, new-number head:** V0 plus a linear head on the final normed state predicting "a new number starts at the
  next byte" (a digit after a non-alphanumeric byte whose value does not occur earlier in the window; computed from
  the bytes, no annotation), binary cross-entropy added to the loss with weight 1;
- **V2, look ahead:** V0 plus three linear heads predicting the bytes 2, 3 and 4 ahead (through the tied embedding),
  their mean cross-entropy added with weight 1;
- **V3, larger (capacity control):** next-byte only, d = 64, 2 layers;
- **V4, both:** V1 + V2.

**Evaluation.** R12's data (GSM8K prose rewrites and the original equation-style solutions, 1,382 train / 275 test
problems), R10's candidates and computed-value labels. For each variant: the state at each candidate (the final
normed state predicting that byte, plus its entropy), read in 128-byte windows as paper 2's patcher reads the corpus.
A logistic probe (as R13: auto-labels "a number not seen earlier in the text", all positives, a seeded 25% of
negatives, standardized, PCA-32, lambda 10) fitted on the train problems of each format, evaluated on its test
problems. 2,000 bootstrap resamples of test problems; differences paired.

**Verdicts (prose test; the equation test reported alongside).**
- **U1, a training change strengthens the signal:** AUC for computed-value starts among candidates, probe on Vk minus
  probe on V0, for k in 1, 2, 4. **Holds** if at least one has an interval above 0 and also beats V3 (Vk - V3 interval
  above 0), so the gain is not capacity alone; **fails** if none of V1, V2, V4 is above V0; otherwise **mixed** (above
  V0 but not above V3).
- **U2, the same for results vs copies:** the within-number AUC (computed values vs other numbers), same rule.

**Predictions (written now).** U1 holds through V1 (the head trains the representation the probe reads; the label
transfers from equation text to prose as R12's did); V2 adds little alone; V3 some. U2 holds for V1 but smaller.

**Reported without a verdict.** The new-number head's own output as a signal (V1, V4); entropy's AUC per variant; the
variants' next-byte loss on held-out corpus text (does the change cost prediction?).

Script: patcher_variants.py -> results/patcher_variants.json (PV_SMOKE=1: 200 updates, 40 / 20 problems, not used).

## Smoke test (2026-10-28, 200 updates, 40 / 20 problems; results not used)

The pipeline ran end to end; no change. Before it: the window reader was checked against direct reads (identical) and
the new-number targets on a test string.

## Results (2026-10-28; results/patcher_variants.json, .log)

Check: V0 retrained here reproduces paper 2's patcher to float precision on the Mac (held-out bits 3.22140 vs
3.22140; prose probe AUC 0.83085 vs 0.83083).

Probe AUC for computed-value starts, and its difference from V0 (95% interval over test problems):

| | prose, all candidates | prose, within numbers | equation text, all | equation text, within numbers | held-out bits/byte |
|---|---|---|---|---|---|
| V0 standard | 0.831 | 0.655 | 0.966 | 0.886 | 3.22 |
| V1 new-number head | -0.008 [-0.020, +0.003] | **-0.027 [-0.043, -0.013]** | +0.010 | +0.024 | 3.24 |
| V2 look ahead | -0.007 [-0.018, +0.004] | **-0.018 [-0.033, -0.004]** | +0.014 | +0.041 | 3.26 |
| V4 both | -0.002 [-0.013, +0.008] | +0.004 [-0.008, +0.015] | +0.014 | +0.046 | 3.31 |
| V3 larger (d 64, 2 layers) | **+0.022 [+0.013, +0.032]** | -0.004 [-0.018, +0.010] | **+0.020** | **+0.062** | **2.66** |

- **U1 FAILS, U2 FAILS:** no training change beats the standard patcher on prose; the new-number head and the
  look-ahead heads make the within-number separation worse. Predictions (U1 and U2 hold through V1) wrong.
- On equation text (the corpus's own style) every change helps a little, but each less than the larger model
  (Vk - V3 below 0 for all k), and each costs next-byte prediction (+0.02 to +0.09 bits per byte); the larger
  model predicts much better (2.66).
- The new-number head's own output is a weak signal on prose (AUC 0.58).

**Reading.** In paper 2's tiny patcher, extra training objectives compete for very little capacity: they help a little
in the training text's own style, do not transfer to prose, and are beaten by simply making the patcher larger. The
signal looks like a product of general language competence (size, data), which fits BLT-1B's much larger patcher
reading it far better (prose 0.96-0.97 vs 0.83 here). Untested: auxiliary objectives on a patcher large enough to
have spare capacity.
