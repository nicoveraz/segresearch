# R12: can the computed-value probe be trained without labels? Decision rules

Written 2026-10-27, before any auto-label was computed or probe fitted.

**Why.** R10's best probe (BLT-1B patcher state, trained on prose) used labels from the rewrite markers, which only
exist because the data were generated. Equation text gives labels for free (`= `), but that probe recovered only 62% of
the gap in prose. This test replaces the labels with ones computed from raw text alone.

**Auto-label.** Same candidates as R10 (positions in the solution whose previous byte is ASCII and not alphanumeric,
before the final-answer line). A candidate is positive if a number starts at that byte (a digit run, commas and one
decimal point allowed) whose value (commas removed) does **not** occur as a number anywhere earlier in the text
(question or solution so far). Everything else is negative. The rule reads the number itself, so it is a training
label only; the probe it trains reads the state before the number, as in R10. Agreement with the true (marker) labels
is reported (precision, recall).

**Probes** (as R10: logistic on PCA-32 of the patcher state, all positives and a seeded 25% of negatives, out-of-fold
scores on the training format's own train problems): **auto_prose** (auto-labels on prose train), **auto_orig**
(auto-labels on equation train), and **marker_prose** (R10's rp_prose, refitted here as the reference).

**Layouts** (BLT-1B, 10%, as R10): entropy; hybrid(X) with X's top 3% forced (threshold on the format's own train
problems), rest entropy; for X in marker_prose, auto_prose, auto_orig. Measured on both test formats: exact on
computed values and final answers, bits per byte of the solution, detection AUC for true computed-value starts.
Intervals: 2,000 bootstrap resamples of test problems, differences paired.

**Verdicts (prose test).**
- **Z1, a label-free probe finds computed values without the cue:** hybrid(auto_prose) - entropy, exact on computed
  values: **holds** if the interval is above 0.
- **Z2, as good as the supervised probe:** hybrid(auto_prose) - hybrid(marker_prose): **holds** if the interval lies
  above -5 points.

**Predictions (written now).** Z1 holds; Z2 holds (computed results are almost always numbers new to the text, and
operands are mostly copies from the question).

**Reported without a verdict.** auto_orig on both formats; the auto-labels' precision and recall; bits per byte.

## Smoke test (2026-10-27, 40 train / 20 test; results not used)

The pipeline ran end to end; no change. Auto-label agreement on the 40 train problems: precision 0.56 (orig) / 0.66
(prose), recall 0.83 / 0.87 (computed results that equal an earlier number are missed; new numbers that are not `= `
results are included).
