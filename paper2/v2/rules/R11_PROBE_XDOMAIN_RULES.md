# R11: is there a general learned rule for where a patch start pays? Decision rules

Written 2026-10-27, before any probe was fitted on these data.

**Why.** The cross-domain test (paper3/XDOMAIN_RULES.md) asked whether one rule predicts where an extra patch start
pays in six domains. No two-byte map fitted on five domains beat entropy on the sixth (X3 failed). R10 then showed
that a linear probe on BLT-1B's patcher state reads far more than two bytes (it finds computed values with or without
`=`). This test repeats X3 with that probe in place of the two-byte map.

**Data.** Unchanged from the cross-domain test: results/xdomain_gains_{domain}.npz (six domains, 50 FIT + 60 TEST
documents each; per-position gains of a start against entropy@10%, bits on the next 16 bytes and exact match of the
next unit). New here: BLT-1B's patcher hidden state (last layer, normed) at every position, recomputed from the stored
bytes; the state at token t-1 is the decision state for a start at token t.

**Probes.** Ridge regression (lambda 10) on the patcher state, standardized and reduced to 32 principal components on
the training positions, predicting the gain:
- bits objective: trained on positions without a start in B (where an extra start would go);
- unit objective: trained on unit starts without a start in B.
Training sets (FIT documents only): **P_dom** (the domain's own), **P_lodo** (the other five domains), **P_math**
(math only). At most 60,000 training positions (seeded subsample) per fit.

**Comparison.** Exactly the cross-domain analysis (xdomain_analysis.py functions): per TEST document the top k =
round(0.03 n) positions without a start in B (unit objective: unit starts), capture = summed gain / the oracle's,
1,000 bootstrap resamples of TEST documents, differences paired, against **E** (entropy). The two-byte maps G_dom /
G_lodo are recomputed alongside for reference.

**Verdicts (each objective separately).**
- **Y1, a learned signal beats entropy within a domain:** P_dom - E above 0 in >= 5 of 6 domains: **holds**; <= 2:
  **fails**; otherwise **mixed**.
- **Y2, a general learned rule (the main question):** P_lodo - E above 0 in >= 5 of 6 domains: **holds**; <= 2:
  **fails**; otherwise **mixed**, with the domains listed.
- **Y3, algebra from math:** P_math - E on the unit objective in algebra above 0: **holds**.

**Predictions (written now).** Y1 holds on bits and is mixed on units; Y2 mixed (math-like domains help each other,
prose and clinical do not); Y3 does not hold (algebra's values were not entropy's blind spot).

**Reported without a verdict.** P - G (probe vs two-byte map) per domain; Spearman of each probe with the bits gain;
per-domain capture tables.

## Before the run (2026-10-27)

Patcher states computed and cached for all 660 documents (no probe fitted, no capture computed). No smoke test of the
analysis beyond compiling it: it reuses xdomain_analysis.py's selection and bootstrap code.
