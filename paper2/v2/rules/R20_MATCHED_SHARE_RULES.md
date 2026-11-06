# R20: force as many starts as there are numbers, on fresh problems. Decision rules

Drafted 2026-10-29 (58cd709); approved by the PI and fixed 2026-10-29, before any fresh problem was rewritten or
scored (spend approved up to $8).

**Why.** In R19 (BLT-1B at 10%, prose), the layout chosen on train problems, a fused ranking, removed the detector's
final-answer cost but kept only two thirds of its gain (+11.3 vs +16.7). Not selected, a 1.5% forced share kept nearly
all of it (+15.9) with no final-answer loss; the cost appeared only when the forced share exceeded the share of
numbers in the text (prose 1.53% of positions, equation text 2.68%, where 3% stayed best). That was seen on R19's test
problems, so it needs problems none of R10-R19 has scored.

**Fresh problems.** GSM8K test problems 300-1,318 (1,019; R9 rewrote only the first 300), rewritten as prose exactly
as R9 (paper2v2/prose_rewrite.py: same model, system prompt, effort, schema and validation), saved apart from R9's data.
Expected from R9's rates: ~935 valid, ~750 with computed values, ~2,400 values. Nothing is tuned on them: the
detector, its thresholds and the entropy fill are fitted on R12-R19's 1,382 train problems, as in R19.

**Rule M (label-free).** Forced share q_f = number starts among candidate positions / all positions in format f's
train text: **prose 1.53%, equation text 2.68%** (measured on the train problems on 2026-10-29, no labels). The
detector's top q_f forced (threshold on train out-of-fold scores, as R19's D_q), the rest by entropy.

**Layouts** (BLT-1B, 10%): E (entropy), M, D_3 (R12's), F_2 (R19's selected fused ranking).

**Measures.** As R19: exact match on computed values and final answers (teacher-forced greedy), bits per byte of the
solution, rate, forced share; 2,000 bootstrap resamples of fresh problems, differences paired.

**Verdicts (fresh prose).**
- **N1, no final-answer cost:** M - E on final answers. **Holds** if the lower bound of the interval is above -2
  points.
- **N2, the gain is kept:** M - D_3 on computed values. **Holds** if the lower bound is above -3 points.
- **N3, better than the fused ranking:** M - F_2 on computed values. **Holds** if the interval is above 0.

**Predictions (written now).** N1, N2 and N3 hold.

**Reported without a verdict.** M - D_3 on final answers; every layout on fresh equation text (where M forces 2.68%,
close to D_3); whether R19's ordering (D_3 > M > F_2 > E on computed values in prose) repeats.

**Cost.** API: ~1,019 batched rewrites at R9's measured $0.0033 each (747 input + 185 output tokens at $2 / $10 per
million) ~ $3.40, plus retries at standard prices; retries skip "too long" failures, which almost never fit on a
second try (R9: 133 of 159 stayed too long), ~ $0.10. Expected total ~ $3.50, hard cap $8. Compute: the Mac, ~1 hour.

Scripts: prose_rewrite.py `fresh` mode -> data/prose_rewrites_fresh/test.jsonl;
realblt_matched_share.py -> results/blt_matched_share.json.

## Before the data (2026-10-29)

- Rewrite batch [redacted: batch ID] submitted: 1,019 requests (823 with computed values in the
  original), projected $3.40; no overlap with R9's 300 test questions.
- Smoke test of realblt_matched_share.py (R20_SMOKE=1: 40 train / 20 of R9's test problems, not fresh; results not
  used): ran end to end; E, D_3 and F_2 reproduce R19's smoke values exactly; M forces 1.8% (prose) and 2.9%
  (equation text) on those 20 problems. No change.

## Fresh data (2026-10-29)

The batch ran in about 5 minutes: 915 valid of 1,019 (100 too long for BLT-1B, 4 with marked values that differ);
the 4 eligible retries were refused by the API (HTTP 400, not billed) and dropped. 745 problems have computed
values (2,448 values). Cost $3.46 (cap $8). Saved in data/prose_rewrites_fresh/test.jsonl; all served by
claude-opus-5-5.

## Results (2026-10-29; results/blt_matched_share.json, .log)

Fresh prose, 915 problems (2,448 computed values), BLT-1B at 10% (points; differences with 95% intervals):

| layout | forced share | exact, computed values | final answers | bits/byte | vs E, exact | vs E, final |
|---|---|---|---|---|---|---|
| E entropy | 0 | 21.7 | 59.0 | 0.802 | | |
| M matched share (1.53%) | 1.6% | 34.0 | 57.2 | 0.796 | +12.3 [+10.2, +14.5] | -1.9 [-3.5, -0.1] |
| D_3 (R12) | 3.1% | 36.1 | 54.5 | 0.792 | +14.4 [+12.2, +16.7] | -4.5 [-6.4, -2.5] |
| F_2 (R19's L\*) | | 32.8 | 58.4 | 0.791 | +11.1 [+9.3, +12.9] | -0.7 [-2.2, +0.8] |

- **N1 DOES NOT HOLD:** M - E on final answers -1.9 [-3.5, -0.1]; the lower bound is below -2.
- **N2 DOES NOT HOLD:** M - D_3 on computed values -2.1 [-3.4, -0.8]; the lower bound is below -3.
- **N3 DOES NOT HOLD:** M - F_2 on computed values +1.2 [-0.1, +2.5].
- All three were predicted to hold. Reported: M - D_3 on final answers +2.6 [+1.0, +4.4]; M - F_2 on final answers
  -1.2 [-2.3, -0.1].
- **R12 and R19 replicate on fresh problems:** D_3 +14.4 on computed values and -4.5 on final answers (R12's test:
  +16.7, -4.7); F_2 +11.1 and -0.7 (R19's test: +11.3, +0.7).
- Fresh equation text (reported): M forces 2.8%, ties D_3 (exact -0.8 [-1.6, +0.1], final answers -0.1
  [-1.1, +0.9]), both +54 points on computed values and +14.5 on final answers over E; F_2 is 13.3 points lower on
  computed values there.

**Reading.** R19's test-set observation (a 1.5% share keeps the whole gain at no cost) does not replicate: on fresh
problems the forced share trades computed values against final answers along one curve, and the share of numbers is
a midpoint on it, not a free fix (it keeps 85% of D_3's gain and halves its final-answer cost). R19's selection rule
picked well: the fused ranking keeps 77% of the gain with no measurable final-answer loss. No single layout wins in
both formats: in prose the fused ranking, in equation text a forced share (3% or the matched share, which tie there).
