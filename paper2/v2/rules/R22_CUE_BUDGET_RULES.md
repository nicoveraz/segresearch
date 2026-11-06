# R22: what R9 measured, and where the detector's final-answer cost comes from. Decision rules

Drafted 2026-10-29 (7c52736), before any layout in this test was run. **Approved by the PI and fixed on 2026-10-29,
with amendment A1**, after the smoke test (below) and
before any fresh problem was scored. Amendment A1 replaces D3R with D3F in verdict F2. Issue #34;
ANALYSIS_2026-10-29.md section 3 items 1 and 4, section 5 (O7); review findings DET-1, DET-2, P2:M1, M2, M3. (R21 is
reserved for the second-writer test, #36.)

**Why.**
- **What R9 measured (O7).** A split of R9's rows by the bytes just before each computed value (`r9_cue_split.py` ->
  `results/blt_prose_cue_split.json`): BLT-1B's entropy at 10% starts a patch at 0.0% of values after `$` in both
  writings (167 and 163 values), at 17.8% of values after `= ` (544) and at 74.3% of prose values after a plain space
  (548). These three classes reproduce R9's 13.6% and 57.2%. A start at every value raises exact match from 4.2% to
  43.7% (`$`, equation text), from 9.2% to 45.4% (`$`, prose) and from 20.4% to 56.6% (`= `), but only from 25.4% to
  31.4% (plain space).
  - Proposed mechanism, the **ceiling**: after a cue that fixes the type (only a digit can follow), the next byte's
    entropy is at most log2 10 = 3.32 bits, and BLT-1B's 10% threshold on GSM8K text is 3.31 bits
    (`patcher_budget_by_kind.json`).
  - Competing account, **budget per value** (P2:M1): R9 fixes 10% of bytes in both writings, but prose has 1.43x the
    bytes per value (1.86x in the solution body, train problems).
- **Disclosure.** The split was found after the fact (`r9_cue_split.py`), on rows already used for R9's verdicts. A
  generator has also joined R12's and R20's rows by cue (opportunities.md C7, evidence (4)). On the fresh prose,
  D_3 - E is +47.1 points on `$` values (n = 575) and +4.4 on plain-space values (n = 1,873). So the exact-match splits
  of E, M, D_3 and F_2 on the fresh problems are not unseen: they are reported here, not tested.
  - Nobody has computed on the fresh problems: coverage (R20's rows have none), the patcher's next-byte distribution,
    or any other layout.
  - A clean confirmation of the split needs new rewrites (about $3.5 after the API cap lifts on 2026-11-01, with the
    PI's OK). That is not part of this rule.
- **The detector.**
  - **DET-1:** its final-answer cost may be one displaced start at the final answer.
  - **DET-2:** its bits per byte count only the solution.
  - **M2:** every detector result is at one budget (10%).
  - **M3:** it lacks its natural comparators.

**Problems and texts.**
- **Fresh, for the verdicts:** R20's 915 problems (`data/prose_rewrites_fresh/test.jsonl`). 745 have computed values
  (2,448 values); all 915 count for final answers and bits.
- **Train, for fitting only:** R9's 1,382 train problems in each format. They fit:
  - the thresholds;
  - the detector (R12's probe, as R19/R20);
  - the fused layouts' percentiles and the random null's rate;
  - the dependence tables (first 400 problems);
  - the matched rates and D3R's body threshold.

  No value labels are used, except where the table below says so and in the matched rates, which count computed
  values on train problems.
- **Pilot:** R9's 275 test problems (221 with values), patcher only, no verdicts. Gate: its entropy coverage by cue must
  reproduce `blt_prose_cue_split.json` exactly, or the run stops before any fresh problem is scored.
- **Three writings:**
  - **orig**, the stripped GSM8K text;
  - **prose**, the R9/R20 rewrite;
  - **orig_is**, orig with every `= ` right before a computed value (or before its `$`) replaced by `is `. The spans
    shift. The same is done to the train problems, and the thresholds are refitted on them.
- **Cue of a value:** the bytes just before its first digit, classified as in `r9_cue_split.py`: `= `, `$`, plain space,
  other. In orig_is the former `= ` values are labelled `is `.

**Measures.**
- **Per computed value:** covered (a patch start at its first digit), exact (teacher-forced greedy, every byte), cue.
- **At each value's first digit**, read from the patcher's prediction at the byte before (the prediction that entropy
  reads):
  - H, the patcher's entropy, which the layouts use (bits);
  - P(digit), the mass on the ten digits, and P($);
  - H_digits, the entropy of the ten digit probabilities renormalized (at most 3.32 bits);
  - max-prob.

  The same is recorded at the final answer's first digit and, in orig_is, after `is `. The thresholds are reported in
  bits at every budget.
- **Final answer:** covered (a start at its first byte), exact, and exact given covered or not.
- **Bits (DET-2):** for each problem and layout, the bits of the question line, the solution body and the final-answer
  line. Whole text = their sum; "solution" = body + final line, as in R10-R20.
- **Intervals:** 2,000 bootstrap resamples of fresh problems; differences are paired by problem and value.

**Budget per value (one definition, M1).**
- **Definition:** entropy starts in the solution body per computed value. The body is the text after the question line
  and before the final-answer line, where every computed value lies. Counts are pooled over a format's train problems
  with values, at that format's train-fitted thresholds.
- **Matched rates:** R_o* is the orig rate at which orig's budget per value equals prose's at 10%; R_p* is the prose rate
  at which prose's equals orig's at 10%. Grid of 2-40% in steps of 0.25 points, linear interpolation; fixed on the
  train problems before any fresh problem is scored.
- **Gaps:** coverage gaps on the fresh values, prose - orig, paired by value:
  - G_10, both writings at 10%;
  - G_A, orig at R_o*;
  - G_B, prose at R_p*.

  Density shares: s_A = 1 - G_A / G_10 and s_B = 1 - G_B / G_10.
- **Reported, not used for a verdict:** the same at rates matched on whole-text bytes per value (R9's 1.43x), and
  coverage-by-cue curves from 5% to 25%.

**Layouts.** BLT-1B with 512-byte windows; thresholds fitted on each format's own train problems, in train mode, as in
R10-R20.

| | rule | labels |
|---|---|---|
| E | entropy | |
| D3 | the detector's top 3% of positions forced, the rest by entropy (R12) | |
| F2 | entropy + 2 x the detector's calibrated probability (R19's L*) | |
| D3R | DET-1's variant: D3 inside the solution body only (below) | |
| VU | training-free fused layout: a start where entropy or value uncertainty T passes its train-percentile threshold (`max:entropy,T`). T = P(likeliest byte class) x within-class entropy / log2 class size (paper 1) | |
| MP | the same with U = 1 - max-prob (`max:entropy,U`) | |
| with | E plus a start at every computed value (R9) | values |
| eq | equation text only: a start at the first digit of every number right after `= ` (with or without `$`), the rest by entropy. This is the `=` rule; in prose it has nothing to fire on | hand-written cue |
| dep | boundary dependence (paper 1's label-free rule), refitted per format on its first 400 train problems by `blt_screens/dependence_screen.py`'s procedure | |
| R3 | 3% of positions forced at random among the detector's candidates, the rest by entropy (the candidate-only null) | |

- **At 10%:** every layout.
- **At 15% and 20% (M2):** E, D3, F2 and D3R.
- **At the matched rates:** E (orig at R_o*, prose at R_p*).
- **orig_is:** E at 10%.

That is about 35,000 BLT-1B forward passes.

**DET-1's variant.**
- **Why not the proposed rule.** The proposed rule, "forced starts may not displace entropy starts above a high
  entropy percentile", is D3 itself at a fixed budget.
  - D3's entropy fill already keeps the highest-entropy positions. So forced starts displace only the lowest entropy
    starts, those between E's threshold and D3's fill threshold.
  - The final answer's first byte is one of those in about 8% of problems. In R10's rows, entropy covers it in 80.0%
    of problems and the 3% hybrids in 71.3-73.1%.
  - No percentile can protect it without also protecting every displaced start, and that leaves no budget to force.
- **What can change** is where the displaced starts come from.
- **D3R.** Forced starts may displace entropy starts only inside the region the detector's candidates come from, the
  solution body.
  - In the question line and the final-answer line, every start of E is kept.
  - In the body: D3's forced starts, plus an entropy fill. The fill's threshold is fitted on train so that the body
    gets as many starts as E gives it.
- **Properties.** Its rate equals E's, and its question bits equal E's by construction. No start is forced after "The
  final answer is ". It uses no structure beyond the candidate region that R10-R20 already use (candidates stop before
  the final line).

**Verdicts (fresh problems; prose unless stated).**

Primary (O7):
- **C1, the ceiling:** of the values after `= ` or `$` (both writings) that E at 10% misses, the share with
  P(digit) >= 0.9 at their first digit. **Holds** if the lower bound is at least 0.70.
- **Q1, `$` values are blind in both writings:** E at 10%, coverage of `$` values. **Holds** if the upper bound is
  below 5% in each writing.
- **Q2, the cue within one text:** E at 10%, coverage of plain-space values minus coverage of `$` values. **Holds** if
  the lower bound is above +40 points.
- **Q3, a start pays where entropy is blind:** (with - E) on `$` values minus (with - E) on plain-space values, exact
  match. **Holds** if the lower bound is above +15 points.
- **D1, budget per value:** read from the point estimates of s_A and s_B; the intervals are reported.
  - **budget** if both are at least 0.67;
  - **cue** if both are below 0.33;
  - **mixed** otherwise.

DET-1 and the detector:
- **F1, the cost is the start:** the problems where D3 loses E's start at the final answer's first byte account for at
  least half of D3 - E on final answers (point estimates). The contribution of the other problems is reported with its
  interval.
- **F2, the cost removed:** D3R - E on final answers with the lower bound above -2 points (R20's N1), and D3R - D3 on
  computed values with the lower bound above -3 points (R20's N2). **Holds** if both.
- **W1, whole-text bits (DET-2):** D3 - E in bits per byte of the whole text. **Holds** (lower) if the interval is below
  0.
- **G1, higher budgets (M2):** D3 - E on computed values at 15% and at 20%. Each **holds** if its lower bound is above
  0.
- **K1, the candidate-only null (M3):** D3 - R3 on computed values. **Holds** if the lower bound is above 0.

Secondary:
- **S1, the `is ` swap (equation text):** coverage of the former `= ` values at 10%, orig_is - orig.
  - **unblinds** if at least +20 points with the lower bound above 0;
  - **no change** if below +10;
  - **partial** otherwise.

  If the median P(digit) after `is ` is at least 0.9, `is ` fixes the type as well, and S1 does not test the cue.
- **V1, a training-free fix:** VU - E on `$` values (exact) at least +20 points with the lower bound above 0, and VU - E
  on final answers with the lower bound above -2 points. **Holds** if both.

There are thirteen readings (G1 counts twice). Each is read on its own, with no correction for multiplicity, as in
R9-R20.

**Check before any reading.** E, D3 and F2 at 10% share R20's code path and seeds, so they must reproduce its fresh
numbers (prose: 21.7 / 59.0, 36.1 / 54.5 and 32.8 / 58.4). A mismatch is reported and investigated first.

**Predictions (written now).**
- C1, Q1, Q2 and Q3 hold (Q2 about +70 points, Q3 about +30).
- **D1 mixed:** R_o* is 15-25% and R_p* 4-7%.
  - s_A at least 0.67: giving equation text prose's budget per value lowers its threshold below log2 10, where capped
    values become visible.
  - s_B below 0.67: a higher threshold keeps capped values at 0.
- F1 and F2 hold: D3R keeps D3's gain within 2 points, at no final-answer cost.
- W1 does not hold in prose: whole-text bits are about equal, because the question's rise cancels the solution's fall.
  In equation text D3 - E is below 0 (reported).
- G1 holds at 15% and at 20%, with gains smaller than at 10% (about +8 and +4).
- K1 holds (D3 - R3 at least +8).
- S1 is partial (+10 to +20 points), and the median P(digit) after `is ` is between 0.7 and 0.9.
- V1 does not hold: VU gains on `$` values but costs final answers. Value uncertainty alone took equation-text final
  answers from 51.4% to 32.0% (`blt_window_sweep.json`).

**Reported without a verdict.**
- **Every layout, both writings:**
  - exact on computed values (overall, by cue, given covered or not);
  - final answers (coverage, exact given covered or not);
  - bits (question, body, final line, whole);
  - rate and forced share.
- **Seen before (see the disclosure):** D3 - E and F2 - E by cue.
- **Comparators:** in equation text D3 - eq, D3 - with and D3 - dep; in prose D3 - dep.
- **Final answers:** each layout's final-answer change, decomposed by what happened to the start (kept, lost, gained,
  neither).
- **The density readout:** exact match at the matched rates.
- **The patcher measures:** by cue, after `is ` and at final answers. orig_is coverage is also split by P(digit) after
  `is ` (above or below 0.9).
- **The pilot.**

**What paper 2 says, by outcome (fixed now).**
- **Q1 and Q2 hold:** C28, C31 and C38 say "without the `= ` cue", not "without the cue". In prose the `$` cue still
  blinds entropy, and the detector's prose gain is reported by cue.
- **C1:**
  - if it holds, the blind spot is stated as a ceiling for BLT-1B: a cue that fixes the type caps entropy near
    log2 10, which sits at its 10% threshold. It is not stated as a law for every entropy-gated method.
  - if it fails, the mechanism sentence is dropped and the split is reported descriptively.
- **D1:**
  - budget: E20 and C28 add "at a tight budget per value";
  - cue: R9 stands as a cue test at equal compute per byte;
  - mixed: both readings are given, with the matched-rate numbers.
- **F2:**
  - if it holds (on D3F, amendment A1), C37 and C38 say the final-answer cost is a displaced start in the final-answer
    line, removable at no cost, and R19/R20 shrink to a paragraph;
  - if it fails, C38 stands.
- **W1 does not hold:** C29, C31, C37 and C38 say "bits per byte of the solution", and "both objectives" is dropped.
- **V1 holds:** the training-free fused layout joins the detector section as a comparator.

Restated claims are drafted in a separate file. `paper2v2/manuscript.md` stays frozen until R2 returns.

**Cost.** $0: the Mac, about 3.5 hours, no download.

Script: `realblt_cue_budget.py` -> `results/blt_cue_budget.json` and `.log`. `R22_SMOKE=1` runs 40 train and 20 of R9's
test problems; its results are not used.

## Smoke test (2026-10-29; 40 train / 20 of R9's test problems; results not used)

- **The pipeline ran end to end in 14.2 minutes.**
- **R20 reproduction:** E, D3 and F2 reproduce R20's smoke rows exactly (0 exact-match differences; bits within
  0.00005).
- **Pilot gate:** printed MISMATCH, as expected with thresholds from 40 train problems (orig `$` 0.0%, `= ` 12.2%; prose
  `$` 0.0%, plain space 69.2%). The gate stops only the full run.
- **Rates:** every 10% layout between 0.097 and 0.102, except "with" (0.106 in equation text), which adds starts by
  design.
- **Train-fit diagnostics (40 problems):**
  - budget per value at 10%: 5.65 starts per value in equation text, 12.15 in prose;
  - R_o* = 19.3%, R_p* = 5.2% (matched on bytes per value: 14.4% / 6.9%);
  - D3's forced starts take 83% (equation text) and 55% (prose) of E's starts in the solution body, so D3R's body fill
    threshold rises to 4.19 and 3.65 bits (E's 10% threshold is about 3.3).

## Amendment A1 (2026-10-29, after the smoke test, before approval and before any fresh problem was scored)

- **Why.** D3R puts all of D3's displacement into the solution body. The train-fit diagnostic above shows that this
  thins the body far more than D3 does: D3 spreads the displacement over the whole text. That mixes a second effect
  into the DET-1 test.
- **Disclosed.** D3R's smoke numbers were seen before this change: on 20 of R9's problems, computed values were 8.5
  points below D3 in each writing.
- **D3F (added).** Forced starts may not displace entropy starts in the final-answer line. Everything before that line
  (the question and the body) absorbs D3's displacement, as in D3, with an entropy fill fitted on train so that this
  region gets as many starts as E gives it. So D3F is D3 with E's own starts kept in the final line, and nothing is
  forced there.
- **Changes to the rule:**
  - F2 is read on D3F (same bars: D3F - E on final answers above -2 points, D3F - D3 on computed values above -3).
  - D3R is reported with the same numbers, without a verdict.
  - D3F runs at 10%, 15% and 20%. That adds about 5,500 forward passes, for about 40,500 in all and about 4 hours.
- **Prediction (written now, after the smoke test):** F2 holds on D3F. The earlier prediction for D3R ("keeps D3's gain
  within 2 points at no final-answer cost") stands as written and is scored as reported.

## Results (2026-10-29; results/blt_cue_budget.json and .log; script at 2944f63; 180 minutes on the Mac)

**Checks.** Both passed:
- the pilot reproduced R9's split exactly: 0.0% / 17.8% / 0.0% / 74.3% of 167 / 544 / 163 / 548 values;
- E, D3 and F2 at 10% reproduced R20 on all 915 fresh problems: 0 exact-match differences, bits within 0.0001
  (prose 21.7 / 59.0, 36.1 / 54.5, 32.8 / 58.4).

The 10% thresholds fitted on train are 3.36 bits in equation text and 3.29 in prose; log2 10 = 3.32.

**By cue.** Fresh problems, BLT-1B, entropy at 10%. P(digit) and H are read at each value's first digit; exact match
is in %.

| writing | cue | values | entropy covers | P(digit), median | H, median (99th pct), bits | exact, E | exact, with |
|---|---|---|---|---|---|---|---|
| equation text | `$` | 586 | 0.0% | 0.996 | 2.76 (3.13) | 7.0 | 45.6 |
| equation text | `= ` | 1,862 | 18.9% | 0.966 | 3.00 (4.13) | 23.1 | 59.5 |
| prose | `$` | 575 | 0.0% | 0.997 | 2.77 (3.13) | 11.7 | 45.9 |
| prose | plain space | 1,873 | 75.5% | 0.384 | 3.63 (4.44) | 24.8 | 29.3 |

At the final answer's first digit (after "The final answer is "), the median P(digit) is only 0.25 in equation text
and 0.15 in prose. Entropy covers that digit in 94.1% and 77.0% of problems.

**Primary verdicts (O7).**
- **C1 HOLDS:** 92.2% [90.8, 93.5] of the 2,672 values after `= ` or `$` that entropy misses have P(digit) >= 0.9.
  By class: `= ` 86.3%, `$` in equation text 100%, `$` in prose 99.7%.
- **Q1 HOLDS:** entropy covers 0.0% of `$` values in both writings (586 and 575 values).
- **Q2 HOLDS:** in prose, plain space minus `$` is +75.5 points [+73.5, +77.6].
- **Q3 HOLDS:** +29.7 [+25.4, +34.3]. With - E is +34.3 on `$` values and +4.5 on plain-space values.
- **D1 mixed.** G_10 = 43.4 points; equation text covers 14.3% at 10% and prose 57.8%.
  - Equation text at R_o* = 18.5% has a threshold of 2.52 bits, below log2 10. It covers 82.8%, more than prose:
    G_A = -25.1, s_A = 1.58 [1.48, 1.68].
  - Prose at R_p* = 5.4% (threshold 3.74 bits) covers 30.2%: G_B = +15.8, s_B = 0.63 [0.60, 0.67].
  - Reported: rates matched on whole-text bytes per value (14.2% and 7.0%) give shares of 0.76 and 0.36, also mixed.

**DET-1 and the detector.**
- **F1 DOES NOT HOLD.** D3 - E on final answers is -4.5 [-6.4, -2.6].
  - In the 67 problems where D3 loses E's start at the final answer, E is right in 31.3% and D3 in 1.5%. These
    problems contribute -2.2 [-3.3, -1.3].
  - The other problems contribute -2.3 [-3.9, -0.7].
  - The lost start's share is 0.49, just under the 0.5 bar.
- **F2 DOES NOT HOLD (D3F).** D3F - E on final answers is -2.5 [-4.4, -0.8]; D3F - D3 on computed values is -0.2
  [-0.5, +0.2]. Keeping E's starts in the final line recovers +2.0 [+0.9, +3.1] of the cost.
  - Reported, D3R: -0.3 [-2.0, +1.3] on final answers. But D3R - D3 on computed values is -3.8 [-5.0, -2.6], which
    would fail F2's bar too.
  - D3R's question bits equal E's, by construction.
- **W1 DOES NOT HOLD.** D3 - E in whole-text bits per byte is +0.0085 [+0.0063, +0.0107]: the question rises by
  +0.048 and the solution falls by 0.010.
  - F2: -0.0027 [-0.0039, -0.0014].
  - Equation text: D3 -0.0134 [-0.0167, -0.0101], F2 -0.0188.
- **G1 HOLDS at 15% and at 20%:** D3 - E on computed values is +6.0 [+4.4, +7.6] at 15% and +0.9 [+0.04, +1.8] at
  20%.
- **K1 HOLDS:** D3 - R3 is +12.9 [+10.8, +15.1].

**Secondary verdicts.**
- **S1 unblinds.** For the former `= ` values, coverage rises from 18.9% to 65.7% (+46.9 [+44.5, +49.3]).
  - Among these values, the median P(digit) after `is ` is 0.736: `is ` does not fix the type, so S1 does test the
    cue. The script printed 0.847 because it also pooled the 586 values whose cue stays `$` (a labelling slip in
    `realblt_cue_budget.py`; recomputed from the stored rows in `test_r22`, and the reading is unchanged).
  - After the swap, coverage is 11.9% where P(digit) >= 0.9 and 81.5% where it is lower.
- **V1 DOES NOT HOLD.** VU - E is +36.3 [+31.8, +41.2] on `$` values but -5.0 [-7.2, -3.0] on final answers. VU
  covers the final answer in 66.4% of problems; entropy covers it in 77.0%.

**Reported without a verdict.** Points; 95% intervals in `diff` in the JSON.
- **Detector by budget, computed values (final answers):**

  | | 10% | 15% | 20% |
  |---|---|---|---|
  | prose, D3 - E | +14.4 (-4.5) | +6.0 (-0.4) | +0.9 (-0.8) |
  | prose, F2 - E | +11.1 (-0.7) | +5.4 (-0.7) | +0.8 (-0.5) |
  | equation text, D3 - E | +54.3 (+14.6) | +29.6 (+3.0) | +8.5 (+0.7) |

  At 20%, entropy covers 92.2% of prose `$` values.
- **Comparators in equation text** (computed values):
  - D3 - eq: +20.0 [+17.9, +22.0];
  - D3 - with: +17.4 [+15.5, +19.5];
  - D3 - dep: +2.0 [+0.2, +3.7]. Dependence never starts at the final answer there: final answers 2.3% against E's
    48.5%.
- **Comparators in prose:**
  - D3 - dep: +2.6 [+0.5, +4.7]. Dependence covers every final answer (62.8%), but costs +0.155 bits per byte on the
    question.
  - D3 - with: +2.9 [+1.3, +4.5].
- **By cue, prose (seen before, see the disclosure):**
  - D3 - E: +47.1 on `$` values and +4.4 on plain-space values;
  - F2 - E: +37.2 and +3.0;
  - VU - E: +36.3 and -1.7.
- **E at the matched rates:** equation text at 18.5% reaches 66.9% exact; prose at 5.4% reaches 4.9%.

**Predictions.**
- **Right:**
  - C1, Q1, Q2 (about +70; +75.5 measured) and Q3 (about +30; +29.7);
  - D1 mixed, with R_o* and R_p* in the predicted ranges, s_A >= 0.67 and s_B < 0.67;
  - W1 (in equation text D3 - E is below 0);
  - G1, though both gains are smaller than predicted (+6.0 vs +8, +0.9 vs +4);
  - K1;
  - the median P(digit) after `is ` (0.736);
  - V1 (fails on final answers).
- **Wrong:**
  - F1 (0.49);
  - F2 on D3F;
  - D3R's prediction: no final-answer cost, but it gives up 3.8 points;
  - S1 "partial": it unblinds, +46.9.

**Reading.**
- **The blind spot after a type-fixing cue is a ceiling.**
  - At the values entropy misses after `$` or `= `, the patcher is at least 90% sure a digit comes. After `$`, the
    entropy never reaches the 10% threshold (99th percentile 3.13 bits).
  - Within one prose text, at one budget, values after `$` get 0% and values after a plain space 75.5%.
  - Swapping `= ` for `is ` lowers P(digit) and unblinds the values, and coverage after the swap tracks P(digit).
  - So "without the cue" in R9, R12 and R20 means "without `= `". The `$` cue survives the rewrite, and the
    detector's prose gain sits on it.
- **Budget per value matters too.**
  - Giving equation text prose's budget per value pushes its threshold below log2 10, and it then covers more values
    than prose does.
  - Taking prose down to equation text's budget per value closes about two thirds of R9's gap.
  - The ceiling is what makes the budget matter this steeply. The density-free part of the cue effect is the
    within-text contrast.
- **The detector's prose gain is a tight-budget effect.** It is +14.4 at 10%, +6.0 at 15% and +0.9 at 20%, because
  entropy's threshold falls below the ceiling and covers `$` values. In equation text the gain stays large.
- **DET-1.** The final-answer cost is two displaced-start effects of about equal size: the start at the final answer
  itself (D3F recovers +2.0) and starts in the question.
  - Keeping both (D3R) removes the cost, but the body then absorbs all the displacement and the detector gives up
    3.8 points.
  - No variant here keeps the whole gain at no cost. The fused ranking (F2: +11.1, -0.7) and D3R (+10.6, -0.3) sit at
    the same trade-off.
- **DET-2.** On the whole text, the 3% detector costs bits in prose and lowers them in equation text. "Lower bits per
  byte" holds in prose only for the solution.
