# R18: a patch start at the second digit where entropy stays high. Decision rules

Written 2026-10-28, before any layout in this test was run.

**Why.** R17 (descriptive): at a number's first digit BLT-1B's patcher entropy is alike for computed values and
copies, but from the second digit copies become predictable and computed values do not (computed vs copy AUC 0.84 at
the second digit, 0.90 at the third; no training). A start at the second digit is one byte late for the first digit,
but could still help the rest of the number. Asked by the PI to test it.

**Rule (label-free, causal).** At byte i, a *second digit position* is one whose previous byte is a digit that begins
a number (the byte before it is not a letter or digit), inside the solution (after the question, before the final
answer line). The decision uses only bytes before i: the patcher's entropy for byte i, H(i). **Persistence rule P:**
start a patch at every second-digit position with H(i) > tau, where tau is the median of H over second-digit
positions in the format's train problems (no labels). The rest of a 10% budget is filled by entropy, as in R12.

**Layouts** (BLT-1B, 10% of bytes, forced starts plus entropy for the rest; thresholds fitted on each format's 1,382
train problems, evaluated on its 275 test problems):
- **E**: entropy only;
- **P**: the persistence rule;
- **P_all**: a start at every second-digit position (control: does the entropy filter matter?);
- **D**: R12's label-free detector at the decision point, its top 3% forced (reference);
- **D+P**: both forced sets (does the late signal add to the early one?);
- **O1, O2** (label oracles, upper bounds, not rules): starts at the first byte, or at the second byte, of every
  computed value.

**Measures.** Exact match on computed values (all bytes right, teacher-forced, as R12); exact match on the rest of the
value after its first byte ("tail"); final answers; bits per byte of the solution; patch rate. 2,000 bootstrap
resamples of test problems, differences paired.

**Verdicts (prose test; the equation test reported alongside).**
- **T1, a late start can help at all:** O2 - E on computed values. **Holds** if the interval is above 0.
- **T2, the label-free persistence rule helps:** P - E on computed values. **Holds** if the interval is above 0.
- **T3, it adds to the detector:** (D+P) - D on computed values. **Holds** if the interval is above 0.

**Predictions (written now).** T1 holds but far below O1 (a wrong first digit, predicted without a fresh step, stays
wrong under teacher forcing; a late start can only fix the rest); T2 holds, smaller than D; T3 does not hold (the
detector's forced starts already cover most computed values). P beats P_all on computed values.

**Reported without a verdict.** Tail exact match for every layout; P - P_all; the share of P's forced starts that
fall in computed values, other new numbers and copies.

Script: realblt_second_digit.py -> results/blt_second_digit.json (R18_SMOKE=1: 40 / 20 problems, not used).

## Smoke test (2026-10-28, 40 / 20 problems; results not used)

The pipeline ran end to end; no change.

## Results (2026-10-28; results/blt_second_digit.json, .log)

tau (median entropy at second digits, train): prose 1.22, equation text 1.47 bits. P forces 0.8% of tokens in prose
(41% at computed values, 14% other new numbers, 45% copies); P_all 1.6%.

Prose test, BLT-1B at 10% (exact match on computed values; "tail" = the value after its first byte; differences in
points with 95% intervals):

| layout | exact | tail | final answers | bits/byte | vs E, exact | vs E, tail |
|---|---|---|---|---|---|---|
| E entropy | 21.7 | 48.1 | 62.2 | 0.799 | | |
| P persistence | 25.3 | 64.4 | 62.9 | 0.798 | **+3.7 [+1.3, +6.1]** | +16.3 |
| P_all every 2nd digit | 27.6 | 68.4 | 61.5 | 0.798 | +5.9 [+3.1, +8.9] | +20.3 |
| D detector | 38.4 | 61.5 | 57.5 | 0.787 | +16.7 [+12.7, +20.9] | +13.4 |
| D+P | 40.9 | 71.1 | 54.5 | 0.798 | +19.2 | +23.0 |
| O1 oracle, 1st byte | 33.1 | 57.1 | 62.2 | 0.794 | +11.4 | +9.1 |
| O2 oracle, 2nd byte | 25.7 | 64.2 | 63.3 | 0.798 | **+4.1 [+2.2, +6.1]** | +16.1 |

- **T1 HOLDS** (O2 - E +4.1 [+2.2, +6.1]), far below a first-byte start (O1 +11.4), as predicted.
- **T2 HOLDS** (P - E +3.7 [+1.3, +6.1]), smaller than the detector (+16.7), as predicted.
- **T3 HOLDS, against the prediction:** (D+P) - D +2.5 [+0.3, +4.7]; the late start adds a little to the detector
  (tail +9.6 [+6.9, +12.4]). Final answers fall with every forced number layout at a fixed 10% budget (D 57.5,
  D+P 54.5 vs E 62.2), as starts move away from other positions.
- **The entropy filter does not help, against the prediction:** P - P_all -2.3 [-4.6, +0.1] exact, -4.0 [-6.9, -1.0]
  tail. Starting at every second digit costs only 1.6% of tokens and does at least as well.
- Equation text (reported): P - E +6.2, P_all - E +7.6, D - E +54.6, (D+P) - D +2.1 [-0.3, +4.5]; P_all lowers bits
  per byte (0.755 vs 0.814).

**Reading.** A start one byte late helps the rest of a number a lot (tail +16 to +20 points) but whole values only a
little (+4 to +6), because the first digit is already decided. R17's persistence shape is real but not needed as a
rule: positions inside numbers are few enough to cover all of them. The loss that matters is the first digit, which
only an early signal (the detector) fixes.
