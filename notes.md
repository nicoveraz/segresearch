# Lab notebook

## Current beliefs (keep this section updated)

- **MAIN_STEPS raised 2000 -> 8000 (human-approved, commit 6fab685).** At 2000 steps seed noise (~0.5 bits) swamped every rule difference and no mask learned ANS_LONG. At 8000 the baseline spans 1.23-1.28 over 3 seeds. Results from before this change are not comparable to results after. (high confidence)
- **Alignment dominates local sums.** Oracle record-starts (8.5% of bytes) solves local sums (0.02-0.05 bits) on 2/3 seeds vs 0.39-0.42 for the 25% entropy baseline. The third seed failed to learn them (1.28), so occasional non-learning runs remain. (medium-high)
- **Allocation is what moves long-range answers.** ANS_LONG only drops (2.12-2.31 vs 2.45-2.59) when a patch starts exactly at each answer (oracle rec+ans1). Record starts alone do not help long answers. First evidence that the two mechanisms separate: alignment for local, a boundary at the answer for long-range. (medium, 2 seeds)
- Working practice: because some runs fail to learn local sums outright, I replicate any candidate that would be kept with --seed 1, not only changes under 0.02 bits.

## Log

### Setup (sep26)
MLX port, reference at 64k steps; reference ANS_LONG 0.09 / 0.26 / 0.16 bits (A / B / C). One run = 51s on GPU.

### Baseline and reference rules (seed 0, val, 25% budget)
| rule | ans | local | long | rate |
|---|---|---|---|---|
| blt_entropy (baseline) | 2.436 | 2.268 | 2.678 | 0.245 |
| stride | 2.598 | – | 2.635 | 0.250 |
| excess_vs_reference | 2.603 | – | 2.680 | 0.250 |
| trajectory_gated | 2.638 | – | 2.739 | 0.248 |

Baseline mechanism: on format A it spends the budget on HEX (start rate 1.00) and never starts at answers (0.00); on B answers get 0.77. Excess-vs-reference does hit answers (0.95–1.00) and is not better — pilot finding reproduced (detecting learnable bytes ≠ good boundaries).

### Oracle diagnostics (use role labels; NOT a usable rule — ceilings only)
| mask | rate | ans (s0 / s1 / s2) | long |
|---|---|---|---|
| record starts | 0.085 | 1.957 / 2.286 / 2.020 | ~2.63 |
| record starts + first byte of long answers | 0.093 | 2.586 / 1.765 / 2.384 | ~2.60 |
| record starts + first byte of every answer | 0.105 | 2.026 | 2.651 |
| first byte of every answer only | 0.020 | 2.630 | 2.648 |
| baseline (entropy) | 0.245 | 2.436 / 2.080 / – | ~2.65 |

Interpretation: seed variance (up to ~0.8 bits on local sums for one mask) dominates every between-mask difference except "answers only", which is clearly bad. Long answers are never learned.

Next: testing whether a longer harness training budget (8000 steps, scratch override, no repo change) reduces variance and lets ANS_LONG be learned. If so, propose raising MAIN_STEPS to the human (fixed file).

### Longer harness budget (8000 steps; scratch override before the change was committed)
| mask | ans s0 / s1 / s2 | local | long |
|---|---|---|---|
| baseline (entropy) | 1.276 / 1.231 / 1.254 | 0.39-0.42 | 2.45-2.50 |
| oracle record starts | 1.026 / 1.090 / 1.761 | 0.016 / 0.048 / 1.281 | 2.45-2.59 |
| oracle record starts + answer starts | 0.987 / 0.892 / – | 0.04-0.07 | 2.12-2.31 |

Variance of the baseline dropped ~7x; local sums get learned; long answers start to respond, only to an answer-start boundary. Human approved raising MAIN_STEPS to 8000. One run now takes ~6.5 min.

### Official baseline at 8000 steps (seed 0)
val_ans_bits 1.2691, local 0.435, long 2.471, rate 0.245. Per-role starts unchanged from before (same mask): A spends everything on HEX (1.00) and never starts at answers.

### H1: entropy jump instead of entropy level
Hypothesis: score = H[i] - H[i-1] (small model, final checkpoint). A patch starts at the first byte of each unpredictable field rather than throughout random ids. Expect HEX start rate << 1.00; TEXT/STRUCT (record starts) and answer start rates up; local bits below 0.435; long maybe slightly better since answers now get starts.
Result: KEEP. val_ans_bits 1.180 (s0) / 1.138 (s1), mean 1.159 vs baseline ~1.250 (1.269 s0, 1.231 s1 from scratch check).
Mechanism, as predicted: record starts 0.99, first answer byte 1.00, inside hex ids ~0 (baseline: 1.00). The whole gain is long-range: ANS_LONG 2.20 / 2.08 vs ~2.47. Local sums slightly worse (0.47-0.48 vs 0.435), plausibly because VAR/VALUE/OPERAND also get starts (0.6-1.0), splitting local sums into more patches.
An entropy-level rule wastes the budget inside noise; an entropy-jump rule marks the first byte of each unit and gives answers a fresh global context. Consistent with the oracle result that long answers need a boundary at the answer.

### H2: entropy jump at a 15% budget (bool mask, threshold fitted on train)
Screen (no training): record starts 0.90 (A) / 0.99 (B), answer starts 1.00, VAR/VALUE/OPERAND starts drop to 0.17-0.50. At 10%, A loses all answer starts, so 15% is the floor.
Hypothesis: the extra starts at 25% are not needed. Expect val_ans_bits equal or better than H1 (local sums less split), long about the same, at 60% of the compute.
Result: KEEP. 1.154 (s0) / 1.175 (s1), mean 1.164 vs H1 1.159: equal within noise at 60% of the compute (rate 0.150).
Local improved (0.41-0.42 vs 0.47-0.48): fewer starts inside local sums. Long slightly worse (2.23-2.26 vs 2.08-2.20); the dropped VAR/VALUE starts may have been helping long-range answers a little. Watch this.

### H3 (negative test): reference-model entropy jump at 15%
Screen: record starts 0.92 (A) / 1.00 (B) but answer starts 0.00 / 0.05 — the strong model already knows the answers, so its entropy does not jump there. Same family, same budget as H2; differs mainly in dropping answer starts.
Hypothesis (alignment-only for long-range should fail): local ≈ H2 (~0.4), long back to baseline level (~2.45+), val_ans_bits worse than H2. If long stays ~2.2 instead, answer starts are not what drives long-range answers and my current belief is wrong.
Result: DISCARD (as intended). 1.348 vs H2 1.154 (seed 0). Answer starts 0.00/0.06; long 2.49 = baseline level; local 0.56 (worse than H2's 0.41).
**Prediction confirmed: alignment alone does not help long-range answers.** Same family and budget as H2, record starts equally covered (0.92/1.00), and the long-range gain vanishes completely once answers lose their patch starts. Answer starts also help local sums somewhat (0.41 vs 0.56).
Side note: the reference model's entropy is a clean record-boundary detector (it has learned the answers, so its jumps sit only at unit starts), but that is exactly why it is a bad patching signal on its own.

### Oracle diagnostics at 8000 steps: what else helps long-range answers?
H1 (25%, many VAR/VALUE starts) had slightly better long (2.08-2.20) than H2 (2.23-2.26). Testing oracle masks record starts + answer starts + {first VALUE byte | first VAR byte} to see which extra placement helps long before designing a rule for it.
Result (seed 0): record+answer starts: local 0.07, long 2.31, ans 0.987. **+ value starts**: local 0.05, long 2.53, ans 1.063. **+ variable starts**: local 0.07, long 2.37, ans 1.010.
Neither extra placement helps long; value starts hurt it. Long seems capped around 2.1-2.3 at this harness budget regardless of mask. The larger remaining gap is local sums: oracles ~0.05 vs H2 0.41.

### H4 screen: separate alignment (reference jump) + allocation (small-model jump where reference is confident)
Allocation half works: answer starts 0.96-1.00, nothing else. Alignment half fails: reference jump catches only ~57% of record starts at a 9% budget, because hex/value/operand/variable starts jump as much or more. Not run.
Feature diagnostic (roles used only to rank features): no available signal (entropies, jumps, previous-byte entropy/surprisal, early-checkpoint entropy, previous-byte frequency or branching entropy from train counts) cleanly separates record starts from other field starts. Every field start looks the same: a big jump after a fully predictable byte. Weak hint: the early (step-200) checkpoint entropy is lower at record starts (3.6-3.8 vs 4.2-4.5).
Next: oracle record+answer+operand starts, to test whether H2's operand starts are what cost it on local sums.
