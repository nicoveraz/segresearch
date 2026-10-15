# Lab notebook

## Current beliefs (keep this section updated)

**Best rule: H12 (commit 6847b72)** — patch after a separator learned from the train split, plus at the most learnable unit starts (small-model jump > 1 bit, reference confident, small-minus-reference gap above an Otsu split learned on train). **val_ans_bits 0.708 (2 seeds, 16k) = the label-using oracle (0.717)**, at ~10% of bytes; baseline 1.17 at 25%. Not yet through the final test: the learned separator is the part most at risk on held-out format C.

**Headline: keep every computation's inputs inside one patch, and give the output a fresh global step.** H12 is the direct implementation of that principle and reaches the oracle.
- Local sums: operands in one patch (operand starts cost 0.4-0.55 bits, 2 seeds at 8k).
- Long-range answers: each stored fact in one patch (value starts: A long 1.08 -> 2.03) AND the query's variable names in one patch (variable starts: A long 1.08/1.32 -> 2.14/2.00, 2 seeds) AND a start at the answer (removing it: long back to baseline, 2 seeds). Only the clean record+answer oracle learns long answers (A 1.08 / 1.32 at 16k); B has not learned them by 16k under any mask.
- At 8000 steps every mask plateaus ~2.2 on long and this is invisible; hence MAIN_STEPS 16000.
- So the answer to "alignment or allocation?" here: **alignment to units of meaning is the dominant factor for both short- and long-range dependencies; allocation of a fresh global step to the answer is a necessary complement for long-range ones.** Allocating global steps to hard bytes per se (baseline entropy, excess-vs-reference, every answer digit) does not help.

- **MAIN_STEPS raised 2000 -> 8000 (human-approved, commit 6fab685).** At 2000 steps seed noise swamped every difference and nothing learned long answers. Earlier numbers are not comparable. (high)
- **Local sums are an alignment problem.** They are solved (~0.04 bits) when each record is one patch and the answer starts its own patch. Splitting the operands into separate patches costs ~0.4 bits (oracle 0.07 -> 0.49), which is exactly H2's remaining deficit (0.41). Patches must also stay record-sized: at ~40 bytes per patch (H5) sum-pooling blurs the operands and local sums collapse to 2.25. (high)
- **Long-range answers need a patch start at the answer, and that is not enough on its own.** Removing answer starts while keeping record starts (H3) returns long to baseline (2.49). With answer starts plus complete record starts (oracles, H1, H2), long reaches ~2.1-2.3; with answer starts but patchy record starts (H5, H6) it stays at ~2.6. Long plateaus around 2.1-2.3 for every mask tried, probably the 2-layer global model's limit on cross-patch retrieval at this budget. (medium: long is noisy, up to ~0.35 bits between seeds for the same mask)
- **So: alignment and allocation are complementary, not rival.** Alignment (record-sized patches with a computation's inputs together) drives local sums; allocation (a fresh global step at each answer) is necessary for long-range answers. The entropy jump gets both cheaply because it marks the first byte of every unit, which includes records and answers. (medium-high)
- **Detecting learnable bytes is not the same as good boundaries** (pilot finding reproduced: excess-vs-reference 2.60 at 2000 steps). But the strong model is useful as a *filter*: "small-model jump while the strong model is confident" finds answer starts almost perfectly (0.96-1.00, nothing else).
- **What blocks further progress:** no available signal separates record starts from operand/value/variable starts (all look like "big entropy jump after a predictable byte"). Rules that trade record coverage for fewer operand starts lose more than they gain (H6). Min-gap suppression works only with a format-tuned gap, likely to fail on held-out C.
- Practice: every kept rule is replicated on >= 2 seeds; long-range differences under ~0.3 bits on one seed are treated as unproven. Rules only use signals at positions <= i, so the mask cannot leak future bytes (and avoid surprisal, which would leak byte i itself).

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
Result (seed 0): record+answer+**operand** starts: local 0.49 (vs 0.07 without operand starts), long 2.12, ans 1.160 — right at H2's level. **Operand starts are what cost H2 on local sums.** Splitting a sum's inputs across patches hurts even though the global model can see every earlier patch.
H5 screen (early-checkpoint entropy to pick record starts among field starts): rec starts only 0.65-0.74 on A, prefers VAR starts. Worse than H2. Not run.
Principle I'm keeping: rules only use signals at positions <= i (causal), so the mask cannot leak future bytes into the model.

### H5: answer starts only (~2.5% of bytes)
Rule: start where the small model's entropy jumps by > 1 bit AND the reference entropy < 1 bit (learnable unit start). Screen: answer starts 0.96-0.99; nothing else except VAR 0.15; rate 0.030 / 0.022.
Hypothesis: local sums improve (operands never split; A's baseline without answer or operand starts already had local 0.125). Long gets worse: with ~2.5% starts each patch sum-pools ~40 bytes, which should blur the stored values long answers must retrieve. Tests how low the rate can go.
Result: DISCARD. 2.401 (seed 0), rate 0.026. Local collapsed too (2.25), not only long (2.62). **Prediction wrong on local.** With ~40 bytes per patch, sum-pooling blurs the operands: the answer's global context no longer carries them cleanly.
Mechanism (sharpened by H5 + operand oracle): a computation's inputs are cheap to use when they sit together in one small patch; across patches the 2-layer global model must retrieve and combine them, which it does poorly. Long answers always need cross-patch retrieval, which is why they plateau ~2.1-2.3. Patches must also stay small (record-sized) for pooling to preserve content.

### H6 screens
- Suppression window (after a random-field start, block further random-field starts for K bytes; always allow learnable starts): works only for some K and the best K differs by format (A: K=8 gives record starts 0.93, operands only at record starts; B: 0.67). Choosing K amounts to fitting record length; likely to break on held-out C. Not run.
- Top-X% jumps + forced learnable starts: at X=10%, rate 0.13/0.11; record starts 0.84 (A) / 0.61 (B); operands 0.21 / 0.41; answers 0.99-1.00. Running as H6.

### H6: top 10% entropy jumps + all learnable starts (small-model jump > 1 bit with reference < 1 bit)
Hypothesis: fewer operand starts (A) should help local sums; fewer record starts (B) should hurt them. Expect roughly H2's level (~1.16) at lower compute (~12%). If worse, record-start coverage matters more than avoiding operand starts.
Result: DISCARD. 1.529 (seed 0), rate 0.118. A: local 0.19 (better than H2's 0.34, fewer operand starts, as predicted), long 2.58. B: local 1.41 (record starts 0.61, as predicted), long 2.58.
**Surprise: long went back to baseline level on both formats although every answer still gets a start.** Record-start coverage by record kind on A is similar to H2 (0.83-0.88 vs 0.88-0.91), so it does not explain A's long drop. The clearest A difference is value starts (H2 0.30, H6 0.00), but the oracle said value starts hurt long. Either long is noisier than assumed (runs cluster at ~2.1-2.3 "partly learned" vs ~2.5-2.6 "not learned", which looks bimodal), or value starts matter in context.
Evidence so far on what long needs: answer starts are necessary (H3, H5, and H6's missing record coverage all ~2.5-2.6 without full record+answer coverage), but not sufficient. Confidence on long-range claims lowered until replicated.
Next: H2 with seed 2 to measure long's run-to-run noise.
Result: H2 seed 2: 1.196, long 2.320 (A 2.40, B 2.24). **Long is not bimodal for H2**: 2.23 / 2.26 / 2.32 over three seeds (A 2.28-2.40, B 2.18-2.24). Run-to-run noise on long is ~±0.1, so H6's 2.58 on A is likely a real drop. H2 3-seed mean 1.175.
Leading suspect for H6's A-long drop: value starts (H2 starts ~60% of values on A, H6 none). This contradicts the single-seed oracle (value starts hurt long, 2.53 vs 2.31). Replicating that oracle with seed 1.

### Oracle replication, seed 1
record+answer: long 2.14, ans 0.900. + value starts: long 2.19, ans 0.922. **Value starts make no clear difference** (seed 0 said they hurt: 2.53 vs 2.31). The same mask moved long by 0.34 bits across seeds, so single-seed long comparisons are unreliable. H6's A-long drop is not explained by value starts and may be partly noise.

### Screen: entropy jump from earlier patcher checkpoints (15%)
Step-200 model: record starts 0.03-0.05, answers 0.40 (A). Step-1000: record starts 0.54 (A) / 0.89 (B). Final (step 4000, = H2): 0.90 / 0.99. The jump only marks unit starts once the small model has learned the templates, so the final checkpoint is best. The entropy-jump family looks saturated at H2.

### Final test run (human request, before going to sleep)
Human asked me to run `test_final.py` here and keep advancing. To keep the held-out test clean I saved the output to `test_final.log` **without reading it** and will not read it while I keep searching. Tested rule: H2 at commit in `test_final.commit`. Any rule found after this point has NOT been through the final test.

### H7: H2 + a patch start at every learnable byte (small-model entropy exceeds reference by > 1 bit, reference < 1 bit)
Screen: rate 0.186 / 0.176; record starts 0.90 / 0.99 (as H2); answer bytes covered 0.81 (H2: 0.38, first digit only).
Hypothesis: long plateaus at ~2.2 because later answer digits are predicted with a stale global state. Giving every answer digit a fresh global step should lower long (target < 2.1 on both seeds). Local may get slightly worse (the answer's digits are split from each other) or better (each digit gets global context). This is excess-vs-reference used as an *addition* to alignment instead of a replacement.
Result: DISCARD. 1.188 (seed 0), rate 0.181. Long 2.29 — inside H2's range (2.23-2.32). **Hypothesis wrong**: a fresh global step at every answer digit does not help. The long plateau is not stale context for later digits; the global model struggles to retrieve stored values at all. More compute, no gain.

### H8 screen: suppress jumps at a fixed offset from the previous jump (learned per 2-byte left context in fit())
Idea: fields inside a template sit at fixed distances; record starts follow variable-length records. Screen: operands still 0.19-0.46, record starts drop to 0.78-0.83. Fails because 2-byte contexts are shared by fields at different offsets (in B, "s " precedes both the second operand, 8 after the first, and the second query variable, 7 after the first). Not run. Every direction in program.md has now been tried; the entropy-jump family looks saturated at H2.

### Overnight plan: replicate the single-seed claims
H3 (seed 1), operand oracle (seed 1), H6 (seed 1), baseline (seed 2). Rules evaluated from saved copies via a scratch script; boundary.py stays H2.
Replications (seed 1):
- H3 (no answer starts): 1.312, long 2.43, local 0.54. vs H2 seed 1: 1.175, long 2.26, local 0.42. **Confirmed on 2 seeds**: dropping answer starts costs long ~0.2-0.25 bits and local ~0.1-0.15.
- Oracle record+answer+operand: local 0.60 (vs 0.04 without operand starts, same seed). **Confirmed on 2 seeds**: operand starts cost local sums 0.4-0.55 bits.
- H6 seed 1: 1.504, long 2.46 (A 2.44), B local 1.49. **H6's long drop confirmed** (vs H2 2.23-2.32); cause still unexplained (not value starts, per the oracle replication). Open question.
- Baseline seed 2: 1.256. **Baseline 3-seed mean 1.252 vs H2 1.175**: H2's lead (~0.08 bits) holds.
- Side finding: the baseline gets A's local sums to 0.13 on every seed (better than H2's 0.34). On A it never starts at answers, so each sum's answer sits in the same patch as its unsplit operands and the local decoder computes it. Local sums work either way as long as operands are not split.

### Diagnostic: entropy at the byte before a unit start
Only ~13-17% of record starts have a slightly uncertain previous byte (small model > 0.05 bits, e.g. where an answer's length is uncertain); the rest look exactly like field starts. The strong model is ~0 before every start. **Not a usable record-start detector.** The blocker stands.

## Where things stand (end of overnight session)
- Best rule: H2 (tested by test_final.py; output in test_final.log, unread by me).
- Open questions: why H6 loses long on A; whether any label-free signal can separate record starts from operand starts (all 8 attempts failed); whether long can go below ~2.1 with any mask at this harness budget.

### After the final test
test_final.py finished (exit 0, 6 lines). Output still unread by me.

### Diagnostic: is the long plateau (~2.1-2.3) a budget limit or a capacity limit?
H2 and oracle record+answer at MAIN_STEPS=16000 (scratch override; repo stays at 8000), seed 0. If long falls well below 2.1 for both, the plateau is the training budget; if it stays ~2.2, it is the 2-layer global model's capacity.
Result (seed 0, 16000 steps): H2: ans 1.100, local 0.36, long 2.17 (A 2.35, B 1.98). Oracle record+answer: ans 0.673, local 0.02, **long 1.61 (A 1.08, B 2.15)**.
**The 8000-step long plateau was a training-budget limit, and it was hiding a big difference.** With more training, the clean aligned mask starts learning long-range answers (A: 1.08), while H2 barely moves. So long-range answers need alignment too, not only a start at the answer.
New hypothesis: retrieval needs each *stored fact* (assignment: variable + value) in one patch. H2 starts patches at values/variables (A: VALUE 0.30, VAR 0.17), splitting facts across patches. Test at 16000: oracle record+value+answer (splits var from value; predict A long well above 1.08) and oracle record+answer seed 1 (is 1.08 repeatable?).
Result (16000 steps): oracle record+**value**+answer, seed 0: long **A 2.03** (vs 1.08 without value starts), B 2.48 (vs 2.15); ans 0.939 vs 0.673. Oracle record+answer, seed 1: long A **1.32**, B 2.33; ans 0.760.
**Stored-fact hypothesis supported (paired, seed 0, both formats): splitting an assignment's variable from its value wrecks long-range learning.** The clean oracle's long-range learning on A is repeatable (1.08, 1.32); B has not learned long by 16000 on either seed.
Recommendation for the human: at MAIN_STEPS=8000 every mask plateaus near 2.2 on long, which hides this effect. Raising MAIN_STEPS to 16000 (~13 min/run on GPU) would let the loop see it. Not changed (fixed file; human asleep).

### H9 screen: keep jumps only after bytes that lead into varied entropy patterns (learned per preceding byte in fit())
Some settings do exactly the right thing on A (value starts 0.00, record starts 0.90, answers 1.00) but on B drop every record start (the space before operands/values/answers leads into more varied patterns than the separator). Not robust; not run. Ninth failed attempt at a label-free record-vs-field detector.

### 16000-step comparison: baseline (seed 0) and H2 (seed 1)
Does H2's lead over the baseline hold with more training, and is H2's 16k result repeatable?
Result (16000 steps): baseline seed 0: 1.170 (local 0.32, long 2.39). H2 seed 1: 1.050 (local 0.36, long 2.04); with seed 0 (1.100) H2 mean 1.075. Oracle record+answer mean 0.717 (0.673 / 0.760).
**Ranking holds with more training and the gaps widen**: baseline 1.17 > H2 1.08 > oracle 0.72. H2 beats the baseline on long (2.04-2.17 vs 2.39). The remaining H2-to-oracle gap (~0.35 bits) is the alignment H2 cannot find: values and operands split out of their records.

## Summary for the human (morning)
1. **Best label-free rule: H2** (entropy jump, top 15%). Beats the BLT-entropy baseline at 8000 steps (1.175 vs 1.252, 3 seeds each) and at 16000 (1.08 vs 1.17), with 40% less compute. Final test on held-out formats ran on H2 (`test_final.log`, unread by me).
2. **Main finding: alignment dominates for both local and long-range dependencies.** Local sums need their operands in one patch; long-range answers need each stored fact (variable + value) in one patch, plus a patch start at the answer. Allocation of global steps to hard or learnable bytes per se does not help (baseline, excess-vs-reference, H7).
3. **The 8000-step harness hides the long-range effect** (every mask plateaus ~2.2). Recommend raising MAIN_STEPS to 16000 (~13 min/run) if the loop continues.
4. **Open problem:** a label-free signal that separates record starts from mid-record field starts. Nine attempts failed; all entropy-based signals see every field start the same way. This is where the remaining ~0.35 bits (at 16k) are.

### MAIN_STEPS raised to 16000 (human-approved, commit cdb8fa9)
New reference numbers (from the 16000-step scratch runs, same code path): baseline 1.170 (s0); **H2 1.100 / 1.050, mean 1.075**; oracle record+answer 0.673 / 0.760. All earlier results.tsv rows are 8000-step and not comparable. One run ~13 min.
Goal now: close the H2-to-oracle gap by avoiding value/operand starts without losing record starts.

### H10 (16k): top 13% entropy jumps + every learnable start
Screen: A: record starts 0.90, VALUE 0.01, OPERAND 0.25, answers 1.00 (close to the oracle). B: record starts 0.85, VALUE 0.50, OPERAND 0.46. In B values jump more than record starts, so no threshold drops values without dropping records first; in A the reverse.
Hypothesis: at 16k, keeping values with their variables should pay off on A's long answers (H2 A long 2.10-2.35 -> clearly lower); B slightly worse (record starts 0.85 vs 0.99). Net: better than H2 if the stored-fact effect is as large as the oracle suggested.
Result: DISCARD. 1.238 / 1.128, mean 1.183 vs H2 1.075. A: local 0.18 / 0.19 (better than H2's 0.31-0.32: fewer operand splits), but **A long 2.40 / 2.19 — same as H2 (2.35 / 2.10) despite values now staying with their variables.** B: local 0.89 / 0.73 (record starts 0.85), as predicted.
Prediction on A long was wrong: unsplit stored facts are not sufficient. Remaining A differences from the oracle: VAR starts 0.33 (incl. the second variable of each query), 10% of record starts missed, hex-id starts.
New hypothesis: queries need their *inputs* (the two variable names) in one patch, just as local sums need their operands together. Test: oracle record+answer+variable starts at 16k. Prediction: A long far above the clean oracle's 1.08.
Result (16k): oracle record+answer+**variable** starts: A long 2.14 (s0) / 2.00 (s1) vs 1.08 / 1.32 without variable starts; local stays 0.02-0.04; ans 0.956 / 0.847 vs 0.673 / 0.760.
**Confirmed on 2 seeds: variable starts wreck long-range learning.** Assignments stay "z=26" together under this mask, so the damage is most likely the query split ("z+" | "x="): the query's two inputs land in different patches — the operand problem again.
This explains H10: it kept values with their variables but still split query variables (VAR 0.33), so A long did not improve.

### H11: learned unit separator + learnable-unit starts
Idea: the ideal mask is record starts + answer starts, nothing else. Answer starts are already detectable. For record starts, learn the separator from the train split: the most frequent byte value that is followed by a >1-bit entropy jump at least 95% of the time. Stats on train: A picks newline (follow-jump 0.99, freq 0.095; "=" "+" ":" also always followed by a jump but rarer); B picks ";" (1.00, 0.074; space is more frequent but only 0.51). By the same logic C should pick its separator. No byte value in the code; a statistic learned in fit(), which program.md allows. Flagging it anyway: this is the rule that most directly "finds the delimiter", so the held-out format C test is what will tell whether it is generic.
Screen: record starts 1.00 / 1.00, answer starts 0.99 / 0.96, VALUE 0.00, HEX 0.00, rate 0.126 / 0.097 — essentially the oracle.
Hypothesis: approaches the oracle at 16k (ans ~0.7-0.8 vs H2 1.075); local ~0.03; A long far below H2's 2.1-2.35.
Result: **KEEP. 0.941 / 0.900, mean 0.920 vs H2 1.075 (-0.155), at rate 0.112 vs 0.150.** Local sums solved: 0.024 / 0.024 (H2 ~0.36). Long 2.26 / 2.16 — only slightly better than H2 (2.17 / 2.04), still far from the oracle on A (1.08 / 1.32).
Suspected remaining gap: the learnable-start term also fires at the second variable of queries (VAR 0.35), splitting "z+" | "x=". Next: separate answers from query variables by the small-minus-reference gap (answers ~3.3 bits; a variable should be less).
Diagnostic: at learnable starts the small-minus-reference gap is <= 2.01 bits for variables (one of ~4) and >= 3.22 for answers (one of 10 digits), both formats.

### H12: H11, but keep only the upper group of learnable starts (Otsu split of the gap, fitted on train)
Learned cut: 2.55 (A) / 2.53 (B) bits — not hand-set. Screen: VAR 0.35 -> 0.18 (remaining are query variables at record starts), record starts 1.00, answers 0.95-0.99, rate 0.118 / 0.091. This is essentially the oracle record+answer mask.
Hypothesis: removing the query split lets long-range learning happen as in the oracle: A long well below H11's 2.13-2.33 (oracle 1.08 / 1.32); overall ~0.7-0.8.
Result: **KEEP. 0.648 / 0.768, mean 0.708 — matches the label-using oracle (0.673 / 0.760, mean 0.717)** at rate 0.105. vs H11 0.920, H2 1.075, baseline 1.17.
Prediction confirmed on both seeds: removing the query split unlocks long-range learning. Long A 1.22 / 1.56 (H11 2.33 / 2.13), B 1.82 / 2.12 (B now learns too; the oracle had not by 16k). Local 0.04 / 0.02.

### Robustness of H12 (rule frozen after this)
Sensitivity: JUMP 0.7 / 1.5, ALWAYS 0.9 / 0.99, CONFIDENT 0.5 / 1.5 each change at most 6 patch starts out of ~40k val bytes (A, B). The learned separator and Otsu cut absorb the constants; no retraining needed. Third seed running. Final test on H12 (commit in test_final_h12.commit) running; output saved unread until the rule is frozen.
H12 seed 2: 0.701 (A long 1.33, B 2.03). **3-seed mean 0.706** (0.648 / 0.768 / 0.701) vs oracle 0.717 (2 seeds). **H12 is frozen**: no more tuning on formats A/B, so reading the final test is now safe.

## Harder corpus (SEGR_SUITE=hard; formats D, E dev, F held out)
Two separators per format (chosen at random per record), 2-3 operand sums, derive records (set y=x+13) making two-hop queries. Cache built with the same scorers (reference 64k steps).
- Format D reference: ANS_LOCAL 0.03, **ANS_LONG 0.69** (above the 0.5 sanity bar; two-hop queries are harder). Proceeding; noted as a limitation.
- **Screen on D: H12 learns the wrong separator, "="** (it appears in 4 of 6 record kinds and is always followed by a jump; each true separator now covers only half the records). H12 on D: record starts 0.00, value starts 0.38, answer starts 1.00, rate 0.061. H2 on D: record starts 0.59, answer starts 0.51 (local) / 1.00 (long), operands 0.32.
- Prediction: H12 does badly on the hard suite (splits stored facts, no record alignment); the oracle shows what the principle is worth here.
- **Fix (commit on master, merged):** derived variables inherited their own record index, so some queries depended on values up to ~12 records (~180 bytes) back, beyond the 64-127 bytes of context. Format E's reference could not learn long answers (1.84 bits). Derived variables now carry the index of the oldest record they depend on. Cache rebuilt. Format D reference after fix: **ANS_LONG 0.31** (was 0.69).

## FINAL TEST (read after H12 was frozen)
test_final.py on H12 (commit cb04a18, MAIN_STEPS 16000), test splits, seed 0:
| rule | A | B | **C (held out)** | long C | rate |
|---|---|---|---|---|---|
| **H12** | 0.471 | 0.703 | **0.477** | 1.22 | 0.10 |
| blt_entropy | 0.973 | 1.284 | 1.207 | 2.59 | 0.25 |
| trajectory_gated | 1.597 | 1.587 | 0.993 | 2.07 | 0.25 |
| excess_vs_reference | 1.388 | 1.445 | 1.447 | 2.55 | 0.25 |
| stride | 1.780 | 1.760 | 1.358 | 2.24 | 0.25 |
**H12 generalizes to the held-out format**: 0.477 on C vs baseline 1.207 (2.5x lower) at 40% of the compute; long 1.22 vs 2.59. Best rule on every format.
Earlier run (H2, MAIN_STEPS 8000, test_final.log): H2 on C 0.998 vs baseline 1.262 — also generalized.
- Format E reference after fix: ANS_LONG **1.09** (was 1.84). Still above 0.5: E's records are ~15 bytes, so 6 records back plus a two-hop chain often exceeds the 64-127 bytes of context. E's long-range numbers are hard for every rule, and H12's answer detection (which needs the reference to be confident) may be weaker there. D (0.31) is the clean comparison.
- **Format D results (16k, seed 0)**: oracle record+answer **0.706** (local 0.10, long 1.52); H12 1.821 (local 1.53, long 2.21; record starts 0.00, rate 0.061); H2 1.825 (local 1.57, long 2.17; record starts 0.56); baseline 2.315. **The principle holds with a larger margin than on A-C (oracle 2.6x better than the best rule), but no label-free rule finds the records.** H12 ties H2 at 40% of H2's compute; both beat the baseline.
- **Format E results (16k, seed 0)**: oracle **0.980** (local 0.15, long 2.21 — long not learnable on E for any mask, see reference); baseline 1.692 (local 1.35, record starts 0.85); H2 1.910 (local 1.57, record starts 0.94); **H12 2.366 (worst: learns newline, one of two separators; record starts 0.49, rate 0.046, answer starts 0.79)**.
- **Hard-suite summary (D+E average, 1 seed)**: oracle 0.843, H2 1.867, baseline 2.003, H12 2.094. **The principle holds (oracle 2.2x better than the best rule); H12's single-separator trick does not survive a second separator; H2 (separator-free) is the most robust label-free rule.** The open problem is a label-free unit detector for messy data.
- Held-out format F cache built for future use; no final test run on the hard suite (no rule was developed on it).

## Pooling ablation (reviewer's top priority): is "inputs in one patch" an architectural artifact?
Harness switches (commit on master): SEGR_POOL=xattn (BLT-style cross-attention pooling), SEGR_LOCAL=window (local decoder sees the previous 32 bytes across patch boundaries). Oracle masks, 16k steps, seed 0, formats A / B.
Long-range bits (A / B):
| setting | clean | +value | +variable | +operand |
|---|---|---|---|---|
| sum, patch-local (original) | 1.08 / 2.15 | 2.03 / 2.48 | 2.14 / (8k-era) | (8k: local 0.49-0.60) |
| sum, cross-patch window | 1.76 / 2.05 | 1.26 / 1.94 | 1.93 / 2.06 | 2.05 / 2.18 |
| xattn, patch-local | 1.94 / 1.91 | 1.65 / 1.82 | 2.03 / 2.15 | 1.78 / 2.00 |
| xattn, cross-patch window | 2.37 / 2.28 | 1.96 / 2.62* | 1.94 / 2.09 | 1.90 / 2.47 |
*B's local-sum skill failed to form on that run (local 1.30); unrelated to the split since the decoder sees across patches.
Local sums: operand splits cost nothing with the cross-patch decoder (0.005 / 0.005); with xattn + patch-local decoder A 0.003, B 1.04.
**Tentative (1 seed): in a BLT-like model the split penalties vanish or fall within noise.** The local-sum alignment effect is caused by the patch-local decoder; most of the long-range effect by sum pooling + patch-local decoder. Long-range learning itself is erratic across runs (clean masks 1.08-2.37). Seed-1 replicates (xattn patch-local, 4 masks) and sum patch-local + operand at 16k (2 seeds) running.
- Replicates so far: xattn patch-local clean seed 1: A long 1.32 (seed 0: 1.94), B local 0.73 (skill partly failed). xattn patch-local + operand seed 1: A local 0.013, **B local 1.30** (seed 0: 1.04). With a patch-local decoder, operand splits hurt on B even with BLT-style pooling; not on A. With the cross-patch decoder they cost nothing.

### Rho-1 at 16k (validation, reviewer point 1)
excess_vs_reference: 1.519 (s0), 1.410 (s1), mean **1.46** vs baseline 1.17 (s0). Confirmed worse than raw entropy at the full budget.

### Learned chunking (H-Net-style, simplified; learned_chunking.py; original model: sum pooling, patch-local decoder; 1 seed)
Original formats: 1.414 (A 1.08, B 1.74; rate 0.13 / 0.15) vs baseline 1.17, H12 0.71. Hard corpus: 2.325 (D 2.38, E 2.27; rate 0.10 / 0.12) vs baseline 2.00, H2 1.87, label-based mask 0.84. Weakest rule on the hard corpus in this architecture; its fair test is the BLT-like model (xattn + cross-patch), where splits are not penalized.

### 32k convergence check (reviewer point 6; seed 0; original model)
H12 0.527 (A 0.484, B 0.569; long 1.26), label-based mask 0.579 (long 1.38), H2 1.050 (long 1.96). At 16k: 0.648 / 0.673 / 1.100. **Ranking stable** (H12 ~ mask << H2); H12 and the mask keep improving on long-range answers, H2 barely moves.

### Wide variable alphabet (reviewer point 2)
Cache built (SEGR_SUITE=wide). Reference ANS_LONG: A 0.16, B 1.00 (above 0.5), C 0.28.
**Screen: H12's answer filter breaks as predicted.** With 16 names a variable's small-minus-reference gap (median 3.9 bits) exceeds an answer digit's (3.3). Otsu still splits cleanly (cut 3.64 / 3.58) but keeps the variables: answer starts 0.00 on both formats; record starts still 1.00. Training runs: H12, H11 (no filter) and the label-based mask on wide A/B.
Wide-alphabet results (seed 0, 16k, original model): label-based mask 0.861 (long 1.99 / 1.98); H11 (no filter) 0.999 (long 2.20 / 2.45); **H12 1.077 (long 2.58 / 2.48 = baseline level; no answer starts)**. Reviewer point 2 confirmed: H12's filter relies on the 4-name generator artifact, and with 16 names H12 falls below the simpler H11.

### BLT-like model (xattn + cross-patch decoder): answer starts and learned chunking
- Record starts only: 1.01 (s0) / 0.96 (s1); long A 2.19, B 2.58 (means). Record + answer starts: 0.97 / 0.95; long A 2.21, B 2.40. **Answer starts make little difference here**, but no mask learns long-range answers well in this model by 16k (2.1-2.6 bits), so long-range effects have little room to show. The BLT-like model at this budget is insensitive to patching.
- Learned chunking (H-Net-style) in the BLT-like model: **1.334** (A 1.27, B 1.40; local 0.36 / 0.57; long 2.57 / 2.61; rate 0.10). Still worse than hand-placed masks in the same model (~0.95-1.01).

## Revised conclusions (after review)
1. In the simplified model (sum pooling, patch-local decoder), aligning patches to units and starting a patch at each answer matters a lot; H12 exploits it and matches the best label-based mask.
2. Most of that effect is architectural: with BLT-style cross-attention pooling or a cross-patch local decoder, splitting a computation's inputs mostly stops mattering, and in the full BLT-like model even answer starts barely matter at this budget.
3. Allocating patches to hard or learnable bytes (Rho-1) is worse than raw entropy at the full budget; learned chunking (simplified H-Net) did not beat hand-placed masks in either model.
4. H12 does not transfer: it fails with two separators and with a 16-name variable alphabet.

### BLT-like model at 48k steps (is its insensitivity to patching a budget effect?)
xattn + cross-patch decoder, seed 0, 48000 steps: clean record+answer, record only, +value, +variable, +operand, and baseline raw entropy. Running.
Results (48k, seed 0; answer bits, long in parentheses): clean record+answer **0.566 (1.37)**; +value 0.618 (1.51); +variable 0.631 (1.54); record only (no answer starts) 0.736 (1.79); +operand 0.810 (1.97); baseline raw entropy 0.899 (2.16) at 25%. Local sums solved everywhere (<= 0.03).
**The 16k insensitivity was a budget effect.** Once the BLT-like model learns long-range answers, placement matters again: answer starts +0.17 bits, operand splits +0.24 (on long-range answers, not local), value/variable splits +0.05-0.07 (much smaller than the ~1 bit in the simplified model). Clean alignment beats raw entropy by 0.33 bits at less than half the compute. One seed.

## Conclusions (final for this round)
The principle survives in a BLT-like model with enough training, at reduced strength: start a patch at each unit and at each output, keep a computation's inputs together; do not spend patches on surprising bytes. Its large size in the simplified model was partly architectural (sum pooling, patch-local decoder) and at 16k partly masked by undertraining.

## Real-text check (GSM8K)
realtext.py: GSM8K worked solutions; answer spans = computed result inside <<expr=R>>, its copy after >>, final answer after ####. Small entropy model trained on GSM8K train. Screen (val): BLT entropy at 25% covers only **2%** of answer-span starts (its boundaries fall mostly inside words); entropy+ans (same budget) and words+ans (19.7%) cover 100%. Queued: 4 masks x 2 seeds, BLT-like model (xattn + cross-patch decoder), 32k steps.
**GSM8K results (BLT-like model, 32k steps, 2 seeds; bits per byte):**
| mask | rate | answers | computed R | final | copy | text | bpb |
|---|---|---|---|---|---|---|---|
| entropy (BLT) | 0.25 | 2.20 | 2.44 | 1.38 | 0.52 | 1.80 | 1.688 |
| **entropy+ans** | 0.25 | **1.86** | 2.17 | **0.82** | 0.12 | 1.80 | 1.674 |
| words | 0.185 | 2.10 | 2.40 | 1.10 | 0.33 | 1.69 | 1.581 |
| **words+ans** | 0.197 | **1.65** | 1.94 | **0.67** | 0.05 | 1.68 | **1.562** |
**The answer-start effect holds on real text**: at the same compute, moving boundaries to answer starts cuts answer loss by 0.34 bits (entropy) / 0.45 (words), final answers most (1.38 -> 0.82), with no cost on other text. Word-aligned patches beat BLT entropy outright (1.58 vs 1.69 bpb at less compute). Seeds agree within 0.01-0.08.

### BLT-like 48k replicates and label-free rules
Clean record+answer: 0.566 / 0.594 / 0.592 (mean **0.584**). Record only: 0.736 / 0.746 / 0.731 (**0.738**; answer starts +0.15, 3 seeds). Operand splits: 0.810 / 0.572 / 0.548 (seed 0 was an outlier; **no consistent penalty**). Value splits: 0.618 / 0.712 / 0.730 (0.687). Variable splits: 0.631 / 0.755 / 0.683 (0.690). Baseline raw entropy: 0.899 / 0.835 / 0.798 (**0.844**).
Label-free rules (seed 0): **H2 0.508 (beats the label-based mask)**, H12 0.639, learned chunking 0.996. In a model where splits barely hurt, H2's extra starts at every unit (including answers) seem to help. Needs replication.
- **H2 replicates (BLT-like, 48k):** 0.508 / 0.747 / 0.708, mean **0.654**. Seed 0 was lucky (A long 0.93). H2 does not beat the label-based mask (0.584); it sits between the mask and the raw-entropy baseline (0.844), about 0.19 bits better than the baseline at 15% of bytes.

## Label-free answer starts on real text (GSM8K)
Screen (val): BLT entropy *level* catches 2% of answer starts at 25%; the entropy *jump* catches 58% at 15% and 85% at 25% (copies and final answers 100%, computed results 5% / 66%). Answer starts are less uncertain for the small model than typical word starts (2.9 vs 4.2 bits) but its uncertainty rises sharply there. Word starts + top 1.5% non-word jumps (20%, matched to words+ans): copies 0.99, finals 1.00, computed 0.01; + top 6.5% (25%): 81% of answer starts.
Runs (BLT-like, 32k, seeds 0/1): jump25, words+jump20, words+jump25. Compare with entropy 2.20, words 2.10, entropy+ans 1.86, words+ans 1.65 (answer bits).
**Results (2 seeds, means; answers / final / computed / overall bpb):** jump25 **1.78** / 0.75 / 2.09 / 1.662; words+jump20 **1.82** / **0.66** / 2.17 / **1.579**; words+jump25 1.82 / 0.68 / 2.16 / 1.624. Seeds agree within 0.02-0.06.
**Label-free works on real text.** At BLT's own budget, switching entropy *level* to entropy *jump* cuts answer loss 2.20 -> 1.78, beating even the labeled entropy+ans (1.86), and overall bpb improves (1.688 -> 1.662). Word starts + non-word jumps matches the labeled words+ans on final answers (0.66 vs 0.67) and overall (1.579 vs 1.562) at the same compute, recovering ~60% of its advantage on all answers. Remaining gap: computed results (2.17 vs 1.94), whose starts show little entropy rise.

## Real text 2: Python code (SEGR_REALDATA=code)
Stdlib source; "answers" = identifiers repeating a name seen <= 120 bytes earlier in the file. BLT-like, 32k, 2 seeds (means; identifiers / overall bpb):
entropy (25%) 2.27 / 1.996; jump25 (25%) 2.22 / 1.973; entropy+ans (24%, labels) 2.17 / 1.992; **words (10.5%) 2.17 / 1.842**; words+jump (11%) 2.19 / 1.862; words+ans (11%, labels) 2.13 / 1.858.
**Word alignment is the big win on code** (0.15 bits better overall than BLT entropy at < half the compute). The entropy jump again beats BLT's level rule, by less than on GSM8K (0.05 on identifiers), consistent with BLT entropy already catching 77% of identifier starts in code. Answer starts add a little on identifiers, nothing overall: identifier starts mostly coincide with word starts already.

## Specialized-math experiment (mathexp.py): GSM8K + MATH, accuracy
Corpus 9.7 MB train / 0.85 MB val (GSM8K + MATH interleaved). BLT-like, D=64, 32k steps. Exact match of greedy decoding given the true prefix, 300 targets per type, boundaries decided online by each rule.
**Seed 0** (final answer bits / exact match; computed acc; boxed acc; overall bpb):
| rule | rate | final | computed acc | boxed acc | bpb |
|---|---|---|---|---|---|
| entropy (BLT) | 0.25 | 2.81 / **13%** | 5.3% | 4.7% | 1.767 |
| jump (label-free) | 0.25 | 1.28 / **70%** | 4.3% | 4.7% | 1.782 |
| syntax+jump | 0.25 | 1.02 / 76% | 7.7% | 4.0% | 1.785 |
| words | 0.17 | 1.49 / 62% | 3.7% | 4.0% | 1.703 |
| words+jump (label-free) | 0.185 | 1.06 / 70% | 3.0% | 3.7% | 1.711 |
| **words+syntax** | 0.185 | **0.80 / 81%** | 7.3% | 3.3% | **1.691** |
Final answers (a copy of the last computed result) go from 13% to 70% correct just by switching BLT's entropy level to entropy jump at the same compute; 81% with word starts + math syntax. Syntax boundaries roughly double computed-result accuracy (~7.5% vs 3-5%), but arithmetic is mostly beyond this model. MATH boxed answers ~4% for every rule (too hard at this size). Seed 1 running.

## FLAW FOUND: word-start rule leaked the predicted byte (commit 7338f28)
`realtext._word_starts` flagged byte t when byte t-1 was a space AND byte t was not a space: the flag for t depended on t itself, revealing information about the byte being predicted. In code it marked exactly where indentation runs end (4.9% of bytes flagged differently from the causal rule, about half of all word starts); in GSM8K 0.6%. Fixed to SpaceByte's causal rule (patch after a space/newline not preceded by one).
**Invalid until rerun:** every word-based result on GSM8K (words, words+ans, words+jump20/25), code (words, words+jump20, words+ans), and math (words, words+jump, words+syntax). Code "+ans" masks are also invalid in principle (the flag reveals that a repeated identifier starts; no causal version) and are dropped. Unaffected: entropy / jump / syntax rules, all synthetic results, GSM8K entropy+ans (answer starts there are fixed by syntax).
Reruns queued (causal rule, 2 seeds): math words / words+jump / words+syntax; GSM8K words / words+jump20 / words+ans; code words / words+jump20.

## New strategy screen: retrieval-triggered boundaries (attention reaching > 16 bytes back)
Negative with the current small model (1 layer, 32 dims): catches copies and MATH boxed answers but 0% of GSM8K final answers even at 25%. The small model cannot retrieve, so its attention does not mark retrieval moments. Parked; revisit with a stronger scorer.
Other math strategies to test next: a fresh patch per result digit inside computed results; operand-aligned patches (each number and operator its own unit).

## Causal reruns (word-start leak fixed) — all conclusions survive
Math (2 seeds; final acc / computed acc / bpb): entropy 11% / 5.5% / 1.771; words 58% / 4.8% / 1.709; jump 68% / 4.2% / 1.780; words+jump 68% / 4.1% / 1.714; syntax+jump 72% / 6.2% / 1.771; **words+syntax 76% / 7.0% / 1.697**.
GSM8K (2 seeds; answers / final / bpb): words 2.01 / 1.02 / 1.584; words+jump20 1.80 / 0.69 / 1.580; words+ans 1.61 / 0.66 / 1.562 (leaky: 2.10, 1.82, 1.65 on answers; essentially unchanged).
Code (2 seeds; identifiers / bpb): words 2.21 / 1.889; words+jump20 2.20 / 1.895 (leaky: 2.17 / 1.842). The leak inflated code's word-start gain; corrected, words still beat BLT entropy by 0.11 bpb and 0.06 on identifiers at < half the compute. Jumps add nothing on top of words for code.

## New math strategies (2 seeds, BLT-like, 32k)
words+syntax+digits (fresh patch per result digit; rate 0.192) and words+syntax+ops (operand-aligned; rate 0.207), against words+syntax (0.185): final 76%, computed 7.0%, bpb 1.697. Running.
Results (2 seeds; computed acc / computed bits / final acc / bpb): words+syntax 7.0% / 2.42 / 76% / 1.697; **+digits 8.9% / 2.36 / 75% / 1.695** (both seeds above the reference on computed results: 8.7, 9.0 vs 7.0, 7.0); **+ops 7.5% / 2.39 / 79% / 1.688** (both seeds above on final answers: +2.7, +3.4 points).
Consistent direction but not significant: +1.9 / +3 points on 600 samples are within about one standard error of the difference, and both rules spend slightly more compute (+0.7 / +2.2 points of bytes). Next if pursued: matched-compute controls (same number of extra boundaries at random positions), more seeds and samples, or a larger model where computed results are off the floor.
**Matched-compute controls** (extra boundaries at pseudo-random positions, same count; 2 seeds; computed acc / computed bits / final acc / bpb):
rand07 (twin of +digits) 6.2% / 2.49 / 74% / 1.712; rand22 (twin of +ops) 7.5% / 2.45 / 75% / 1.716.
**Placement matters, not just compute:** random extra boundaries make overall loss worse than no extra boundaries (1.712-1.716 vs 1.697); targeted ones make it slightly better. +digits beats its twin on computed results on both seeds (acc 9.0 vs 5.3, 8.7 vs 7.0; bits 2.36 vs 2.49); +ops beats its twin on final answers on both seeds (79% vs 75%). Accuracy gains are ~1.6-1.8 standard errors on 600 answers: suggestive, not conclusive. Next: more seeds / larger evaluation / larger model.

## Literature check (2026-09-30)
- **Scratchpad Patching** (Zheng et al., arXiv 2605.09630, May 2026): names the mechanism we kept finding, "patch lag" (predictions inside a patch rely on a stale representation from the previous patch). Fix: transient scratchpads inside patches, triggered when next-byte entropy exceeds a threshold (a *level* trigger, like BLT's).
- **What still looks new:** entropy-*level* triggers have a blind spot at the start of outputs (low entropy, high dependency). On math it costs most final-answer accuracy (11% vs 68-76%); jump or syntax triggers fix it. Not found reported; likely applies to entropy-triggered scratchpads too (testable).
- The two math-specific rules are byte-patching analogs of known tokenizer practice: per-digit patches ~ single-digit tokenization (LLaMA-style; "Tokenization counts", arXiv 2402.14903); number/operand alignment ~ digit place-value alignment studied for tokenizers (TokEval, arXiv 2608.18062, COLM 2026). If confirmed: "tokenizer wisdom transfers to patching", useful but not a new idea.
- "Disentangling Language Modeling and Boundaries" (arXiv 2608.03599): boundary placement can be changed post hoc, nearly independently of capability. Suggests a cheap test on a real trained byte model: re-place boundaries at answer starts without retraining.
- Also relevant: SpaceByte (word-aligned patches), BLT (includes the entropy-jump "approximate monotonic" rule), H-Net (learned chunking), ByteSpan.
- **Verified by hand 2026-10-01 (#13): see literature.md and refs.bib.** All IDs check out. Precisions: released BLT-1B uses the global threshold (monotonicity off), while the BLT paper's main model uses the jump rule; Nawrot et al. 2023 also used entropy spikes, so the jump is prior art; Scratchpad Patching has no code release and reports no math accuracy (its E.2 ablation finds entropy best on BPB); 2608.03599 is a position paper. No prior report of the blind spot found.

### Confirmation, 5 seeds (seeds 2-4 scored on 1000 computed / 660 final / 700 boxed)
Pooled (computed acc n=3600 / final acc n=2580 / computed bits / bpb):
words+syntax 9.7% / 76.1% / 2.40 / 1.695; +digits 7.8% / 75.1% / 2.39 / 1.699; rand07 7.9% / 76.2% / 2.47 / 1.706; +ops 9.7% / 77.5% / 2.38 / 1.687; rand22 7.9% / 76.2% / 2.46 / 1.715.
**Per-digit patches: NOT confirmed** (accuracy = random twin, z -0.1; computed acc below the reference, 7.8 vs 9.7%). Only consistent effect: lower loss on result digits than its twin (5/5 seeds), which does not become correct answers.
**Operand alignment: small consistent loss gain, no significant accuracy gain** (bpb below its twin 5/5, below the reference 4/5 by ~0.008; final +1.4 points vs reference, z 1.2).
**Solid:** random extra boundaries hurt on every seed (bpb +0.011 / +0.020 vs reference; computed acc 9.7 -> 7.9%). Placement matters, but these targeted placements barely beat the reference.
Lesson (rule 3 in program.md): the 2-seed "promising" gains were noise; do not report 2-seed accuracy differences of 2-4 points as effects.

## Blind spot on Scratchpad Patching (simplified; harness SEGR_SCRATCH=1)
Model: sum pooling, cross-patch local decoder, patches every 8 bytes (12.5%) + 6% transient scratchpads (global step over the partial current patch; later patches do not attend to them). Checked: default model unchanged, no-scratchpad case matches, causal.
Screen (share of answer starts that get a scratchpad): entropy level (the paper's trigger) 5%, random 2%, entropy jump 23%, math syntax + jumps 87%.
Runs (2 seeds, 32k, accuracy on 1000 computed / 660 final): sp:none (12.5%), sp:dense5 (20% patches, no scratchpads), sp:entropy, sp:jump, sp:syntax, sp:random (each 18.5% total).
Prediction if the blind spot carries over: sp:entropy final-answer accuracy near sp:random / sp:none; sp:syntax (and partly sp:jump) clearly higher.
(Note: a first launch was stopped within minutes because a git step left the scratchpad harness code in a stash; relaunched after committing it.)
**Results (2 seeds; final acc n=1320 / computed acc / bpb):** sp:none 7.7% / 5.4% / 1.871; sp:random 7.35% / 5.6% / 1.861; **sp:entropy 7.45%** / 4.9% / 1.856; sp:jump 8.9% / 5.1% / 1.857; **sp:syntax 38.4%** / 6.5% / 1.851; sp:dense5 (20% patches) 44.4% / 5.3% / 1.828. Seeds agree within ~1 point except dense5 (40.5 / 48.2).
**The blind spot carries over to Scratchpad Patching:** the paper's entropy-level trigger is no better than random scratchpads (or none) for final answers; the same 6% of compute at answer starts gives 5x the accuracy. The jump trigger barely helps at a 6% budget (covers 23% of answer starts). Caveat: in this simplified implementation (sum pooling, partial-patch sum), denser regular patches beat every scratchpad trigger on final answers at slightly more compute, so scratchpads here may be weaker than the paper's.

## Round "do all" (2026-10-01): faithful SP, learned trigger, larger model
Larger model runs started: D=128, 4 global layers, BLT-like (xattn + cross-patch), math, 32k, seeds 0/1: entropy, jump, words+syntax.
- Faithful Scratchpad Patching: scratchpads pooled by cross-attention with the mean as query (as in the paper), patches every 16 bytes (the paper's headline setting); sp16:{none, dense8, entropy, jump, syntax, random} x 2 seeds queued. Screen: entropy trigger gives 5% of answer starts a scratchpad, syntax 93%.
- Learned trigger (gaintrigger.py): measures where a scratchpad lowers the loss on the rest of its patch, fits a ridge regression on causal features (no answer labels); sp16:learned x 2 seeds queued after fitting.
- Real model (realblt.py): Meta's BLT-1B (itazap/blt-1b-hf, 15.4 GB) with patch boundaries changed at inference; four layouts (default, +answer, -answer, jump at equal patch count) on GSM8K final answers. Downloading; will run when the GPU is free.
- **Learned trigger fitted (gaintrigger.py, 16k-step random-scratchpad model, 400 windows = 48k positions):** gains are very skewed (mean 0.026, p90 0.92 nats); ridge prediction correlates only 0.18 with measured gain. Largest weights on rare previous bytes (X +1.10, Y +0.53, ! +0.49), likely overfitting; '=' +0.12, newline +0.15, '#' -0.15. Screen: gives a scratchpad to 14% of answer starts (entropy 5%, random 2%, syntax 93%). Runs queued as-is. Refinement if it helps: stronger regularization or a minimum byte count, a classification target (gain > threshold), and saving the measured gains for refits.

### Larger model (D=128, 4 global layers; BLT-like; 2 seeds)
final acc / computed acc / bpb (means): entropy 44.8% / 10.9% / 1.442; jump 74.0% / 9.0% / 1.453; **words+syntax 79.4% / 32.3% / 1.412** (seeds: computed 33.8, 30.8).
- The blind spot shrinks but persists: BLT entropy 11% -> 45% on final answers at 4x size, still ~30 points behind jump / words+syntax.
- **New at this size: alignment changes arithmetic.** words+syntax gets 3x the computed-result accuracy of both entropy rules (32% vs 9-11%); at D=64 every rule was at 4-10%. Arithmetic was capacity-limited; with capacity, placement matters for the computation itself.
- Attribution open: word alignment (numbers as units) vs the boundary after '='. Queued: words and syntax-only at D=128.

### Faithful Scratchpad Patching (xattn pooling, 16-byte patches, 6% scratchpads; 2 seeds; final acc / bpb)
none 6.5% / 1.934; random 5.8% / 1.919; **entropy (paper's trigger) 6.0%** / 1.927; jump 6.0% / 1.917; learned 8.8% / 1.914; dense8 (12.5% patches) 9.5% / 1.870; **syntax 40.3%** (49.2, 31.4) / 1.906.
**Confirms the blind spot with the paper's pooling and patch size:** the entropy trigger is indistinguishable from random or no scratchpads for answers; answer-start scratchpads give ~7x. Resolves the earlier caveat: with faithful pooling, syntax scratchpads beat denser patches at equal compute (40% vs 9.5%); the sum-pooled version had underpowered scratchpads. Computed results ~5% for every setup (small model).
Learned trigger (14% of answer starts covered): 8.8%, slightly above chance; needs the refinements noted above.

### Attribution on the larger model (2 seeds; computed acc / final acc / bpb)
words 16.5% / 61% / 1.440; stride6+syntax 16.5% / 73% / 1.568; **words+syntax 32.3% / 79% / 1.412**.
**Interaction:** either part alone gives the same 16.5% on computed results; together 32%. Arithmetic needs numbers aligned as whole units AND a fresh step at the result. Final answers come mostly from the result boundary.

### Real model: BLT-1B (in progress)
Format: GSM8K question + solution (calculator annotations removed) + "The final answer is " (+ "$" if the solution used it); target = the answer as the solution last wrote it. 10-problem check: BLT-1B's own entropy patches already start at the answer in 100% of cases (its entropy model is uncertain after "The final answer is "); removing that boundary drops exact match 100% -> 70% and raises answer loss 0.39 -> 1.12 bits.
**BLT-1B results (300 GSM8K test problems):** default entropy patches already start at the answer in 100% of problems; default 87.7% exact / 0.64 bits per answer byte; +answer identical; **-answer (boundary removed) 60.3% / 1.56**; jump at equal patch count 80.0% / 0.93 (post-hoc layout change; the model was trained with entropy patches).
- **Mechanism confirmed in a real trained model:** removing the patch start at the answer costs 27 points of exact match.
- **No blind spot at BLT's default budget (~30% of bytes):** its patcher starts patches at 100% of final answers and 97% of in-line results after '= ' (word starts 81%).
- **Blind spot appears under tight budgets, for computed results:** ranking by BLT-1B's own entropy at 15 / 10 / 6% budgets covers in-line results after '= ' only 48 / 26 / 18% of the time, below word starts (54 / 39 / 25%); final answers stay covered (96 / 94 / 91%). Median entropy: final 3.16, result 2.15, word 2.25 nats. A computed result's TYPE is predictable after '=', so its entropy is lower than an ordinary word start even though its content must be computed.
- Revised claim: entropy-level triggers under tight compute budgets (scratchpads, larger patches) systematically skip positions whose content is hard but whose type is predictable. At generous budgets they do not.

### BLT-1B tight-budget test (realblt_budget.py; 294 problems, 796 in-line results after '= ')
results covered / in-line results exact / bits / final exact:
default (28%) 83% / 73.0% / 0.67 / 67.3%; entropy@15 48% / 39.8% / 1.71 / 56.8%; **results@15 100% / 61.3% / 1.04 / 57.1%**; entropy@10 24% / 18.1% / 2.58 / 40.1%; **results@10 100% / 45.2% / 1.59 / 51.0%**.
**The blind spot is real in a trained 1B model at tight budgets:** at equal patch counts, forcing a boundary after each '=' raises exact computed-result accuracy by +21 points at 15% and +27 at 10%, and cuts loss on those bytes ~40%; final answers do not get worse. Inference-time layout change only (BLT-1B was trained at ~30%), so absolute numbers at tight budgets are out of distribution for every layout; the comparison is at equal budget.

## Go-for-all round 2 (2026-10-01): label-free trigger, code, tight-budget training
**Label-free trigger screens on BLT-1B (blt_screens/), coverage of in-line results after '= ' at 15/10/6% budgets:**
- Entropy (BLT): 42 / 22 / 14%.
- Kind vs value split (entropy over byte kinds vs within-kind): no help (results' kind entropy 0.31 vs words 0.23 nats; within-kind 1.82 vs 1.91). Negative.
- "Difficulty ahead" (main-model loss on next 4 bytes under default patches, per 2-byte context): 0% for results. Negative: under default patches BLT already predicts results well (its patches start there), so measured difficulty is low. It measures difficulty GIVEN a boundary, not the value of one.
- **"Boundary dependence"** (main-model loss on next 4 bytes under a 10% entropy layout minus under default, per 2-byte context, fitted on 400 GSM8K TRAIN problems; causal lookup, no labels): **100 / 100 / 100%**. Top contexts are arithmetic ('=$', '0*', '5=', '0+', '3=', '2=' ...); after '= ' 0.91 vs 0.17 average. Entropy + dependence: results 100%, words 41% at 15%.
Running: BLT-1B accuracy test with dep@R and entdep@R layouts (300 problems), then the same on Python code (realblt_code.py), then tight-budget TRAINING at 10% (D=128): entropy10, jump10, syntax+entropy10 x 2 seeds.

### BLT-1B label-free trigger, accuracy (realblt_budget.py, 294 GSM8K test problems, 796 in-line results)
| layout | results covered | in-line exact | final exact |
|---|---|---|---|
| default (28%) | 83% | 73.0% | 67.3% |
| entropy@15 | 48% | 39.8% | 56.8% |
| results@15 (label) | 100% | 61.3% | 57.1% |
| dep@15 | 100% | 70.5% | 35.0% |
| **entdep@15** | 100% | **71.2%** | **57.5%** |
| entropy@10 | 24% | 18.1% | 40.1% |
| results@10 (label) | 100% | 45.2% | 51.0% |
| dep@10 | 100% | 64.2% | 9.2% |
| **entdep@10** | 100% | **60.3%** | **53.7%** |
- Entropy + boundary dependence, with no labels, keeps in-line results near the default (71% vs 73%) at about half the patches, and beats the label-based results@R reference (it also cuts at operands and operators, not only after '=').
- Dependence alone wrecks the final answer: "The final answer is " is not an arithmetic context, so it puts no boundary there. The combination keeps both.
- Caveat: the table is fitted with BLT-1B on GSM8K train (same domain as the test); inference-time layout change only, trained at ~30%.

### Beyond math: BLT-1B on Python code (realblt_code.py, 300 stdlib test chunks, 5229 repeated identifiers)
| layout | covered | exact | bits |
|---|---|---|---|
| default (~28%) | 69% | 73.0% | 0.526 |
| entropy@15 | 43% | 67.9% | 0.643 |
| dep@15 | 59% | 67.2% | 0.705 |
| entdep@15 | 60% | 69.2% | 0.627 |
| oracle@15 (label) | 100% | 70.7% | 0.599 |
| entropy@10 | 32% | 64.0% | 0.738 |
| dep@10 | 44% | 64.0% | 0.800 |
| entdep@10 | 46% | 66.3% | 0.705 |
| oracle@10 (label) | 100% | 67.0% | 0.676 |
- The blind spot is much weaker on code: even forcing a boundary at every repeated identifier gains only 3 points (vs 21-27 for math results). A repeated name is a copy, so its value is not hard the way a computed number is; the patch start matters less.
- entdep again helps most among label-free rules (+1.3 / +2.3 points over entropy, close to the oracle at 10%); dependence alone does not.
- Conclusion: the effect is specific to positions whose value needs computation, not to "predictable type" alone. One evaluation pass, no seeds (inference only).

### Tight-budget TRAINING (D=128, 4 global layers, xattn + window, 32k steps, 10% patch budget, 2 seeds)
computed acc / final acc / boxed acc / bpb (means; seeds in brackets):
- entropy10: 7.0% [6.7, 7.3] / 12.9% [15.5, 10.2] / 2.8% / 1.655
- jump10: 6.9% [6.6, 7.1] / 63.9% [61.4, 66.4] / 3.2% / 1.639
- **syntax+entropy10: 25.3% [25.3, 25.3] / 72.4% [72.3, 72.4] / 3.0% / 1.633**
Reference, same model at its usual budget: entropy (25% of bytes) 10.9% / 44.8% / bpb 1.442; words+syntax (18.5%) 32.3% / 79.4% / 1.412.
- **The advantage survives training at a tight budget:** with the result boundary (after '=', '#### ', '\boxed{'), the 10%-budget model computes 3.6x more results correctly than entropy at 10%, and beats BLT entropy at 25% on both computed results (25% vs 11%) and final answers (72% vs 45%) with 2.5x fewer patches. Overall bpb is also the best of the three at 10%, so the gain is not bought elsewhere.
- Jump rescues final answers (64%) but not computed results (7%), as before.
- MATH (\boxed) stays at ~3% for all: capacity-limited at this size.
- Seeds agree closely; a third seed is not needed for the ranking (standard 3 gate: spread << gaps).
- Caveat: syntax is a hand-written rule; the label-free trigger (entropy + dependence) has not yet been trained into a model (next: M5 step 1, fitting dependence with the harness's own model).

### Beyond arithmetic on BLT-1B (realblt_reason.py): program traces
300 generated straight-line programs; targets after '# v is now ': 2469 computed (after an arithmetic line), 1716 copied (after a constant line). Exact match:
| layout | computed | copied |
|---|---|---|
| default (~28%) | 20.5% | 82.5% |
| entropy@15 | 11.2% | 66.8% |
| dep@15 (own table) | 9.3% | 46.0% |
| entdep@15 | 10.9% | 66.7% |
| entdepM@15 (math table) | 9.6% | 61.2% |
| oracle@15 | 12.8% | 73.8% |
| entropy@10 | 9.6% | 58.4% |
| entdep@10 | 8.9% | 62.1% |
| oracle@10 | 9.4% | 61.8% |
- **Inconclusive for computed values (floor):** BLT-1B computes only 20% of trace values even with default patches, and a boundary at each value barely helps (+1.6 points at 15%, none at 10%). The model mostly cannot do this arithmetic in its head, so placement has little to act on. Contrast GSM8K, where it gets 73% of in-line results.
- Copied values gain more from a boundary (+7 points at 15%) than computed ones here: when the skill is present, the boundary helps; when it is absent, it cannot.
- The trace dependence table picks up the 'is now ' context ('w ', 'ow'); dep alone hurts copies badly. The math table does not transfer (0% coverage of trace values: '= ' is not the context here).
- Logic first run: 'True/False' answers were unusable (BLT-1B continues 'Answer: ' with 'Yes/No' or a sentence; exact 0% everywhere). Rerunning with Yes/No answers and a forced-choice score.

### Beyond arithmetic on BLT-1B: logic (Yes/No answers, forced-choice score)
300 generated theories (one person, 2 facts, a 3-5 rule chain plus distractors), balanced Yes/No (142/158; always "No" = 52.7%).
- **Answers are at chance for every layout**: forced choice 52-58% with proof (logic) and without (direct), default included (52.0% / 56.0%); SE ~2.9 points. BLT-1B cannot answer these questions, so patching has nothing to act on. Inconclusive (floor), not negative.
- Proof steps (a rule lookup: "X is a, so X is b"): default 69.2%; at 15% entropy already covers 98% of them (66.6% vs oracle 67.3%); at 10% entropy 45.6% vs oracle 50.2% (+4.6). Small, like code.
- Label-free: entdep on its own table hurts steps (59.2% / 40.9%); the math table (entdepM) matches entropy (66.8% / 45.7%).

### Where "beyond arithmetic" stands
- The effect is large where BLT-1B has the skill and the value is computed (GSM8K results: +21-27 points), small where the value is a lookup or copy (code identifiers +3, proof steps +1-5, trace copies +3-7), and unmeasurable where BLT-1B lacks the skill (trace arithmetic 20% at default, logic answers at chance).
- Standard 2 applies: test where the skill is learned. Next: train MLX models on the generated logic and trace tasks (which they can learn) and compare patching rules in training, instead of probing BLT-1B.
- Dependence tables do not transfer across formats (math -> traces 0% coverage); a refit per domain is needed, and on its own a table can hurt (dep alone). entropy + dependence is the safe combination where the table fits the format.

### Label-free trigger trained into the model (deptrigger.py + mathexp.py entdep10/dep10; D=128, 10% budget, 32k steps, 2 seeds)
Dependence fitted with the harness's own model, no labels:
- v1 ("remove", as for BLT-1B; reference trained on entropy 25%): failed the screen. Top contexts were LaTeX; arithmetic below average; entdep10 covered 3% of answer starts, dep10 0%. The reference rarely had a boundary at results (13% of answer starts), so it never learned to use one. Not trained.
- v2 ("marginal"; reference trained on random 25% patches, gain of adding one boundary at t to a random 10% layout, bytes t..t+3, 4096 windows x 32 candidates): top contexts include '<<', '=6', '=<', 'x=', '>>'. Answer-start coverage: entropy10 8%, entdep10 13%, **dep10 51%**, syntax+entropy10 100%.
computed acc / final acc / bpb (means; seeds in brackets):
- entropy10: 7.0% / 12.9% / 1.655 (earlier)
- entdep10: 9.7% [7.2, 12.2] / 22.5% [13.2, 31.8] / 1.676
- **dep10: 19.2% [17.3, 21.0] / 68.8% [68.0, 69.5] / 1.676**
- syntax+entropy10 (hand-written): 25.3% / 72.4% / 1.633 (earlier)
- **With no labels and no hand-written syntax, dependence alone recovers most of the hand-written gain at a 10% budget**: computed results 7% -> 19% (hand-written 25%), final answers 13% -> 69% (hand-written 72%). It also beats BLT entropy at 25% of bytes (10.9% / 44.8%) with 2.5x fewer patches.
- Cost: overall bpb 1.676 vs 1.655 for entropy10 (+1.3%); the hand-written rule has no such cost (1.633).
- entdep10 (z-sum) is dominated by entropy, which is low at answers; it barely helps. The combination that worked on BLT-1B does not transfer; on BLT-1B dependence alone hurt final answers, here it is the best label-free rule. Rules were fixed before training (no tuning on answer coverage).

### Compute per byte (flops.py; matmul FLOPs, forward)
- Harness (D=128, 4 global layers): the local parts dominate (0.57 MFLOPs/byte vs 1.64 per patch), so a 10% budget costs 76% of entropy at 25%; a lookup rule (dependence table, no entropy model) 72%. The small model saves little.
- BLT-1B: 299 MFLOPs per byte (local), 2,653 per patch (global), and its entropy model 199 per byte. At 15% with the entropy model 72% of default; at 10% 62%. A dependence-table rule also drops the entropy model: 56% at 15%, 46% at 10%.
- So on BLT-1B, the label-free layout that kept computed results at 71% (vs 73% default) at 15% would cost 72% of default compute with entropy in the rule (entdep), or 56% if a lookup-only rule worked as well (dep alone did on the trained small model, not on BLT-1B's final answers).

## Causality test (#14, test_causal.py)
`uv run test_causal.py` (~25 s, CPU) perturbs everything a rule may not see (bytes[t:], entropy[t+1:], surprisal[t:]) with real or random values and requires mask[:t+1] unchanged; models must give the same logits for bytes <= t+1 when later bytes and later mask entries change.
- **Pass:** boundary.py (H12), all 4 baselines, realtext words / words+jump20/25, all 31 mathexp rules (incl. dep10/entdep10 and every sp:/sp16: trigger), the harness model in all 8 pool/local/scratchpad settings, learned_chunking, and the entropy scorer. Negative controls caught: surprisal rules and the pre-7338f28 word-start rule (at t itself).
- **Split-level (threshold refit on the evaluated split; reported, not failed).** On the real splits this fires only for excess_vs_reference on B-val / C-val (rate 0.252 / 0.253, clipped to 0.25) and for realtext code entropy / jump25 (0.292 / 0.265, refit on val). One quantile over a whole split; negligible information about any single byte. No final-test split affected.
- **Not covered, worth fixing:** the BLT-1B scripts (realblt_budget.py, realblt_code.py, realblt_reason.py) pick the top R% of positions per problem, and entdep z-scores per problem, so whether t gets a boundary depends on entropies later in the same ~500-byte problem, answer bytes included. Same for every layout, so comparisons are fair-ish, but the fix is cheap: thresholds fitted on train problems (as BLT itself uses a global threshold).
- Not a leak, noted: in mathexp._accuracy the sp:* stride grid is relative to the sliding decode window, so its phase shifts each generated byte (same for every sp rule).

## BLT-1B layouts: train-fitted thresholds (#26, code change; reruns pending)
New `blt_layouts.py`, used by realblt_budget / realblt_code / realblt_reason. Default `SEGR_BLT_THRESH=train`: one threshold per layout and budget (and z-score parameters for entdep), fitted on the TRAIN problems so the mean patch rate there is exactly R, then applied position by position; tied scores (dependence tables are lookups) are accepted at the fraction needed, by a fixed hash of the position. `SEGR_BLT_THRESH=problem` reproduces the old top-R-per-problem selection.
- realblt_budget.py gained a train pass (entropy model only) over N_TRAIN GSM8K train problems (2nd argument, default 300); code and reason reuse their existing train passes.
- Checks: test_causal.py (problem mode = old inline code on 800 synthetic cases; train mode causal 0/480; problem mode changes earlier starts in 301/480; train rate = R). Old vs new scripts against a deterministic stub of BLT-1B: problem mode gives identical RESULT lines for budget, code, trace and logic; train mode runs, train rates = R.
- Not changed: realblt.py (its "jump at equal patch count" layout is also per problem) and the blt_screens/ coverage screens.
- The python env used for the BLT-1B runs is not recorded (transformers >= 5 with BltForCausalLM; `pytorch` env lacks it). Record it (#16).

### Budget sweep (D=128, 32k steps; 3 seeds at 10%, 2 at 15/20%): computed acc / final acc / bpb, means
| budget | BLT entropy | dependence (no labels) | hand-written syntax+entropy |
|---|---|---|---|
| 10% | 7.1 / 10.4 / 1.654 | 17.1 / 67.4 / 1.678 | 24.7 / 72.5 / 1.640 |
| 15% | 7.6 / 28.5 / 1.555 | 14.2 / 68.3 / 1.635 | 20.1 / 74.0 / 1.529 |
| 20% | 8.2 / 44.2 / 1.475 | 17.8 / 68.9 / 1.571 | 26.5 / 76.1 / 1.451 |
Seeds: dep10 computed 17.3/21.0/12.9, final 68.0/69.5/64.7; entropy10 final 15.5/10.2/5.5; syntax+entropy10 computed 25.3/25.3/23.4.
- Placement beats budget: at every budget, ordering is hand-written > dependence > entropy on both accuracies. Dependence at 10% beats entropy at 20% on final answers (67% vs 44%) with half the patches.
- Entropy needs budget to reach answers (final 10 -> 28 -> 44%); result-aware rules are flat (67-76%) from 10% up.
- Computed results do not improve with budget for any rule (arithmetic capacity-limited at this size; MATH ~3% everywhere).
- Dependence costs overall bpb (+1.5% at 10%, +5% at 15%, +6.5% at 20% vs entropy): it spends patches at arithmetic contexts that entropy would put at hard prose/LaTeX bytes. The hand-written rule has a lower bpb than entropy at every budget.
- Compute (flops.py): at this model size 10% costs 76% of entropy at 25%; on BLT-1B 62% (46% with a lookup rule).

### Logic and program traces with models trained here (reasonexp.py; D=128, 32k steps, 10% unless noted, 2 seeds)
computed / copy / logic answer / proof step exact (means):
- entropy (25%): 20.7 / 100 / 49.5 / 6.5
- entropy10: 20.6 / 100 / 49.9 / 6.5
- jump10: 26.4 (21.4, 31.4) / 100 / 49.2 / 7.0
- syntax+entropy10 (hand-written: after 'is now ', ', so X is ', 'Answer: '): 21.8 / 100 / 50.7 / 7.0
- entdep10: 20.5 / 100 / 49.9 / 7.2  (dep10 covered 0% of targets in the screen; swapped for entdep10 before training)
- **Null result: placement does not matter on these tasks at this size.** Entropy at 10% skips 83% of computed values and all answers, yet matches the hand-written rule that covers all of them.
- The logic skill is not learned (answers at chance, proof-step lookups 7%), so logic tests nothing here.
- Trace values are partly learned (~21%) but the operands sit within the local decoder's 32-byte window ('a = a + b\n# a is now '), so the local model sees them without a global step. Contrast GSM8K, where results depend on quantities stated far back. Hypothesis: the blind spot needs the value to depend on context beyond the local window. Test queued: same runs with SEGR_WINDOW=8.

### Window test: program traces with the local decoder window cut from 32 to 8 bytes (SEGR_WINDOW=8)
computed exact (bits) / copy / answer / step:
- entropy (25%, 1 seed): 24.5% (2.51) / 100 / 48.0 / 8.2
- entropy10: 20.4% [19.9, 20.9] (2.94) / 100 / 50.3 / 6.5
- **syntax+entropy10: 26.9% [28.6, 25.1] (2.40)** / 99.7 / 50.3 / 6.7
- With the operands outside the local window, a boundary at each traced value now helps: +6.5 points exact and -0.54 bits, seeds separated (at W=32 the gap was 1.2 points). Supports the refined claim: the blind spot needs a value computed from context beyond the local model's reach. Effect moderate; third seed queued.
- **Third seed weakens it (correction):** syntax+entropy10 s2 18.8% (2.82 bits) vs entropy10 s2 19.5% (2.83). 3-seed means: entropy10 20.1% [19.9, 20.9, 19.5], 2.91 bits; syntax+entropy10 24.2% [28.6, 25.1, 18.8], 2.54 bits. Gap 4 points, seeds now overlap; the bits gap (0.37) is more consistent. entropy at 25%: 24.5 / 34.6 (high variance). **Verdict: suggestive, not confirmed** -- the window explanation is plausible but this test is underpowered; the two-seed version overstated it (standard 3 again).

### #26 reruns: train-fitted thresholds (BLT-1B, env bltenv: torch 2.14.1, transformers 5.18)
Problem mode reproduces the old tables exactly (budget, code, trace), so the refactor and env are faithful. Train mode, exact match (patch rate):
| GSM8K in-line results | @15 train | @15 problem (old) | @10 train | @10 problem (old) |
|---|---|---|---|---|
| entropy | 33.9% (0.146) | 39.8% | 14.8% (0.096) | 18.1% |
| results (label) | 58.9% (0.146) | 61.3% | 41.5% (0.097) | 45.2% |
| dep | 71.7% (0.152) | 70.5% | 65.2% (0.101) | 64.2% |
| entdep | 72.1% (0.150) | 71.2% | 60.7% (0.098) | 60.3% |
Final answers (train mode): entropy 52.7 / 37.8%, results 56.5 / 49.0%, dep 32.0 / 4.1%, entdep 59.9 / 50.7% (@15 / @10).
- **Headline holds with causal thresholds:** results − entropy +25.0 points at 15% (old +21.5) and +26.7 at 10% (old +27.1); entdep − entropy +38.2 / +45.9 (old +31.4 / +42.2). Entropy loses most under train thresholds (covers 40.8% of results vs 48.4%), partly because test problems run slightly below the train rate for entropy (0.146 / 0.096 vs 0.150 / 0.100); the label-free rules are at or above R. Gaps are far larger than that compute difference.
- dep alone still wrecks final answers (4.1% at 10%); entdep keeps them (50.7%).
- Code and traces: train and problem modes agree within ~1 point on every layout; conclusions unchanged (small effect on copies, floor on computed trace values).
- The per-problem selection did not inflate the published gaps; if anything it favored the entropy baseline.

### Results registry (#15)
Every script now records each RESULT line as a JSON record (`registry.py`; live records in `results/registry/<script>.jsonl` with commit, config and metrics). The 347 results in the old logs are backfilled (`build_registry.py` -> `results/registry/backfill.jsonl`). `make_tables.py` rebuilds the headline tables (`results/tables.md`; checked against the tables above: larger model, budget sweep, BLT-1B train thresholds match) and `make_figures.py` the accuracy-vs-compute figure (`results/figures/acc_vs_compute.svg`). One log (mlx_math_small/words+jump_s1) is empty: that run was cut off and later replaced by the word-start-fix reruns.

### Statistics (#18) and end-to-end accuracy (#17)
- `stats.py` -> `results/stats.md`: seed audit (41 configurations below 3 seeds), headline training comparisons with seed separation, Welch and pooled intervals, and paired McNemar / problem-bootstrap tests for BLT-1B once per-target files exist (`results/items/`, now saved by the BLT-1B scripts). All main math claims separate across seeds (e.g. dep10 vs entropy10 on final answers +57.0 points, pooled 95% [+54.5, +59.5]); Scratchpad entropy-vs-random (+0.2) and window-8 traces (+4.1) do not.
- #17: BLT-1B cannot solve GSM8K end to end. With its own default patches and greedy one-shot decoding it copies the example solution or degenerates into repetition (0/4 in a check; `realblt_e2e.py` gives 0/2 under every layout, ~2 min per problem per layout). Teacher-forced scoring is the only informative measure at 1B; end-to-end needs a stronger model (deferred; MLX models are not saved after training, so their end-to-end eval needs retraining with a generation eval).
- **Paired tests on BLT-1B** (train-fitted thresholds; reruns reproduce the #26 tables exactly; `results/stats.md` section 3). GSM8K in-line results: hand-written results@15 vs entropy@15 +25.0 points [problem bootstrap +21.4, +28.7], McNemar p = 1e-44; entdep@15 vs entropy@15 +38.2 [+34.0, +42.4]; entdep vs results +13.2 [+9.8, +16.5]; at 10% +26.6, +45.9, +19.2 (all p < 1e-20). Final answers: entdep vs entropy +7.1 [+2.7, +11.9] p = 0.004 at 15%, +12.9 [+7.1, +18.7] at 10%; entdep vs results not significant (p = 0.13, 0.6). Code: entdep +2.4 / +2.0 (p < 1e-4), oracle +3.0 / +4.0: small but real. Logic: entdep hurts proof steps (-6.2 at 15%, p = 4e-6), answers at chance. Traces: computed values null for every layout (forced boundary +1.7 at 15%, p = 0.001; -0.3 at 10%).

### Third seeds (18 runs; results/logs/mlx_seed3/, stats in results/stats.md)
- Confirmed with 3 seeds (all seeds separated): budget sweep at 15% and 20% (dependence vs entropy +35.6 / +24.1 final, +6.5 / +9.0 computed; hand-written above both), dependence 10% vs entropy 20% on final answers (+23.4), larger-model attribution (words+syntax vs entropy +19.4 computed, vs words +14.9; stride6+syntax 16.9% in between), small-model math (words+syntax vs entropy +67.3 final).
- **Scratchpad Patching, correction:** the answer-start (syntax) trigger is unstable. Final accuracy by seed 49.2 / 31.4 / **8.3%** (none 5.2%); mean 29.6%. The entropy trigger equals random across 3 seeds (5.4% vs 5.4%), so "entropy trigger = random" holds, but "answer-start scratchpads give ~7x" does not hold reliably. Seeds 3-4 for syntax, entropy, random and dense8 queued.
- **Scratchpad Patching, 5 seeds (resolves the correction):** answer-start (syntax) trigger final accuracy 49.2 / 31.4 / 8.3 / 49.2 / 52.4% (mean 38.1%); entropy trigger 5.6%, random 5.9%, dense8 7.8%, none 6.1%. Syntax vs entropy +32.5 points, every syntax seed above every entropy seed (Welch 95% [+9.4, +55.6]); entropy vs random -0.3 [-2.2, +1.6]. The effect is real; seed 2 (8.3%) looks like an occasional training failure. Claim for the paper: the entropy trigger is indistinguishable from random for answers, and answer-start scratchpads give ~6x on average with one failed seed in five.

### BLT-1B adapted to a 10% budget (#11; realblt_finetune.py: LoRA r=16 on the global transformer and decoder cross-attention, 16.9M parameters, 1500 steps on GSM8K train, one run per rule)
In-line computed results, exact match given the true prefix (796 results, 294 test problems; rows = training layout, columns = test layout, all at 10% with train-fitted thresholds):
| trained under | entropy@10 | entdep@10 | results@10 |
|---|---|---|---|
| (untrained) | 14.8% | 60.7% | 41.5% |
| entropy | **26.9%** | 63.1% | 48.7% |
| entdep (label-free) | 19.3% | **70.4%** | 47.7% |
| results (hand-written) | 22.0% | 66.3% | **56.9%** |
- **The blind spot survives adaptation.** Each model under its own training layout: entdep 70.4% vs entropy 26.9%, +43.5 points [problem bootstrap +39.2, +47.7], McNemar p = 9e-83; hand-written 56.9% vs entropy, +30.0 [+26.2, +33.9]; entdep vs hand-written +13.4 [+10.3, +16.5].
- Training under entropy helps its own layout (14.8 -> 26.9%) but the same model does better under entdep (63.1%), a layout it never trained on.
- Final answers are not informative after fine-tuning: ~90-98% under most layouts, because the final answer repeats the last computed result and fine-tuning teaches the copy.
- Caveats: one run per rule (LoRA, not full training); teacher-forced scoring; the dependence table is fit on GSM8K train with untrained BLT-1B.
- **Three runs per rule (seeds 0-2, 2026-10-04):** each model under its own training layout, computed results exact: entropy 26.9 / 27.5 / 28.4% (mean 27.6), hand-written 56.9 / 56.0 / 57.9% (57.0), entdep 70.4 / 68.8 / 70.9% (70.0). Pooled paired tests over the three runs: entdep - entropy +42.4 [+38.6, +46.3], McNemar p = 2e-237; hand-written - entropy +29.4 [+26.0, +32.9]; entdep - hand-written +13.1 [+10.2, +15.8]. Run-to-run spread < 2.5 points.

### End-to-end GSM8K with the fine-tuned BLT-1B (#17; realblt_e2e.py, zero-shot in the training format, greedy, patch starts decided online, 50 test problems, seed-0 adapters)
- Final answers right, each model under its own layout: entropy 2.0%, hand-written 2.0%, entdep 2.0% (1/50 each). About half the generations never reach "The final answer is" within 320 bytes (26-27 of 50), and some degenerate into runaway digit strings.
- Patch rates during generation fall below the fitted 10% (6.2-8.3%): generated text is more predictable than the human solutions the thresholds were fitted on.
- **Inconclusive (floor):** even fine-tuned, BLT-1B cannot carry a GSM8K solution end to end, so layouts cannot be compared this way at 1B. The teacher-forced gaps (+42 points on computed results) do not translate into measurable end-to-end differences because whole-solution accuracy is ~2% for every layout. Needs a stronger base model or a longer fine-tune; deferred to #12.

### Learned patcher (#19; neural_patcher.py)
- v1, single-layout targets (262k math samples): per-sample gains are very noisy (sd 0.40 vs mean 0.057); held-out correlation ~0.07 for the 2-byte table and the network alike (16 bytes: 0.076). Answer-start coverage at 10%: table 51%, network 2 bytes 39%, 16 bytes 28%. Failed the screen; not trained.
- v2, targets averaged over 4 random layouts (SEGR_DEP_LAYOUTS=4; 262k samples each from math and from logic/traces), one network for both corpora: correlations double (math table 0.140 vs network 8 bytes 0.140; logic/traces table 0.158 vs network 0.147). Coverage at 10%, mixed 8-byte network: math computed-result starts **83%** (math table 37%), all math answer starts 44% (table 51%); logic answers **100%** (table 0%), proof steps 38%, trace computed values 0%.
- Training ndep10 (mixed 8-byte network) on math, 3 seeds, to compare with dep10 (17.1% / 67.4%) and entropy10 (7.1% / 10.4%).

### Beyond patching (#22; confidence_exit.py, Qwen2.5-0.5B draft vs 1.5B target, teacher forced)
- GSM8K: computed results are the easiest tokens (draft right 97.4%, confident 94%, wrong-when-confident 0.5% vs text 7.1%): likely contamination plus easy arithmetic given the worked line.
- Generated program traces (values not memorizable): computed values: draft-target agreement 52.9% (text 85.7%, copies 99.9%), draft confident on only 6.5%, wrong-when-confident 6.7% (text 0.4%).
- Reading: in token models a computed value's uncertainty shows in the model's own confidence, so speculative decoding pays a speed cost there (acceptance drops) but confidence-based exit is mostly protected (0.4% of computed tokens slip through). Unlike byte patching, where the patch decision for a result is made before its first byte. Partial support; not a correctness blind spot at this scale.
- **ndep10 trained (mixed 8-byte network, 3 seeds): computed 14.3% [8.4, 14.4, 20.2] / final 57.4% [60.6, 53.3, 58.2] / bpb 1.714.** Worse than the 2-byte table rule dep10 (17.1% / 67.4% / 1.678) on every measure, though far above entropy10 (7.1% / 10.4%). Its higher computed-result coverage (83% vs 37%) did not translate into accuracy, and it costs more bits per byte. **Negative for #19 as built:** the learned patcher transfers to logic answers in the screen but does not beat the table where it matters; the table stays the best label-free rule on math. Possible reasons: it covers fewer final answers (44% vs 51% of answer starts) and spends patches inside arithmetic expressions.

## Scaling pilot (#27): 1M -> 12M -> 50M trained from scratch (results/scale/, scale/)
PyTorch port of the harness (checked against MLX: logits 2.6e-6, identical masks and eval), one rented A40 (RunPod), 36 h, ~$19.
Corpus 1.90 GB: OpenWebMath shards 0-3 (221,264 docs; 324 dropped for 13-gram overlap with the eval problems) + GSM8K/MATH train x20 (10%). All rules at 10%, thresholds fitted on the corpus; dep uses the paper's table (a corpus-fitted table covered 0% of computed starts). 28 runs; one failure (1M dep s0 never learned, bpb 4.72), excluded.
| params | entropy final / computed | dependence | hand-written |
|---|---|---|---|
| 1.1M | 7.0 / 4.4 | 5.4 / 4.2 (2 seeds) | 45.5 / 4.6 |
| 11.7M | 46.6 / 5.8 | 56.7 / 7.7 | 76.3 / 12.9 |
| 53.7M | 59.0 / 13.9 | 78.9 / 35.9 | 85.5 / 43.4 (2 seeds) |
- dep - entropy on final answers: -1.6, +10.1, +19.8 (seeds separate at 12M and 50M); computed at 50M +22.0 (seeds separate). bpb at 50M: dep 1.420, entropy 1.439.
- 50M, one seed each: jump10 53.8 / 6.3 (lowest bpb of the 10% rules, 1.410): BLT's monotonic rule no longer recovers final answers at this scale. entropy20 (22.8% eval rate) 80.2 / 23.1: twice the budget matches dep10 on final answers but not computed results.
- End to end: at most 1 of 100 GSM8K problems for any run.
- Practical: entropy runs trained ~1.7x slower than dep runs at the same nominal rate on the shared GPU (patch bursts raise the per-batch maximum); not measured cleanly.

## BLT-1B rerun with the 512-byte window restored (2026-10-07/08)
transformers #49185: the HF conversion of BLT-1B lacks the 512-byte sliding window (entropy patcher, local encoder/decoder). Fix: upstream PR #49188 applied to transformers 5.18 in .venv-blt (patches/), loaded via blt_load.py (window set explicitly; SEGR_BLT_WINDOW=0 = released behaviour). Verified: first 512 bytes bit-identical; past 512 the released patcher collapses to 1.07 bytes/patch vs 3.69 fixed. All BLT-1B experiments rerun on the Mac (~18 h incl. a lid-close pause): dependence table, budget, Figure 1, code/trace/logic, 9 LoRA fine-tunes, answer-boundary test. Paper now uses these (window=512 rows, *_w512 items).
- GSM8K computed results (exact): default 76.8% (25.5% patches; released 73.0% at 28.4%); entropy 47.0 / 19.0, results 66.1 / 51.8, entdep 76.5 / 67.1, dep 77.3 / 67.8 (@15 / @10). Gaps: results - entropy +19.1 / +32.8; entdep - entropy +29.5 / +48.1 (released: +25.0 / +26.6 and +38.2 / +45.9).
- Final answers: at 15% the layouts no longer differ (+2.0 / +2.4, n.s.); at 10% +9.2 / +11.6 (p <= 1.4e-5).
- Fine-tuned (3 runs): entropy 32.9, results 62.7, entdep 72.7; entdep - entropy +39.8 (p 6e-217); entropy-trained models under entdep 66.2%.
- Code: default 83.3% (released 73.0%); forced-identifier gain shrinks to +0.7 / +1.4. Traces unchanged (inputs < 512 bytes). Logic similar.
- Answer boundary: removing it 97.0% -> 78.3% (released 87.7 -> 60.3).
- Figure 1 uses top-10%-per-solution layouts (equal counts, 42 each; train-fitted thresholds gave entdep 74 vs entropy 42 on this problem).

## Reuse vs novelty screen (#31; reuse_screen.py, 2026-10-10)
Idea (from assembly theory's reuse vs construction; framed as LZ77/MDL): computed values are NEW content, copies repeat the context. Signals within the 128-byte context: novelty (forward LZ match, non-causal diagnostic), suffix / ambig (causal match signals), novtab / novnet (causal, self-supervised: predict novelty from the last 2 / 16 bytes; target from the text itself, no model losses). Coverage of starts at a 10% budget, val split of the math and logic/trace corpora; thresholds and predictors fitted on 2M train bytes.
- **Diagnostic confirms the premise:** no 3-byte match in context at 94% of math computed starts (word starts 63%) and 85% of trace computed values vs **9%** of trace copies. Entropy does not track it: math entropy10 covers 0% of computed starts.
- **math:** novtab10 alone covers computed **100%**, copies 0%, words 7% (dep10: computed 37%, copies 100%). Likely rediscovers "after '='", but label-free and needs no reference model. novnet10: 94% computed, 23% copies.
- **traces:** novnet[reason]10 alone separates computed (86%) from copies (**0%**), which neither entropy (17% / 54%) nor entropy+dep (100% / 100%) does; entropy+novnet 75% / 10%. The causal match signal entropy+ambig10 (no fitting) covers 99% computed but also 85% copies.
- **Final answers: novelty marks them as copies** (math: 0% coverage by novtab; the final answer usually repeats the last result), yet boundaries there matter a lot in training (dep10 final 67% vs entropy10 10%). Reuse does not explain the final-answer effect; a rule must keep them (e.g. '#### ').
- Transfer not solved: 2-byte novtab transfers no better than the dependence table (math -> traces 0% alone); novnet[reason] on math 19%.
- Next if pursued: train novtab10 / entropy+novtab10 on math (3 seeds) against dep10 (17.1 / 67.4) and syntax+entropy10, plus a variant forcing final-answer starts; novnet on traces.
- **Trained (1M setup as dep10: D=128, 4 global layers, xattn, window 32, 32k steps, 3 seeds; computed / final / bpb):**
  - **nov10: 22.7% [23.4, 21.9, 22.8] / 76.2% [77.6, 74.8, 76.1] / 1.658**
  - final+nov10: 21.6% / 75.3% / 1.661 (forcing final-answer starts adds nothing)
  - entnov10: 8.6% / 62.9% / 1.662 (adding entropy breaks computed results despite 100% coverage in the screen; unexplained)
  - reference: dep10 17.1% [17.3, 21.0, 12.9] / 67.4% [68.0, 69.5, 64.7] / 1.678; syntax+entropy10 24.7% / 72.5% / 1.640; entropy10 7.1% / 10.4% / 1.654.
  - **nov10 beats dep10 on all three, seeds separating on final answers (min 74.8 vs max 69.5) and computed (min 21.9 vs max 21.0), and beats the hand-written rule on final answers (min 74.8 vs max 72.9)**; computed slightly below it (21.9-23.4 vs 23.4-25.3), bpb +0.018. The best label-free rule so far, and the cheapest: fitted from the text in seconds, no reference model, no losses.
  - Caveats: 1M model, math only; the table likely encodes "after '='" (computed starts are almost all there), so it is closer to a learned syntax rule than a general reuse detector; final answers score well although the rule never starts a patch there (0% coverage), so why final answers recover here but not under entropy10 is open. Next: scaling check (50M, scale/), traces (novnet), and why entnov fails.

## Typed uncertainty screen (local, 2026-10-11; typed_screen.py)
Entropy is capped by the number of options (a digit: at most 3.3 bits), so total ignorance of a computed value reads as less uncertain than a routine word start (math: 2.97 vs 4.14 bits). Split the small model's next-byte distribution over generic byte classes (digits, lower, upper, whitespace, punctuation, other): Hclass = uncertainty about the kind; T = P(likeliest class) x normalized within-class entropy (sure of the kind, ignorant of the item). Causal, label-free, no math markers. Coverage at 10% (val; thresholds on 2M train bytes):
- math: T-Hclass10 (standardized T minus Hclass) covers computed 98.4%, copy 99.9%, final 99.8%, step 0.1%, other words 8.3%; T10 99.5 / 100 / 99.8 / 5.0 / 30.2. entropy10: 0 / 0 / 0 / 60.4 / 46.6. About the hand-written rule's coverage (100 / 100 / 100 / 100 / 39) without knowing any math syntax.
- traces: T10 covers computed 100% (entropy10 17%), copies 100%, logic answers 0% (Yes/No: low within-class ignorance), steps 100%.
- Does not separate copies from computed values: the tiny entropy model (d=32, 1 layer) cannot copy, so it is equally ignorant at both. A stronger model (BLT-1B's entropy model) should be confident at copies, so T might separate them there.
- Next: 1M training runs (T-Hclass10, T10, 3 seeds); T on BLT-1B (post-hoc layouts, GSM8K).
- **Trained (1M setup as dep10, 3 seeds; computed / final / bpb):**
  - typed10: 25.4% [26.6, 28.8, 20.8] / 70.0% [69.7, 69.8, 70.5] / 1.703
  - typedc10: 22.7% [25.2, 22.1, 20.8] / 66.7% / 1.711
  - reference: nov10 22.7 / 76.2 / 1.658; dep10 17.1 / 67.4 / 1.678; syntax+entropy10 24.7 / 72.5 / 1.640; entropy10 7.1 / 10.4 / 1.654.
  - typed10 has the highest mean on computed results of any rule (seeds overlap with the hand-written rule and nov10; one low seed), beats dep10 on final answers narrowly (min 69.7 vs max 69.5) but is clearly below nov10 there (76.2), and has the worst bpb (+0.045 vs nov10): spending patches on every value position, copies included, costs on ordinary text. Complementary to nov10 (computed vs final) -> typednov10 queued.
  - typednov10 (typed + novelty table, standardized sum): 18.2% [17.8, 17.7, 19.1] / 73.7% [74.5, 72.7, 73.9] / 1.700. Worse than either signal alone on computed results and below nov10 on final answers: combining does not give the best of both (as entnov10 before, a standardized sum with the near-binary novelty table changes the ranking for the worse). nov10 stays the best-balanced label-free rule; typed10 the best on computed results.

### Typed uncertainty on BLT-1B (local; realblt_typed.py, 294 GSM8K test problems, 796 in-line results, window 512, train-fitted thresholds)
T read from BLT-1B's own entropy model (bytes = token ids - 4). Baselines reproduce the paper exactly (entropy 47.0 / 19.0, results 66.1 / 51.8, entdep 76.5 / 67.1 at 15 / 10%).
| layout | covers results / final / other numbers | computed exact | final exact |
|---|---|---|---|
| entropy@15 / @10 | 48 / 98 / 71 ; 16 / 94 / 52 | 47.0 / 19.0 | 65.0 / 51.4 |
| typed@15 / @10 | 72 / 3 / 49 ; 66 / 0.3 / 32 | **64.2 / 48.7** | 46.6 / 32.0 |
| typedc@15 / @10 | 60 / 0 / 23 ; 53 / 0 / 17 | 50.0 / 36.4 | 43.9 / 30.6 |
| results@10 (label-based '= ') | 100 / 93 / 50 | 51.8 | 60.5 |
| entdep@10 | 100 / 90 / 52 | 67.1 | 62.9 |
- typed vs entropy, paired: computed +17.2 (15%, p 3e-18) and **+29.7 (10%, p 2e-43)**, label-free and without math syntax; at 10% close to the '=' rule (51.8). Below entdep (67.1), which uses a GSM8K-fitted table.
- **The stronger model separates copies:** typed covers 32% of other number starts at 10% vs entropy 52%, while covering 4x the computed results (66% vs 16%). The tiny model of the screen could not (it cannot copy).
- **Final answers collapse:** typed puts a patch at 0.3% of final-answer starts (BLT-1B can predict the answer, which repeats the last result), and final exact drops to 32.0% (entropy 51.4%, p 7e-8). A patch at the final answer matters even when its value is predictable (consistent with the answer-boundary test: 97.0 -> 78.3% without it). A usable rule needs typed uncertainty plus the answer boundary.

### Combinations on BLT-1B (local; realblt_combo.py, same 294 problems; computed / final exact at 10% and 15%)
| layout | @10 | @15 |
|---|---|---|
| entropy | 19.0 / 51.4 | 47.0 / 65.0 |
| results ('= ', label-based) | 51.8 / 60.5 | 66.1 / 67.0 |
| entdep (GSM8K-fitted table) | **67.1** / 62.9 | **76.5** / 67.3 |
| typed | 48.7 / 32.0 | 64.2 / 46.6 |
| **typed+final** (boundary after 'The final answer is ', rest typed) | 48.4 / **64.6** | 63.9 / **70.4** |
| nov (novelty table fitted on GSM8K train text) | 56.5 / 19.4 | 66.5 / 36.4 |
| max (union of typed and nov top sets) | 52.4 / 36.4 | 64.9 / 33.7 |
| max+final | 51.9 / 66.0 | 64.7 / 68.4 |
- **The marker boundary fixes final answers at no cost to computed results:** typed+final vs typed: final +32.7 (p 2e-22), computed -0.4 (n.s.). vs entropy: computed +29.4 (p 5e-42), final +13.3 (p 7e-7). Final answers at or above every other tight layout (vs entdep +1.7 n.s. at 10%, +3.1 at 15%); computed equal to the label-based '=' rule (-3.4, n.s.) but below entdep (-18.7, p 1e-21).
- Novelty on BLT-1B: computed 56.5% at 10% (above the '=' rule, +4.8, p 0.01) but final answers collapse (19.4%), as typed alone.
- The union does not beat its better member on computed results (52.4 vs nov 56.5); with the final boundary 51.9 / 66.0. No gain from combining typed and novelty here either.
- Reading: the computed-result gap to entdep is about the REST of the budget, not coverage (typed covers 66% of results, nov and entdep 100%, but nov's 100% gives 56.5 vs entdep 67.1). Final answers need a boundary regardless of predictability; a format marker supplies it.

### Follow-ups on BLT-1B (local; 2026-10-11)
**GSM8K, novelty + final boundary** (realblt_combo.py, SEGR_COMBO_NAME=novfinal): nov+final 54.8 / 65.6 at 10%, 66.5 / 69.4 at 15% (computed / final). The marker boundary fixes novelty's final answers (19.4 -> 65.6) at little cost (56.5 -> 54.8). Best label-free layout on BLT-1B: above the '=' rule on both (51.8 / 60.5) and typed+final on computed (48.4); entdep still best on computed (67.1).
**Traces** (realblt_reason.py trace, SEGR_REASON_TYPED=1; 300 programs, 2469 computed / 1716 copy values): computed values stay at the floor for every layout (BLT-1B default 20.5%, oracle 12.8 / 9.7%), so only copies separate. typed@15: computed 12.3% (covers 83% vs entropy 36%), **copies 76.7%: best of all tight layouts** (entropy 66.7, +10.1, p 5e-24; entdep 67.2, +9.6; oracle 72.2, +4.5, p 2e-6). At 10%: copies 62.9 vs entropy 58.3 (+4.6, p 1e-5), entdep 62.3 (n.s.). Typed does not skip trace copies (covers 93% at 15%), but the placement helps them more than any other rule, including the label oracle. Domain-free: the math dependence table covers 0% here (paper), entdep needs a per-task refit.
**Logic** (realblt_reason.py logic; 300 problems, 1196 proof steps, 300 answers): typed@10 steps 52.7% vs entropy 46.3 (+6.4, p 4e-8), entdep 41.5 (+11.2, p 6e-13), oracle 49.4 (+3.3, p 0.01): **best of all tight layouts at 10%**. At 15% all within ~1.5 except entdep (-7). Yes/No answers near chance for every layout (BLT-1B cannot do the chaining); typed+final@10 choice 59.0% (vs entdep +8.3, p 3e-4).
**Overall:** typed uncertainty is the only label-free rule that is never worse than entropy and is the best or tied-best tight layout on traces and logic, where fitted tables fail or need refits; on GSM8K the GSM8K-fitted dependence table stays best on computed results. With a format-cue boundary for the final answer it is competitive everywhere.
**Code** (realblt_code.py, SEGR_CODE_TYPED=1; 300 Python stdlib chunks, 5490 repeated identifiers = copies): every layout within ~3 points (default 83.3%). typed covers fewer repeated identifiers than entropy (39.9 vs 43.4% at 15%, 26.7 vs 30.1% at 10%), as intended (BLT-1B can copy them), and loses a little: 82.3 vs 83.1 (15%, -0.8, p 0.01), 80.7 vs 81.7 (10%, -1.0, p 0.006); oracle 83.8 / 83.1. Consistent with copies gaining little from a boundary; unlike traces, where typed's placement helped copies (+10).

### Domain-free final-answer fixes on BLT-1B (local; realblt_general.py, same 294 problems; computed / final exact)
| layout | @15 | @10 | final starts covered @15 / @10 |
|---|---|---|---|
| typed | 64.2 / 46.6 | 48.7 / 32.0 | 3% / 0.3% |
| typed+final (format cue) | 63.9 / **70.4** | 48.4 / **64.6** | 100% / 100% |
| typed+stale (z(T) + gap/8) | 65.5 / 51.7 | 53.9 / 25.2 | 3% / 2% |
| typed+cap (start after 20 bytes without one) | 64.8 / 43.5 | 52.9 / 26.2 | 3% / 4% |
| typed+dec (decoder's own typed uncertainty under typed@0.8R, added to R) | **67.2** / 50.3 | 50.4 / 31.3 | 28% / 4% |
- **None recovers final answers** (best +5.1 at 15% for stale, p 0.02; at 10% stale and cap lose 6-7 points). Staleness and the cap shorten gaps but do not land on the answer's first byte; spreading starts helps computed results instead (stale +5.2 at 10%, p 0.002; cap +4.1).
- **The decoder rarely flags final answers** (28% / 4% of final starts) although it gets most of them wrong (32% exact under typed@10): its errors there are confident, not uncertain, so no uncertainty signal (patcher's or decoder's) can find them. Final answers need a boundary for a reason other than ignorance: plausibly structural (the answer must be read off the whole solution; a fresh global step realigns the decoder), which the format cue, the dependence table (90% coverage, learned from loss) and entropy (93%) supply.
- Takeaway: keep the format cue (or a loss-learned boundary signal) for answer starts; typed uncertainty for computed values. A domain-free answer detector would need a loss-based (dependence-style) signal, not an uncertainty one.

### Must retrieve vs must compute on BLT-1B (local; realblt_gain.py, same 294 problems; computed / final exact)
| layout | @15 | @10 | final starts covered @15 / @10 |
|---|---|---|---|
| typed | 64.2 / 46.6 | 48.7 / 32.0 | 3% / 0.3% |
| typed+final (format cue) | 63.9 / **70.4** | 48.4 / **64.6** | 100% / 100% |
| gain (context gain, W = 32 tokens) | 29.4 / 5.4 | 19.0 / 3.4 | 1% / 0.3% |
| typed+gain (union of top sets) | 59.4 / 22.8 | 43.2 / 14.6 | 0.3% / 0% |
| typed+dis (patcher confident, decoder disagrees; added to typed@0.8R) | 64.7 / 39.1 | 47.9 / 25.9 | 0.7% / 0% |
- **Both fail, and context gain is harmful** (final answers 5.4%, below every other layout; eval rate also under budget, 13.9 / 9.1%). The last result is usually within the 32-token short context, so the final answer is not "far back"; the bytes with high context gain are elsewhere and the budget spent there starves the rest.
- **Disagreement does not flag final-answer starts** (0.7% / 0%): at the answer's first byte the decoder and the patcher agree. The errors come later in the answer, or from a stale global state that byte-level comparison does not reveal.
- Reading (with realblt_general.py): the benefit of a boundary at an answer start is for the whole multi-byte unit that follows, decoded from one fresh global state; no per-byte signal of ignorance, retrieval distance or disagreement marks it. Answer starts are a structural kind of boundary (a segment start), distinct from computation points. Signals that do find them: the format cue (100%), entropy (93%: BLT-1B's entropy spikes there), and the loss-learned dependence table (90%).
- Takeaway for the second paper: two kinds of boundary. Computation points: typed uncertainty finds them domain-free. Segment starts (answers): need structure, from a format cue or a loss-learned signal. Uncertainty-based signals of any kind (patcher, decoder, disagreement, context gain) do not find them.
- **final+typed10 trained (1M, 3 seeds; answer cue '#### ' / '\\boxed{' + typed):** computed 25.1% [24.6, 26.0, 24.8] / final 70.3% / boxed 2.9% / bpb 1.704. Same as typed10 (25.4 / 70.0 / 2.3 / 1.703), with less seed spread on computed results. At 1M the cue adds nothing: the tiny patcher is ignorant at GSM8K answers, so typed already covers 99.8% of them, and MATH boxed answers are at the floor for every rule (~2-3%). The answer-start problem is a property of strong patchers (BLT-1B's can predict the answer); the 1M setup cannot show it. nov10 stays best on final answers here (76.2).

### Literature check: confidence-based compute allocation beyond patching (local, 2026-10-12)
Prompted by fast "System One" decision models (typesafe.ai Jev: typed outputs, one parallel pass, calibrated probabilities; architecture undisclosed).
- Confidence-gated CoT is an active area: Lewis-Lim et al. 2025 (arXiv 2510.21007) "Can Confidence Estimates Decide When CoT Is Necessary?": first systematic study; training-free confidence measures reduce redundant reasoning but no measure is consistent across settings. Many fast/slow routers (AdaptThink, CDR 2508.16636, CogRouter 2602.12662).
- Uncertainty-triggered extra computation exists: Kim et al. ACL Findings 2025 (DIT, 2506.03616) insert [PAUSE] tokens at low-likelihood positions in fine-tuning (GSM8K +4.7); Galashov et al. 2025 (2510.13879) let the model request pauses. ARES (2510.08457) EXCLUDES digits from its entropy triggers as noise: the opposite of our finding that digit positions after '=' are where computation is needed.
- LLM overconfidence on math is well documented (Xiong et al. 2306.13063: verbalized confidence near chance for GSM8K failure detection; sampling agreement fixes much of it); the most confident errors are the hardest to catch.
- Computed vs copied values: CoT tokens act as program variables (2505.04955); small models copy the last CoT number on readout (2605.22870); extraction studies route arithmetic to code. No paper found that uses a copied-vs-computed field split as the axis where confidence fails.
- No paper found defining uncertainty WITHIN the expected output type (kind vs item, a chain-rule split of next-token entropy over byte/token classes). Semantic entropy (Farquhar et al., Nature 2024) splits over meaning classes of whole answers, a different object. Search not exhaustive.
- Implication: the routing experiment is crowded; a contribution would be (1) computed vs copied fields as the axis where confidence-based allocation fails, (2) typed uncertainty as a gate compared with the measures of Lewis-Lim et al., (3) confident errors on computed values needing a structural signal. Position against Lewis-Lim et al., Kim et al. and ARES.

### H-Net blind-spot check (#21; local; scale/cloud/hnet_check.py on the RunPod A40, 2026-10-12)
Public checkpoints hnet_1stage_XL and hnet_2stage_XL (goombalab/hnet), 1292 GSM8K test solutions (<= 1000 bytes; 3547 in-line results), native chunk rates, boundaries not changed. results/hnet/.
- H-Net chunks like BPE tokens: a chunk starts AT the space before a word or number (2-stage stage 0: 69% of word starts have the chunk start 1 byte back). Counting the preceding space:
  - 1-stage (chunk rate 0.27): computed results 100%, final answers 100%, other numbers 81%, word starts 98%.
  - 2-stage, stage 0 (0.445): everything 100%, words 98%.
  - 2-stage, stage 1 (outer, 0.184): final answers 100%, words 81.5%, other numbers 60%; computed results 0.1% at the result or its space, but **100% have an outer chunk start exactly 2 bytes back, at the '='**: the outer chunk is "= 19". H-Net's main network takes a step at '=' with the whole expression in view, so it does allocate a coarse boundary at every computation point.
- **No blind spot in H-Net at its native rates:** the end-to-end learned chunker discovered '=' as a chunk start (the hand-written rule, learned). Supports the paper's claim that the blind spot is specific to entropy-triggered patching; a chunker trained on its own loss finds computation points (like the loss-based dependence signal).
- H-Net is confident at computed results (next-byte entropy 1.18 bits vs words 2.11; 47.5% exact), plausibly GSM8K-like data in FineWeb-Edu; its exact match cannot be split by outer-chunk coverage (all covered).
- Caveat: native rates only (18-45% of bytes); H-Net's boundaries are learned, so a tight-budget test would need changing its routing threshold.

### 1M runs of the new rules on Kaggle (local; T4 x2, fp16; scale pipeline, unclipped like the pilot; 2026-10-12)
Session 2 (session 1 failed: -1e9 mask value overflowed fp16; fixed in scale/model.py). Corpus and masks identical to the pod (same SHA-256, thresholds and rates). 13 min per 1M run on a T4; 1.8 h for the session. computed / final / bpb (3 seeds):
- nov10 3.7 / 12.6 / 2.125; typed14 4.2 / 3.9 / 2.196; final+typed14 4.7 / 4.8 / 2.167
- pilot: entropy10 4.4 / 7.0 / 2.093; dep10 2.8 / 3.6 (one failed seed); syntax+entropy10 4.6 / 45.5 / 2.109
- At the floor, as the pilot found at 1M: computed results 3-5% for every rule. The answer cue alone (final+typed14) does not give the hand-written rule's final answers (45.5%), although both start a patch after '#### ': at this size the hand-written rule's other markers ('=', '>>') matter too. 12M is the informative size on Kaggle.

### Copied vs computed fields, fast vs slow (local; fields_bench.py, Qwen2.5-7B-Instruct MLX 4-bit, 50 invoices, 500 fields)
fast / slow accuracy: copied 99 / 95.5; lookup (most expensive item) 88 / 94; computed 48 / 60 (item count 100 / 100, total quantity 64 / 78, subtotal 0 / 52, total 0 / 32, due date 76 / 36).
- The copied-vs-computed split is large in fast mode (99 vs 48), as predicted; writing out the working helps computed fields (+12) but hurts date arithmetic (due date 76 -> 36).
- Confident errors (fast answer probability > 0.8 while wrong): all 6 wrong lookups; 7 of 18 wrong total quantities (small sums); 4 of 12 wrong due dates; 3 of 50 subtotals; 0 of 50 totals. Hard computations (multi-digit sums) give uncertain errors, which confidence catches; easy-looking ones (comparisons, small sums, date offsets) give confident errors, which it cannot.
- Routing at 30% escalated: mean entropy 82.2%, answer probability 80.0, typed uncertainty 79.6, field kind 79.4; oracle 85.4. For this model typed uncertainty and field kind add nothing over plain confidence: its computed errors are mostly uncertain.
- Revised reading: the blind spot of confidence-based routing is not "computed values" as such but computations that LOOK easy (the model attempts them in one step and is confidently wrong).

### 12M runs of the new rules on Kaggle (local; T4 x2, fp16, unclipped like the pilot; 2026-10-13)
~1 h per 12M run; 5.7 h session. computed / final / bpb (3 seeds):
- nov10 10.1 [14.2, 9.6, 6.5] / 71.5 [72.3, 71.2, 70.9] / 1.674
- typed14 6.2 / 18.9 [26.2, 19.1, 11.4] / 1.678; final+typed14 5.9 / 25.2 [12.9, 48.9, 13.6] / 1.712
- pilot: entropy10 5.8 / 46.6 / 1.657; dep10 7.7 / 56.7 / 1.673; syntax+entropy10 12.9 / 76.3 / 1.645
- nov10 - entropy on final answers +24.8 [+14.7, +35.0], nov10 - dep10 +14.8 [+8.6, +21.0] (seeds separate). typed14 - entropy -27.7 (seeds separate): the trained-typed failure is present at 12M without clipping, so it is the rule, not clipping or instability. The answer cue does not rescue it (as at 1M).

### Novelty fitted on the training corpus itself, 12M (local; Kaggle T4, unclipped like the pilot, 2026-10-13)
novelty_corpus.npz: the novelty table fitted on 4M bytes sampled from the 1.9 GB training corpus (mostly OpenWebMath), not on GSM8K/MATH solutions. Coverage on the eval text: computed starts 99%, final 100%, copies 0%. Eval rate at a 10% corpus budget 8.4%, at 13% 11.4% (matched to entropy10 11.2%). final / computed / bpb (3 seeds):
- novc13 69.8 [71.2, 69.8, 68.5] / 10.2 / **1.619** (best bpb at 12M); novc10 (eval rate 8.7%) 67.4 / 8.1 / 1.732
- vs math-fitted nov10 71.5 / 10.1 / 1.674; entropy10 46.6 / 5.8 / 1.657; dep10 56.7 / 7.7 / 1.673; syntax+entropy10 76.3 / 12.9 / 1.645
- novc13 - entropy10 final +23.2 [+12.7, +33.8], - dep10 +13.2 [+8.1, +18.3], seeds separate. A corpus-fitted dependence table covered 0% of computed starts (pilot), so novelty is the rule that survives fitting on raw training text. Main claim candidate: a patching rule fitted from the training text alone (no labels, no curation, no model run) beats entropy patching with no cost in bits per byte.

### Outside math: code and prose, 12M (local; scale/generic.py on Kaggle T4, unclipped, 2026-10-13)
Per domain: entropy model and novelty table fitted on the domain's own text; novelty's budget set so its EVAL patch rate matches entropy10's. Bits per byte on held-out text (seeds 0 / 1):
- **prose (WikiText-103):** entropy10 1.810 / 1.854 (mean 1.832); **novelty 1.801 / 1.738 (mean 1.770)**: better by 0.062, every novelty seed below every entropy seed. Rates matched in training (10.0 vs 10.5%) and eval (10.1 vs 10.3%). Prose is little repetitive (11.6% of val bytes copied).
- **code (installed Python files, ~600 MB):** **entropy10 1.652 / 1.647 (mean 1.650)**; novelty 1.680 / 1.689 (mean 1.684): worse by 0.034, on new (+0.023) and copied (+0.056) bytes. Confounded: matching the eval rate (7.6 vs 7.7%) left novelty with a third fewer patch starts in TRAINING (6.6% vs 10.0%). Code is repetitive (36.6% copied).
- Reading: novelty beats entropy on general prose at equal compute (not math-specific); code inconclusive until novelty trains at the same 10% budget. Only bits per byte here: no accuracy target outside math.
- **Code rerun, training-matched pairs (same corpus, SHA-256 f510d0fd...):** novelty_t10 (10% training budget, eval rate 13.1%) 1.537 / 1.545 (mean 1.541) vs entropy10 (10%, eval 7.6%) 1.650; entropy_tnov (6.6% training, eval 5.9%) 1.694 / 1.728 (mean 1.711) vs novelty (6.6%, eval 7.7%) 1.684. At equal TRAINING budget novelty wins in both pairs (seeds separate); at equal EVAL budget entropy wins (1.650 vs 1.684, novelty trained with a third fewer patches). On code novelty puts relatively more starts on held-out files than on the training corpus, so no single threshold matches both budgets; settling it needs entropy at several budgets (accuracy-vs-compute curves). Outside math overall: prose clean win; code better at equal training compute, undecided at equal inference compute.

### 50M clipped runs complete (local; RunPod A40, 11 runs, pod terminated 2026-10-14 00:29 UTC after verified download)
computed / final / bpb (clipped, seeds):
- nov10 57.5 [56.3, 58.5, 57.7] / 89.5 [90.6, 89.4, 88.6] / 1.446
- typed14 8.9 [9.9, 7.7, 9.0] / 35.1 [50.6, 43.5, 11.1] / 1.410; final+typed14 9.5 / 65.7 [68.0, 63.9, 65.2] / 1.405
- clipped checks (seed 0): entropy10 12.1 / 61.4 / 1.414; dep10 33.4 / 78.9 / 1.422 (G1 passed; G2 not met, see prereg)
- typed14 fails as a training rule at 50M on every seed (seed 2 final answers 11.1%); the answer cue rescues final answers to ~66% but not computed results (9.5%). typed rules have the best bits per byte (1.405-1.410) and the worst computed accuracy: patches spent on value positions help general prediction, not the arithmetic.
- **Code, entropy at more budgets (Kaggle v7, same corpus f510d0fd...):** entropy_t14 (eval 9.5%) 1.611 / 1.603 (mean 1.607); entropy_t17 (eval 11.7%) 1.581 / 1.551 (mean 1.566). Entropy curve on code (eval rate -> bpb): 5.9% 1.711, 7.6% 1.650, 9.5% 1.607, 11.7% 1.566; extrapolated to 13.1% ~1.540, i.e. tied with novelty_t10 (1.541 at 13.1%); at 7.7% entropy ~1.648 vs novelty 1.684. Verdict: on code, at equal INFERENCE compute novelty is tied or worse; it wins only at equal training budget. Outside math: prose clean win for novelty, code no win. (Extrapolation 1.4 points beyond the last entropy point; two seeds each.)
- **Does novelty catch structure? (scale/novstruct.py, Mac, seconds.)** Math (val, 3 MB): nov10's top contexts are "word?" (end of a question -> the solution starts); it covers 75% of line starts (10% of its boundaries vs 1% chance) but only 25% of the bytes after '=' (4% of boundaries vs 2% chance); novc13 similar (57% of line starts). Code (1.7 MB local Python, table fitted on half, 10% train rate): 0% of statement starts, 1% of right-hand sides ('= ', 'return ', '('); 62% of its boundaries are mid-identifier (chance 50%), top contexts inside hex/escape literals ('x8', '\\x', '{0'): it finds genuinely unrepeated data, not structure. Reading: the 2-byte context cannot see "start of a statement" (indentation contexts are ambiguous), so on code novelty spends its budget on literals. In math it catches segment starts (lines, after questions), which may be why it helps final answers. Caveat: small local code sample, not the Kaggle corpus.
- **Label-free structure screens (scale/learned_boundary.py on the Mac; scale/cloud/hnet_code.py on the novc pod).** Coverage at 10% rate on held-out bytes (statement starts / right after '=','return ','(' / mid-identifier share of boundaries vs chance):
  - Code, novelty: 2-byte table 0% / 0% / 57% (chance 50); 128-byte network (predicts the target better: rank corr 0.52 vs 0.39) 9% / 2% / 68%; **the true target itself 18% / 12% / 49%**. In code, "new content" is not structure: the target, not only the predictor, misses it.
  - Math, novelty: table 75% line starts / 26% after '='; network 38% / 21%; true target 52% / 15%. Novelty catches segment starts in math.
  - Gain target (6000-update reference): held-out rank corr 0.03-0.10 for table and network, i.e. too noisy to learn at this size; inconclusive (paper's reference used 16000 updates, and ndep10 was already negative on accuracy).
  - **H-Net (public checkpoints, 500 stdlib chunks):** 2-stage outer level (rate 26%) starts chunks at 45% of statement starts and 35% of right-hand sides (with preceding space), only 4% inside identifiers; stage 0 / 1-stage chunk at nearly every token start (88-100% with space). Learned end to end from LM loss on web text, it recovers word-level structure in code; the proxies (novelty, 2-byte gain) do not.
  - Reading: label-free structure comes from end-to-end learned chunking, not from a hand-chosen proxy target. Novelty is a cheap proxy that happens to align with segment starts in math only.
- **Cross-rule test (Kaggle v8; 12M trained with nov10, then evaluated with other patchers, 2 seeds):** own layout nov10: bpb 1.668 / 1.650, final 68.9 / 78.3%, computed 12.0 / 12.2%. Switched at inference to entropy10: bpb 2.56 / 2.51, final 0 / 0%, computed 3.8 / 2.6%; typed14: bpb 2.90 / 2.83, final 0.9 / 0%; final+typed14 likewise. **Collapse:** a 12M model trained with one layout cannot read another (no fine-tuning). "Novelty for training, typed at inference" does not work at this size. Contrast: BLT-1B accepted typed layouts at inference (trained on entropy boundaries, much larger, longer training); the swap needs either scale or a short adaptation fine-tune (realblt_fullft-style).
- **50M novc13, clipped, 3 seeds (RunPod A40, ~9.7 h, ~$5.7):** computed 54.7 / 58.0 / 59.1 (mean 57.3), final 86.4 / 85.3 / 87.6 (mean 86.4), bpb 1.436. vs nov10 clipped 57.5 / 89.5 / 1.446; entropy10 clipped 12.1 / 61.4 / 1.414. Computed: tie. Final: -3.1 points (seed spreads do not overlap; ~3 SE of the difference). bpb: novc13 better by 0.010. Same pattern as 12M (bpb better, final slightly lower). Reading: the novelty rule does not need GSM8K/MATH-solution fitting; fitted on the raw corpus it keeps the whole computed-result gain and nearly all the final-answer gain over entropy (+25 vs +28 points). H-Net code check ran on the same pod (results/learned_boundary/hnet_code.json).
- **Code accuracy (Kaggle v9; 12M, equal eval rate 7.6 vs 7.7%, 2 seeds; teacher-forced exact match of whole tokens, second half of each 128-byte window):** entropy10 vs novelty, diff +- binomial SE (seeds agree within ~0.5): copied identifiers 24.3 vs 24.1 (-0.2 +- 0.2, n 31k); **new identifiers 15.0 vs 15.8 (+0.9 +- 0.1, n 61k)**; new numbers 14.2 vs 13.4 (-0.8 +- 0.6); **values after = / return / ( that were copied 21.4 vs 19.6 (-1.8 +- 0.5)**; new values 16.0 vs 15.7 (-0.3 +- 0.5). bpb 1.655 vs 1.684. Reading: code is a tie with a small trade-off (novelty slightly better on new names, slightly worse on copied values); no hidden accuracy win. Code stays "equal" in paper 2.
- **Boundary enrichment (SOMBRERO's B, scale/enrichment.py; math eval 0.9 MB, surprisal from the bundle's small byte LM, exact 50M thresholds):** entropy10 B 1.61 (Z 215), covers 1% of computed-result starts, 50M computed 12.1%; syntax+entropy10 B 1.52, 100%; dep10 B 1.02, 37%, 33.4%; nov10 B 1.13, 100%, 57.5%; novc13 B 1.03 (near chance), 99%, 57.3%. Computed-result starts have average surprisal (3.34 vs 3.12 bits all bytes): not hard for the patcher's model, so surprisal-seeking boundaries skip them. **High enrichment goes with the WORST computed accuracy; the best rules sit near B = 1.** Direct counterpoint to SOMBRERO's objective (raise B).
- **50M clipped baselines, seeds 1-2 (RunPod A40, ~12.3 h, ~$7.3):** entropy10 computed 12.1 / 27.5 / 22.6 (mean 20.7 ± 7.9), final 61.4 / 59.5 / 58.9 (59.9 ± 1.3), bpb 1.416; dep10 computed 35.1 ± 1.8, final 78.2 ± 0.9, bpb 1.420. Novelty (57.5 ± 1.1 / 89.5 ± 1.0) beats both by wide margins with every seed separating; entropy's computed accuracy is noisy across seeds but its best seed (27.5) is far below novelty's worst (56.3). All 50M cells now 3 seeds.
- **Execution traces (Kaggle v10; synthetic straight-line programs printing values after ' -> '; 12M, eval rate 10.0 vs 10.3%, 2 seeds):** entropy10 computed 61.2 / 49.0 (mean 55.1), copied 60.7 / 53.1 (56.9), bpb 1.104 / 1.151; novelty computed 50.4 / 50.0 (50.2), copied 63.5 / 58.6 (61.1), bpb 1.167 / 1.147. Seed spread for entropy is large (12 points), so computed is a tie within noise (-4.9, seeds overlap); novelty slightly better on copied values (+4.2), slightly worse bpb (+0.03). Reading: in text with little prose, entropy's budget is not drawn away from computed values (digits are the hardest bytes), so the blind spot does not appear and novelty has nothing to fix. Consistent with the structure account: the math gain comes from entropy being pulled toward prose/word starts. Trace domain = no advantage (scope limit), not a failure.
- **Trace coverage check (scale/trace_coverage.py, Mac, rules rebuilt as in prep):** at ~10% rate, NEITHER rule starts a patch at computed values (0%) or copied values (0%). Entropy spends its budget on operands after the operator (82% covered) and literals (53%); novelty on line starts (100% of variable names, 73% of its boundaries). **My earlier reading (entropy covers the values in prose-free text) was wrong:** the tie is because both rules miss the values. Novelty's 2-byte context ("> ") cannot tell a computed line from a copy line, and entropy finds operands more uncertain than results. Reading: novelty is not a general computation finder; in math it helps through segment starts (line / solution starts), which carry over to final answers and in-line results there but not to the ' -> ' value slot here. Consistent with the structure account (Section 6): a structural boundary after ' -> ' (or a learned chunker) would be needed in traces.
- **Context-aware novelty on traces (scale/trace_novnet.py, Mac, 2 min):** coverage of computed / copied value starts at 10%: 2-byte table 0% / 0% (100% of line starts); **128-byte network 74% / 0%** (also 55% of literals, 35% line starts); the true target itself only 20% / 1% (its top 10% is mostly ties among all unrepeated bytes; the network's EXPECTED novelty ranks computed values first). The predictor, not the target, was the problem in traces: with context, novelty separates computed from copied values without markers. Next: train with it (novelty_net rule in generic.py) against the v10 entropy/novelty runs.
- **Traces, trained with novelty_net (Kaggle v11; same corpus and thresholds as v10, eval rate 10.0%, 2 seeds):** computed 47.7 / 48.1 (mean 47.9), copied 36.4 / 36.8 (36.6), bpb 1.195 / 1.196. vs v10 entropy10 55.1 / 56.9 / 1.128 and novelty (table) 50.2 / 61.1 / 1.157. **Negative:** placing boundaries at 74% of computed values did not improve computing them (slightly worse than both) and hurt copied values badly (-20 to -25) and bpb (+0.04-0.07). Reading: a boundary AT a computed value is not what helps; the table's line-start boundaries (segment starts) serve copies and computation better. Consistent with the math picture being about segment structure, not about marking results. Learned novelty predictor = not the fix; trace domain stays a limitation.
