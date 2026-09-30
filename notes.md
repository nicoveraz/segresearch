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
