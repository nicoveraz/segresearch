---
title: "Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses"
author: "Nicolás Vera Zúñiga — Independent researcher, Chile — nicovera@quetru.cl"
bibliography: refs.bib
link-citations: true
header-includes:
  - \usepackage{needspace}
---

## Abstract

Byte-level language models such as the Byte Latent Transformer (BLT) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high, so global compute goes where the next byte is hard to predict. We show that this rule has a systematic blind spot: positions whose *type* is predictable but whose *value* must be computed, such as the number after `=` in a worked math solution. Under tight patch budgets, entropy-triggered layouts skip these positions, and accuracy on them collapses. In Meta's BLT-1B with patch starts on 10% of bytes, the entropy rule puts a patch start at 21% of the computed results in GSM8K solutions and gets 14.8% of them exactly right; a boundary after each `=` at the same patch count gets 41.5%, and entropy combined with a label-free *boundary-dependence* signal gets 60.7% (default layout at 28% of bytes: 73.0%). The gap survives adapting BLT-1B to the budget with low-rank fine-tuning (27.6% vs 70.0%, three runs per rule, paired $p < 10^{-230}$) and appears in small byte models trained from scratch at 10–20% budgets (3 seeds each). The entropy trigger of Scratchpad Patching is likewise indistinguishable from random scratchpads on final answers (5.6% vs 5.9%, 5 seeds), while answer-start scratchpads give 38.1%. The effect is specific to computed values: copies and lookups gain little from a patch start, and values the model cannot compute gain nothing. Boundary dependence, the rise in the model's own loss when a patch start is removed, measured per two-byte context, recovers most of the loss without labels and, as a lookup, also removes the entropy model's cost.

## 1. Introduction

Byte-level models avoid a fixed tokenizer by working on raw bytes, and recover efficiency by grouping bytes into variable-length patches [@pagnoni2025blt; @slagle2024spacebyte; @hwang2025hnet]. In BLT, a small byte-level language model estimates the entropy of each next byte; a patch starts where that entropy crosses a threshold, and the large global transformer runs once per patch. The intuition is that compute should go where prediction is hard.

Entropy measures uncertainty about the next byte. What a patch start actually provides is different: a fresh global state for the bytes that follow. These coincide in prose, where a surprising byte often begins a new word. They come apart at a computed result. After `16 - 3 - 4 = `, the next byte is certainly a digit, so the entropy model may be confident about its type, while its value requires computation over context the local model cannot see. When patch budgets are tight, an entropy threshold spends patches on hard-to-predict prose and skips exactly these positions.

We make four contributions:

1. **A blind spot in a trained 1B model.** At 10–15% patch budgets, BLT-1B's own entropy rule skips most computed results in GSM8K solutions; forcing a boundary after each `=` at the same patch count raises exact accuracy on them by 25–27 points (§4).
2. **The gap survives training and adaptation.** Small BLT-style models trained from scratch at 10–20% budgets show the same ordering across 3 seeds (§5), and BLT-1B adapted to a 10% budget with low-rank adapters keeps a 42-point gap (§7).
3. **A label-free trigger.** *Boundary dependence*, the increase in the model's own loss when a patch start is removed, averaged per preceding two-byte context, finds computed results without hand-written rules (§6).
4. **Scope.** The effect needs a computed value and a model able to compute it: copies and lookups gain 1–7 points, and values the model cannot compute gain nothing (§8). Scratchpad Patching's entropy trigger shows the same blind spot (§9).

## 2. Background and related work

**Patching rules.** BLT [@pagnoni2025blt] describes two entropy rules: a global threshold, $H(x_t) > \theta_g$, and an approximate monotonic rule that starts a patch where entropy rises, $H(x_t) - H(x_{t-1}) > \theta_r$. Its final version compares both at 8B scale and defaults to the global threshold when not specified, as do the released BLT-1B checkpoint and the official code (`monotonicity: false`, threshold 1.335); we call that rule *entropy* and the monotonic rule *jump*. Entropy spikes as boundaries go back to dynamic pooling [@nawrot2023dynamic]. SpaceByte [@slagle2024spacebyte] starts patches at word boundaries; H-Net [@hwang2025hnet], AU-Net [@videau2025aunet], MrT5 [@kallini2024mrt5] and ByteSpan [@goriely2025bytespan] learn or derive boundaries in other ways. AU-Net reports GSM8K accuracy as a benchmark, rising with the number of hierarchy stages; none of these papers examines where boundaries fall on computed outputs or measures answer accuracy as a function of where boundaries are placed.

**Compute between patches.** Scratchpad Patching [@zheng2026scratchpad] decouples compute from patch size by inserting transient global steps ("scratchpads") inside patches, triggered by an absolute next-byte entropy threshold, and names the stale-context problem *patch lag*. Its trigger ablation compares strategies on validation bits per byte (code, natural-language and math splits) and finds the entropy trigger best; its downstream evaluation (code pass@1, multiple-choice understanding) includes no math answer accuracy. Fast BLT [@kallini2026fastblt] speeds up decoding without changing where boundaries go.

**Tokenization and arithmetic.** For static tokenizers, how numbers are segmented affects arithmetic [@singh2024tokenization; @meister2026tokeval]. Our setting differs: boundaries are decided online by a model, and the question is which positions receive a fresh global step.

**Selection signals.** Rho-1 [@lin2024rho1] selects training tokens by excess loss against a reference model; we test a patch rule built on the same signal as a baseline (§11). Mixture-of-Depths [@raposo2024mod] routes compute per token by a learned router.

## 3. Setup

**BLT-1B.** We use the released BLT-1B (the HF conversion of `facebook/blt-1b`), which accepts explicit patch lengths, so layouts can be changed at inference without retraining. Its default layout starts a patch on about 28% of bytes on GSM8K. We build tight-budget layouts at R = 10% and 15% of bytes. Thresholds are fitted once on 300 GSM8K training problems so that the mean patch rate there is R, then applied position by position at test time; whether a byte starts a patch never depends on later bytes. (Choosing the top R% within each test problem instead gives the same conclusions; §12.)

**Layouts.** *entropy*: BLT-1B's next-byte entropy. *results* (hand-written): a boundary right after every `= `, the rest filled by entropy. *dep*: boundary dependence (§6). *entdep*: the sum of standardized entropy and dependence.

**Evaluation.** 294 GSM8K test problems (question, worked solution with calculator annotations removed, and "The final answer is N"), containing 796 in-line computed results: the number right after `= `. A target counts as exactly right when every byte is the model's top prediction given the true preceding text (teacher forcing). We report final answers as well, and test paired differences with exact McNemar tests and 95% intervals that resample whole problems.

**Small models trained from scratch.** A BLT-style model in MLX: a local encoder (byte embedding plus in-patch offset embedding), cross-attention pooling with the patch mean as query, a global transformer (D = 128, 4 layers unless noted), and a local decoder layer that attends to the previous 32 bytes across patch boundaries. Patch starts come from a fixed rule applied to the bytes and to a small entropy model (one layer, 32 dimensions) trained on the same corpus. Training: GSM8K and MATH training solutions [@cobbe2021gsm8k; @hendrycks2021math] (9.7 MB), 32,000 steps of 32 windows of 128 bytes, AdamW with warmup and cosine decay. Evaluation: 660 GSM8K and 700 MATH test problems held out from training, with greedy exact match given the true prefix on 1,000 computed results, 660 final answers and 700 MATH `\boxed{}` answers; patch boundaries during decoding are decided online by the same rule. Three seeds per configuration.

## 4. The blind spot in BLT-1B

At BLT-1B's default budget its patcher covers computed results well (83% of them start a patch), and removing the patch start at the final answer costs 27 points of exact match (87.7% → 60.3%), confirming that the boundary matters. Under a tight budget, entropy drops computed results faster than other positions: at 15% it covers 41% of them, at 10% only 21%.

**Table 1.** BLT-1B, patch layouts changed at inference (train-fitted thresholds). Exact match on 796 in-line computed results and 294 final answers.

{{table:table1_blt1b}}

At equal patch counts, a boundary after each `=` beats entropy on computed results by +25.0 points at 15% (95% interval +21.4 to +28.7; 217 results right only under *results* vs 18 only under *entropy*; McNemar $p = 10^{-44}$) and +26.6 at 10%. Figure 1 shows a single solution: at 10%, entropy places its patch starts at word beginnings and misses all six computed results; entropy plus dependence starts a patch right before each of them.

![BLT-1B patch starts in one GSM8K solution at a 10% budget, with equal patch counts for the two layouts. Bars mark patch starts and shading marks in-line computed results. Entropy starts no patch at any of the six results; entropy plus dependence starts one before each.](figures/fig1_patch_starts.pdf)

## 5. Training at tight budgets

Changing layouts at inference puts BLT-1B out of distribution. To test whether the blind spot survives a model that learned with the tight layout, we train small byte models from scratch with each rule.

**Table 2.** Trained at a fixed budget (D = 128, 4 global layers, 3 seeds). Computed results / final answers / bits per byte. Every entropy seed is below every seed of the other two rules on both accuracies; the hand-written and dependence seeds overlap on computed results at 20%.

{{table:table2_budget}}

The ordering hand-written > dependence > entropy holds at every budget. Placement matters more than budget: dependence at 10% beats entropy at 20% on final answers by 23.4 points (pooled 95% interval +20.4 to +26.6) with half the patches. Entropy needs budget to reach the answers (final-answer accuracy 10% → 33% → 44%); the result-aware rules are flat from 10% up. Computed-result accuracy does not improve with budget for any rule, and MATH answers stay near 3% for every rule: at this size, arithmetic capacity, not placement, caps them. The hand-written rule also has the lowest bits per byte at every budget, so its gain is not bought elsewhere; dependence costs 1.5–7% in bits per byte against entropy.

The same pattern holds at entropy's usual 25% budget. At D = 128, word starts plus math syntax (18.5% of bytes) gets 30.4% of computed results against 11.0% for entropy (+19.4 points, 3 seeds); word starts alone get 15.5% and a 6-byte stride plus math syntax 16.6%, so word alignment and the result boundary each reach only about half the combined accuracy. At D = 64, BLT's entropy rule gets 9.8% of final answers, its jump variant 67.5%, and word starts plus math syntax 77.1% (3 seeds each).

## 6. A label-free trigger: boundary dependence

The hand-written rule uses domain knowledge (`=`). We look for a signal that finds the same positions from the model alone.

**Definition.** For BLT-1B: on 400 GSM8K training problems, compute the per-byte loss under the default layout and under a 10% entropy layout. For each position t, take the mean rise in loss over bytes t…t+3, and average it per preceding two-byte context $(b_{t-2}, b_{t-1})$. At inference the score is a causal table lookup. The top contexts are arithmetic (`=$`, `5=`, `0+`, `3=`); after `= ` the mean rise is 0.91 nats against 0.17 on average.

For the small trained models the same recipe fails: a reference model trained with entropy patches rarely had a boundary at a result, so it never learned to use one, and removing one costs little. Training the reference model with random patch starts (25%) and measuring the drop in loss from *adding* one patch start at t to a random 10% layout fixes this; the top contexts then include `<<`, `=6`, `x=` and `>>`.

**Results.** On BLT-1B (Table 1), entropy plus dependence beats entropy on computed results by +38.2 points at 15% (+34.0 to +42.4) and beats even the hand-written rule by +13.2 (+9.8 to +16.5): the table also places boundaries at operands and operators, not only after `=`. On final answers it beats entropy by +7.1 points at 15% (p = 0.004) and +12.9 at 10%, and does not differ significantly from the hand-written rule. Dependence alone fails on final answers on BLT-1B (32.0% and 4.1%), because "The final answer is " is not an arithmetic context; in the trained models dependence alone is the better variant (Table 2; entropy plus dependence is dominated by entropy there: 9.7% / 22.5%).

## 7. Adapting BLT-1B to the budget

To test whether BLT-1B can learn its way around the blind spot, we train low-rank adapters (rank 16, 16.9M parameters, on the global transformer and the decoder's cross-attention) for 1,500 steps on GSM8K training solutions under each 10% layout, three runs per rule. Each model is tested on the same 796 results under its own training layout.

**Table 3.** BLT-1B adapted to a 10% budget, computed results exact.

{{table:table3_finetune}}

Adaptation helps every layout but closes none of the gap: entdep − entropy is +42.4 points (pooled over runs, +38.6 to +46.3; $p = 2 \times 10^{-237}$), hand-written − entropy +29.4, and entdep − hand-written +13.1. The models trained under entropy do better under entdep (61.8% on average), a layout they never trained on, than under their own (27.6%). Final answers saturate near 97% after fine-tuning, because the final answer repeats the last computed result and fine-tuning teaches the copy; they no longer separate the layouts.

## 8. Which positions are affected

If the blind spot is about computed values, copies and lookups should gain little from a patch start. We repeat the BLT-1B test on three other target types at 10% (train-fitted thresholds).

**Table 4.** BLT-1B at 10%, exact match. *Forced*: a patch start at every target, the rest by entropy.

{{table:table4_scope}}

Copies and lookups gain 2–4 points from a patch start (code: +4.0, $p < 10^{-17}$), against 26.7 for computed results. Where BLT-1B cannot compute the value (program-trace arithmetic, 20.5% even with default patches) or answer at all (logic Yes/No answers at chance under every layout), placement has nothing to act on. Small models trained on the logic and trace tasks give the same picture: logic is not learned (answers at chance), and on traces, where each value's operands sit inside the local decoder's 32-byte window, the hand-written rule matches entropy (21.8% vs 20.6%). With the window cut to 8 bytes the hand-written rule leads by 4 points (24.2% vs 20.1%, 3 seeds), but the seeds overlap. Our reading is that the blind spot needs a value that is computed from context beyond the local model's reach, by a model able to compute it.

## 9. Scratchpad Patching

We reimplemented Scratchpad Patching in the small-model harness (16-byte fixed patches, scratchpads on 6% of bytes, pooled by cross-attention over the partial patch with the mean as query) and compared triggers at equal scratchpad counts (D = 64, 2 global layers).

**Table 5.** Final-answer exact match on GSM8K by scratchpad trigger, with the accuracy of each seed.

{{table:table5_scratchpad}}

The entropy trigger is indistinguishable from random scratchpads (−0.3 points, Welch 95% interval −2.2 to +1.6). Answer-start scratchpads give about six times the accuracy on average, with one of five seeds failing to train (8.3%). This does not contradict the paper's ablation, which found entropy best on bits per byte: averaged bits per byte can hide a failure at a few positions that decide answer accuracy.

## 10. Compute

Fewer patches save global compute, but local layers run on every byte. For BLT-1B, counting matrix-multiply FLOPs, the local layers cost 299 MFLOPs per byte, the global model 2,653 MFLOPs per patch, and the entropy model 199 MFLOPs per byte. A 10% budget then costs 61% of the default's forward compute and 15% costs 72%. A dependence-table rule is a lookup and needs no entropy model: 45% at 10% and 56% at 15%. On computed results, dependence alone reaches 71.7% at 56% of default compute, against 73.0% for the default (Figure 2), though it fails on final answers (§6). In our small models the local layers dominate, so a 10% budget saves only 24% of compute.

![BLT-1B exact match on in-line computed results (left) and final answers (right) against forward compute per byte, relative to the default layout, for layouts at 10% and 15% of bytes. Dependence alone is a lookup and pays no entropy-model cost.](figures/blt1b_acc_vs_compute.pdf)

![Models trained at 10-25% patch budgets (D = 128): exact match against forward compute per byte, relative to entropy at 25%. Points are seed means; bars span the seeds.](figures/acc_vs_compute.pdf)

## 11. Negative results

**Spending patches on hard or learnable bytes does not help.** A Rho-1-style rule (patches where a small model's loss exceeds a reference model's) and simplified learned chunking lost to hand-placed boundaries in every setting tried.

**Finer arithmetic boundaries do not help.** Per-digit patches after `=` and operand-aligned patches looked promising at two seeds and vanished at five seeds against matched random controls.

**A learned patcher does not beat the table.** A small network that predicts the dependence gain from the last 8 bytes, trained on math and on logic/trace data together, matches each domain's own table on held-out correlation and covers 83% of computed-result starts (table: 37%), but trained into a model at 10% it gets 14.3% / 57.4% against the table's 17.1% / 67.4%.

**Token models are partly protected.** With a Qwen2.5-0.5B draft and a 1.5B target on generated program traces, the draft agrees with the target on only 53% of computed-value tokens (86% on text), so speculative decoding loses speed there; but the draft is rarely confident on them (6.5%), so a confidence-based early exit would let few errors through. On GSM8K, computed results are the easiest tokens for these models, likely through training-data overlap. In token models a computed value's uncertainty shows in the model's own confidence; in byte patching, the decision to start a patch before a result is made from bytes whose type is predictable.

**The original synthetic result was mostly architectural.** In an earlier synthetic benchmark with sum pooling and a patch-local decoder, splitting a computation across patches cost up to a full bit; with cross-attention pooling and a decoder that sees across patch boundaries, most of that penalty disappeared. All results above use the cross-attention model.

## 12. Limitations

**Scale.** Models trained from scratch are small (D ≤ 128). BLT-1B results come from inference-time layout changes and low-rank adaptation, not from full training at a tight budget; a scaling study at 10–40M parameters is feasible on our hardware but not done.

**Teacher forcing.** All accuracies score each target given the true preceding text. End to end, BLT-1B cannot solve GSM8K by itself: even after adaptation it gets about 2% of final answers under every layout, so the gaps above could not be measured end to end.

**Domain.** The effect is established for arithmetic in worked math solutions. Code shows a small, significant effect; logic and program traces were inconclusive because the models lacked the skill or the local model already saw the operands.

**Label-free trigger.** The dependence table is fitted on the same domain it is tested on and does not transfer across formats (the GSM8K table covers 0% of program-trace values). On BLT-1B, the best variant (entropy plus dependence) differs from the best variant in trained models (dependence alone).

**Thresholds.** Our original BLT-1B layouts chose the top R% of positions within each test problem, so a position's selection could depend on later bytes. Train-fitted, position-by-position thresholds reproduce every conclusion, with gaps as large or larger (results − entropy at 15%: +25.0 vs +21.5 points).

## 13. Conclusion

Entropy tells a byte-level model where the next byte is surprising, not where its decoder needs the global model. The two diverge at computed outputs: positions whose type is predictable and whose value must be computed. Under tight budgets, entropy-triggered patching and scratchpad triggering skip these positions, and accuracy on them collapses, in a trained 1B model, in models trained at the budget, and after adaptation. Measuring the dependence directly, as the loss the model loses when a patch start is removed, finds these positions without labels. Compute allocation in byte models should follow what the decoder needs from the global model, not how surprising the next byte is.

## Declarations

**Data and code.** Code, raw logs and the results registry: https://github.com/nicoveraz/segresearch. `realblt_budget.py`, `realblt_code.py`, `realblt_reason.py` and `realblt_finetune.py` run the BLT-1B experiments (PyTorch); `mathexp.py`, `deptrigger.py` and `reasonexp.py` the trained models (MLX); `flops.py` the compute estimates; `stats.py`, `make_tables.py` and `make_figures.py` rebuild every table and figure from `results/registry/`. GSM8K, MATH and the BLT-1B weights are available from their authors. **Funding:** none. **Competing interests:** none. **Use of AI:** the experiments were designed, run and analysed with an AI coding agent (Claude Code) working in a research loop under the author's direction; the standards in Appendix A came out of that process. **Ethics:** public datasets and generated text only; no human subjects.

## References

::: {#refs}
:::

## Appendix A. Experimental standards

The project ran as a partly autonomous research loop, and several early conclusions were wrong. The standards in `program.md` exist because breaking each produced a wrong result: a word-start rule that read the byte it was predicting (caught by an automatic causality test, `test_causal.py`); effects at two seeds that vanished at five; a synthetic generator that placed dependencies outside the model's context; an effect that was mostly architectural; undertrained models hiding the effect; and a rule that exploited the generator's fixed number of variable names. All results here use three or more seeds for trained models, paired tests for BLT-1B, and rules checked for causality.
