# Literature check (verified, issue #13)

Checked by hand on 2026-10-01 against the arXiv abstract pages, the Scratchpad Patching PDF (appendix B.3 and E.2), the BLT paper and code, and the BLT-1B config. Bibliography: `paper/refs.bib`.

## Citations in notes.md: all verified

| Cited as | Verified | Matches notes.md? |
|---|---|---|
| Scratchpad Patching, Zheng et al., arXiv 2605.09630 | Zheng, Bashlovkina, Dozat, Garrette, Rimell, Maynez; May 2026 | Yes: "patch lag" is their term; the trigger is an absolute entropy threshold |
| Tokenization counts, arXiv 2402.14903 | Singh & Strouse, Feb 2024 | Yes |
| TokEval, arXiv 2608.18062, COLM 2026 | Meister, Aug 2026, COLM 2026 | Yes: includes digit place-value boundary alignment |
| Disentangling LM and Boundaries, arXiv 2608.03599 | Haltiuk, Aug 2026 | Partly: it is a **position paper** with preliminary measurements; it proposes, but does not show, that boundaries can be changed nearly independently of capability. Cite it as a hypothesis, not as evidence |
| BLT-1B checkpoint `itazap/blt-1b-hf` | Community HF-transformers conversion of `facebook/blt-1b`, Apache-2.0 | Cite the official `facebook/blt-1b` and say we used the conversion |

## Corrections and precision needed in the write-up

1. **"BLT entropy" must mean the global-threshold rule, and the paper must say so.** The BLT paper describes two rules: a global threshold, H(x_t) > θ_g, and an approximate monotonic constraint, H(x_t) − H(x_{t−1}) > θ_r. It uses the monotonic one (with entropy reset at newlines) for its main BLT-Entropy model in arXiv v1; the published ACL 2025 version reports both (BLT-Global and BLT-Mono in Table 1) and defaults to the global threshold when not specified (checked 2026-10-04, `paper/CITATIONS.md`). The **released** BLT-1B and the official code default to the global threshold: `monotonicity: false`, `patching_threshold 1.3354`, `threshold_add null`, in both the HF config and `bytelatent/data/patcher.py`. Our "default" layout is therefore correct for the released model, but a reviewer reading the BLT paper will expect the jump rule. State both facts.
2. **The entropy jump is not ours.** BLT's monotonic constraint is the same idea, and Nawrot et al. (2023) already used "spikes in conditional entropy" as boundaries. Our contribution is showing *where* the level rule fails and why the jump partly fixes it, not the jump rule itself. notes.md already says BLT includes the jump rule; the write-up must not present it as new.
3. **Scratchpad Patching reimplementation: close to the paper, with differences to list.**
   - Matches: trigger = absolute next-byte entropy threshold; scratchpads pooled by local cross-attention with the mean-pooled segment as query; fixed 16-byte patches is one of their settings (p ∈ {2, 4, 8, 16}).
   - Differs: their trigger is a fixed threshold (τ_SP = 1.5 for fixed-size patching), so the scratchpad rate varies; ours picks the top 6% to match compute across triggers. Their entropy comes from an LM head on the encoder (two extra layers); ours from a separate small model. Their models are ~2B parameters trained on ~400B bytes.
   - **No code release** is linked in the paper, so #23 cannot use the authors' code. List these differences in the paper instead.
4. **Their own ablation (appendix E.2) found entropy triggering best**, ahead of fixed stride and whitespace, on validation BPB. They evaluate BPB, code pass@1 (MBPP, HumanEval) and multiple-choice NLU, with **no math answer accuracy**. Our result (entropy = random for final-answer accuracy on GSM8K) does not contradict theirs; it shows that averaged BPB can hide a failure at specific positions. That is the framing to use.

## Novelty: is the blind spot already reported?

Searched for prior reports of entropy-based patching, scratchpads or learned chunking failing at computed outputs (numbers after "=", final answers), of patch boundary placement measured by math accuracy, and of the same type/value confusion in confidence-based adaptive compute (early exit, Mixture-of-Depths).

**Found none.** Closest work:
- **BLT:** no analysis of boundaries on numbers or math.
- **Scratchpad Patching:** entropy-triggered compute; no math accuracy; ablations on BPB only.
- **Fast BLT** (Kallini et al., 2026): speeds up decoding (self-speculation past patch boundaries); does not change where boundaries go.
- **Tokenization counts, TokEval:** the tokenizer-side version (number segmentation affects arithmetic), for static tokenizers, not entropy-driven patches.
- **Rho-1:** a token-selection signal from a reference model; we use it as a negative baseline (excess-vs-reference), not a boundary rule.
- **H-Net, AU-Net, MrT5, ByteSpan, SpaceByte:** other boundary mechanisms; none evaluate boundaries at computed outputs.

**Claim that looks new:** entropy-level boundary or compute triggers systematically skip positions whose type is predictable but whose value must be computed. Under tight budgets this costs most final-answer and computed-result accuracy, both in a trained 1B model and in tight-budget training, and a label-free boundary-dependence signal recovers most of it.

**Caveat:** this is a web search, not an exhaustive review; arXiv moves fast. Search again right before submission (keywords: byte patching math, patch boundary arithmetic, entropy trigger accuracy, dynamic chunking numbers).

## Useful for other issues

- **H-Net checkpoints are public** (`goombalab/hnet`, on HF under cartesia-ai): `hnet_1stage/2stage_L/XL` on FineWeb-Edu (English), plus Chinese and code variants. That makes #21 doable; boundary extraction needs code (no documented API).
- **AU-Net** (Videau et al., 2025) is another candidate for #21; weights not checked.
- **Fast BLT** self-speculation drafts past patch boundaries, so it is a natural target for the speculative-decoding part of #22.

## Re-check, 2026-10-02 (quick; the full search is still due right before submission)

New candidates found, neither prior work on the blind spot:
- **ATDC**, Adaptive Targeted Dynamic Chunking for Tokenization-Free Hierarchical Model (Dang, Nakagawa, Kobayashi, Shirahata; arXiv 2605.30080, May 2026): curriculum on the compression ratio for H-Net-style chunking; evaluated on FineWeb-Edu bits per byte and downstream tasks; no analysis of boundaries at numbers and no math accuracy. Cite as related dynamic chunking.
- **ReinPatch**, Dynamic Tokenization via Reinforcement Patching (Wu et al.; arXiv 2603.26097, March 2026): patch boundaries learned with policy gradients, evaluated on time-series forecasting; no text or math, no entropy-patching comparison.
Also seen, already covered: Fast BLT (2605.08044), Scratchpad Patching (2605.09630), H-Net (2507.07955), ByteSpan (2506.18639). Confidence-based early exit papers (e.g. 2605.05222, 2509.23666) discuss overconfidence in general but not the type/value split at computed outputs; relevant to #22.

## Venue versions and searches outside arXiv, 2026-10-04

- **Published versions checked** (`paper/CITATIONS.md`): BLT (ACL 2025), Nawrot et al. (ACL 2023), SpaceByte (NeurIPS 2024), Rho-1 (NeurIPS 2024, retitled *Not All Tokens Are What You Need for Pretraining*). MrT5 and TokEval's arXiv versions are the camera-ready copies; ByteSpan was non-archival. H-Net (ICLR 2026) and AU-Net (NeurIPS 2025): OpenReview PDFs downloaded by hand and checked; same claims.
- **BLT correction:** the ACL version reports BLT-Global and BLT-Mono side by side and "default[s] to the global entropy threshold when not specified"; the manuscript no longer says the monotonic rule is the main model.
- **ACL Anthology / OpenReview search:** two byte- and character-level papers not in the bibliography, *From Characters to Tokens: Dynamic Grouping with Hierarchical BPE* (Dolga et al., Findings of EMNLP 2025) and *Retrofitting Large Language Models with Dynamic Tokenization* (Feher et al., ACL 2025). Neither evaluates math, arithmetic or computed outputs. Still no prior report of the blind spot.

## Final novelty search, 2026-10-07 (before arXiv, #13)

Searched: byte-level patching and arithmetic/math accuracy; computed results and patch boundaries; Scratchpad Patching follow-ups; dynamic tokenization/chunking (Aug-Oct 2026); entropy-driven adaptive compute and number tokens; "boundary/patch dependence"; H-Net and math.
- **No prior report of the blind spot found.** New since the last check, none overlapping: **EntropyMoE** (Liu et al., arXiv 2608.06398, Jul 2026; patch entropy routes experts in tokenizer-free models; bits per byte and downstream accuracy, no math or computed outputs, no failure analysis of entropy routing), **ReconSpan** (Li, arXiv 2608.12756, Aug 2026; reconstruction-guided chunking, no math), *Hierarchical Continuous Diffusion LMs* (2610.02193; diffusion, not patching), *A Four-Stage Decomposition of Word-Problem Solving* (2609.17804; mechanistic, finds GSM8K bottlenecked by arithmetic computation, consistent with our framing), ANI (2609.39294; number representations). EntropyMoE is related work worth a sentence (entropy as a compute-allocation signal in tokenizer-free models).

**Implementation issue that affects our BLT-1B experiments (not a novelty issue).** huggingface/transformers issue #49185 (opened 2026-09-29; fix PR #49188 open, not merged): the HF conversion of BLT (`itazap/blt-1b-hf`, which we used) dropped the 512-byte sliding window of the entropy patcher and of the local encoder/decoder (config: `sliding_window: None`). Positions before byte 512 are unaffected (the window covers all earlier bytes); beyond 512 every layout runs outside BLT-1B's trained attention pattern and the default patcher over-segments. In our GSM8K test set 121 of 294 problems exceed 512 bytes; 598 of 796 in-line results and 173 of 294 final answers lie inside the first 512 bytes. Recomputed on those targets only (per-target files in results/items/), every BLT-1B conclusion holds with larger gaps:
| computed results | all targets | inside 512 bytes |
|---|---|---|
| results − entropy @15 / @10 | +25.0 / +26.6 | +29.6 / +31.9 (McNemar p 9e-40 / 7e-49) |
| entdep − entropy @15 / @10 | +38.2 / +45.9 | +45.2 / +53.7 (p 1e-69 / 1e-82) |
| fine-tuned (§7) entdep vs entropy | 70.0 vs 27.6 | 73.2 vs 26.0 |
Final answers inside 512: results − entropy +5.8 / +14.5, entdep − entropy +8.7 / +13.9 (p ≤ 0.02). Code chunks (800 bytes) and longer trace/logic inputs are affected the same way and were not rechecked. The paper must state this and report the inside-512 numbers.
