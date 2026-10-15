# Related work scan for paper 2 (2026-10-14, web search; abstracts only unless noted)

**Bottom line:** no prior work found that sets byte-level patch boundaries from LZ-style reuse / novelty. The
closest work either (a) learns boundaries end to end, (b) steers boundaries TOWARD high surprisal (the opposite of
our finding), or (c) uses LZ compression to merge tokens, not to place compute. Search was not exhaustive (web search
does not index arXiv fully); re-check arXiv / Semantic Scholar citing papers of BLT (2412.09871) and H-Net
(2507.07955) right before submission.

## Must cite and position against

| Work | What it does | Relation to paper 2 |
|---|---|---|
| BLT, Pagnoni et al. (arXiv 2412.09871, ACL 2025) | entropy patching (global threshold, monotonic) | the rule we replace |
| Our paper 1 (arXiv 2610.11790) | entropy skips computed results; boundary dependence (loss-learned table) | direct predecessor; cite in third person for TMLR anonymity |
| **SOMBRERO** (arXiv 2601.22805, Jan 2026) | metric "boundary enrichment" (chunk starts on high next-byte surprisal) + loss steering H-Net-style boundaries toward hard-to-predict positions; 1B scale; English/German, code, math | **closest and opposite**: optimizes boundaries toward surprisal; our result says surprisal-aligned boundaries miss computed results, and novelty (structure) wins on math. Must discuss; ideally evaluate their enrichment metric on our rules |
| H-Net, Hwang, Wang, Gu (arXiv 2507.07955) | end-to-end dynamic chunking | our Section 6: learns structure, no blind spot in math, word-level structure in code |
| ByteFlow (arXiv 2603.03583, ICLR 2026) | "compression-driven segmentation based on the coding rate of latent representations", learned, Top-K static graph | compression framing in learned form; **read full paper** (is coding rate ≈ novelty? any math eval?) |
| Disentangling Language Modeling and Boundaries (arXiv 2608.03599) | boundary distribution can be changed nearly independently of the LM | motivates inference-time layout changes; our cross-rule collapse at 12M is a counterpoint at small scale |
| Scratchpad Patching (arXiv 2605.09630, May 2026) | fixed-size patches + entropy-triggered scratchpads inside patches; text and code | another entropy-triggered compute allocator: the blind spot argument applies; novelty could trigger scratchpads |
| zip2zip (arXiv 2506.01084, ICML 2025) | LZW merges recurring tokens into hypertokens at inference | uses LZ to compress repeats (fewer tokens); we use LZ to decide where compute goes. Complementary |
| ByteSpan (arXiv 2506.18639, TokShop 2025) | groups predictable byte spans (external byte LM) into a subword vocabulary | predictability-based grouping for tokenization, not patching |
| Adaptive Targeted Dynamic Chunking (arXiv 2605.30080) | curriculum on H-Net compression ratio | learned-chunking training detail; minor |
| SpaceByte, MegaByte, MBLM (arXiv 2502.14553) | whitespace / fixed / hierarchical byte models | baselines in the landscape |
| Fast BLT (arXiv 2605.08044), EntropyMoE (arXiv 2608.06398) | speed / MoE on top of entropy patching | inherit entropy patching; brief mention |

## Full-text notes (read 2026-10-14)

**SOMBRERO** (Neitemeier et al., Aleph Alpha; arXiv 2601.22805v1)
- Metric: boundary enrichment B = mean next-byte surprisal at chunk starts / mean surprisal overall (B > 1 =
  boundaries on hard bytes), with a rate-matched circular-shift null (Z_B).
- Method: confidence-alignment boundary loss (1 - sg[P_{t+1}] - p_t)^2 pushes the router's boundary probability
  toward low next-byte confidence; byte-level EMA smoothing; linear sigmoid router. Two-stage HAT (Mamba-2 local,
  Transformer global), ~1B (and 2.2B), 839B bytes incl. math (DeepMind Mathematics, Orca-Math, Algebraic Stack) and code.
- Results: bpb only per domain (math 0.410 vs H-Net 0.413; code 0.370 vs 0.373); B 3.0 vs 1.2. At 2.2B bpb ties
  H-Net though B is higher (3.13 vs 2.00); downstream log-likelihood tasks mixed (TriviaQA, MMLU lower). Single run
  per configuration. No BLT experiment, no math/code accuracy, no breakdown at digits or computed results.
- Their own qualitative failure case: on retrieval (number -> character) both models spend compute on the
  unpredictable next number, not the predictable answer character.
- **Position for paper 2:** SOMBRERO optimizes exactly the property we show is misaligned with computation: our
  entropy rule is surprisal-aligned (high B by construction) yet scores 12% on computed results at 50M, vs 57% for
  novelty. Higher B did not buy bpb at scale for them either. Their failure case is our blind spot seen from the
  other side (predictable-looking outputs that need compute). **Cheap addition: report B for entropy / dep /
  novelty / H-Net on our eval set to show high enrichment does not predict computed-result accuracy.**

**ByteFlow Net** (arXiv 2603.03583v1, ICLR 2026)
- Boundaries: marginal coding-rate gain of the encoder states, dR_t = R(h_1:t) - R(h_1:t-1),
  R = 1/2 logdet(I + d/eps^2 H H^T): how much byte t's latent adds a NEW direction to the span of the prefix.
  Top-K positions kept (fixed K; default ~2.56x compression); training uses the full-sequence profile (causality at
  inference not fully specified; an L2 approximation for streaming).
- 600M / 1.3B on FineWeb-Edu; zero-shot commonsense (+1.7 / +3.0 avg over LLaMA), CUTE character tasks. No math or
  code evaluation. Ablation inside their model: coding-rate 0.86 bpb vs entropy 0.91 vs cosine (H-Net-like) 0.92.
  Qualitative: high scores at word-initial letters and key entities. Code not released (legal review).
- **Position for paper 2:** conceptually the closest relative. Coding-rate gain is a latent, learned "novelty":
  boundaries where the representation adds information not spanned by the prefix. Our rule is the explicit,
  causal, byte-level version (LZ reuse in the last 128 bytes), costs a table lookup, and is tested where it
  matters (computed-result and final-answer accuracy, 3 seeds, scale trend). Their entropy ablation also favours a
  novelty-like signal over entropy: independent support. Cite as concurrent evidence that "new information"
  beats "uncertainty" as a boundary signal.

## Compression and segmentation (for the LZ framing)
- Ziv and Lempel 1977 (LZ77): parse into copies and literals.
- MDL / description-length segmentation: Kit and Wilks 1999 (description length gain, ACL W99-0701); Chen, Tsai,
  Hsiang 2012 (regularized compression, SIGMORPHON W12-2304); Chen 2013 (improved MDL compressor, ACL P13-2030).
  Classic evidence that compression-driven cuts align with words; our novelty boundary is the online, causal analogue
  (where the LZ parse would stop copying).
- Optional: compression as prediction (Delétang et al. 2024, "Language Modeling Is Compression") — verify citation.

## Gaps confirmed (worth stating in the paper)
- No LZ / reuse-based patch boundaries for byte-level models found.
- No analysis of where learned chunkers (H-Net) place boundaries on code or math found; our H-Net code/math
  measurements are new.
- SOMBRERO pushes toward surprisal; nobody reports the computed-result blind spot except paper 1.

## To do before submission
1. ~~Read ByteFlow and SOMBRERO full texts~~ (done, notes above).
2. Compute SOMBRERO's boundary enrichment B for entropy / dep / novelty (and H-Net) on our eval set: shows B does
   not predict computed-result accuracy (cheap, Mac).
3. Citing-paper sweep for BLT and H-Net on Semantic Scholar.

Sources: arxiv.org/abs/{2412.09871, 2610.11790, 2601.22805, 2507.07955, 2603.03583, 2608.03599, 2605.09630,
2506.01084, 2506.18639, 2605.30080, 2502.14553, 2605.08044, 2608.06398}; aclanthology W99-0701, W12-2304, P13-2030.
