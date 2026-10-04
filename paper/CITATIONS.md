# Citation ledger

Why this file exists (from textca's ledger): a prior-art gate there refuted 13 of 74 extracted claims for
overreaching their own sources, so summaries are not citable. Every work cited in `manuscript.md` has an entry:
where it is cited, what the manuscript claims about it, and the supporting quote with its basis.

**Basis.** `LOCAL FULL TEXT`: the PDF was fetched to this machine (2026-10-04), extracted with
`pdftotext -layout`, and the quote was read in that extraction; the line number refers to that extraction, and
the version and SHA-256 prefix identify the file. Works cited by a venue are checked against the **published
version** where it could be fetched (ACL Anthology, NeurIPS proceedings); OpenReview blocks automated download, so
the ICLR 2026 and NeurIPS 2025 versions of entries 6 and 7 are checked against their latest arXiv version only.
`CODE`: the released artifact was read directly.

**Status: 16 works; 14 carry a claim, all 14 verified at LOCAL FULL TEXT (one also at CODE); 2 cited as datasets.**
Published versions checked for BLT, Nawrot et al., SpaceByte and Rho-1; MrT5 and TokEval's arXiv versions are the
camera-ready copies; ByteSpan was non-archival; H-Net and AU-Net are pending their venue PDFs (OpenReview). The
check corrected the manuscript in three places (entries 1, 2 and 7).

---

## 1. `pagnoni2025blt`: ACL 2025 (aclanthology 2025.acl-long.453), sha256 69fdaab9d6b4; also arXiv:2412.09871v1, sha256 23e0cc90e9e2

- **Cited in:** §1, §2.
- **Claim:** BLT starts patches where a small model's next-byte entropy crosses a threshold; it describes a global-threshold rule and an approximate monotonic rule; the final version compares both and defaults to the global threshold, as do the released checkpoint and code (`monotonicity: false`, threshold 1.335).
- **Source, ACL version (l. 196-206):** "We experiment with two methods to identify patch boundaries given entropies H(xi). Global finds points above a global entropy threshold … Approximate Monotonicity, identifies points that are high relative to the previous entropy … Global H(xi) > θg / Approx. Monotonic H(xi)−H(xi−1) > θr".
- **Source, ACL version (l. 338-340):** "For models using entropy patching, we estimate a patching threshold that achieves a desired average patch size on the pretraining data mix. We default to the global entropy threshold when not specified."
- **Source, ACL version (Table 1, l. 377-411):** columns "Llama 3, BLT-Space, BLT-Global, BLT-Mono"; "we compare BPE Llama 3 tokenizer-based model, and three variants of BLT: space-patching, global, and approx. monotonic entropy".
- **Source (CODE, `itazap/blt-1b-hf` config.json):** `"patching_mode": "entropy"`, `"patching_threshold": 1.335442066192627`, `"monotonicity": false`, `"patching_threshold_add": null`.
- **Correction made:** the draft said the monotonic rule was "used for the paper's main BLT-Entropy model". That is true of arXiv v1 (l. 657: "BLT-Entropy … with approx. monotonicity constraint"), but the published ACL version, which `refs.bib` cites, reports both rules and defaults to the global threshold. §2 now says so.
- **Novelty check:** Table 1 evaluates Arc-E, Arc-C, HellaSwag, PIQA, MMLU, MBPP and HumanEval; 0 GSM8K mentions in either version.

## 2. `zheng2026scratchpad`: arXiv:2605.09630v1, sha256 c96d7e1193eb

- **Cited in:** §2, §9.
- **Claim:** scratchpads (transient global steps inside patches) are triggered by an absolute next-byte entropy threshold; the paper names stale patch context "patch lag"; its trigger ablation compares strategies on validation bits per byte and finds entropy best; its downstream evaluation includes no math answer accuracy.
- **Source (l. 246):** "… exceeds a predefined threshold: pn := 1[Hn > τSP]." **(l. 1328-1329):** "Our default policy issues a scratchpad update whenever the encoder's next-byte prediction entropy exceeds a threshold, where we use τSP = 1.5 by default."
- **Source (l. 166):** "… the most recent patch-level representation available to it as patch lag."
- **Source (l. 1438-1440, Figure 12):** "Ablations of scratchpad triggering strategies on validation BPB versus training FLOPs … Entropy-based triggers (E > τSP), fixed-stride updates (S), and whitespace-based heuristics are compared." **(l. 1450):** "entropy-based triggering achieves the best compute-quality trade-off across all three domains".
- **Source (l. 314-316):** "We evaluate (i) Bits-Per-Byte (BPB) on held-out validation data, (ii) estimated pass@1 on code generation with MBPP … and HumanEval …, and (iii) accuracy on multiple-choice natural language understanding benchmarks." Math appears only as a validation BPB split (l. 1300, Fig. 12c); 0 GSM8K mentions.
- **Correction made:** the draft said the ablations compared triggers "on bits per byte, code pass@1 and multiple-choice tasks". The trigger ablation (appendix E.2) uses validation BPB only; §2 now says so.
- **Reimplementation detail (§9):** "Our default patchifier computes a mean-pooled summary over the byte-level hidden states within each patch, which then serves as the query in the cross-attention mechanism" (appendix B.1); fixed widths "p ∈ {2, 4, 8, 16}" (B.2).

## 3. `kallini2026fastblt`: arXiv:2605.08044v1, sha256 7957dd70eb71

- **Cited in:** §2. **Claim:** Fast BLT speeds up decoding without changing where boundaries go.
- **Source (l. 9-21):** BLT Diffusion "generates multiple bytes in parallel per decoding step"; BLT Self-speculation, "in which BLT's local decoder continues generating past its normal patch boundaries to draft bytes, which are then verified". **(l. 101):** "… while preserving the main benefits of BLT: operating directly on bytes, using dynamic patching, …"

## 4. `nawrot2023dynamic`: ACL 2023 (aclanthology 2023.acl-long.353), sha256 026430901b8c; also arXiv:2211.09761v2, sha256 9db799b97dcc

- **Cited in:** §2. **Claim:** entropy spikes as boundaries go back to dynamic pooling.
- **Source (ACL version l. 52; arXiv l. 57-58):** boundary predictors "supervised by spikes in the conditional entropy of the predictive distribution".

## 5. `slagle2024spacebyte`: NeurIPS 2024 (proceedings PDF), sha256 3eb7971bc74d; also arXiv:2404.14408v3, sha256 19914f0da857

- **Cited in:** §1, §2. **Claim:** SpaceByte starts patches at word boundaries.
- **Source (NeurIPS version l. 21, 69):** "larger blocks only after certain bytes, such as space characters, which typically …"; "The intuition is that the first character of a word is typically the hardest [to predict]".

## 6. `hwang2025hnet`: ICLR 2026; checked against arXiv:2507.07955v2, sha256 04a2b32e6e82 (venue PDF pending: OpenReview blocks automated download)

- **Cited in:** §1, §2. **Claim:** H-Net learns boundaries.
- **Source (abstract):** "a dynamic chunking mechanism which automatically learns content- and context- dependent segmentation strategies learned jointly with the rest of the model".

## 7. `videau2025aunet`: NeurIPS 2025; checked against arXiv:2506.14761v1, sha256 f6b6ec25b416 (venue PDF pending: OpenReview blocks automated download)

- **Cited in:** §2. **Claim:** AU-Net derives boundaries in another way (word-level pooling); it reports GSM8K accuracy as a benchmark, rising with the number of stages, but not as a function of boundary placement.
- **Source (l. 20):** "network reads raw bytes, pools them into words, then pairs of words, then up to 4 words". **(l. 385):** "GSM8k performances continue to improve with increased stage, even at fixed scale."
- **Correction made:** the draft said none of the boundary papers "measures answer accuracy as a function of boundary placement" without qualification. AU-Net does report GSM8K (by number of stages), so §2 now names it and narrows the claim to where boundaries fall on computed outputs.

## 8. `kallini2024mrt5`: arXiv:2410.20771v3 (ICLR 2025), sha256 4ba6187fc440

- **Cited in:** §2. **Claim:** MrT5 learns which bytes to keep.
- **Source (l. 26-30):** "a token deletion mechanism in its encoder to dynamically shorten the input … learned delete gate determines which tokens are to be removed and which are to be retained".

## 9. `goriely2025bytespan`: arXiv:2506.18639v1 (TokShop 2025), sha256 363bcd0af02a

- **Cited in:** §2. **Claim:** ByteSpan derives boundaries from information.
- **Source (title, l. 25):** "ByteSpan: Information-Driven Subword Tokenisation"; "We propose a new information-driven tokeniser".

## 10. `singh2024tokenization`: arXiv:2402.14903v1, sha256 762b049e1ca2

- **Cited in:** §2. **Claim:** number segmentation in static tokenizers affects arithmetic.
- **Source (title; l. 27-29):** "Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs"; "we enforce right-to-left (R2L) tokenization for the same addition prob[lems]".

## 11. `meister2026tokeval`: arXiv:2608.18062v3 (COLM 2026), sha256 dedfb2f3754f

- **Cited in:** §2. **Claim:** same, including digit place-value alignment.
- **Source (l. 27):** "… boundary integrity and digit place-value boundary alignment for mathe[matical text]". **(l. 126):** "demonstrated that digit tokenization directly impacts models' arithmetic capabilities".

## 12. `lin2024rho1`: NeurIPS 2024, published as *Not All Tokens Are What You Need for Pretraining* (proceedings PDF), sha256 7b29297863a7; also arXiv:2404.07965v4, sha256 ae9dcc41c4ff

- **Cited in:** §2, §11. **Claim:** Rho-1 selects training tokens by excess loss against a reference model.
- **Source (NeurIPS version l. 110-112; arXiv l. 114-116):** "SLM uses the reference model to score each token … high excess loss between the reference and the training model, selectively learning the tokens".

## 13. `raposo2024mod`: arXiv:2404.02258v1, sha256 a64ff37ceb25

- **Cited in:** §2. **Claim:** Mixture-of-Depths routes compute per token with a learned router.
- **Source (l. 76):** "MoD transformers learn to route intelligently (i.e., skipping computations that are [unnecessary])"; routing by a router with top-k selection (l. 105-117).

## 14. `haltiuk2026disentangling`: arXiv:2608.03599

- **Not cited** in the manuscript. `../literature.md`: a position paper; cite only as a hypothesis if it is added.

## 15-16. `cobbe2021gsm8k` (arXiv:2110.14168v2, sha256 a52417d8fcd0), `hendrycks2021math` (arXiv:2103.03874v2, sha256 a6f448173e57)

- **Cited in:** §3, as the datasets. No claim about the papers beyond the data.

---

**The novelty claim** (§1-2: no prior report of entropy-based patching, scratchpads or learned chunking failing at
computed outputs) also rests on the searches in `../literature.md` (2026-10-01, re-check 2026-10-02). The full
texts above confirm it for the cited boundary papers: only AU-Net reports GSM8K, and none examines where
boundaries fall on computed outputs. Re-run the search right before submission (#13).
