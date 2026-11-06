# arXiv submission metadata (paper 2)

Paste these by hand into the arXiv form. `./build.sh` builds and verifies `arxiv-submission.tar.gz`.

## Version 2 (replacement of arXiv:2610.22100; prepared, not submitted)
Prepared on the local branch `paper2-v2`; nothing is pushed or posted. The author reviews it and posts it as a
replacement of 2610.22100 (not a withdrawal and not a new submission): see `ARXIV_V2.md` for the steps and the order
(the v2.1.0 release and its Zenodo record first, then the replacement). Paste the fields below by hand.

Checklist:
- [x] References checked against their arXiv abstract pages (2026-11-02), including BLT's §4.2, Fig. 3(f) and §5.1.
- [x] `./build.sh` ends with `OK -- verified from the tarball's own contents.` (29 pages)
- [x] `tests/test_paper2_numbers.py` passes (the paper's result numbers recomputed from the files of release v2.1.0).
- [ ] Release v2.1.0 published and its Zenodo DOI known; `ZENODO_V210_DOI` replaced in the Comments below, in
      `manuscript.md` ("Code and data") and in `v2/README.md`, then `./build.sh` re-run.
- [ ] Upload `paper2/arxiv-submission.tar.gz`; compare arXiv's compiled PDF with `paper2/arxiv.pdf` before confirming.

### Title
```
When the next byte's type is fixed: why entropy patching skips computed values under tight budgets
```

### Authors, categories, licence (unchanged from version 1)
Nicolás Vera Zúñiga (Independent Researcher, Chile; nicovera@quetru.cl). Primary cs.CL, cross-list cs.LG. CC BY 4.0.

### Comments
`ZENODO_V210_DOI` is a placeholder: replace it with the DOI of the v2.1.0 Zenodo record before pasting. The line keeps
version 1's comments (counts updated, the code link) and says what changed, as arXiv asks of a replacement.
```
v2 corrects v1's main claims (new title, abstract and framing; the paper's section "Changes in version 2" lists each change) and is the project's final record; no venue is intended. 29 pages, 2 figures, 8 tables. Code, logs and results: https://github.com/nicoveraz/segresearch (release v2.1.0, doi:ZENODO_V210_DOI)
```

### Abstract (plain ASCII, 1,897 of 1,920 characters)
No `$` in the abstract: arXiv renders `$`-delimited TeX in the title, abstract and comments with MathJax
(<https://info.arxiv.org/help/mathjax.html>), so the dollar sign is written out in words.
```
The Byte Latent Transformer (BLT) runs a large model once per patch and starts patches where a small model's next-byte entropy is high. Under tight budgets this rule skips computed values, such as the number after "= " in math. On BLT-1B at inference we trace the miss to the cue before a value: when the preceding bytes fix the next byte's type, the patcher's entropy is capped near log2 10 bits, close to a 10% budget's threshold. Entropy then starts a patch at 0% of values after a dollar sign and 19% after "= "; writing "is " instead of "= " raises the 19% to 66%. Version 1's two-byte lookup table, motivated by LZ novelty, acts as a learned detector of cues that tend to precede new values in its fitting text: on the solutions rewritten as prose it covers every value after a dollar sign and none after a plain space, and its margin over entropy falls from +42.1 to -8.9 points, partly because prose gives entropy more starts per value. In BLT-style models trained from scratch (1M-50M parameters, 128-byte context, 3 seeds per rule) and scored by teacher-forced exact match on answer spans, the table beats entropy at 50M by 36.8 points on computed results and 29.6 on final answers, which mostly copy the last computed result. A hand-written marker rule is not distinguishable from the table at 1M and 12M; its 50M control was not run. At 1M and 12M the table ran on a T4 GPU in fp16 and the baselines on an A40 in bf16; at 50M all ran on the A40. End to end, no 1M-50M model solves more than 2 of 100 GSM8K problems. A label-free probe of BLT-1B's patcher finds prose values, mostly after a dollar sign, but costs final answers. In six domains, no rule we tested beat entropy on exact match of whole words and numbers in more than one domain. The positive findings concern tight budgets on cue-formatted text. This version corrects version 1; the work is closed, with no venue intended.
```

## Version 1 (submitted 2026-10-15; record)

### Status
**SUBMITTED to arXiv on 2026-10-15: arXiv:2610.22100** (doi:10.48550/arXiv.2610.22100; cs.CL, cross-list cs.LG). Zenodo release v2.0.0: doi:10.5281/zenodo.23400010. Checklist:
- [x] References checked (new ones against their arXiv/publisher pages, 2026-10-15).
- [x] `./build.sh` ends with `OK -- verified from the tarball's own contents.` (10 pages)
- [x] Code public: https://github.com/nicoveraz/segresearch (PR #35)
- [x] GitHub release v2.0.0 + Zenodo DOI (not in Comments; in README and CITATION.cff)
- [x] Upload `paper2/arxiv-submission.tar.gz`; compare arXiv's compiled PDF with `paper2/arxiv.pdf` before confirming.

### Title
```
Patch where the text is new: a compression view of where byte-level models should spend compute
```

### Authors
```
Nicolás Vera Zúñiga
```
Independent Researcher, Chile. nicovera@quetru.cl

### Categories
- Primary: cs.CL
- Cross-list: cs.LG

### License
```
CC BY 4.0
```

### Comments
```
10 pages, 2 figures, 5 tables. Code, logs and results: https://github.com/nicoveraz/segresearch
```

### Abstract (plain ASCII, 1827 of 1,920 characters)
```
Byte-level language models such as the Byte Latent Transformer (BLT) run their large global model once per patch and start a patch where a small model's next-byte entropy is high. This rule rarely starts a patch at in-line computed results, such as the number after "=" in a worked solution, and accuracy on them collapses under tight patch budgets. We propose novelty patching, motivated by LZ77-style compression: start a patch where the upcoming bytes are unlikely to repeat the recent context. The rule is a lookup table over the previous two bytes, fitted in seconds on raw text with no labels, reference model or losses. In BLT-style models trained from scratch on a 1.9 GB math corpus at a matched ~11% patch rate (3 seeds each), it has no clear effect at 1M parameters, raises final-answer accuracy from 47% to 72% at 12M, and at 50M raises computed-result accuracy from 21% to 57% and final-answer accuracy from 60% to 90%; a table fitted on the raw pretraining corpus keeps the gain. Outside math the advantage shrinks: novelty lowers bits per byte on prose and ties entropy on source code and on synthetic execution traces. An analysis of where boundaries fall explains the pattern. Computed results have ordinary surprisal for the patcher's model, so surprisal-seeking rules pass over them: entropy concentrates patch starts on high-surprisal bytes most strongly of all rules (boundary enrichment 1.61) and is the worst on computed results, while the best rules sit near chance enrichment. Novelty works where new content begins at segment starts close to the computation, as in written math; a learned chunker (H-Net) finds such structure on its own, in math and in code. We also report what did not work: typed uncertainty as a training rule, switching rules after training, and a context-aware novelty predictor.
```

### Report number, Journal-ref, DOI, MSC, ACM
Leave empty.
