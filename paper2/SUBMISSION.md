# arXiv submission metadata (paper 2)

Paste these by hand into the arXiv form. `./build.sh` builds and verifies `arxiv-submission.tar.gz`.

## Status
**SUBMITTED to arXiv on 2026-10-15: arXiv:2610.22100** (doi:10.48550/arXiv.2610.22100; cs.CL, cross-list cs.LG). Zenodo release v2.0.0: doi:10.5281/zenodo.23400010. Checklist:
- [x] References checked (new ones against their arXiv/publisher pages, 2026-10-15).
- [x] `./build.sh` ends with `OK -- verified from the tarball's own contents.` (10 pages)
- [x] Code public: https://github.com/nicoveraz/segresearch (PR #35)
- [x] GitHub release v2.0.0 + Zenodo DOI (not in Comments; in README and CITATION.cff)
- [x] Upload `paper2/arxiv-submission.tar.gz`; compare arXiv's compiled PDF with `paper2/arxiv.pdf` before confirming.

## Title
```
Patch where the text is new: a compression view of where byte-level models should spend compute
```

## Authors
```
Nicolás Vera Zúñiga
```
Independent Researcher, Chile. nicovera@quetru.cl

## Categories
- Primary: cs.CL
- Cross-list: cs.LG

## License
```
CC BY 4.0
```

## Comments
```
10 pages, 2 figures, 5 tables. Code, logs and results: https://github.com/nicoveraz/segresearch
```

## Abstract (plain ASCII, 1827 of 1,920 characters)
```
Byte-level language models such as the Byte Latent Transformer (BLT) run their large global model once per patch and start a patch where a small model's next-byte entropy is high. This rule rarely starts a patch at in-line computed results, such as the number after "=" in a worked solution, and accuracy on them collapses under tight patch budgets. We propose novelty patching, motivated by LZ77-style compression: start a patch where the upcoming bytes are unlikely to repeat the recent context. The rule is a lookup table over the previous two bytes, fitted in seconds on raw text with no labels, reference model or losses. In BLT-style models trained from scratch on a 1.9 GB math corpus at a matched ~11% patch rate (3 seeds each), it has no clear effect at 1M parameters, raises final-answer accuracy from 47% to 72% at 12M, and at 50M raises computed-result accuracy from 21% to 57% and final-answer accuracy from 60% to 90%; a table fitted on the raw pretraining corpus keeps the gain. Outside math the advantage shrinks: novelty lowers bits per byte on prose and ties entropy on source code and on synthetic execution traces. An analysis of where boundaries fall explains the pattern. Computed results have ordinary surprisal for the patcher's model, so surprisal-seeking rules pass over them: entropy concentrates patch starts on high-surprisal bytes most strongly of all rules (boundary enrichment 1.61) and is the worst on computed results, while the best rules sit near chance enrichment. Novelty works where new content begins at segment starts close to the computation, as in written math; a learned chunker (H-Net) finds such structure on its own, in math and in code. We also report what did not work: typed uncertainty as a training rule, switching rules after training, and a context-aware novelty predictor.
```

## Report number, Journal-ref, DOI, MSC, ACM
Leave empty.
