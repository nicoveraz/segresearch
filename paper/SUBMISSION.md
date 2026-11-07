# arXiv submission metadata

Everything the arXiv form asks for, in the order it asks, in the same format as the author's
earlier arXiv papers (LLM_Calc/paper/SUBMISSION.md). Nothing here is uploaded automatically:
`./build.sh` builds and verifies `arxiv-submission.tar.gz`; the fields below are pasted by hand.

## Status

**SUBMITTED to arXiv on 2026-10-08: arXiv:2610.11790** (doi:10.48550/arXiv.2610.11790). Checklist used:
- [x] Every citation verified at LOCAL FULL TEXT in `CITATIONS.md`, published versions included (2026-10-04; three corrections made).
- [x] Final novelty search right before submission (#13).
- [x] `uv run pytest tests/test_paper_numbers.py` passes (every load-bearing number in the prose; 9 tests).
- [x] A full read for flow, length and claims (2026-10-04; review fixes applied). Re-read once more before upload.
- [x] Make https://github.com/nicoveraz/segresearch public (the paper links to it), cut a release,
      and archive it on Zenodo; add the DOI to the Comments line, README and `CITATION.cff`.
      Check that the Zenodo DOI resolves to the new version before submitting (textca paper 4's
      release reached Zenodo about 70 minutes after submission).
- [x] Run `./build.sh`. It must end with `OK -- verified from the tarball's own contents.`
- [x] Upload `arxiv-submission.tar.gz` (main.tex, main.bbl, refs.bib, figures/). Check arXiv's
      compiled PDF against `arxiv.pdf` before confirming.

## Title
```
Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses
```
No dashes in the title field (arXiv renders them as two literal hyphens).

## Authors
```
Nicolás Vera Zúñiga
```
Independent Researcher, Chile. `nicovera@quetru.cl`.

## Categories
- Primary: `cs.CL` (same as the author's earlier papers).
- Cross-list: `cs.LG`.

## License
```
CC BY 4.0
```

## Comments
```
13 pages, 4 figures, 6 tables. Code, logs and results: https://github.com/nicoveraz/segresearch (archived: doi:10.5281/zenodo.23238215)
```
arXiv does not allow editing Comments after announcement without a new version, so check it now.

## Abstract (plain text, ready to paste)
arXiv caps this field at 1,920 characters. This version is 1901 characters, plain ASCII (arXiv rejects the Unicode minus sign).
```
Byte-level language models such as the Byte Latent Transformer (BLT) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high, so global compute goes where the next byte is hard to predict. We show that this rule has a systematic blind spot: positions whose type is predictable but whose value must be computed, such as the number after "=" in a worked math solution. Under tight patch budgets, entropy-triggered layouts skip these positions, and accuracy on them collapses. In Meta's BLT-1B with patch starts on 10% of bytes, the entropy rule puts a patch start at 16% of the computed results in GSM8K solutions and gets 19.0% of them exactly right; a boundary after each "=" at the same patch count gets 51.8%, and entropy combined with a label-free boundary-dependence signal gets 67.1% (default layout at 26% of bytes: 76.8%). The gap survives adapting BLT-1B to the budget with low-rank fine-tuning (32.9% vs 72.7%, three runs per rule, paired p < 1e-200) and grows with model size in byte models trained from scratch at a 10% budget: at 1M, 12M and 50M parameters, boundary dependence beats entropy on final answers by -1.6, +10.1 and +19.8 points, and at 50M it gets 35.9% of computed results against 13.9% (3 seeds each). BLT's entropy-jump rule helps neither target at 50M. The entropy trigger of Scratchpad Patching is likewise indistinguishable from random scratchpads on final answers (5.6% vs 5.9%, 5 seeds), while answer-start scratchpads give 38.1%. The effect is specific to computed values: copies and lookups gain little, and values the model cannot compute gain nothing. Boundary dependence, the rise in the model's own loss when a patch start is removed, measured per two-byte context, finds these positions without labels: combined with entropy it beats the hand-written rule on computed results.
```

## Version 2

**Not submitted.** Version 2 replaces arXiv:2610.11790 (a replacement, not a new submission); it waits for the author's
review. Steps, order and what must be published first: `ARXIV_V2.md`. What changed and why: the manuscript's "Changes in
version 2" section. New results, scripts and data: `v2/README.md`.

Checklist:
- [ ] The author has read the whole manuscript (`manuscript.pdf`) and `v2/README.md`.
- [ ] `uv run pytest tests/test_paper_numbers.py` passes (every number new in version 2, and the load-bearing numbers kept
      from version 1).
- [ ] The main session's release steps in `ARXIV_V2.md` ("First") are done: `scale/analyze.load()` keeps the runs with
      gradient clipping apart, Figure 3's legend is corrected and the figures rebuilt, and the root release files are
      updated for v2.1.0.
- [x] Release v2.1.0 of segresearch is on GitHub and its Zenodo record exists; its DOI, 10.5281/zenodo.24023934, is in the line below (the manuscript cites the concept DOI, which resolves to the newest version, so it needs no change).
- [ ] Version 2 of arXiv:2610.22100 is announced, and its title and its description of the two-byte table match
      `refs.bib` (`verazuniga2026patch`) and §13.
- [ ] `./build.sh` ends with `OK -- verified from the tarball's own contents.`; the page count below matches `arxiv.pdf`.
- [ ] Upload `arxiv-submission.tar.gz` through the Replace icon; check arXiv's compiled PDF against `arxiv.pdf`.

### Title
Unchanged:
```
Easy to anticipate, hard to compute: boundary dependence finds the computed outputs that entropy patching misses
```

### What changed (one sentence)
Version 2 scopes the blind spot to cues that fix the next byte's type and, in BLT-1B, to tight budgets (a new section
gives the entropy ceiling after such a cue, coverage by budget and a prose rewrite), shows that final answers mostly copy
the solution's last number and corrects the explanations that assumed otherwise, no longer says that the gap grows with
model size, and adds seed-level intervals and a second training recipe to the scaling study.

### Comments
The DOI of the v2.1.0 release is 10.5281/zenodo.24023934 (Zenodo, 2026-11-06).
```
15 pages, 4 figures, 6 tables. v2: the blind spot scoped to type-fixing cues and, in BLT-1B, to tight patch budgets (new Section 5: entropy ceiling, budget, prose rewrite); final answers shown to be mostly copies under teacher forcing, and their explanations corrected; the gap is no longer said to grow with model size; seed-level intervals and a second training recipe for the scaling study. Code, logs and results: https://github.com/nicoveraz/segresearch (archived: doi:10.5281/zenodo.24023934; all versions: doi:10.5281/zenodo.23238215)
```

### Abstract (plain text, ready to paste)
arXiv caps this field at 1,920 characters. This version is 1,897 characters, plain ASCII
(`tests/test_paper_numbers.py::test_arxiv_abstract` checks both, and that it says what the manuscript's abstract says).
```
Byte-level language models such as the Byte Latent Transformer (BLT) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high. We show that under tight patch budgets this rule has a blind spot: positions whose type the preceding text fixes but whose value must be computed, such as the number after "= " in a worked math solution. A nearly certain digit keeps the next byte's entropy near or below log2(10) = 3.32 bits, about BLT-1B's threshold for a 10% budget. In Meta's BLT-1B with patch starts on 10% of bytes, the entropy rule puts a patch start at 16% of the computed results in GSM8K solutions and gets 19.0% of them exactly right given the true preceding text; a boundary after each "=" at the same patch count gets 51.8%, and entropy combined with a label-free boundary-dependence signal gets 67.1% (default layout at 26% of bytes: 76.8%). The gap survives adapting BLT-1B to the budget with low-rank adapters (32.9% vs 72.7%, three runs per rule). In byte models trained from scratch at a 10% budget on a 1.9 GB math corpus, it is not detectable at 1M parameters, appears on final answers at 12M and on both targets at 50M, as training data grow too: at 50M boundary dependence gets 35.9% of computed results against 13.9% for entropy and 78.9% of final answers against 59.0% (3 seeds each), although final answers mostly copy the solution's last number and entropy starts a patch at every one. In BLT-1B the blind spot narrows at larger budgets (entropy covers about 90% of the values right after "= " at 20% of bytes) and in the same solutions rewritten as prose, where a patch start at each value still pays. Boundary dependence, the change in the model's own loss when a patch start is removed or added, measured per two-byte context on text in the same format, finds these positions without labels.
```
