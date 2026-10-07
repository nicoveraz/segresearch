# arXiv submission metadata

Everything the arXiv form asks for, in the order it asks, in the same format as the author's
earlier arXiv papers (LLM_Calc/paper/SUBMISSION.md). Nothing here is uploaded automatically:
`./build.sh` builds and verifies `arxiv-submission.tar.gz`; the fields below are pasted by hand.

## Status

**DRAFT, not ready to submit.** Before submission:
- [x] Every citation verified at LOCAL FULL TEXT in `CITATIONS.md`, published versions included (2026-10-04; three corrections made).
- [ ] Final novelty search right before submission (#13).
- [x] `uv run pytest tests/test_paper_numbers.py` passes (every load-bearing number in the prose; 9 tests).
- [x] A full read for flow, length and claims (2026-10-04; review fixes applied). Re-read once more before upload.
- [ ] Make https://github.com/nicoveraz/segresearch public (the paper links to it), cut a release,
      and archive it on Zenodo; add the DOI to the Comments line, README and `CITATION.cff`.
      Check that the Zenodo DOI resolves to the new version before submitting (textca paper 4's
      release reached Zenodo about 70 minutes after submission).
- [ ] Run `./build.sh`. It must end with `OK -- verified from the tarball's own contents.`
- [ ] Upload `arxiv-submission.tar.gz` (main.tex, main.bbl, refs.bib, figures/). Check arXiv's
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
13 pages, 4 figures, 6 tables. Code, logs and results: https://github.com/nicoveraz/segresearch (archive DOI to add)
```
arXiv does not allow editing Comments after announcement without a new version, so check it now.

## Abstract (plain text, ready to paste)
arXiv caps this field at 1,920 characters. This version is 1901 characters.
```
Byte-level language models such as the Byte Latent Transformer (BLT) group bytes into patches and run their large global model once per patch. BLT starts a patch where a small model's next-byte entropy is high, so global compute goes where the next byte is hard to predict. We show that this rule has a systematic blind spot: positions whose type is predictable but whose value must be computed, such as the number after `=` in a worked math solution. Under tight patch budgets, entropy-triggered layouts skip these positions, and accuracy on them collapses. In Meta's BLT-1B with patch starts on 10% of bytes, the entropy rule puts a patch start at 21% of the computed results in GSM8K solutions and gets 14.8% of them exactly right; a boundary after each `=` at the same patch count gets 41.5%, and entropy combined with a label-free boundary-dependence signal gets 60.7% (default layout at 28% of bytes: 73.0%). The gap survives adapting BLT-1B to the budget with low-rank fine-tuning (27.6% vs 70.0%, three runs per rule, paired p < 1e-230) and grows with model size in byte models trained from scratch at a 10% budget: at 1M, 12M and 50M parameters, boundary dependence beats entropy on final answers by −1.6, +10.1 and +19.8 points, and at 50M it gets 35.9% of computed results against 13.9% (3 seeds each). BLT's entropy-jump rule helps neither target at 50M. The entropy trigger of Scratchpad Patching is likewise indistinguishable from random scratchpads on final answers (5.6% vs 5.9%, 5 seeds), while answer-start scratchpads give 38.1%. The effect is specific to computed values: copies and lookups gain little, and values the model cannot compute gain nothing. Boundary dependence, the rise in the model's own loss when a patch start is removed, measured per two-byte context, finds these positions without labels: combined with entropy it beats the hand-written rule on computed results.
```
