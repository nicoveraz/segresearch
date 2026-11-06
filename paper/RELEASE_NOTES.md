# Release notes

## v2.1.0 (2026-11-06)
Version 2 of both papers, the project's final release.
- Paper 1 (arXiv:2610.11790v2, `paper/`): the blind spot scoped to type-fixing cues and tight budgets (new §5),
  final answers described as mostly teacher-forced copies, seed-level intervals and a second training recipe in the
  scaling study; support files in `paper/v2/`.
- Paper 2 (arXiv:2610.22100v2, `paper2/`): retitled and corrected; the two-byte table read as a learned cue
  detector; results, scripts, data (LLM-written prose rewrites of GSM8K problems, MIT) and filed decision rules in
  `paper2/v2/`.
- `scale/analyze.load()` keeps the 50M runs with gradient clipping apart, so `make_tables.py` and `make_figures.py`
  rebuild paper 1's Table 3 and Figure 2 from its own runs; Figure 3's legend gives BLT-1B's default patch rate as ~26%.

## v1.0.1 (2026-10-08)
Adds the arXiv identifier (arXiv:2610.11790, doi:10.48550/arXiv.2610.11790) to `CITATION.cff` and the README.
Code and results are unchanged from v1.0.0.

## v1.0.0 (2026-10-08)
First release, submitted to arXiv with the paper: code, run logs, the results registry, per-target results and the
paper source. Archived on Zenodo (concept DOI 10.5281/zenodo.23238215; this version 10.5281/zenodo.23238216).
