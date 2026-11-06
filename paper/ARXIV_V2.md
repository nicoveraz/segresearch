# Replacing arXiv:2610.11790 with version 2

Version 2 of *Easy to anticipate, hard to compute* corrects and scopes version 1 (what changed and why: the
manuscript's "Changes in version 2" section). It replaces the arXiv article; it is not a new submission. Nothing here
has been uploaded, pushed or posted: every step below is the author's.

## First: the v2.1.0 release and its Zenodo record

The paper's new numbers trace to files in `paper/v2/` and to a fixed release of the repository, so the release comes
before the arXiv replacement:

1. Merge this branch (`paper1-v2`) with the paper 2 version 2 branch, as planned; v2.1.0 is the release for both
   papers' version 2 (v2.0.0 already exists, for paper 2 version 1). This branch was created from `origin/master` and
   still tracks it; `git branch --unset-upstream paper1-v2` removes that, so that a bare `git pull` or `git push` does
   nothing unexpected.
2. Files outside this branch's scope, for the main session before tagging:
   - `scale/analyze.load()`: make it skip runs with `cfg["clip"]` (or key them apart). As it is, a `_clip` run can
     replace the unclipped run with the same size, rule and seed, so running `make_tables.py` or `make_figures.py`
     rewrites Table 3 and Figure 2 with the two recipes mixed. The manuscript does not depend on this fix: its
     Declarations name `paper/v2/scripts/scaling_unclipped.py`, which rebuilds both from Table 3's runs alone
     (`paper/v2/README.md`, "Table 3, Figure 2 and the runs with gradient clipping"). After the fix,
     `tests/test_paper_numbers.py::test_table3_and_figure2_rebuild` also checks that `make_tables.py` itself rebuilds
     Table 3 byte for byte.
   - `make_figures.py`, line 46: Figure 3's legend says "BLT-1B default (~28%)", the released conversion's patch rate;
     the figure plots the corrected model, whose default layout starts a patch on 25.5% of bytes (Table 1, §3). Change
     the label to "(~26%)", rebuild the figures and check that only that label changed.
   - The root release files (`README.md`, `CITATION.cff`, `paper/RELEASE_NOTES.md`, `.zenodo.json`) for v2.1.0. The
     release now redistributes GSM8K text (the questions and solutions in `paper/v2/data/prose_rewrites/`, under
     GSM8K's MIT licence, `LICENSE-GSM8K` there), so `.zenodo.json`'s "Datasets (GSM8K, MATH, OpenWebMath) ... are not
     redistributed" needs that exception.
3. Push the tag v2.1.0 and let Zenodo archive it. This needs the author's OK: pushing makes the branch, `paper/v2/`
   (with the prose rewrites) and the release public.
4. Check that the version DOI resolves, then replace the placeholder `ZENODO_V210_DOI` in the Comments line below and
   in `SUBMISSION.md`. The manuscript cites the concept DOI (10.5281/zenodo.23238215), which resolves to the newest
   version, and names release v2.1.0, so the PDF needs no change for the DOI.

## Then: the replacement on arXiv

Replace 2610.11790 only after version 2 of arXiv:2610.22100 is announced (not only submitted), so that §13's and the
"Changes in version 2" sentence, that the follow-up's version 2 corrects the main claims of its first version, holds
when readers follow the citation. Before replacing, check paper 2 version 2's title against `refs.bib`
(`verazuniga2026patch`, which has version 1's title) and its description of the two-byte table, a learned detector of the
cue before values, against §13.

1. Run `./paper/build.sh`; it must end with `OK -- verified from the tarball's own contents.` Check that the page count
   it prints matches the Comments line.
2. Log in to arXiv and open your user page; under the article 2610.11790, click the **Replace** icon. (Only an
   announced article can be replaced this way.)
3. Upload `paper/arxiv-submission.tar.gz`. Keep the title. Paste the version 2 abstract and the Comments line below,
   both from `SUBMISSION.md`.
4. The Comments field must say what changed, and it must repeat the old comments: arXiv replaces the field, so old
   comments that are not merged into the new text are lost. The line below does both (page counts, the change, the code
   link, the archive of this version and the concept DOI that covers all versions, version 1's archive included).
5. Check arXiv's compiled PDF against `paper/arxiv.pdf`, then submit.

What to expect:
- Earlier versions stay public: 2610.11790v1 remains available, and the abstract page lists both versions.
- arXiv asks that an article be replaced no more than once a week, so fix everything in this round.
- The replacement is announced on the regular announcement schedule, like a new submission. Edits made the same day
  before the daily cutoff (14:00 US Eastern on weekdays) do not create another version.
- Changing the comments or other metadata later requires another version (journal reference, DOI and report number
  are the exceptions).

## Comments line for version 2

`ZENODO_V210_DOI` is a placeholder for the DOI of the v2.1.0 release; replace it before submitting.

```
15 pages, 4 figures, 6 tables. v2: the blind spot scoped to type-fixing cues and, in BLT-1B, to tight patch budgets (new Section 5: entropy ceiling, budget, prose rewrite); final answers shown to be mostly copies under teacher forcing, and their explanations corrected; the gap is no longer said to grow with model size; seed-level intervals and a second training recipe for the scaling study. Code, logs and results: https://github.com/nicoveraz/segresearch (archived: doi:ZENODO_V210_DOI; all versions: doi:10.5281/zenodo.23238215)
```

Version 1's line, for reference: `13 pages, 4 figures, 6 tables. Code, logs and results:
https://github.com/nicoveraz/segresearch (archived: doi:10.5281/zenodo.23238215)`. Its DOI is the concept DOI, which
now resolves to the newest release; version 1's own archive is v1.0.0, doi:10.5281/zenodo.23238216.
