# Replacing arXiv:2610.22100 with version 2 (paper 2)

Prepared on 2026-11-02 on the local branch `paper2-v2`. Nothing has been pushed, released or posted; every step below
is the author's to take. arXiv does not remove an announced paper, so version 2 replaces it: version 1 stays public as
`arXiv:2610.22100v1`, and version 2 becomes the default. Version 2 is a corrected and complete record of closed work;
no venue is intended.

## Recommended order

1. **Review.** Read `paper2/manuscript.md` (or `paper2/arxiv.pdf`), `paper2/SUBMISSION.md` (Version 2) and
   `paper2/v2/README.md`, including the open points it lists (redacted rule files, the prose-rewrite licence note, the
   Study A interim look reported as a disclosed deviation).
2. **Release v2.1.0 first** (needs the author's OK to push). Merge `paper2-v2` (and `paper1-v2`, prepared separately)
   into the public repository, update README.md and CITATION.cff, and publish the GitHub release v2.1.0 so that
   Zenodo archives it and mints its DOI. The paper's "Code and data" section and every number test point to files in
   that release, so it must be public before the replacement is announced.
3. **Fill in the DOI.** Replace the placeholder `ZENODO_V210_DOI` with the new DOI in `paper2/manuscript.md`,
   `paper2/v2/README.md` and the Comments line in `paper2/SUBMISSION.md` (and in this file), then run, from the
   repository root:

   ```
   ./paper2/build.sh        # must end with "OK -- verified from the tarball's own contents."
   PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_paper2_numbers.py
   ```

   If the page count changes, update the page count ("29 pages") in the Comments line.
4. **Replace the arXiv paper** (next section).

## Steps on arXiv

1. Log in to arXiv and open the user page (<https://arxiv.org/user>).
2. In the list of your articles, find 2610.22100 and select its **Replace** icon. (Do not withdraw it and do not
   start a new submission: a replacement keeps the identifier and the citation history.)
3. Upload `paper2/arxiv-submission.tar.gz` (main.tex, main.bbl, refs.bib and figures/; built and verified by
   `build.sh`). arXiv compiles it with pdfLaTeX; the package declares the few non-ASCII symbols it uses.
4. Metadata. The title and abstract may change in a replacement; paste them from `paper2/SUBMISSION.md`, Version 2.
   The abstract writes the dollar sign in words: arXiv renders `$`-delimited TeX in the abstract with MathJax, so a
   literal `$` would garble it. Check the abstract in arXiv's metadata preview.
   Authors, categories (cs.CL, cross-list cs.LG) and licence (CC BY 4.0) stay as in version 1.
5. **Comments.** arXiv asks that the reason for a replacement appear in the Comments field of the new version, and
   that the original comments be merged into it ("the old comments will be lost" otherwise). The line below repeats
   version 1's comments (counts updated, the code link) and says what changed:

   ```
   v2 corrects v1's main claims (new title, abstract and framing; the paper's section "Changes in version 2" lists each change) and is the project's final record; no venue is intended. 29 pages, 2 figures, 8 tables. Code, logs and results: https://github.com/nicoveraz/segresearch (release v2.1.0, doi:ZENODO_V210_DOI)
   ```

6. Preview: compare arXiv's compiled PDF with `paper2/arxiv.pdf` page by page (29 pages, 2 figures, 8 tables),
   then submit.

Notes from arXiv's help page on replacements (<https://info.arxiv.org/help/replace.html>, read 2026-11-02):

- "Previous versions of your article will remain publicly available."
- "We ask that articles be replaced no more than once per week." Plan for one replacement; paper 1 (2610.11790) is a
  separate article with its own replacement.
- Replacements are announced on the regular schedule. A replacement edited again on the same day after 14:00 US
  Eastern (Monday to Friday) is not re-versioned but delays its announcement; Friday 14:00 to Monday 14:00 counts as
  one day.

## After the replacement

Check the abstract page of 2610.22100: version 2 is the default, version 1 is listed under "Submission history", and
the Comments line shows the DOI of release v2.1.0.
