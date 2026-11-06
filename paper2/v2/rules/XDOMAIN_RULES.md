# Cross-domain test (issue #30): is there a general rule for where a patch start pays? Decision rules

Written 2026-10-25, before any gain was measured (documents were built and spot-read for format only).

**Why.** Every earlier test of "compute pays where work begins" (v7, kind 2) scored arithmetic values: computed
numbers in GSM8K, synthetic math, traces. The author asked for a general target instead: compare domains and test
whether one rule predicts where a patch start pays everywhere, not only after `=`. Algebra (MATH) is added as the
closest new domain (computed values inside LaTeX, not word problems).

**Model and measure.** BLT-1B (`itazap/blt-1b-hf`, MPS), patch layouts changed at inference only. Base layout B per
document: BLT-1B's entropy at a 10% patch rate (threshold fitted on the domain's FIT documents). The positions
t >= 2 of a document are split into 48 groups by t mod 48; for each group, one forward pass with the starts of that
group toggled (added where B has none, removed where it has one), everything else as in B. The **gain** of a start at
t is the change at t's window when the start is present instead of absent, the rest of the layout fixed:
- **bits objective:** bits on bytes t..t+15 (16 bytes; windows of one group never overlap, and a start only changes
  predictions at and after itself), without the start minus with it;
- **unit objective** (only at *unit starts*: an alphanumeric byte after a non-alphanumeric one): exact greedy match
  of the alphanumeric run starting at t (capped at 16 bytes), with the start minus without it.
The bits objective is the perplexity side of paper 2's two-sided framing, the unit objective the answers side, made
domain-general (any word, number or identifier, not only computed values). Gains measured with 1/24 of positions
toggled at once (interference through context is part of the measure; reported, not corrected).
The spacing 48 was
chosen from a calibration (below) before the full run.

**Domains** (`xdomain_data.py`; 50 FIT + 60 TEST documents each, 300-950 bytes, ASCII): math (GSM8K, stripped),
algebra (MATH Algebra, problem + solution), code (Python stdlib), prose (English Wikipedia), clinical (MedQA-USMLE
vignettes with options and answer), traces (synthetic execution traces). Each domain also has a label-free 200 KB
pool (disjoint from TEST) for its novelty table.

**Rules for spending an extra 3% of patch starts** (scores at the positions B leaves without a start):
- **E** entropy: BLT-1B's next-byte entropy (the next-best entropy positions; BLT's own rule);
- **N** novelty table fitted on the domain's pool (paper 2's table; label-free);
- **H** entropy + novelty (sum of z-scores, z from the domain's FIT documents; v7 in byte form: surprise or new
  material);
- **U** unit start (1 at unit starts, else 0; ties by entropy): a structure-only rule with no fitting;
- **G_dom** gain table: mean bits gain per preceding two-byte context (fallback one byte, then global, if seen < 20
  times) fitted on the domain's FIT gains (in-domain reference; for the unit objective, mean unit gain);
- **G_lodo** the same table fitted on the FIT gains of the other five domains (leave one domain out): the
  general-rule candidate learned without the target domain;
- **G_math** the same table fitted on math FIT gains only (does the arithmetic map transfer?);
- **oracle** the measured gain itself (denominator).
Per TEST document, each rule picks its top k = round(0.03 n) positions among those without a start in B (unit
objective: among unit starts without a start in B, same k). **Capture** = summed gain at the picks / summed gain at
the oracle's picks, pooled over documents. Intervals: 1,000 bootstrap resamples of TEST documents; differences
paired.

**Verdicts** (per domain d, interval of capture(rule) - capture(E)):
- **X1, v7 beyond arithmetic** (unit objective): H - E above 0 in >= 5 of 6 domains: **holds** (new material pays
  wherever the target is a whole unit). Above 0 in <= 2 domains: **fails** (the effect is arithmetic-specific).
  Otherwise **mixed**, with the domains listed.
- **X2, the objective decides, in every domain** (bits objective): N - E *not* above 0 in >= 5 of 6 domains, and
  X1's H - E above 0 in the same domains on the unit objective: **holds**. Otherwise reported domain by domain.
- **X3, one map transfers** (each objective separately): G_lodo - E above 0 in >= 5 of 6 domains, and
  capture(G_lodo) >= 0.8 x capture(G_dom) (point estimates) in >= 5 of 6: **holds** (a general two-byte map of where
  starts pay exists). G_lodo - E above 0 in <= 2 domains: **fails**. Otherwise **mixed**.
- **X4, algebra** (the author's question): on algebra, G_math - E and N - E on the unit objective, each with its
  interval; **holds** if both are above 0.

**Predictions (written now).** X1 holds or is mixed with prose and clinical as the exceptions (their new words are
often not learnable: names, drug names, lab values); X2 holds (entropy is hard to beat on bits); X3 mixed (the map
of value cues differs: `=` in math, ` -> ` in traces, `(` and `=` in code); X4 holds.

**Reported without a verdict.** Mean gain by byte class per domain (after `=` / other digit / word or identifier
start / inside a run / whitespace / other); the top two-byte contexts of G_lodo; Spearman correlation of each score
with the bits gain over all positions; the removal side (gain at B's own starts); patch rates.

**Size.** 6 domains x 110 documents x 49 forward passes. If the smoke test projects more than about 8 hours on the
Mac, TEST is cut to the largest equal number per domain that fits, fixed before the full run. Any change after the
smoke test is disclosed below.

## Calibration and smoke test (2026-10-25, before the full run)

- Smoke test: 2 math FIT and 2 math TEST documents with spacing 24, about 13 s per document; only the gain
  arrays' sanity was checked (signs, magnitudes, the top positions of 2 documents). No rule was scored.
- Calibration: on one math and one prose TEST document, 12 random positions each, the grouped gain was compared with
  the gain of adding that start alone (one forward per position). Correlation with the single-start gain: spacing 24
  0.81 (math) and 0.92 (prose), spacing 48 0.88 and 0.93; mean absolute difference 0.91 / 0.74 bits at 24 and
  0.57 / 0.70 at 48 (grouped gains run 0.1-0.4 bits above single ones: the other added starts of a group also
  help). The design was changed from spacing 24 to 48 (about 5-6 hours in all). Nothing else changed.
- Pipeline test: measurement (spacing 48) and `xdomain_analysis.py` run once end to end on 2 FIT and 1 TEST
  document per domain, about 25 s per document, to check that the code runs; its printed output was not read and
  its result files are overwritten by the full run. The analysis code is committed with these rules.
