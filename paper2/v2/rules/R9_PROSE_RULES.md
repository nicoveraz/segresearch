# R9: is entropy's blind spot a property of computed values or of equation-style text? Decision rules

Written 2026-10-26, before any rewrite was generated or any model run.

**Why.** The cross-domain test (paper3/XDOMAIN_RULES.md) found the blind spot only where values follow `=` in
written arithmetic, and the earlier cue-free test (realblt_cues.py, Part B) found entropy covering 61% of computed
values in templated "varied prose". Templated text is not natural, and its problems differ from GSM8K's. This test
holds the problems and the values fixed and changes only the writing: the same GSM8K solutions, rewritten as natural
prose without equations.

**Text.** Two formats of the same problems:
- **orig**: GSM8K, stripped format (question, solution without `<<>>` annotations, "The final answer is N").
- **prose**: the same question, the solution rewritten by Claude Opus 5.5 (`paper2v2/prose_rewrite.py`, prompt in the
  script) as natural explanatory prose with no `=` and no arithmetic symbols between numbers, every computed result
  of the original kept in order and written with the same digits, then the same final-answer line.
- **Computed values**: in orig, each number right after `= ` (digits only; a `$` is not part of the span); in prose,
  the same values, located by markers the rewrite places around their first statement and removed before scoring.
- **Validation** (automatic, before any BLT-1B run): the marked values equal the original's computed values in order
  (commas and `$` ignored), no `=`, no operator between two numbers, ASCII, the token limit of the BLT-1B scripts;
  one retry, then the problem is dropped from both formats (count reported).
- **Problems**: test = the first 300 GSM8K test problems (as in realblt_cues.py Part A) that pass validation; train =
  up to the first 1,500 GSM8K train problems that pass (thresholds and novelty tables per format). The pilot uses
  the first 20 train problems.

**Model and layouts** (BLT-1B at inference, 10% budget; thresholds and tables fitted on each format's own train
problems, as in realblt_cues.py):
- **entropy**: BLT-1B's entropy;
- **nov**: the novelty table fitted on that format's train text (paper 2's procedure);
- **with / without**: the entropy layout plus / minus a start at every computed value (rest unchanged);
- **default**: BLT-1B's own patching (reference for how hard the values are).
Measured per computed value: covered (entropy starts a patch at its first byte) and exact match (teacher-forced
greedy, all its bytes). Intervals: 2,000 bootstrap resamples of test problems; orig - prose differences are paired
by problem and value.

**Verdicts.**
- **V1, coverage without the cue:** entropy's coverage of computed values, prose - orig. Interval above 0: entropy
  finds the values once the `= ` cue is gone.
- **V2, a start still pays without the cue:** gain = exact(with) - exact(without) at computed values in prose.
  Interval above 0: it pays. Also prose gain - orig gain (paired).
- **V3, the table needs the cue (natural text):** nov - entropy, exact at computed values, in prose. **Holds** if its
  interval is not above 0 (orig reported for reference).
- **Overall reading:** **format-tied** if V1's interval is above 0 and prose gain - orig gain has an interval below 0
  (in natural prose entropy covers the values and a start there is worth less); **follows computed values** if V1's
  interval is not above 0 and the prose gain's interval is above 0; otherwise **mixed**.

**Prediction (written now).** Format-tied: V1 above 0 (the templated test gave 61% vs 16%), the prose gain smaller
than orig's but above 0, V3 holds.

**Reported without a verdict.** Exact match under each layout per format; coverage and exact for the final answer;
patch rates; validation failures by reason; cost of the rewrites; how often the prose states a value in the same
sentence as its operands (a rough difficulty check).

**Budget.** Pilot of 20 problems with standard requests to measure cost per rewrite; the rest by the Message Batches
API only if the projected total is under about $30 (author's limit). Any change after the pilot is disclosed below.

## Pilot (2026-10-26)

20 train problems, standard requests: 17 valid, 3 dropped as too long for BLT-1B's 1,000-token limit (the prose is
longer than the original); $0.14 in all ($0.0070 per rewrite), projected total with the rest batched $6.38. Spot
check of 4 rewrites: values stated in words ("doubling 15 to get 30 dollars"), no symbols. Problems whose original
equations have no space after `=` (`3*2=6`) have no computed values by this definition (the paper's, as in every
earlier test) and serve only as train text. No change to the design.

## Run (2026-10-26)

Batch of 1,780 queued about 8 h before it ran; 1,614 valid (159 too long, 5 marked values differ, 2 non-ASCII); the
collect step retried the 166 failures once at standard prices as written (26 valid; the "too long" ones mostly stay
too long). Valid: 1,382 train, 275 test (221 with computed values, 711 values). Cost: pilot $0.14 + batch $5.94 +
retry $1.48 = $7.56. Rewrites saved in data/prose_rewrites/. Results: NOTES.md (2026-10-26), results/blt_prose.json.
