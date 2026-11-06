# R10: does a learned signal find computed values where the `=` cue is gone? Decision rules

Written 2026-10-27, before any router was trained or any hidden state extracted on these texts.

**Why.** R9 (paper2v2/R9_PROSE_RULES.md) found that in natural prose entropy starts a patch at 57% of computed
values and its layout still leaves about 13 points of exact match (21.7% vs 34.6% with a start at every value); the
two-byte table, which reads the `= ` cue, is below entropy there. Earlier (realblt_router.py, #27) a router on
BLT-1B's patcher state tied the table in equation text, and a probe on Qwen's hidden state flagged errors far better
than output entropy (#20). Question: is there a learned signal of "a computed result begins here" that works
without the cue, and does one learned on equation text transfer to prose?

**Data.** data/prose_rewrites (R9): 1,382 train / 275 test GSM8K problems, each in two formats, **orig** (stripped,
values after `= `) and **prose** (rewrite without equations), with the same computed values marked.

**Candidates and label.** Decision positions in the solution where the previous byte is not alphanumeric (a space,
punctuation, newline or `$`): a patch may start at the next byte. Label 1 if a computed value starts at that byte.
The rule is causal: it uses only bytes before the position.

**Signals** (all read before the candidate byte):
- **entropy**: BLT-1B's patcher entropy (BLT's rule);
- **table**: the novelty table fitted on the format's train text (R9's);
- **router_patcher[F]**: logistic regression on BLT-1B's patcher hidden state (last layer, normed), standardized and
  reduced to 32 principal components on the training set, trained on format F's train problems (F = orig or prose);
- **router_qwen[F]**: the same on Qwen2.5-1.5B's layer-14 state at the last token ending before the candidate's
  preceding byte (see the disclosure below; not
  deployable inside BLT; tests whether a capable model's state holds the signal).
Training uses all positive and a seeded 25% of negative candidates. Scores on the same format's train problems (for
thresholds) are out-of-fold (5 folds by problem); on the other format, from the full fit.

**Layouts** (BLT-1B, 10% patch rate, test problems of each format): entropy; **with** (entropy plus a start at every
computed value; the R9 reference); **hybrid(X)**: X's top positions forced, threshold set on the format's own train
problems so that forced starts are 3% of positions (label-free), the rest by entropy up to 10% (paper 2's hybrid
recipe, R8b), for X in table, router_patcher[orig], router_patcher[prose], router_qwen[orig], router_qwen[prose].
Measured: exact match on computed values and final answers (teacher-forced greedy), coverage, bits per byte of the
solution. Intervals: 2,000 bootstrap resamples of test problems, differences paired.

**Verdicts (prose test).**
- **L1, a learned signal finds values without the cue:** hybrid(router_patcher[prose]) - entropy, exact on computed
  values: **holds** if the interval is above 0.
- **L2, the patcher's signal does not depend on the `=` token:** hybrid(router_patcher[orig]) - entropy, same
  measure: **holds** if above 0.
- **L3, a capable model's state carries a format-independent marker:** detection AUC of router_qwen[orig] on prose
  candidates: **holds** if its lower bound is above 0.75. Reported with router_patcher[orig]'s AUC and their paired
  difference.
- **L4, both objectives (reported, no verdict):** bits per byte of each hybrid minus entropy; share of the with -
  entropy gap each hybrid closes.

**Predictions (written now).** L1 holds (prose has its own cues: "comes to", "leaves", "brings ... to"); L2 fails (a
small patcher trained on equation text keys on `= `); L3 holds; the deployable hybrid closes part of the 13-point gap.

**Reported without a verdict.** Detection AUCs of every signal on both formats; the orig-test results of every
layout (reference: the table and the equation-trained routers should do well there); patch rates and forced shares.

**Compute.** Mac only: Qwen states first, then BLT-1B (sequential, memory caps as before). Any change after a smoke
test is disclosed below.

## Smoke test (2026-10-27, 40 train / 20 test problems; results not used)

Two fixes before the full run, nothing else changed:
- Candidates must be ASCII bytes with an ASCII previous byte (a few orig solutions contain multi-byte characters,
  which broke the byte-to-character alignment).
- **Qwen leak.** Qwen2.5 writes digits as single tokens and splits a preceding space off as its own token, which it
  does not do before a word. The state at that space token therefore reveals that a digit comes next (smoke-test AUC
  0.995 on 20 problems). The Qwen state is now read at the last token ending before the candidate's preceding byte,
  whose boundaries do not depend on the candidate byte. (The same alignment was used in realblt_router.py, #27, and
  should be checked there.)
