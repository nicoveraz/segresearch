# R8 (issue #29): does entropy patching spend its budget where it cannot help? Decision rules

Written 2026-10-23 before any run of `realblt_waste.py` (no smoke test yet). BLT-1B, stripped GSM8K (the 294 test
problems of paper 1 / realblt_cues.py), thresholds fitted on 300 training problems, teacher forcing. Intervals: 95%
bootstrap over problems (1,000 resamples).

## A. What each patch start buys (perplexity currency)

For the entropy layout at a 10% patch rate (and, for comparison, the novelty table's at 10%), on 150 test problems:
assign every patch start (tokens 0 and 1 excluded) at random to one of 5 groups; for each group, remove its starts and
run the model; the **value of a start** is the increase in bits, summed over the bytes of the patch it started in the
full layout, when it is removed. (Downstream effects beyond the patch are not counted.) Each start is labelled by the
position it falls on: question text, or in the solution: computed-result start, final-answer start, other number,
word start, mid-word, space, punctuation, newline.

- **A holds** (a substantial share of entropy's budget buys nothing, by its own measure) if the share of entropy's
  starts with value <= 0 has an interval whose lower bound is at least 20%.
- **A fails** if that share's upper bound is below 10%. Otherwise inconclusive.
- Reported without a verdict: the same share for the table; mean value per start for both; values by position type.

## B. Patch starts needed for equal answers (task currency)

Layouts on all 294 problems: entropy at 5, 10, 15, 20, 25 and 30%; the table at 5, 7.5, 10 and 15%; the `=` rule
(forced at every computed result, rest entropy) at 7.5, 10 and 15%; the table plus a boundary at the answer marker at
10%. Measured: actual patch rate (global steps per byte), computed-result and final-answer exact match.
**k** = the patch rate entropy needs to reach the table's computed-result accuracy at 10% (linear interpolation on
entropy's accuracy-vs-rate curve), divided by the table's actual rate; if entropy never reaches it within 30%, k is
reported as greater than 30% / the table's rate.

- **B holds** (entropy needs clearly more compute for the same answers) if the interval of k has a lower bound of at
  least 1.5.
- **B fails** if k's upper bound is below 1.2. Otherwise inconclusive.
- Reported without a verdict: k for the `=` rule; k for final answers (the table plus the answer marker vs entropy).

## C. The trade-off in bits per byte (reported, no verdict)

From the same layouts: bits per byte over the whole text, over the question and over the solution, against patch rate.
Stated before the data: we expect entropy to be ahead of the table on whole-text bits per byte at equal rate (it spends
on hard bytes). Reported: the rate the table needs to match entropy's whole-text bits per byte at 10%.

## Decision for the framing (#29 F)

Adopt "entropy patching spends compute where it cannot help" only if A and B both hold, and state C's trade-off in the
abstract. Any change to this design after the smoke test is disclosed with the results.
