# R8c (issue #29): the label-free hybrid in training — decision rule (written 2026-10-24 before any run)

**Question.** At inference on BLT-1B, keeping entropy and adding patch starts at the novelty table's strongest
positions got both objectives (R8b H). Does a model trained from scratch under that layout also get both?

**Arm.** `novtop3+entropy10`: a patch start at every byte whose novelty-table value (paper 2's `novelty.npz`, the
table of nov10) is in the top 3% of sampled training positions, the rest of a 10% budget filled by entropy (the
syntax+entropy10 construction with the table's top 3% in place of hand-written markers). 12M, unclipped, seeds 0-2,
same corpus, evaluation and training code as paper 2's 12M runs; the code is the published commit plus one added rule
kind (`paper2v2/pod_patch/`), shown on the Mac to leave every existing rule's thresholds and masks byte-identical.

**Comparisons** (Welch 95% over seeds, against paper 2's existing 12M runs):
- **Holds** if (a) final answers: hybrid minus entropy10 has an interval above 0, and (b) bits per byte: hybrid minus
  entropy10 has an upper bound of at most +0.03.
- **Fails** if the final-answer interval is below 0, or the bits-per-byte lower bound is above +0.03.
- Otherwise inconclusive.
- Reported without a verdict: computed results; the hybrid against nov10 and against syntax+entropy10; the evaluation
  patch rate and the share of forced starts.

Runs last in the rental session (`h_hybrid12.sh` after R2 and D), so a failure cannot affect them.

## Amendment (2026-10-29, before any run of R8c; disclosed with the result)

Written after the project review of 2026-10-29 (`ANALYSIS_2026-10-29.md` section 2; review findings TR-1, MISS-4,
P2-13, INF-11, TR/INF M1, P2/DET M5, TR/INF M4, TR-12) and before the rental; no run of R8c has started. The arm, the
seeds (0-2) and the verdict above are unchanged.

1. **The verdict above is "at equal training budget"** (10% of training bytes in both arms) and is reported under that
   name.
2. **A co-primary reading at equal evaluation rate.** A hybrid can evaluate with more patch starts than entropy10 (R13's
   one-seed pilot: 0.130 against 0.112), and extra starts alone lower bits per byte and raise answers. The hybrid is
   therefore also compared with entropy interpolated linearly at the hybrid's measured evaluation rate, from entropy10
   (paper 2, A40) and D's entropy15 and entropy20 (same session, same A40): for final answers and bits per byte, the
   hybrid's mean minus the interpolated mean, a linear combination of independent arm means with a Welch-Satterthwaite
   95% interval (the t values of `scale.analyze.welch`). The thresholds are those above: holds if the final-answer
   interval is above 0 and the bits-per-byte upper bound is at most +0.03; fails if the final-answer interval is below
   0 or the bits-per-byte lower bound is above +0.03; otherwise inconclusive. If the hybrid evaluates at or below
   entropy10's rate, this reading is the equal-budget comparison, which then favours entropy. If it evaluates above the
   highest entropy arm with 2 or more valid seeds, the reading is not available. This reading holds evaluation compute
   equal, not the training budget.
3. **Wording.** "Gets both in training" is written only if both readings hold. At 3 seeds per arm the bits-per-byte
   test has little power (with no true difference it passes 0.15-0.30 of the time; review TR/INF M1), and the author
   chose not to add seeds (2026-10-29). A bits-per-byte INCONCLUSIVE is therefore reported as "not distinguishable
   (interval a to b)", never as a failure or as a match.
4. **Rates.** Each arm's evaluation rate is reported next to every verdict, and a gap of more than 1.5 points from
   entropy10's is disclosed with the result (R2's convention).
5. **Reported without a verdict** (the list above): the hybrid against nov10 uses D's A40 nov10 runs (same device);
   the comparison with paper 2's T4 nov10 runs is reported beside it.
6. **Failed runs**: as in D's amendment of the same date (the pilot's rule; a bits per byte that is not finite counts;
   so does a run the divergence guard stops; failed runs are reported and not replaced; an arm with fewer than 2 valid
   seeds gets no verdict).
7. **Code**: `paper2v2/rental_verdicts.py` (`hybrid_rule`, `rate_matched`), committed with this amendment, tested and
   dry-run on paper 2's runs (`results/rental_verdicts_dryrun.json`). R2's amendment of the same date (point 5)
   describes the divergence guard, the kept weights and the pod check.
8. **Rates measured on the Mac before the rental** (2026-10-29; the archive's code, masks only, no training; entropy scores on MPS, which reproduce paper 2's CUDA evaluation rates for entropy10 0.1120, nov10 0.1195 and entropy20 0.2279 to four decimals): novtop3+entropy10 evaluates at 0.1163 (train rate 0.1000), 0.43 points above
   entropy10 and below the 1.5-point disclosure line; entropy15 at 0.1754. R8c's rate-matched reading therefore
   interpolates 7% of the way from entropy10 to entropy15 and will be close to the equal-budget one. The pod's own
   rates decide.
