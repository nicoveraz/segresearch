# R13: the label-free probe hybrid in training — decision rule (written 2026-10-27 before any run)

**Question.** At inference on BLT-1B, a probe on the patcher's own state, trained on auto-labels ("a number not seen
earlier in the text"), found computed values better than the novelty table and lowered bits per byte (R12,
paper3/R12_AUTOLABEL_RULES.md). Does a model trained from scratch under a layout chosen by such a probe also gain on
both objectives?

**Arm.** `probetop3+entropy10`: a logistic probe on paper 2's training-time entropy model (1 layer, d = 32; its final
normed state predicting byte t, plus its entropy) for the auto-label, fitted on the first 4 MB of `math_train.u8`
(all positives, a seeded 25% of negatives; no answer labels). A patch start at every candidate position (previous byte
ASCII and not alphanumeric) whose probe score is in the top 3% of sampled training positions; the rest of a 10% budget
filled by entropy (the novtop3+entropy10 construction of R8c with the probe in place of the table). 12M, unclipped,
seeds 0-2, same corpus, evaluation and training code as paper 2's 12M runs; the code is the published commit plus two
added rule kinds (`paper2v2/pod_patch/`), shown on the Mac to leave every existing rule's thresholds and masks
byte-identical (toy corpus). Probe scores over the corpus are stored as float16 and thresholds are fitted on those.

**Feasibility check (before writing this rule; no decision rule, not a result).** On paper 2's evaluation file, with
the tiny entropy model, detection AUC for computed-value starts among candidates: entropy 0.24, novelty table 0.965,
the probe (auto-labels on 4 MB of math_train) 0.989.

**Comparisons** (Welch 95% over seeds, as R8c, against paper 2's existing 12M runs):
- **Holds** if (a) final answers: probe hybrid minus entropy10 has an interval above 0, and (b) bits per byte: probe
  hybrid minus entropy10 has an upper bound of at most +0.03.
- **Fails** if the final-answer interval is below 0, or the bits-per-byte lower bound is above +0.03.
- Otherwise inconclusive.
- Reported without a verdict: computed results; the probe hybrid against novtop3+entropy10 (R8c, same session),
  nov10 and syntax+entropy10; the evaluation patch rate and the share of forced starts.

**Scope.** Paper 2's corpus and evaluation text carry the `= ` / `>>` cues, so this tests the learned, label-free
signal as a training-time patcher, not its advantage without the cue (R10-R12 found that only at inference).

**Prediction (written now).** Holds, close to R8c's hybrid.

Runs last in the rental session (`p_probe12.sh` after R2, D and R8c), so a failure cannot affect them.

**Mac check (2026-10-27, toy corpus).** Against the published code, the 9 existing arms (entropy10, dep10,
syntax+entropy10, nov10, novc10, jump10, typed10, final+typed10, entropy15) give identical thresholds, train masks and
evaluation masks. The probe arm builds (probe fitted on 386,769 candidates, 73,330 auto-label positives; train rate
0.100, evaluation rate 0.150) and a 60-step toy run trains and evaluates end to end.

## Local pilot (written 2026-10-27, before it ran; no verdict)

One seed (0) of `probetop3+entropy10` at 12M on the Mac (MPS; about 9 h of training), from the same archive, corpus
(SHA-256 checked) and bundle as the rental, to test the pod pipeline at full corpus scale (probe fit, probe scoring of
1.9 GB, mask, training, evaluation) and to get an early read. It does not replace the 3-seed rule above, and its run
is not pooled with the rental's seeds (different device). Read against paper 2's 12M seeds (CUDA): final answers
entropy10 42.0 / 49.1 / 48.8%, nov10 72.3 / 71.2 / 70.9%, syntax+entropy10 76.4 / 73.9 / 78.5%; bits per byte entropy10
1.670 / 1.643 / 1.658. Described in advance as: **promising** if final answers >= 60% with bits per byte <= 1.70;
**no sign of a gain** if final answers <= 50%; otherwise **unclear**.

**Pilot result (2026-10-28; results/r13_pilot/).** Read as described above: **promising.** Final answers 73.5% (n = 660),
bits per byte 1.614, computed results 10.5% (n = 1,000); train patch rate 0.100, evaluation patch rate 0.130 (paper 2's
12M: entropy10 0.112, nov10 0.119, syntax+entropy10 0.121). Against paper 2's 12M seeds: final answers above all three
entropy10 seeds (42.0-49.1%) and nov10 seeds (70.9-72.3%), inside syntax+entropy10's range (73.9-78.5%); bits per byte
below every run of the three arms (1.622-1.707); computed results 10.5% against entropy10 4.8-6.4%, nov10 6.5-14.2%,
syntax+entropy10 10.3-14.9%. Not a test: one seed on another device (MPS against CUDA), and the evaluation patch rate
is higher than the comparison arms' (the probe fires more often on the evaluation text than on the corpus), so part of
any gain may be extra patches at evaluation; the rental's three seeds and the rule above decide. Pipeline at full scale,
no change needed: probe fitted on 386,769 candidates (73,330 auto-label positives); entropy scoring 45 min and probe
scoring 39 min on MPS; training and evaluation 5.6 h; the work directory held 13.3 GB for this one arm (entropy
scores 7.6 GB, probe scores 3.8 GB, mask 1.9 GB).

## Amendment (2026-10-29, after the one-seed pilot and before any rental run; disclosed with the result)

Written after the project review of 2026-10-29 (`ANALYSIS_2026-10-29.md` section 2; review findings DET-3, MISS-6,
TR-1, MISS-4, INF-11, TR/INF M1, TR/INF M4, TR-12) and before the rental; no rental run of R13 has started. It follows
the Mac pilot above, so the verdict is kept exactly as committed: changing it after the pilot would be a forking path.

1. **What HOLDS means.** Paper 2's corpus and evaluation text carry the `= ` and `>>` cues (Scope above), and every cue
   rule already beats entropy10 at 12M (the hand-written rule by +29.7 points on final answers, C3). HOLDS therefore
   means only that the label-free detector also works as a training-time rule on text with cues, like any cue rule. It
   does not show that the detector adds anything in training, nor its advantage without the cue: "trains without the
   cue" and "the detector's advantage holds in training" are not written (CLAIMS.md, do-not-write list).
2. **Readings, wording, rates, failed runs and code: as R8c's amendment of the same date** (its points 1-7). The verdict
   above is "at equal training budget"; the co-primary reading at equal evaluation rate compares the probe hybrid with
   entropy interpolated at its measured evaluation rate from entropy10 and D's entropy15 and entropy20; "gets both in
   training" is written only if both readings hold; a bits-per-byte INCONCLUSIVE is reported as "not distinguishable
   (interval a to b)"; the seeds stay 0-2 (the author, 2026-10-29). The pilot evaluated at 0.1303 against entropy10's
   0.1120, a gap of 1.8 points, above R2's 1.5-point disclosure line.
3. **Reported without a verdict** (the list above): against nov10, D's A40 runs (same device), with paper 2's T4 runs
   beside.
