# R2: syntax+entropy at 50M — decision rule (written 2026-10-23, before any run; issue #28)

**Question.** Is paper 2's novelty table more than a learned detector of the cue before a value? Compare it with the
hand-written rule that starts a patch right after every math-syntax marker (`=`, `\boxed{`, `#### `, `>>`) and fills
the rest of the budget by entropy (`syntax+entropy10`, `scale/rules.py`), at 50M parameters, where paper 2's effect is
largest and this control was never run.

**Runs.** `syntax+entropy10`, size `50m`, seeds 0, 1, 2. Everything else as paper 2's clipped 50M runs: same corpus
(SHA-256 checked), same evaluation (1,000 computed results, 660 final answers, online boundaries), gradient clipping
1.0 (`SEGR_SCALE_CLIP=1.0`, tag `_clip`), A40 GPU, bf16, code at the published commit 1985984 (the submodule in this
repo; the only later change to `scale/` is pause/resume, inactive without a deadline). No other arm, no other size.

**Comparison.** nov10 at 50M (paper 2's three clipped seeds, already run: computed 56.3 / 58.5 / 57.7, final
90.6 / 89.4 / 88.6) minus syntax+entropy10 (the three new seeds), separately for computed results and final answers:
the difference of means with a Welch 95% interval over seeds (3 vs 3, `scale/compare_new.py` conventions).

**Outcome per target.**
- **AHEAD**: the whole interval above 0.
- **TIE**: the interval covers 0.
- **BEHIND**: the whole interval below 0.

**What follows (paper2v2/PLAN.md).**
- AHEAD on both targets: the table finds more than the cue; the compression framing stands.
- TIE or BEHIND on computed results: the claim narrows to "a label-free table recovers the format's value cues";
  entropy remains the wrong target.
- Mixed: reported as is, with the narrower wording.
The cue-free laptop test (#21) is read alongside, not instead.

**Secondary (reported, not decisive).** Bits per byte (nov10 vs syntax+entropy10, same interval); evaluation patch
rates (paper 2: syntax+entropy 12.1% vs novelty 11.9% of evaluation bytes at 12M; a gap above 1.5 points is
disclosed with the result).

**Exclusions.** A run is excluded only if its training failed by the pilot's rule (bits per byte > median + 1);
any exclusion, rerun or deviation from this document is reported with the result.

## Disclosure (added 2026-10-23, after the rule above was committed at 42d53fe; the rule is unchanged)

While computing intervals from existing results (paper2v2/intervals.py), two **unclipped** 50M syntax+entropy10 runs
from the earlier pilot were found (seeds 0, 1: computed 36.8 / 49.9, final 85.8 / 85.3). Paper 2's table had listed
the arm as "not run with clipping". Against clipped nov10 they give final +4.0 [+1.3, +6.7] and computed +14.2
[-69.5, +97.8] (2 seeds). They are a preview, not the test: clipping and seed count differ from the rule above. The
three clipped R2 runs decide, and the paper reports both, in this order of discovery.

## Correction and amendment (2026-10-23, still before any R2 run)

**Correction to the disclosure above.** The two unclipped runs were not newly found: they are published as a reference
section of the released v1 tables (`segresearch/paper2/tables/table1_math.md`, "50M without clipping (reference; nov10
was only run clipped)": syntax + entropy 43.3 +- 9.3 / 85.5 +- 0.3, two seeds, not marked as two) but not in the v1
paper text. We overlooked them when writing the rule. The rule above is unchanged.

**Amendment (secondary comparison, added before data).** From the same R2 runs, also report syntax+entropy10 minus
entropy10 at 50M (clipped entropy10 seeds from paper 2), computed and final, Welch 95% over seeds. It shows whether the
hand rule's own gain over entropy grows with scale, whatever the table-vs-rule outcome.

## Note on the code (added 2026-10-24, before any run of this study)

The rental archive overlays `paper2v2/pod_patch/scale/rules.py` and `prep.py` on the published code to add one rule
kind (`novtop{Q}+entropy{R}`, for R8c). On the Mac, every existing rule (entropy10/15/20, nov10/15, novc13, dep10,
syntax+entropy10, jump10) gives byte-identical thresholds and masks with and without the overlay; a toy run trained and
evaluated end to end. The runs of this study therefore use the same computation as the published code.

## Amendment (2026-10-29, before any R2 run; disclosed with the result)

Written after the project review of 2026-10-29 (`ANALYSIS_2026-10-29.md` section 2; review findings P2-5, TR-2, BIG-6,
TR-11, M5, M4, TR-12) and before the rental; no run of this study has started, and none of its data exists. The runs,
seeds (0, 1, 2), comparison and interval above are unchanged; what changes is which target decides.

1. **Final answers decide.** At 3 seeds per arm the computed-result interval is expected to be about ±18 points wide
   (seed SDs at 50M: clipped nov10 1.1 points, the unclipped hand rule 9.3), so a TIE there is the likeliest outcome
   whatever the truth (about 0.65 if the clipped hand rule scores like its unclipped preview). On final answers the
   expected half-width is about ±2.5 points. The outcome on final answers is therefore the verdict, and the outcome on
   computed results is reported beside it as secondary. A computed-result TIE is written "not distinguishable at 3
   seeds (interval a to b)", never "matches"; with a half-width above 10 points it is written "unresolved at 3 seeds
   (interval a to b)". The author chose this over two more hand-rule seeds (about $3-4).
2. **What follows**, from the final-answer verdict. AHEAD: the table's learned boundaries beat the hand-written cue set
   in training at 50M; R9 limits this to text that carries cues, and the compression framing stays superseded
   (PLAN.md; review TR-11). TIE or BEHIND: the claim narrows to "a label-free table recovers the format's value cues";
   entropy remains the wrong target. "Mixed" no longer arises.
3. **Hardware and precision are checked, not assumed.** The job scripts stop before any GPU time unless the pod has an
   NVIDIA A40, bf16 and torch 2.8 built for CUDA 12.8, as paper 2's 50M runs had (`pod_env.sh`, `check_pod`). No other
   GPU substitutes for the A40.
4. **Failed runs.** A bits per byte that is not finite also counts as failed (the median is taken over the finite
   ones), and so does a run stopped by the divergence guard (point 5), which leaves no result. Failed runs are reported
   and not replaced: no other seed is run in their place.
5. **Code.** The archive also overlays `paper2v2/pod_patch/scale/train.py`: the published file plus the divergence
   check of Study A's amendment A-6 (a loss that stays non-finite for 50 updates stops the run before the damaged
   state is saved; checkpoints are taken only while the loss is finite). On the Mac (CPU, toy sizes, with and without
   clipping), the weights, the final checkpoint, the sampler state and the optimizer moments are identical with and
   without it, and a run with a NaN learning rate stops at update 6. Each run's final weights are kept (`model.pt`,
   weights only; `keep_weights.py`). The verdicts are computed by `paper2v2/rental_verdicts.py`, committed with this
   amendment, tested (`tests/test_rental_verdicts.py`) and dry-run on paper 2's runs with stand-ins
   (`results/rental_verdicts_dryrun.json`; R2's stand-in, 50M entropy10, reproduces C5 to the last digit).
