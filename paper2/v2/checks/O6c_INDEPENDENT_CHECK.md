> **Public copy of an internal check report** from the author's private working repository (`analysis/2026-11-01_o6c_result/INDEPENDENT_CHECK.md`,
> last changed at 28e0e85). It is the source of the qualifications that paper 2 v2 attaches to the O6c result. Paths refer to that repository: the files it names under `results/` and
> `followup/` are in this package's `results/` and `rules/`; the check's own scripts and outputs are not included.
> The changes made for this copy (this header, and phrases naming the private repository or a temporary folder) are
> listed in README.md.

# O6c confirmatory run: independent check of the verdict (2026-11-01)

The run under filing 48618dc finished with no stop, no refused start and no failed check. The filing covers followup/O6c_CUTOFF_RULES.md, approved at 4fbd17d, with the code pinned at 3875d32. The outputs are:
- C0 at a075f86: results/blt_copy_latch_cutoff_blocks.json.
- C1 at c5ce090: results/blt_copy_latch_cutoff_score.json. Its patcher outputs, results/blt_copy_latch_cutoff_patcher.npz, are git-ignored (sha256 daa978d1...).
- `verdict` at 3d03679: results/blt_copy_latch_cutoff_verdict.json.
- The run log at 1b1e452: results/blt_copy_latch_cutoff_run.log.

Reported: "the cutoff better", D.L8 - D.SIVPO<620 = -3.17 [-5.00, -1.33] points of exact copies on 600 test texts. Beside it, D.L8 - D.SIVPO = +7.50 [+4.33, +10.67], "L8 better" under O6b's rule (O6b's released row was +6.7 [+4.0, +9.5]).

Four independent parts checked the run, each followed by an adversarial verifier: provenance (a), recompute (b), texts (c) and interpretation (d). A synthesis then combined them. Only findings that survived verification are counted, at the verifier's severity.

Sources used below:

| Short name | What it is |
|---|---|
| R | followup/O6c_CUTOFF_RULES.md at 48618dc (sha256 558e7834..., each output's code.rules_sha256; unchanged through 1b1e452) |
| B, S, V, L | the blocks, score, verdict and run-log files above |
| P | results/blt_copy_latch_cutoff_predict.log |
| code | realblt_copy_latch_cutoff.py at 3875d32 |
| fresh | realblt_copy_latch_fresh.py, unchanged from O6b's pin 4026bec |
| J7 | results/blt_copy_latch_fresh_j7.json (b60f319) |

The check's scripts and small outputs are in check/<part>/ next to this file (scripts/, out/), for the parts a_provenance, b_recompute, c_texts, d_interpretation, their verify_* parts, and synthesis. Copies of repository files, the bounded copies of the GSM8K train file, and outputs over 300 KB were left out. The run's driver, which existed only in a temporary folder outside git, is kept as drive.sh, and the O6 session's run report as O6_RUN_REPORT.md.

## Verdict

**The class "the cutoff better" and its reading stand as filed.** The outputs are the filed test's, and the jobs ran as the rule says. Every value in V follows from S. Every field of S that needs no model rebuilds from B, from GSM8K train idx 0-7176 and from the stored patcher outputs. No finding invalidates the result.

The reading, quoted whole (V:36 = V:sentences[0] = R:185-186):

> A byte rule that stops at token 620, a position fixed from dev texts, beats L8 under the released conversion, on this template.

Four qualifications must travel with it. A verifier confirmed each one at the severity "qualifies".

1. **Exact copies only.** The primary readout is the teacher-forced exact copy (R:172-173). On bits per copy, the other copy readout in the same output, L8 and the cutoff do not differ:
   - D.L8 - D.SIVPO<620 = -0.0136 [-0.0744, +0.0444] bits per copy (S:pairs['D.L8 vs D.SIVPO620'].id_copy.bits_per_unit). The mean is 2.629 bits against 2.642.
   - Under O6b's 0.5-bit margin (fresh:80) this would be a tie.
   - The row is not in O6c's rule, so it is reported here after the data.
   - "Beats L8" must not be read as cheaper copies.
2. **BLT-1B's default patch rate only.** R:140 scores "at D only". Every layout is refitted to D's fit rate, 0.3146 (R:151-153; S:train_rate_default), and entropy's test rate is 30.2% (S:by_layout.D.rate). Neither the reading nor "What it cannot show" (R:200-206) names a rate. The rate must therefore be named wherever the class is reported; Table 5b's label is "default (30.2%)". Nothing was run at 10% or at any other rate.
3. **One placement, which must be disclosed.**
   - 620 was fixed from the dev rows by the copy's position, after the J7-based predictions had been seen. The PI set it from the pre-approval check's recommendation (R:96-110, R:210-221; DECISION.md:3-16; :14-16 lists what must be disclosed).
   - Only 620, and only SIV+PO, were tested (R:202).
   - After the data (exploratory), the rule's own approximation (R:229-232) on O6c's rows gives:

     | Cutoff (token) | 560 | 600 | 615 | 617-621 | 622-630 | 665 | 700 |
     |---|---|---|---|---|---|---|---|
     | D.L8 - cutoff (points) | +2.17 | -2.17 | -3.00 | -3.50 | -3.33 | -0.50 | +1.50 |

   - So 620 sits on these texts' own optimum (synthesis scripts/extra.py; d and verify_d agree), and the class is specific to it.
4. **The discordant split as the verdict prints it.** V:sentences[2] says that of the 25 copies L8 lost to the cutoff, L8 did not fire at c1 on 22 and fired on 3. R:194-195 calls the second group "side effects of the refit", but that does not describe O6c's three:

   | Lost text | c1 | What decided the copy |
   |---|---|---|
   | idx 7035 | 790 | L8's own veto masks the copy past token 620, where the cutoff leaves it to D. D and the cutoff keep it; D-all loses it. |
   | idx 6734 | 615 | The only difference at or before the copy is token 613, the 't' of the copy line's "Ticket". SIV+PO masks it and L8 does not. |
   | idx 6722 | 299 | Differences only at token 47 (the threshold alone) and token 89 (L8 masks), far before the copy. |

Notes to report beside the class. These are not qualifications of it.
- **Strength.** The class is neither a seed nor a method artifact, but its margin is moderate. About 9% of bootstrap replications of the whole test would tie (see Robustness).
- **The predictions.** The rule predicted a tie. It gave this class about 1 in 4 on J7 and about 1 in 2 on dev (R:242-248; P:17-18, :31).
- **New by index.** Test text idx 6691 asks the question of O6b text idx 2483 (an O6b F1 and F6 test text, scored in J7) with only the name changed. It is concordant: exact under none of the five layouts. Dropping it leaves -3.17 (-19/599) and moves D.L8 - D.SIVPO to +7.51.
- **No cost bars.** O6c's rule reports no cost row, so nothing about costs is licensed. Read after the data as context only, no exact-match cost row of D.L8 against D.SIVPO<620 has an interval that excludes 0 (S:pairs['D.L8 vs D.SIVPO620']):

  | Cost row | D.L8 - D.SIVPO<620 |
  |---|---|
  | computed values | +0.12 [-0.28, +0.53] points |
  | final answers | -0.50 [-1.17, +0.17] points |
  | copied numbers | -0.04 [-0.37, +0.27] points |
  | bits per byte outside the copy | +0.0003 [-0.0001, +0.0007] |

- **Section 6's draft is stale until edited.** It says the length-cutoff rival is untested and that the O6c rule has no result (paper2v2/drafts/S6_copy_starts.md:115-118, :531-534). The changes it needs are listed at the end.

## What was checked, and by whom

| Area | Checked by | Result |
|---|---|---|
| Code, rule and model records: 12 covered files, script, segresearch gitlink, rule sha256, snapshot, weights blob, modeling_blt.py, library versions, threshold mode, the window path, no python -O | a. Synthesis: the 12 hashes in an archive of 3875d32, and B, S and V records equal except head (e4a3933, a075f86, c5ce090) | hold |
| Filing chain: approval, filing diff, go-ahead, first job | a, verify_a | hold |
| Job order, commits and pushes; nothing committed between jobs; outputs unchanged since commit | a. verify_a: a disk-wide search for writes at 18:50-19:30Z, and the tool-call inputs of the O6 session | hold |
| The driver (O6C_FILED, pgrep, caffeinate); the run log rebuilt byte for byte | a, verify_a | hold |
| C1's implementation checks, rerun from the npz with the pinned code | a. Synthesis: scripts/c1_checks.py | equal to S:checks |
| D's fit rate, the thresholds, the rates, and every row field that needs no model, rebuilt from the npz | a, b, d, verify_b, verify_d, synthesis | equal; 0 mismatches |
| The 900 texts, rebuilt by F1's builder, by an implementation written from R alone, with the real tokenizer, and from an exact bounded copy of the train file | a, c, verify_a, verify_c | 900 of 900 equal |
| Overlap with earlier problems; readers of idx >= 6253 since O6b's filing | c, verify_c | new by index; one renamed duplicate |
| Every value in V, recomputed from S's 600 rows; intervals, class, sentences | b, verify_b, d. Synthesis: scripts/recheck.py, which uses no repository code | 124 of 124 equal; intervals bit-identical |
| Robustness to seeds, methods and the margin | b, verify_b, verify_d, synthesis | the class holds; about 9% of replications tie |
| The readings against R, character for character | d, verify_d, synthesis | equal |
| Predictions, mechanism, artifacts, consistency with J7 | d, verify_d, b, verify_b | sections below |
| The Section 6 draft and paper2v2/PLAN.md | d, verify_d, synthesis | stale; changes listed below |
| The cost readouts in S, which no part had read | synthesis | outside the rule (note above) |

## Recomputed numbers

| Number | Committed | Recomputed | By | Match |
|---|---|---|---|---|
| Rule sha256 | V:code.rules_sha256 558e78342ed6... (the same in B and S) | 48618dc:R and the file on disk. `git diff 48618dc 1b1e452` on R is empty, as V:316 ("") says | a, b, c, d, synthesis | yes |
| Script sha256 | V:code.script_sha256 69afabdea19d... | 3875d32:realblt_copy_latch_cutoff.py | a, b, c, d, synthesis | yes |
| Covered code | V:code.files_sha256, 12 files | 12 of 12 equal the blobs at 3875d32 and at segresearch 19859849. fresh equals 4026bec (a48b9f82...) | a, b, c, synthesis | yes |
| Model record | V:code: snapshot 91aa6b8e..., weights_blob fdc3e6f1..., modeling_blt_sha256 df8ecdd4..., transformers 5.18.0, torch 2.14.1, numpy 2.5.3 | The only cached snapshot. Its blob link, file hash and venv versions are equal, and the record is the same as J7's | a | yes |
| Patcher npz | S:patcher_sha256 daa978d1c10a... | The file on disk, unchanged after every part's reads | a, b, c, d, synthesis | yes |
| Block header | B:block: seed 6301, visited 924, idx 6253-7176, pool_ended false | The same | a, c, verify_a, verify_c | yes |
| Texts | B:block: 300 fit (idx 6253-6556) and 600 test (6557-7176), with a sha256 each | 900 of 900 equal on idx, ticket, distance, n_bytes and sha256. n_tokens = n_bytes + 1. The npz holds the same 900 texts byte for byte | a, c, verify_a, verify_c, synthesis | yes |
| Rejections | 924 - 900; not listed in B | 24, all on length alone (1,001-1,359 tokens). If rejected problems gave their draws back, 879 texts would differ | c, verify_c | yes |
| Pool file | B:pool_file: 17f347dc..., 7,473 lines | Compared with O6b J0's record (a2d8c95) only. Recomputing it would read idx >= 7177 | a, c | record only |
| C1 checks, fit | S:checks.fit: 0 changed, 0 misaligned, 188,045,712 comparisons, 633,632 at boundaries, 318,016 decisions, 40,623 confident positions | Equal | a, synthesis | yes |
| C1 checks, test | S:checks.test: 0, 0, 356,164,876, 1,238,424, 621,612, 81,052 | Equal | a, synthesis | yes |
| D's fit rate | S:train_rate_default 0.31461939021936003 | Equal | a, b, d, verify_b, verify_d, synthesis | yes |
| Thresholds | S:thresholds_nats: entropy [1.34375, 1.0]; entl8 [1.3203125, 0.88797]; entsivpo [1.296875, 0.78088]; entsivpo620 [1.3125, 0.4] | Equal | a, b, d, verify_b, verify_d, synthesis | yes |
| Test patch rates | S:by_layout.*.rate: D 0.30222, D-all 0.29622, D.L8 0.30118, D.SIVPO 0.30197, D.SIVPO620 0.30069 | Equal. Minus D: -0.0060 (D-all, which has no refill), -0.0010, -0.0002, -0.0015 | a | yes |
| Row fields (600 rows x 5 layouts) | S:rows: c1, copy_len, fires_c1, starts_c1, rate | 0 mismatches | a, b, d, verify_b, verify_d, synthesis | yes |
| Exact copies | S:by_layout.*.id_copy.exact: 0.2417, 0.665, 0.7417, 0.6667, 0.7733 | 145, 399, 445, 400 and 464 of 600 | b, d, synthesis | yes |
| Primary | V:primary -0.03166666666666667 [-0.05, -0.013333333333333334] | (6 - 25)/600. boot_ratio with seed 6 and 2,000 resamples (realblt_copy_veto.py:60) gives a bit-identical interval | b, d, verify_b, synthesis | yes |
| Class | V:class "the cutoff better" | -3.17 <= -2 and -1.33 < 0, with n 600 >= 400 (R:174-177; fresh:1021-1028) | all parts | yes |
| Reading | V:reading and V:sentences[0] | R:185-186 with the line wrap removed (127 characters); code:430-431 | d, verify_d, synthesis | yes |
| D.L8 - D.SIVPO | V:beside: +0.075 [+0.043333, +0.106667], "L8 better" | (74 - 29)/600, with the same interval and class | b, d, synthesis | yes |
| D-all - D | V:beside: +0.423333 [+0.369958, +0.476667] | 254/600, the same | b, d, synthesis | yes |
| D.SIVPO<620 - D | V:beside: +0.531667 [+0.491667, +0.571708] | 319/600, the same | b, d, synthesis | yes |
| By-length table | V:beside['by length'], 35 cells | All equal; the sums equal the totals (table below) | b, d, synthesis | yes |
| Primary discordant split | V:beside: lost 25 (L8 did not fire on 22, fired on 3; the cutoff starts at c1 in 1); kept 6 (1 not fired, 5 fired); joint 22/1/2/5/1; the idx lists | The same; the lists are equal in order | b, d, synthesis | yes |
| D.L8 vs D.SIVPO discordant split | lost 29 (24 not fired, 5 fired); kept 74 (66, 8) | The same | b | yes |
| Printed sentences | V:sentences and L:398-401 | -3.17 [-5.00, -1.33]; +7.50 [+4.33, +10.67]; +42.33 [+37.00, +47.67]; +53.17 [+49.17, +57.17]; 22 and 3 | b, d, synthesis | yes |
| L8 fires at c1 | S:fires_c1: L8 0.73167; SIVPO620 0.80167 | 439 of 600; 481, which is the number of copies with c1 < 620 | b, d | yes |
| Bits per copy, D.L8 - D.SIVPO<620 (not in R) | S:pairs: -0.013570 [-0.074377, +0.044410] | Bit-identical | verify_b, synthesis | yes |
| Hash label at DECISION.md:47 | e57dde51... is called the hash of "line 13 to the end" | e57dde51... is the whole file at 4fbd17d; lines 13 to the end hash to 43258a1b... | a, c | no (wording only) |

The by-length table (V:70-116), as counts with shares in brackets, recomputed from the rows:

| Group | n | L8 fires at c1 | D | D-all | D.L8 | D.SIVPO | D.SIVPO<620 |
|---|---|---|---|---|---|---|---|
| tokens 0-600 | 426 | 416 (0.977) | 71 (0.167) | 365 (0.857) | 358 (0.840) | 361 (0.847) | 361 (0.847) |
| tokens 600-700 | 83 | 22 (0.265) | 25 (0.301) | 34 (0.410) | 40 (0.482) | 39 (0.470) | 55 (0.663) |
| tokens 700-2000 | 91 | 1 (0.011) | 49 (0.538) | 0 | 47 (0.516) | 0 | 48 (0.527) |
| copy starts before 620 | 481 | 438 (0.911) | 78 (0.162) | 398 (0.827) | 380 (0.790) | 400 (0.832) | 398 (0.827) |
| copy starts at 620 or later | 119 | 1 (0.008) | 67 (0.563) | 1 (0.008) | 65 (0.546) | 0 | 66 (0.555) |

The bins are [lo, hi) on n_tokens (code:60, :342-353). No text has 600 or 700 tokens, and none has c1 = 620. The two copies that straddle 620 count under "before 620", as R:94-95 says: idx 6597 (c1 619) and idx 6818 (c1 618). All five layouts lost both.

## Robustness of the class

- **Seed.** The class holds in 1,000 of 1,000 bootstrap seeds (seeds 0-999), with the upper end between -1.66 and -1.17 points (b, verify_b). verify_d and synthesis repeated seeds 0-199 and found it below 0 every time. The point estimate does not depend on the seed.
- **Method.** Every method gives the same class:
  - exact two-sided sign test on the 31 discordant texts (25 lost, 6 kept): p = 0.00088 (942,649/2^31 one-sided, doubled);
  - McNemar 11.65, or 10.45 with continuity correction;
  - Wald -3.17 +/- 1.96 x 0.919 = [-4.97, -1.37], z = -3.446;
  - Tango [-5.20, -1.44];
  - Clopper-Pearson, given the 31 discordant texts: [-4.40, -1.29].
- **Margin.** The result is net -19 texts, against the -12 a class needs (2 points of 600).
  - The class survives up to 7 lost copies turned concordant and becomes a tie at the 8th; it becomes a tie at the 4th lost-to-kept swap.
  - P(resampled net <= -12) = 0.916 (exact over the multinomial).
  - A nested bootstrap gives "the cutoff better" 0.912-0.914, tie 0.086-0.088 and "L8 better" 0 (b, verify_b).
  - The interval's upper end, -1.33, lies inside the margin: the true difference may be as small as about 1.3 points.
- **Texts.** Dropping the renamed duplicate (idx 6691) leaves -3.17. Counting the 7 texts decided away from the copy as ties (exploratory) gives -3.67 [-5.33, -2.17] (d, verify_d).
- **Placement.** See qualification 3: the class is specific to 620.

## Predictions (R:208-252; P), held and missed

Held:
- "L8 better" near 0 (R:246): the class is not "L8 better".
- D.L8 - D.SIVPO about +6.7 (+3.5 to +10), "L8 better" (R:249): +7.50 [+4.33, +10.67], "L8 better".
- The cutoff's exact share, about 0.777 (R:235): 0.773.
- Texts (R:252): 600 test texts; 900 of the 924 visited problems kept (97.4%), against about 97.5%.
- The by-length table near J7's (R:250): mostly. The largest departures are in the 600-700 row (83 texts against 73): D-all 41.0% against J7's 24.7% (|z| 2.21), and the cutoff 66.3% against J7's approximated 49.3% (|z| 2.17). No cell of the 18 departs by more than |z| 2.21.
- The kept side of the discordant split (about 7): 6.

Missed:

| Prediction | Predicted | Observed |
|---|---|---|
| Class | "tie" (J7: 0.71-0.78; P:17-18). "The cutoff better" about 1 in 4 on J7 (0.22-0.29). Dev alone, in-sample at 620, slightly favoured it (0.51-0.56; P:31) | "the cutoff better" |
| Primary | about -1.3 (J7; R:239) and -2.0 (dev; R:225-226) | -3.17. J7's -1.33 equals the observed upper end exactly; dev's -2.05 lies inside the interval |
| Discordant split | about 15 lost, 12 of them where L8 did not fire (R:239-240, :250-251) | 25 and 22 |
| Below token 575 | a tie, 6 lost and 6 kept (R:240) | 11 lost and 5 kept |
| At 575-619 | 9 lost and 1 kept, on 34 copies | 13 and 1, on 42 copies |
| D-all - D | about +50 (+45 to +55; R:249-250) | +42.33 [+37.00, +47.67] |

Why:
- **The primary row's miss comes from L8**, not from the cutoff or the approximation:
  - L8 kept 445 copies against J7's 458 (P:3; below token 575, 0.838 against 0.877, P:9). The cutoff kept 464 against the 466 approximated for J7.
  - The rule's approximation applied to O6c's own rows gives -3.50, so the approximation does not explain the miss.
  - L8's 445 against 458 is within sampling error (z about -0.87).
  - The sensitivity lines (R:244-245) perturbed only the cutoff's side.
- **D-all - D.**
  - Composition explains about -3.7 points: there are more long texts here (91 against 79 at 700+ tokens), and J7's per-bin gaps weighted by O6c's bins give +46.15.
  - Differences within the bins explain the other -3.8.
  - The +/-5 range is J7's own interval ([+44.8, +55.0], J7:F5.pairs['D-all vs D']): one sample's error, not an interval for a new sample.
- The O6 session's report (O6_RUN_REPORT.md) states these comparisons correctly. It leaves out the by-length table and the number of texts, which both held.

## Mechanism (after the data, exploratory)

This section uses the per-token masks rebuilt from the npz (d, verify_d) and S's rows (b, synthesis).

**Who vetoes the four copy tokens:**

| Pattern | Texts | Discordant |
|---|---|---|
| Both rules veto every copy token | 438 | 7 (2 lost, 5 kept) |
| Only the cutoff vetoes (c1 254-616) | 41 | 23 (22 lost, 1 kept) |
| Only L8 vetoes (idx 7035) | 1 | 1 (lost) |
| Neither rule vetoes | 118 | 0 |
| The copy straddles 620 (idx 6597, 6818) | 2 | 0 (all five layouts lost both) |

**The class rests on the 22 copies L8 missed before token 620.**
- On all 22, L8's agreement condition failed: the patcher's argmax at c1-1 was not the copy's first digit. It was a letter in 13 texts and '=' in 1, all at c1 569-615, and a wrong digit in 8 (7 at c1 254-524, plus idx 6672 at c1 612).
- D's starts on these copies equal D.L8's; D kept 1 of the 22, and D-all kept 18.
- Why the argmax fails there is outside what the test can show (R:204-206).

**Where the difference comes from:**
- By the copy's first token: net -15 of -19 at c1 569-619 (16 lost, 1 kept, in texts of 603-650 tokens), -3 at c1 < 569 and -1 at c1 790.
- At c1 575-619, L8 fires on 12 of 42 copies and keeps 12 exact; the cutoff keeps 24.
- By text length: -3 under 600 tokens, -15 at 600-699 and -1 at 700 or more.

**The texts decided by starts away from the copy:**
- The 3 L8-fired losses are described under qualification 4.
- The 5 L8-fired copies L8 kept (idx 6763, 6833, 6934, 7036, 7171; c1 267-394) are all decided by starts before c1:
  - idx 6934 by the copy line's 'T' (c1-7), which SIV+PO masks;
  - idx 7171 by the thresholds alone (c1-82, c1-80);
  - idx 6763, 6833 and 7036 by mask differences 36-325 tokens before c1.

**Bits per copy** (verify_b, synthesis):
- At c1 575-619, L8's copies cost fewer bits than the cutoff's (-0.86 [-1.41, -0.32]), though they are exact half as often (12 against 24).
- D-all against D shows the same split there: exact 20 against 7, bits +0.77 [+0.27, +1.28].
- Elsewhere: +0.036 [+0.009, +0.069] at c1 < 575, and +0.10 [-0.00, +0.28] at c1 >= 620.

## Artifacts

Ruled out:
- **The threshold refit.**
  - L8's and the cutoff's thresholds are one bf16 step apart: 1.3203125 with a tie share of 0.888, against 1.3125 with 0.4.
  - Swapping them changes no start at any copy token in any of the 600 texts.
  - Of the 31 discordant texts, only idx 7171, which L8 kept, is decided by the thresholds alone.
  - For scale: D.SIVPO and D.SIVPO<620 have identical masks before 620 and thresholds two steps apart. They flip 8 of the 479 copies that lie wholly before 620, 5 one way and 3 the other (d, verify_d).
- **A rate mismatch.**
  - Fit rates are within about 0.00005 of D's 0.314619: D.L8 -0.000031, D.SIVPO<620 +0.000019, D.SIVPO +0.000050.
  - On test texts, D.L8 - D.SIVPO<620 differs by +0.00049 (S:pairs['D.L8 vs D.SIVPO620'].rate).
  - Every matched layout is within O6b's 0.005. O6c's rule names no tolerance among C1's checks (R:142-145), so these are context rows, not a gate.
- **Starts away from the copy.** They decide 7 discordant texts, 5 to 2 in L8's favour, which is against the result. The base rate is 5 flips in 309 texts whose layouts differ only further back than 8 tokens before c1.
- **Causality and scoring.**
  - Layouts identical at every token up to the copy's end never differ in copy outcome: 0 of 240 texts in the primary pair, and 0 in every other pair.
  - SIV+PO<620 was recomputed from bytes 0..t-2 at every t in [c1-8, cL], 7,200 decisions, and differs from the stored mask in 0. So PO's possible read of bytes t-1 and t (R:89-94) never arose there (verify_d).
  - d also reports 0 differences in 15,918 decisions at t < 620 in the 31 discordant texts. Its verifier did not repeat this.
- **The implementation.** C1's checks passed, with 0 changed decisions and 0 misaligned positions on fit and test (S:checks). Rerun from the npz with the pinned code, they give the same record (a; synthesis, scripts/c1_checks.py).
- **The texts.** They were rebuilt twice, and 900 of 900 are equal. Rejections were on length alone. No kept index is at or below 6252. One question repeats an earlier one with a renamed person, and that text is concordant.
- **The verdict's arithmetic.** V copies S's aggregates (code:442-470, from score at :376-381). Those equal a recomputation from the rows: 124 of 124 values, with bit-identical intervals.
- **The bootstrap seed and method** (Robustness).

Not ruled out, or not checkable here:
- **The placement** (qualification 3). It is disclosed, and only 620 was tested.
- **Per-text id_copy exactness.** It needs BLT-1B forward passes, which this check may not run. Only consistency was checked: no layout starts at a masked c1, D-all never starts at c1, and identical layouts score identically.
- **The pool file's sha256 and line count.** These were compared with O6b J0's record only.
- **Other jobs during the score job.** The one-job guard (code:387) checks only at start. No other BLT-1B job or result file left a trace between 18:50 and 19:30Z, but a job that wrote nowhere would leave none.
- **Nothing posted to #35.** No network read was made. The O6 session's tool-call inputs after 18:58Z use no gh, curl or WebFetch and do not mention #35 (verify_a).

## Disagreements between parts, resolved from the files

1. **"Side effects of the refit."** b's finding on the label was kept at "qualifies" by its verifier; d's was lowered to a note by its verifier.
   - In the rule family, the label covers any difference outside the copy. O6b's F2a sentence reads "Neither rule touches the copy; a difference is a side effect of the refit." (7e9c6fa:followup/O6b_COPY_LATCH_RULES.md:1026). O6b's split is "copies L8 missed" against "side effect of the refit" (:1044-1046; fresh:1269-1270).
   - Under that use, idx 6722 and 6734 fit, since both rules veto every copy token there. idx 7035 fits in no sense.
   - Resolved: the class is unaffected. R:194-195 itself carries the label, so the wording of the row beside the class is a qualification (number 4).
2. **The pool sentence in Section 6.** d said S6:55-58 must add idx 6253-7176 and a LoRA caveat for O6c.
   - "From index 1800 on" (S6:55) already covers those indices. The audit chose that wording knowing about the LoRA draws (analysis/2026-10-31_p2copy/AUDIT.md:111-120, M7).
   - The LoRA overlap also applies to O6b's pool (7e9c6fa:followup/O6b_COPY_LATCH_RULES.md:758).
   - Resolved for the verifier: the range needs no change, and a LoRA caveat, if added, covers both pools.
3. **The severity of the prediction miss and of the band wording.** d rated both "qualifies"; its verifier lowered them to notes. The predictions are exploratory (R:208) and play no part in the class rule (R:172-177). Neither the reading nor the rows beside it say where the win comes from (R:185-196). Resolved: notes.
4. **The window record** (a). O6b's J7 also records code.window = 512, read before the switch, beside a literal window 0 (b60f319). In both runs, window 0 rests on code:395-398, the runtime check that the windows are None. Resolved: O6c records the window exactly as O6b did.
5. **Text novelty.** c said no kept problem repeats an earlier question; its verifier found one renamed duplicate. Synthesis read train idx 2483 and 6691 from the copy truncated at idx 7176. The questions are identical except for the name (Josie, Amanda), and both answers are 7. Resolved for the verifier. The index claim (R:115-117) stands.
6. **Small numbers.** Resolved for the verifiers; synthesis recomputed the placement and z (scripts/extra.py).
   - The D-all/D proxy peaks at c = 604-621, not only at 610-620.
   - The approximation's minimum is a plateau at 617-621.
   - The veto partition is 438 + 118 + 41 + 1, plus 2 straddling copies, = 600.
   - L8 - D fell from +56.8 to +50.0 because D rose 4.67 points and L8 fell 2.17, not only because D is higher.
   - z = -3.446.

## Run record and hygiene

**The run** (a, verify_a):
- **Filing chain.**
  - The PI approved the rule on 2026-11-01 (DECISION.md:43-44), before the filing at 48618dc (14:48:49Z).
  - The filing diff changes only the heading, the quoted approval and the Status (DIFF_CHECK.md:613-628).
  - The PI gave the go-ahead for the run on 2026-11-01 (DECISION.md:56). It was committed at e4a3933 and pushed at 18:58:45Z.
- **Jobs.** All three exited 0 (L:3, :15, :80):

  | Job | Ran (UTC) | Output | Output pushed |
  |---|---|---|---|
  | blocks | 19:01:09-19:01:17Z | a075f86 | 19:01:34Z |
  | score | 19:01:41-19:24:15Z | c5ce090 | 19:24:35Z |
  | verdict | 19:24:43-19:24:49Z | 3d03679 | 19:25:14Z |
  | run log | - | 1b1e452 | 19:25:16Z |

  Each output was committed and pushed before the next job (origin/main reflog), and no output changed after its commit.
- **The driver.** Every job ran with O6C_FILED=48618dc, PYTHONDONTWRITEBYTECODE=1 and HF_HUB_OFFLINE=1. `pgrep -fl realblt` ran before each job, and `caffeinate -i -w $$ &` ran in the driver's shell (R:159-163).
- **No refused start, crash or failed check**, so no dated note was needed. R is unchanged since filing (V:316). paper2v2/ and NOTES.md are untouched in 48618dc..1b1e452.

Notes on the run. None has any effect on the result.
- **The pool hash.** C0 hashed and line-counted the whole pool file, as R:134-135 requires (code:314-316). So the bytes of idx 7177-7472 went through a hash, though no problem there was parsed or built into a text. Word it "never parsed or used", not "never read".
- **Files outside git.** The driver (drive.sh, in a temporary folder) and the npz (.gitignore:5) exist only outside git. Keep a dated copy of both.
- **The run log's header.** The header (L:1) was written before the first job. For "pushed before the next job", cite the reflog and the driver, not the header.
- **The guard log.** The guard log that DIFF_CHECK.md section 18 cites ends in an AssertionError (6a1b4c9:analysis/2026-11-01_o6c_check/diff_check/final/guard_filing_48618dc.log:34-38). Its positive control therefore never ran on the filed text. The code's own filing check passed at run time in all three jobs (code.filed_commit 48618dc...).
- **Wording slips.**
  - DECISION.md:47 calls e57dde51... the hash of lines 13 to the end; it is the whole file's hash.
  - V's key "D.L8 - D.SIVPO (O6b's released row, replicated)" (code:458) differs from R:190's "repeats". ST7 changed the rule's wording (R:360-361), and the code key was left as it was by decision (DIFF_CHECK.md:210).
- **The rejected problems.** The block file does not list them. They are idx 6274, 6306, 6371, 6477, 6566, 6660, 6663, 6669, 6712, 6714, 6715, 6732, 6747, 6755, 6795, 6799, 6821, 6841, 6951, 7037, 7043, 7110, 7157 and 7163.
- **Readers since O6b's filing.** Since 7e9c6fa, no repository reader parsed or recorded train idx >= 6253 before C0, except the O6c script (verify_c's audit of the 115 Python files added since and of the committed outputs). Earlier readers rest on O6b's own Data audit.

**This check:**
- Both repositories were only read. At the end, git status shows the same 8 untracked *_smoke files as at the start, HEAD is 1b1e452, segresearch is at 19859849 and clean, and no new .pyc exists.
- Repository code ran only from git archives: 3875d32, or 3d03679, whose covered code is the same. segresearch and the npz were symlinked read-only, and the npz's sha256 is still daa978d1....
- GSM8K train was read only from copies truncated at idx 7176 (sha256 7466fd38...). One was made with head, whose buffer may hold bytes past line 7,177, though none were parsed; one was made by bounded reads. The full file was only stat'ed.
- The tokenizer was loaded only to call the filed builder (c, verify_a). No model weights were loaded, no blocks, score or verdict command ran, and O6C_FILED was never set.
- No Study A or Study B result was opened, and no other session was contacted.
- b's specification, said to be written from R before V was read, was not blind: its copies of V and the run log are stamped 93 s earlier, and the brief gave the numbers. The recompute is deterministic, so this changes nothing.
- Synthesis scripts: synthesis/scripts/recheck.py (numpy and JSON only), extra.py and c1_checks.py, with outputs in synthesis/out/.

## Refuted or corrected findings (one line each)

- c, "no kept problem repeats a question from train idx 0-6252": refuted. Test idx 6691 is train idx 2483 (an O6b F1 and F6 test text) with the name changed; the index claim stands.
- b, "the specification was written before the verdict's values were read": not supported (V was copied 93 s earlier, and the brief gave the numbers); no consequence.
- a, "O6c dropped a window record O6b had": wrong. O6b's code.window also read 512 before the switch; window 0 rests on code:395-398 in both runs.
- d, "the label fits none of the 3 cleanly": overstated. Under the family's use, 6722 and 6734 fit; only 7035 clearly does not.
- d, "S6:55-58 must add idx 6253-7176 and a LoRA caveat for O6c": "from index 1800 on" covers those indices, and the LoRA caveat concerns both pools.
- d, the proposed Table 5b row leaves out D.SIVPO<620 - D (+53.2 [+49.2, +57.2]), which R:191 lists. Its L8 - entropy (+50.0) comes from S:pairs['D.L8 vs D'], which is not a row R lists.
- d, "the D-all/D proxy peaks at 610-620": it peaks at 604-621.
- d, "438 + 118 + 41 + 1": that is 598; the 2 straddling copies make 600.
- d, "L8 - D and D-all - D are lower because D is higher": incomplete. D rose 4.67 points, D.L8 fell 2.17 and D-all fell 2.83.
- d, "fit rates matched to 0.00005": D.SIVPO is +0.0000503 off.
- b, "z = -3.44": it is -3.446.
- d, the severity "qualifies" for the prediction miss and for the band wording: lowered to notes (the predictions are exploratory, R:208).

## Changes the Section 6 draft needs

These apply to paper2v2/drafts/S6_copy_starts.md at 1b1e452 (last changed at 065dc92). Do not merge the draft until they are made.

| Lines | Change |
|---|---|
| S6:3-11 | Add O6c's sources (R at 48618dc and its sha256; B, S, V and L with their commits; this check). S6:4 must name O6c's rule. |
| S6:15-17 | Introduce O6c as a second filed run: on 600 new test problems (idx 6557-7176; fit idx 6253-6556), under the conversion as released. |
| S6:21-22 | Define token positions where O6c is quoted: token t is byte t-1, and token 0 is BOS. |
| S6:35-37 | Keep O6b's released reading whole, and replace "[O6c: ...]" with O6c's reading quoted whole. Add the primary row (-3.2 [-5.0, -1.3], the cutoff better), "at BLT-1B's default patch rate", and the disclosure. O6b's "that is untested" must not stand alone. |
| S6:37-38 | O6b's reading 5 stays O6b's. O6c's reading has no sentence about reading the model (R:179-186). |
| S6:45-47, :539-540 | Five licensed readings become six; recount the words. |
| S6:53-54 | "except the last scored row" becomes "the last two scored rows"; name O6c's rule in the caption (S6:533). |
| S6:55-58 | No change to the range. Use "new by GSM8K index" for O6c, because of idx 6691 and 2483. A LoRA caveat, if added, covers both pools. |
| S6:58-65 | Define SIV+PO<620, and say how 620 was fixed (R:96-110). |
| S6:67, :74 | Fill O6c's row (shown after this table) and add a SIV+PO<620 column. |
| S6:76-111 | Add an O6c note: the by-length and by-position cells, and the split printed as 22 and 3, not "side effects of the refit". Note (d) stays O6b's. |
| S6:115-118 | Replace "untested [O6c: ...]" with O6c's outcome inside R:200-206 and the default rate. Name both byte-rule comparisons. |
| S6:122-126, :141-143 | Add O6c's printed sentences (V:sentences[0..2]) as a separate list; O6b's reading 4 stays whole. |
| S6:146-161 | Keep O6b's list whole (including :158), and add O6c's "What it cannot show" (R:200-206) and the rate scope (R:140). |
| S6:163-317 | Add O6c's ledger rows (V', S', B', R'). |
| S6:318-331 | Add do-not-write items: an unqualified "a byte rule (length cutoff) beats L8"; other cutoffs or "620 is optimal"; other rates; cheaper copies in bits; "reading the model is not needed"; "replicates"; "side effects of the refit"; anything about the windows; "at no cost"; "new problems" without "by index". |
| S6:333-527 | Add O6c number tests: filed_commit 48618dc; the sha256 of R's first 35,913 bytes; window 0; checks passed; class, reading and rows. |
| S6:529-535 | Stale (:531-532 and :534); remove once filled. |
| S6:549-556 | :549-553 stays. Add O6c's outputs to test_paper2v2_numbers_traced. |
| paper2v2/PLAN.md:40, :106-107, :115 | Mark "Pending: O6c" done with the class, and add O6c's readings to the claims. |

O6c's row for Table 5b (S6:74):

| Column | Value |
|---|---|
| Label | 4-digit ticket, as released, new problems (O6c), default (30.2%) |
| n | 600 |
| entropy | 24.2 |
| no start on the copy | 66.5 |
| L8 | 74.2 |
| SIV+PO | 66.7 |
| SIV+PO<620 | 77.3 |
| no start - entropy | +42.3 [+37.0, +47.7] |
| L8 - SIV+PO | +7.5 [+4.3, +10.7] |
| L8 - SIV+PO<620 | -3.2 [-5.0, -1.3] |
| SIV+PO<620 - entropy | +53.2 [+49.2, +57.2] |

## Follow-ups recorded by the main session (2026-11-01)

- The driver (drive.sh) is now kept here. The patcher npz stays git-ignored on this disk; its sha256 (daa978d1...) is in
  the score output.
- DECISION.md's hash label and the guard log's ending (Run record, notes) are corrected by dated notes in DECISION.md and
  DIFF_CHECK.md section 18.
- The Section 6 draft and paper2v2/PLAN.md are updated separately (Changes the Section 6 draft needs).

