# O6c: L8 against a byte rule with a fixed position cutoff (token 620), under the conversion as released. Decision rules

Written 2026-10-31, after O6b's confirmatory run (filed 7e9c6fa; verdict af15140). A follow-up to O6b
(followup/O6b_COPY_LATCH_RULES.md, whose filed section defines L8, SIV+PO, the layouts and the class rule used here).
The PI asked for this test (2026-10-31) after a proposal that the deciding experiment was the length-cutoff rival on
the published-model texts, with a short pre-registration first. The design is the
main session's, except the cutoff, which the PI set after an independent check (2026-11-01); the O6 session drafted this
text and lists the concerns in the Status. Its result is planned for paper 2 v2's Section 6: paper2v2/PLAN.md,
"Decision 2026-10-31", lifts the earlier rule that nothing from O6/O6b goes into paper2v2 and lists O6c's result among
what must come in before the framing is fixed; paper2v2/drafts/S6_copy_starts.md holds its placeholders. Its readings
are quoted whole there, as everywhere (Readings, below).

## O6c confirmatory rule, filed (2026-11-01)

**The PI's approval** (2026-11-01). The PI approved the rule at 4fbd17d for filing after a plain summary of the rule
(analysis/2026-11-01_o6c_check/DECISION.md, "The PI's approval of the rule"). The approved text is this section at 4fbd17d, from its heading to
the end of the file, with the code pinned at 3875d32.

**Status: filed and approved on 2026-11-01.** Written 2026-10-31 by the O6 session from the main session's design,
and amended 2026-11-01 after an independent check (analysis/2026-11-01_o6c_check/VERDICT.md) and the PI's choice of
the cutoff (DECISION.md there), as the Change log lists. Until the PI approved this text, no GSM8K train problem from
idx 6253 on was read, and no O6c job ran on new problems (dev smoke runs excepted; see the end of this section).

Concerns found while drafting (the O6 session's) and in an independent check before approval:
- **The cutoff is on the copy's position; [JW0] bins by the text's length.** In F1's template the copy's first token
  sits 34-40 tokens before the end of the text (33 plus the answer's bytes), so [JW0]'s 700-token boundary of text
  length falls at token 661-665, and token 700 is about 735 tokens of text. Binned by position, [JW0]'s own rows put
  the boundary earlier, at 612-627 (The rule and its rival). The first draft (99f2550) fixed the cutoff at token 700;
  by position, dev predicted that cutoff to lose to L8 by about +3.8 points from its placement alone, and J7
  (approximated) by +1.8, against -2.0 (in-sample) and -1.3 at 620. The cutoff was moved to 620 after those J7-based
  predictions had been seen (Disclosure, under Predictions).
- **What a tie and an "L8 better" can show.** The tie reading says the cutoff "is not classed as differing from L8";
  like O6b's tie, a tie is not an equivalence test, and a 600-text row can tie up to about 2.5 points apart (with about
  65 discordant texts; with the 20-22 predicted here, under 2). "L8 better" is against this one cutoff only: another
  cutoff or byte rule might not be classed as differing from L8.

**Filing.** The O6 session makes the filing commit, and only after the main session relays the PI's approval in the
PI's words.
- The filing commit changes only this heading line and the Status lines, which then say filed and approved, with the
  date. In it this heading becomes `## O6c confirmatory rule, filed (YYYY-MM-DD)`, a line of its own, and the PI's
  words are quoted under it.
- Before the first job, the main session checks that `git diff <approved commit> <filing commit> --
  followup/O6c_CUTOFF_RULES.md` shows nothing else.
- After filing, this section changes only by dated notes appended at its end, apart from a re-filing commit's changes
  (below). None changes the rival, the cutoff, the texts, the class rule, a prediction or a reading, and results are
  read under the filed text. Every output records this file's sha256, and `verdict` prints its changes since the
  filing commit (`git diff --stat`).
- A re-filing takes two commits, as O6b's does: the fix, in a commit of its own; then a re-filing commit that changes
  only the Code section's pin line, edited in place to name the fix commit, the Status lines, and one new line
  `## O6c confirmatory rule, filed (YYYY-MM-DD)` appended at the end of this section with the disclosure and the PI's
  words quoted under it. Before the first rerun job, the main session checks that `git diff <previous filing commit>
  <re-filing commit> -- followup/O6c_CUTOFF_RULES.md` shows only those and the dated notes appended since. Every
  re-filing needs the PI's approval.

**What the code checks** (filed_check, provenance and gate in realblt_copy_latch_cutoff.py, as O6b's guard). Every
command that reads or scores the new problems refuses unless O6C_FILED names a commit that:
- is an ancestor of HEAD and has one parent;
- introduced a line `## O6c confirmatory rule, filed (YYYY-MM-DD)`, outside code fences and HTML comments, dated no
  earlier than 2026-10-31, that its parent's O6c rules file lacks;
- changes no covered code file and no submodule against its parent;
- holds the covered code unchanged from the code commit its Code section pins, with realblt_copy_latch_fresh.py
  unchanged from O6b's pin, 4026bec; and the working tree's covered code is unchanged since. The covered code is the
  O6c script, the O6b script and every repository module they import.

Every output records the filing commit, HEAD, each code file's sha256, the submodule's commit, this file's sha256,
the threshold mode and the model record (snapshot, weights' blob, library versions, modeling_blt.py's sha256, device).
The script stops before reading anything under python -O, a SEGR_BLT_WINDOW other than 512 or a SEGR_BLT_THRESH other
than train (the score job switches the windows off itself; the last two stop the import with an AssertionError). The job
gate: each job runs once per filing (an output committed and pushed still counts after rm, a reset or `git rm`); every
output that exists is committed, unmodified and pushed, and HEAD is not behind its upstream, before the next job starts;
the score job needs the block file and `verdict` the score output, both under the filing. The gate cannot see an output
deleted before it was committed, or a commit removed before or after it was pushed; each output is therefore committed
and pushed as soon as its job ends, and none is deleted, reset away or force-pushed away. This is kept by procedure.

### The question

Under the conversion as released (window 0), O6b's released paired verdict was "L8 better": D.L8 - D.SIVPO = +6.7
[+4.0, +9.5] points of exact copies on 600 fresh F1 texts. Its reading added: "A byte rule with a length cutoff might
match it; that is untested." O6c tests one such rule, its cutoff placed on the token position.

### The rule and its rival (frozen)

- **L8**, exactly as in O6b (followup/O6b_COPY_LATCH_RULES.md, section 1 and the Implementation section).
- **SIV+PO**, the bytes designer's design 2 (siv(kmin = 5) OR po(), from
  analysis/2026-10-30_copyfix/bytes/final_rules.py), unchanged, as in O6b.
- **SIV+PO<620, the rival.** SIV+PO applied only at token positions t < 620, where token t is byte t-1 and token 0
  is BOS. It masks no patch start at t >= 620. It reads no model output. Its decision at token t reads bytes up to t-2,
  as SIV's does, with one exception: PO's source-token test (analysis/2026-10-30_copyfix/bytes/final_rules.py:68-69) can
  also read bytes t-1 and t when the token it matches reaches byte t-2. That changes no decision on the 4,200 texts of
  J1's stored patcher outputs (O6b's six templates) or on the 293 dev texts, and it cannot arise at a copy's tokens in
  F1's template, where the token PO matches is the source ticket, which ends long before the copy
  (analysis/2026-11-01_o6c_check/DIFF_CHECK.md, MF2). A copy that straddles token 620 is masked only at its tokens
  before 620, and the by-length table counts it under "before 620".
- **The cutoff, 620, is fixed from dev only, by the copy's position.** [JW0]
  (analysis/2026-10-30_copyfix/judge_cost/out_w0check.txt) bins the released conversion's dev texts
  (results/blt_copy_gate_w0.json; the 285 of its 293 rows whose copy is within 512 bytes of its source) by text
  length: masking every copy start helps under 700 tokens (0-600: D 0.139 exact against D-all 0.912; 600-700: 0.227
  against 0.341) and destroys copies at 700 tokens and over (0.553 against 0.000). The rule acts on position, so all
  293 rows were binned by the copy's first token c1, rebuilt from F1's template (every row's token count and distance
  match; analysis/2026-11-01_o6c_check/d_synthesis/dev_positions.log; the prediction log repeats the numbers below but
  for the bootstrap). Masking every copy start keeps 182 of 200 copies against D's 27 at c1 < 575, 10 of 24 against 3 at
  575-626, and 0 of 69 against 38 from 627 on. Scored as D-all where c1 < c and D elsewhere, a cutoff c keeps the most
  dev copies, 230 of 293, for every c from 612 to 627; 620 is the middle (over 4,000 bootstrap resamples of the texts,
  the middle of that range lies in 589-627, 95%). The PI asked for this test on 2026-10-31 (local time), after a byte rule that switches off past about 700 tokens, a text length, had
  been described; the option that first proposed the rival (2026-10-30, not chosen then) said "the cutoff
  comes from dev data". The PI chose 620 from the independent check's recommendation (2026-11-01). No other
  cutoff is tried.

### Texts (C0, before any model run)

- **Pool.** GSM8K train problems from idx 6253 to the end of ~/.cache/segresearch-gsm8k/train.jsonl (idx 7472; 7,473
  lines with sha256 17f347dc..., O6b's J0 record), in file order: 1,220 problems. None of them has been kept for
  scoring or few-shot use in this project, or for fitting anything this test uses: O6b's J0 stopped at idx 6252, and
  every other reader that keeps problems for those uses stops at idx 1799 or below (O6b's Data paragraph, "Who has
  touched the pool before this test"). Other reads did reach them: [S3] (analysis/2026-10-30_copyfix/synth/s3_fresh.py)
  built O6's template on every problem from idx 1800 on and printed counts; whole-file reads format or tabulate every
  problem, and the novelty table is fitted on all 7,473 training solutions, these included
  (paper2v2/drafts/S6_copy_starts.md, notes for the merge); and the LoRA fine-tuning runs of BLT-1B trained adapters on
  problems drawn at random from idx 300 on, about half of these among them (554 of the 1,220 when the draws are replayed
  from their indices, as O6b's Data paragraph replays them; a toy dry run of realblt_fullft.py may also have drawn a
  few). None of these selects a text here, and this test loads the base weights, never those adapters.
- **Template.** O6's F1 template, built by the filed builder (realblt_copy_latch_fresh.make, which is
  realblt_quadrant.build): the question, "Ticket NNNN. " before the solution, and the copy line "\nTicket NNNN closed."
  before "\nThe final answer is". Tickets are 4 digits, 1000-9999, drawn with seed 6301 in problem order and redrawn if
  they occur in the problem.
- **Filter.** O6's: at most 1,000 tokens, one token per byte plus BOS. Rejected problems are skipped and keep their
  draws.
- **Split.** The first 300 kept are the fit texts; the next, up to 600, are the test texts. O6b's F1 kept 900 of 922,
  so about 1,190 should be kept: 300 fit and 600 test. If the pool runs out before 300 fit texts, nothing runs. With
  fewer than 400 test texts the test is underpowered: the numbers are reported with no class.
- C0 writes results/blt_copy_latch_cutoff_blocks.json: every text's index, ticket, distance, byte count, token count
  and sha256, the visited range, and the pool file's sha256 and line count. It is committed and pushed before the score
  job, which rebuilds the texts and checks them against it.

### Scoring (C1)

- **As O6b's F5:** the conversion as released (window 0), at D only, teacher-forced. The patcher outputs, L8, the
  masks and the thresholds are computed under that model.
- **Before scoring, the implementation checks** of O6b's J7, on every fit and test text: causality of L8 and L8c,
  alignment, and the subset asserts. A failure is a bug, not a result: C1 writes its checks and stops before any
  scoring. There is no redo unless the PI approves a re-filing, and a second failure stops the test. A failed subset
  assert ends C1 with no output; by procedure it counts as a failed check.
- **Which result counts.** After a failed implementation check (C1 stops before scoring), the re-filed run, if the PI
  approves one, is the result. Once C1 has written an output that passed its checks, a bug found later does not
  replace it: the filed outputs remain the confirmatory result, and a corrected rerun under a re-filing the PI
  approves is reported beside it as a deviation, never in its place. `verdict` reads only the current filing's
  outputs, so this is kept by procedure.
- **Layouts:** D; D-all (no start at any byte of the copied string, no refill); D.L8; D.SIVPO; D.SIVPO<620 (D.SIVPO620
  in the code and its outputs, threshold kind entsivpo620). Each X.R is the entropy with rule R's tokens set to -inf,
  its threshold refitted on the 300 fit texts (blt_layouts.Fit, train mode) at D's pooled train rate there, as in O6b.
- **Readouts and pairs:** O6 step 2's paired bootstrap over problems (realblt_copy_veto.boot_ratio, 2,000 resamples,
  seed 6); every interval is the 2.5th-97.5th percentile.
- **Jobs and Mac time.** C0 (blocks: no model, under a minute); C1 (score: the patcher on about 900 texts, the
  checks, then 5 layouts on up to 600, about 20-30 min; O6b's J7 took 18 min with 4 layouts); `verdict` (seconds).
  One BLT-1B job at a time (`pgrep -fl realblt`), each output committed and pushed before the next job, O6C_FILED set
  for every job. The score job's one-job guard (realblt_copy_latch.alone) refuses to start while any process other than
  the job and its parents has "realblt" in its command line. caffeinate is never given the job's command: it then runs
  as the job's child and names the script, and the guard refuses the job (O6b's J1 first start; the first attempt of
  this rule's 2026-11-01 smoke rerun). It is started first, in the shell that then starts the job, waiting on that
  shell: `caffeinate -i -w $$ &`. A start stopped before it reads any new problem, by that guard, by the filing check,
  the model record (BLT-1B must load offline from snapshot 91aa6b8e) or the job gate (a failed `git fetch` included), or
  by the environment checks (python -O, a SEGR_BLT_WINDOW other than 512, a SEGR_BLT_THRESH other than train; the last
  two stop the import with an AssertionError), writes no output and is not a run, a crash or a failed check: its message
  is recorded in a dated note, and the job starts again once the cause is removed. A job that crashes before writing its
  output reruns once under the same filing; a second crash is a bug. Cost: $0.

### The verdict

- **Primary row:** D.L8 - D.SIVPO<620 on id_copy exact (every byte of the copied ticket right under teacher-forced
  argmax), paired over problems; positive means L8 is better.
- **Class** (O6b's class rule): "L8 better" if the point estimate is at least +2 points and the 95% interval lies
  above 0; "the cutoff better" if it is at most -2 points and the interval lies below 0; otherwise "tie". An interval
  end of exactly 0 is a tie, as O6b's "-3 [-7, 0]" is. With fewer than 400 test texts no row has a class,
  D.L8 - D.SIVPO beside it included.
- **Readings**, quoted whole wherever the result is reported:
  - tie: "Under the released conversion, a byte rule that stops at token 620, a position fixed from dev texts, is not
    classed as differing from L8 (a class needs a difference of at least 2 points with its 95% interval on that side
    of 0; this is not an equivalence test). L8's model-reading gate shows no advantage over a fixed position cutoff on
    this template. With O6b's windowed result, no setting tested shows that reading the model is needed."
  - L8 better: "L8's gate helps beyond a byte rule that stops at token 620, a position fixed from dev texts, under the
    released conversion, on this template; reading the model earns something there."
  - the cutoff better: "A byte rule that stops at token 620, a position fixed from dev texts, beats L8 under the
    released conversion, on this template."
  - fewer than 400 test texts: "Fewer than 400 test texts (<n>): the numbers are reported with no class."
  - a failed check: "The implementation failed under window 0; nothing was tested."
- **Reported beside the class**, without a verdict:
  - D.L8 - D.SIVPO, which repeats O6b's released row (+6.7 there), with its class under O6b's rule;
  - D-all - D, and D.SIVPO<620 - D, on id_copy exact;
  - the by-length table: exact copies per layout and L8's firing at c1, by text length (tokens 0-600, 600-700,
    700-2000, O6b's F5 bins) and by the copy's first token (before 620, or 620 and later);
  - the discordant split of the primary row: of the copies L8 lost to the cutoff, those where L8 did not fire at c1
    (copies L8 missed) and those where it fired (side effects of the refit), with the counts crossed by D.SIVPO<620's
    start at c1 and the texts' problem indices; the same for D.L8 against D.SIVPO.
- `verdict` prints "O6c result (filing <commit>)", the reading, the primary row with its numbers and class, and the
  rows beside it.

### What it cannot show

- Another cutoff or another byte rule: only 620, and only SIV+PO, are tested.
- Equivalence: a tie is not an equivalence test.
- Other templates, other models or the official BLT weights (only BLT-1B's base weights, itazap/blt-1b-hf at snapshot
  91aa6b8e, as released, window 0), free generation, natural text, or why the patcher's argmax at the copy fails in
  long texts.

### Predictions (dev and O6b's J7; exploratory)

**Disclosure.** O6b's J7 numbers (results/blt_copy_latch_fresh_j7.json, read with the verdict at af15140) were read
before this pre-registration was written. The first draft (99f2550) fixed the cutoff at token 700, from [JW0]'s
text-length bins and from the main session's design brief, which wrongly said that the option the PI chose had said
"~700 tokens" (the PI chose no such option; "~700 tokens" was the main session's description of a text length;
analysis/2026-11-01_o6c_check/QUOTES.md). An independent check before approval (2026-11-01) binned [JW0]'s dev rows
by the copy's position, which puts the boundary at 612-627, and computed J7-based predictions for cutoffs 600-720
(D.L8 - cutoff, approximated: -1.3 points at 620, +0.7 at 665, +1.8 at 700). The PI saw a summary of these
predictions and then set the cutoff at 620, the middle of the dev range (2026-11-01); the number comes from
the dev rows only. The predictions below use dev and J7, and copy_latch_cutoff_predict.py
(results/blt_copy_latch_cutoff_predict.log) computes both from stored outputs and dev problems only: J7's 600 rows,
each copy's first token from J1's patcher outputs (sha256 checked against J1's record), and [JW0]'s dev rows, whose
copies' positions it rebuilds from GSM8K test problems 0-299; no model and no train row.

- **Dev ([JW0]'s 293 rows, by the copy's position).** [JW0]'s argmax canary masks the copy's starts only where the
  patcher's argmax at c1 is right, and so stands in for L8; it keeps 224 of the 293 dev copies (0.765). The cutoff,
  scored as D-all where c1 < 620 and D elsewhere, keeps 230 (0.785): L8 - cutoff about -2.0 points (the stand-in lost
  8 copies the cutoff kept and kept 2 the cutoff lost; these rows also placed the cutoff, so at 620 this line is
  in-sample). The same calculation gives +0.7 at token 665 (where [JW0]'s 700-token text length falls) and +3.8 at token
  700.
- **J7, per text.** The cutoff's outcome is approximated from J7's own layouts: where the copy's first token c1 is
  before 620, SIV+PO<620 masks the copy as SIV+PO does, so it takes D.SIVPO's outcome; where c1 is 620 or later it
  masks nothing there, so it takes D.L8's outcome when L8 did not fire at c1, and D's when it did. The refitted
  threshold's small shift is ignored.
  - In F1's template the copy's first token sits 34-40 tokens before the end of the text. 111 of J7's 600 copies
    start at token 620 or later; 1 straddles 620 and takes D.SIVPO's outcome.
  - Exact copies, J7: D 0.195, D-all 0.693, D.L8 0.763, D.SIVPO 0.697, the cutoff about 0.777.
  - By length (n; D, D-all, D.L8, D.SIVPO, the cutoff): tokens 0-600 (448; 0.138, 0.888, 0.879, 0.879, 0.879); 600-700
    (73; 0.247, 0.247, 0.384, 0.329, 0.493); 700 and more (79; 0.468, 0.000, 0.456, 0.000, 0.456). L8 fired at c1 on
    0.982, 0.205 and 0.013 of the copies there.
  - D.L8 - D.SIVPO<620, approximated: -1.3 points (L8 lost 15 copies the cutoff kept, 12 of them where L8 did not
    fire at c1, and kept 7 the cutoff lost). Below token 575 the two tie (6 and 6); the rest comes from the 34 copies
    at 575-619, where L8 fires on 9 and keeps 0.265 of them, against SIV+PO's 0.500 (9 lost, 1 kept).
- **Primary class on 600 new texts** (exact over the (lost, kept) counts, each count's class one seeded bootstrap
  draw): with J7 as truth, tie 0.78, the cutoff better 0.22, L8 better 0.00; with J7's uncertainty (Jeffreys), tie
  0.71, the cutoff better 0.29, L8 better 0.002. If the cutoff loses 2, 5 or 10 more copies per 600 than the
  approximation says, P(the cutoff better) is 0.13, 0.05 or 0.01 and P(L8 better) 0.00, 0.00 or 0.04. Predicted:
  "tie", with "the cutoff better" about 1 in 4; "L8 better" near 0. Dev alone (by position; in-sample at 620) predicts
  "tie" or "the cutoff better", about evenly (tie 0.44-0.49, the cutoff better 0.51-0.56). At 700 the same calculations
  gave "L8 better" 0.40-0.42 (J7) and 0.74-0.87 (dev).
- **Beside the class:** D.L8 - D.SIVPO about +6.7 (+3.5 to +10), L8 better (J7: +6.7 [+4.0, +9.5]); D-all - D about
  +50 (+45 to +55) (J7: +49.8); the by-length table near J7's above; the discordant split near 15 lost, 12 of them
  where L8 did not fire at c1.
- **Texts:** about 1,190 kept of 1,220 (O6b's F1: 900 of 922), so 600 test texts; P(fewer than 400) about 0.

### Code

- realblt_copy_latch_cutoff.py, at commit 3875d32. It imports realblt_copy_latch_fresh.py, at commit 4026bec
  (O6b's pin), unchanged: F1's builder and O6's filter, the masks and their subset asserts, the implementation checks,
  the matched-rate fit, the layouts, the scorer, the readouts, the bootstrap and the class rule. It adds only:
  - the cutoff mask (cut_mask) and the D.SIVPO<620 layout;
  - the builder for idx 6253 on (take_texts, pool_rows) and the block file;
  - the score job (cmd_score: window 0, the patcher, the checks, the five layouts, the by-length table and the
    discordant split);
  - O6c's verdict (verdict_from, READINGS, cmd_verdict);
  - the guard and the job gate (filed_headings, filed_check, provenance, gate, failures), as O6b's.
- **Self-test, on bytes and dev problems only** (results/blt_copy_latch_cutoff_selftest.json; no model):
  - the builder on dev problems is deterministic and gives F1's template with the ticket twice; it records where the
    copy's first token sits (34-39 tokens before the end on 250 dev texts, 76 of them longer than 620 tokens);
  - the cutoff mask equals SIV+PO below token 620 and masks nothing from 620 on, on dev texts and on a joined text
    longer than 620 tokens;
  - on 70 prefixes of 21 texts, a prefix of p bytes gives the full text's mask at tokens 0..p+1, as it must if token t
    reads bytes up to t-2 only; none of the 70 has a veto at token p+1, so the check never meets a later byte that
    removes a veto, as PO's can (The rule and its rival);
  - with stand-in patcher outputs, D.SIVPO<620 equals D.SIVPO where no text reaches token 620, and starts patches
    from token 620 on where SIV+PO's mask holds;
  - the guard refuses here (an empty name, HEAD, a tree, the index, an unknown name), this file has no filing heading
    yet, and in a scratch repository the guard accepts the filing commit and refuses: no heading, O6b's heading, a
    heading dated before 2026-10-31, a filing commit that also edits code, code changed after filing, the O6b code
    changed from its pin, and a pin that is not its code.
- **Branch test** (copy_latch_cutoff_test.py; results/blt_copy_latch_cutoff_verdict_test.log; 14 hand-set cases):
  each class of the primary row and its reading; a point estimate at +2 and at -2 points with and without 0 in its
  interval, and interval ends of exactly 0; 399 and 400 test texts, with the class of the row beside it at 399; the
  rows beside the class by name and value, and the "Beside it" sentence; the discordant split's sentence; the
  two-decimal print; a failed check and a missing score job.
- **Predictions** (copy_latch_cutoff_predict.py; results/blt_copy_latch_cutoff_predict.log): above.

### Change log (2026-10-31)

Amendments before filing are listed here, dated.

**2026-11-01, before approval: the independent check and the PI's cutoff decision** (analysis/2026-11-01_o6c_check/:
VERDICT.md, CHECKS.md and QUOTES.md at 85c8141; DECISION.md at 065dc92). The code is at 3875d32 (pinned above); its
self-test, branch test and predictions at 8c5fdf0; the dev smoke rerun at 3cf51f4 (note below). Nothing was filed, no
problem from idx 6253 on was read, and BLT-1B ran on dev texts only.
- **The cutoff: token 620, not 700** (the PI's decision, 2026-11-01; VERDICT.md section 4, lists A and B).
  The rival is renamed SIV+PO<620 (it was named SIV+PO<=700 but defined as t < 700; B-F11), and its layout
  D.SIVPO<620. Changed with it: the title; the Concerns lead line; the concern bullet on position and length; the rival
  and the basis paragraph; the primary row; the readings; the rows beside the class; What it cannot show; the
  Disclosure; the dev line, the whole J7 part, the primary class's odds and the discordant split beside it in the
  Predictions, recomputed by copy_latch_cutoff_predict.py at 3875d32; the Code section. The text-length bins 0-600,
  600-700 and 700-2000 stay (O6b's F5 bins). DECISION.md's three replacements stand in for list A's texts where they
  differ, and correct the misquoted "~700 tokens" (QUOTES.md): the basis paragraph's last two sentences (with the
  request's time given in UTC beside its local date, 2026-10-31, which the opening paragraph uses), and the Disclosure's
  second and fourth sentences.
- **M1** (code): the row beside the class is reported under its own name, D.SIVPO<620 - D (keyed "D.SIVPO620 - D" in the
  verdict output); it was reported under "D.SIVPO700 - D.SIVPO", a different quantity. The branch test checks its name
  and value.
- **M2**: the Pool paragraph says which reads reached the reserved problems: none kept one for scoring or few-shot use,
  and the LoRA runs that trained adapters on about half of them are never loaded here.
- **M3**: O6c's result is planned for paper 2 v2's Section 6 (it said nothing here goes into paper2v2); the same in the
  code's docstring.
- **M4**: the dev prediction is computed by the copy's position, which the rule acts on, not by text length; at 620,
  dev alone predicts "tie" or "the cutoff better" (it said "tie").
- **M5**: the position facts in the concern bullet, the basis paragraph and the Disclosure (list A).
- **S1** (code): `verdict` prints its output whole, keeps D.L8 vs D.SIVPO's discordant record beside the class, and
  gives D.L8 - D.SIVPO's class and D.SIVPO<620 - D in its "Beside it" sentence.
- **S2** (text and code): with fewer than 400 test texts no row has a class, D.L8 - D.SIVPO included.
- **S3**: an interval end of exactly 0 is a tie.
- **S4**: Which result counts after a re-filing (under Scoring).
- **S5** (branch test): -2 points with and without 0 in the interval, interval ends of exactly 0, the rows beside by
  name and value, and the class of the row beside at 399 texts.
- **S6** (code): the verdict prints points with two decimals, and an interval end of exactly 0 prints 0.00.
- **A14**: the tie reading says the byte rule "is not classed as differing from L8" and that this is not an
  equivalence test (it said the rule "matches L8 within the class rule"); the concern bullet is worded the same way.
- **B-F12**: a 600-text row can tie up to about 2.5 points apart (it said about 2).
- **A10**: a copy that straddles the cutoff is masked only before it and counted under "before 620".
- **C-F10**: the rows beside the class are named with their metric, id_copy exact.
- **A6** (code): the self-test's causality check compares a p-byte prefix's mask at tokens 0..p+1, since token t may
  read bytes up to t-2 (it compared tokens 0..p, and the text said "every prefix").
- **A7** (code): take_texts stops before the pool yields another row (it parsed one row past the last it used). The
  texts are unchanged: the smoke rerun's block equals the first smoke run's.
- Also changed to match: the opening paragraph says the cutoff is the PI's; the question says the cutoff is on the
  token position; the basis paragraph names [JW0]'s 285 texts and the log of the position bins; the Disclosure says
  the prediction script also rebuilds the dev line; the Status; the Code section's pin, moved from ae0266a to 3875d32,
  the self-test's numbers, from its new JSON, and the branch test's cases (S5).
- The check's other style items (A11, A12, A13, B-F9, C-F8) are left as they are.

**2026-11-01, after the diff check of these amendments** (analysis/2026-11-01_o6c_check/DIFF_CHECK.md). Text only: the
code stays as pinned above, and its outputs at 8c5fdf0 and 3cf51f4 stand.
- **Pool**: the LoRA runs trained adapters on about half of the reserved problems (the paragraph said none was kept for
  fitting and listed those runs among reads that keep nothing); the M2 line above says the same.
- **Causality**: the rival and the self-test say that PO's source-token test can read bytes t-1 and t, which changes no
  decision on the texts checked (they said the cutoff rule is causal).
- **SF1**: Jobs and Mac time names the score job's one-job guard, says how caffeinate is started so that the guard does
  not refuse the job, and says that a start stopped before it reads any new problem by that guard, the filing check, the
  model record, the job gate or the environment checks writes no output and is not a run, a crash or a failed check, and
  that its message is recorded in a dated note (it covered crashes only).
- **SF2**: the line on the cutoff above names three more paragraphs it changed: the Concerns lead line, and the primary
  class's odds and the discordant split beside it in the Predictions.
- **ST1**: the layouts and the M1 line above give the names that the code and its outputs use (D.SIVPO620, entsivpo620,
  "D.SIVPO620 - D").
- **ST2**: the concern bullet on what a tie can show quotes the tie reading's words, as the A14 line above says.
- **ST3**: the opening says the O6 session drafted "this text" (it said "it", which could be read as the independent
  check).
- **ST4**: the basis paragraph says which of its numbers the prediction log repeats (it said the log repeats
  dev_positions.log but for the bootstrap).
- **ST5**: the concern bullet, the dev line and the dev odds say that dev's numbers at 620 are in-sample: the same 293
  rows placed the cutoff.
- **ST6**: What the code checks names three settings that stop the script before it reads anything: python -O,
  SEGR_BLT_WINDOW and SEGR_BLT_THRESH (it named python -O and the threshold mode).
- **ST7**: D.L8 - D.SIVPO beside the class is said to repeat O6b's released row, +6.7 there (it said the row replicates
  O6b's +6.7).
- **ST8**: the S6 line above says it is the verdict that prints two decimals (the score job's console line prints one).
- **ST10**: added: the number of discordant texts behind the 2.5-point tie, and the widest tie with the 20-22 predicted
  here (concern bullet); that the first smoke run had classed D.L8 - D.SIVPO "tie" (smoke rerun note); and the branch
  test's cases, in the "Also changed to match" line above.

**2026-11-01, after the check of 5e73f1d** (analysis/2026-11-01_o6c_check/DIFF_CHECK.md, section 16). Text only:
- **Refused starts**: Jobs and Mac time also lists the model record's refusal (BLT-1B must load offline from snapshot
  91aa6b8e), which comes before any new problem is read; the SF1 line above now names the causes the paragraph lists,
  and the ST6 line says "three settings" (it said "every setting").

### Smoke run on dev problems (dated note, 2026-10-31; a code check, not a result)

- **When.** 2026-11-01 01:19:52-01:20:58 UTC (2026-10-31, -0300), after this rule and its predictions were committed
  (99f2550), on the code at ae0266a.
- **What.** `O6C_SMOKE=1 realblt_copy_latch_cutoff.py` ran blocks, score and verdict on dev problems, GSM8K train idx
  0-20: 10 fit and 10 test F1 texts, scored under the released conversion (window 0). BLT-1B ran on these dev texts
  only; no new problem was read.
- **Result.** Every command exited 0. The implementation checks passed on all 20 texts (0 changed decisions, 0
  misaligned positions). The five layouts, their thresholds (entsivpo700 refitted beside the others), the by-length
  table, both discordant records and the verdict were written. With 10 test texts the verdict has no class, as the
  rule says.
- **Outputs.** results/blt_copy_latch_cutoff_{blocks,score,verdict}_smoke.json are committed; the smoke patcher
  outputs are git-ignored. Their numbers rest on 10 test texts and are not read. Nothing in the rule was changed for
  them.

### Smoke rerun on dev problems (dated note, 2026-11-01; a code check, not a result)

- **When.** 2026-11-01 11:31:19-11:32:20 UTC, on the code at 3875d32 (HEAD 8c5fdf0), after the cutoff and the layout
  names changed and before the 2026-11-01 amendments to this text were committed.
- **What.** As in the note above: blocks, score and verdict with O6C_SMOKE=1 on dev problems, GSM8K train idx 0-20
  (10 fit and 10 test F1 texts), scored under the released conversion (window 0). BLT-1B ran on these dev texts only;
  no new problem was read.
- **Result.** Every command exited 0. The implementation checks passed on all 20 texts (0 changed decisions, 0
  misaligned positions). The five layouts, their thresholds (entsivpo620 refitted beside the others), the by-length
  table, both discordant records and the verdict were written; the row beside the class is named D.SIVPO620 - D, and
  with 10 test texts no row has a class (the first run had classed D.L8 - D.SIVPO beside it "tie"; S2). The block file's
  texts equal the first smoke run's.
- **A refused start.** A first attempt at 11:30:45 UTC ran blocks and then stopped at the score job's one-job guard
  (L.alone), before reading or scoring anything: caffeinate, started around the job to keep the Mac awake, named the
  script and is not one of its parents, as at O6b's J1 first start. The three commands were then rerun with caffeinate
  waiting on the shell instead.
- **Outputs.** results/blt_copy_latch_cutoff_{blocks,score,verdict}_smoke.json now hold this run (3cf51f4); the first
  run's stay in the history at 069c10b. The smoke patcher outputs are git-ignored. Their numbers rest on 10 test texts
  and are not read. Nothing in the rule was changed for them.
