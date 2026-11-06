# Study A: interim look at seed 0, stop-only rule (amendment A-5; written 2026-10-28 before any Study A evaluation was opened)

**Why.** Study A (segresearch/prereg/2026-10-13_scale150_bltft.md, amendments A-3 and A-4: entropy10 vs nov10 at
145M, 3 seeds each, Kaggle T4 fp16) is paused on the weekly Kaggle quota. Seed 0 of both arms has finished and been
evaluated (the result files exist; no value has been read). Seed 1 is part-trained (nov10 72,664 / entropy10 60,132
of 91,500 updates) and seed 2 has not started. Left: about 5-6 Kaggle sessions (~60 GPU-hours, ~2 weekly quotas,
finishing mid-November). The PI asked whether that time is worth spending. The preregistration has no interim
analysis, so this one is added and disclosed.

**What is opened.** Seed 0 of entropy10 and nov10 only (result.json). Seed 1's paused state and logs stay unopened
beyond training loss and progress, as before.

**The look can only stop the study; it never supports a claim.** The A-3 decision rule (Welch 95% over 3 seeds
against the stated margins), the arms, the metrics and the margins are unchanged. Stopping for futility cannot make
a HOLDS verdict more likely, so it does not inflate false positives.

**Stop rules** (n = the evaluation's target counts: 660 final answers, 1,000 computed results):
- **S1, broken training.** Either seed-0 run has bits per byte >= 1.45 (no better than every 50M run, 1.414-1.451)
  or any non-finite metric. Action: stop the chain and diagnose before any further quota is spent.
- **S2, the main question is already settled against the claim.** g = nov10 - entropy10, final answers, seed 0, in
  points. Optimistic bound U = g + 1.96 x 100 x sqrt(p1(1-p1)/660 + p2(1-p2)/660) + 3 (the 3 points allow for seed
  variation: at 50M, seed SD ~1 point per arm). If U < +10 (A1's margin), A1 cannot plausibly hold, nor A3 (needs a
  145M gap of at least the 50M gap minus 5, about +25.5). Action: stop; report seed 0 as exploratory (one seed,
  problem-level interval) with this amendment, and paper 2 drops the claim that the gap grows with size.
- **Otherwise continue unchanged.** The remaining seeds are what makes the result confirmatory. Seed 0's values are
  not reported or used in the manuscript until all three seeds are in.

**Reported regardless:** that this look happened, its date, and which rule (if any) fired.

## Outcome (2026-10-28, same day, after 23d20a2)

Seed 0 (145M, Kaggle T4, 91,500 updates each): nov10 final answers 91.2%, computed results 68.3%, bits per byte
1.813 (evaluation patch rate 0.119); entropy10 68.9%, 29.3%, 1.492 (0.112).
- **S1 FIRED:** both runs have bits per byte >= 1.45 (50M runs: 1.414-1.451).
- **S2 did not fire:** g = +22.3, U = +29.4 (> +10).
- **Action taken:** the daily Kaggle tick [redacted: task name] is disabled, so no quota is spent when it resets.
  Seed 1's paused checkpoints stay in the kernels' latest outputs, so the chain can resume unchanged.

Diagnosis so far (seed 0 only; seed 1 not opened):
- Training itself looks sound: the final training loss (nov10 1.227, entropy10 1.243) is below the 50M runs'
  (1.318 / 1.299) and falls smoothly across every pause and resume; answers are better than at 50M in both arms.
- The regression is on the ~98% of evaluation bytes that are not answer targets: derived bits per byte there
  1.824 (nov10) / 1.493 (entropy10) against 1.456 / 1.414 at 50M; the answer bytes improved.
- Not the T4 or fp16 path alone: paper 2's 12M nov10 runs trained on the same Kaggle T4s with normal bits per byte
  (1.653-1.707).
- 145M is the first size to see the corpus more than once: 3.0 GB of samples from a 1.9 GB corpus (1.58 epochs;
  50M 0.79, 12M 0.26). A candidate cause, not tested.

## Diagnosis, continued (2026-10-28)

Deviation, disclosed: this part opened seed 1's paused checkpoints (bits per byte only, on three texts), which the
rule above kept closed. Nothing from it is used for any verdict.
- **entropy10 seed 1 diverged.** Training loss 1.374 at update 32,500, NaN from 33,000 on; the fp16 loss scale fell to
  0, and the chain kept saving the damaged model until update 60,132 (about 27,000 wasted updates, one session). That
  checkpoint scores ~5.2 bits per byte on every text (near random). The tick did not see it: it counts updates. The
  last good checkpoint (update 29,279) is in kernel B's latest output (copied to the Mac). Seed 0 (both arms) and
  nov10 seed 1 logged no non-finite loss in any session.
- **Memorization of the repeated GSM8K training problems.** The corpus repeats GSM8K's training solutions (9.7 MB) 20
  times to make its 10% math share, so each problem is seen ~5 times at 12M, ~16 at 50M and ~32 at 145M. nov10 seed 1
  at update 72,664 (~25 exposures), bits per byte: **0.844 on training problems, 1.566 on new problems** from the same
  source (the evaluation text), 1.729 on corpus web text (paper2v2/diag_repetition.py,
  results/studyA_diag_repetition.json). This accounts for seed 0's worse bits per byte at 145M without broken
  training; final answers and computed results are scored on new problems and improved.
- So S1's premise (broken training) holds for one run, entropy10 seed 1, and not for seed 0.

## Amendment A-6 (written 2026-10-28; CONFIRMED by the PI the same day)

- **Code** (segresearch 962400b, local): training stops when the loss stays non-finite for 50 updates, the fp16 loss
  scale falls below 1, or a session ends on a non-finite loss, and never saves while the loss is non-finite; the
  Kaggle job restarts such a run once in the session from its last good checkpoint with 2,000 batches skipped; the
  daily tick reports DIVERGED and stops on a second divergence. Runs that never diverge train bit-identically
  (checked on CPU against the previous code). Job and tick logic checked with mock sessions.
- **entropy10 seed 1** restarts from update 29,279 with 2,000 batches skipped; its 27,000 damaged updates are
  discarded. All other runs continue unchanged.
- **Analysis unchanged** (A-3 rule and margins). A4 (bits per byte) is reported with the measured seen/new gap, and
  its reading notes the memorization of the repeated GSM8K problems, which grows with size in this corpus design.
  The corpus is not changed mid-study (that would break comparability with seed 0 and with 1M-50M).
- **Scheduling** (segresearch e60510d, local): instead of fixed pairs, each session gets every unfinished run in the
  order nov10 s1, entropy10 s1, entropy10 s2, nov10 s2, trains two at a time and, when a run ends with more than an
  hour left, starts the next on the freed GPU; runs not reached carry their state forward. This changes only when
  runs train, not how.
- **Rollback in place:** the checkpoint at update 29,279 (healthy: finite weights, loss scale 131,072) is in the
  private Kaggle dataset [redacted: dataset name], attached once to the next session; the queue records the
  rollback (backup of the previous queue: queue.json.bak_20261028_preA6). Code uploaded to the bundle dataset; the
  daily tick is re-enabled and resumes after the weekly quota resets (~October 31).
- **Remaining cost:** about 5 Kaggle sessions (~55 GPU-hours, ~2 weekly quotas), finishing around mid-to-late
  November (7 with the old fixed pairs).

## Bookkeeping: final models kept (2026-10-29, before the next session; no Study A result opened)

- **Why.** A-6 reports A4 with the measured seen/new gap, which needs each run's final model. The Kaggle job never
  kept a finished run's final model on purpose:
  - It copied a run's checkpoint into its output when the run paused or diverged.
  - Since 2026-10-22 (segresearch 236d6dc) it also copied one once an hour while the run's process was alive (in
    236d6dc, for every run of the session until its last run ended).
  - The final checkpoint stayed in the session's work folder, which is discarded. So a finished run's output held
    only its last hourly copy: the final model if a copy fell after the last update, otherwise an earlier one.

  Found by the project review of 2026-10-29 (ANALYSIS_2026-10-29.md, TR-3).
- **Seed 0.** Only the update counts in the checkpoint sidecars were read (`ckpt.pt.json`, which holds k and steps;
  read 2026-10-29, nothing else opened). They show:
  - nov10's final checkpoint (update 91,500 of 91,500) in the outputs the tick downloaded on 2026-10-24 at 19:21
    (kernel B) and 21:43 (kernel A);
  - entropy10's copy in the second of these from update 89,205, 2,295 updates before its end.

  Those kernel versions are no longer the latest. The installed kaggle CLI (2.2.4) cannot download an older version's
  output (it ignores the version), so retrieving them needs Kaggle's web interface or a newer client. Not attempted.
- **Code** (segresearch aaef0b0, local; prereg note 504305f):
  - `kaggle_chain_job.py`: when a run finishes, its final model is saved in the session's output as `model.pt`.
    It holds the weights, the update count and the config, about 0.6 GB, extracted in a separate process from the
    run's checkpoint, and only if that checkpoint is from the last update. If the extraction fails, the full
    checkpoint is kept instead. A model already kept for a run is never replaced.
  - The session log records the update, the model's SHA-256 and the SHA-256 of the run's `result.json`. This pairs
    each model with its result without opening either.
  - Every session copies the `model.pt` files of runs finished earlier from the previous output into its own, so the
    latest output normally holds all of them.
  - `kaggle_chain_tick.py`: after its decision, the daily tick starts a background download into
    `~/.cache/segresearch-scale/kaggle_chain/models/` and prints a MODELS line. It fetches `model.pt` for runs
    finished since this change (seeds 1 and 2), plus the session's `rules.json`, with at most 4 attempts per run.
    While a re-run is pending, the runs of the re-run session are not fetched, since the re-run trains them again.
    The download never changes the queue or the decision.
- **Unchanged:**
  - training, evaluation, data order, the run queue and every decision rule;
  - the training and evaluation code in the bundle dataset (e60510d), which is not re-uploaded.

  One scheduling detail: the extraction runs in the job's main loop after the run's process has ended. The rule for
  starting the next run on the freed GPU still uses the run's end time, but the start itself waits for the extraction
  (seconds; it is stopped after at most 10 minutes). As with A-6's scheduling, this changes when a run trains, not how.

  Checked with mock sessions against the previous code:
  - the job's outputs are identical apart from the model files;
  - the tick's queue, STATUS line and Kaggle calls are identical apart from the download.
- **Use.** The kept models are copied to the Mac as their runs finish, but none is loaded or evaluated before all six
  runs have finished. They serve only A-6's seen/new reading in the final analysis.
- **Not covered.** A final model can still be lost:
  - if the session is stopped in the seconds between the run's end and the model's save;
  - if the kernel holding the newest models is pushed again by hand before the next session copied them and before
    the tick fetched them;
  - if a copy into a later session fails (logged) and the older output is replaced before the fetch.

  If the tick re-runs a session, runs that finished in it are trained again from the previous checkpoint, as before
  this change. The analysis then uses the last training's result and its model, paired by the hashes in the session
  log.
