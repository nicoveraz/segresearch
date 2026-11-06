> **Public copy of an internal check report** from the author's private working repository (`analysis/2026-10-31_o6b_result/INDEPENDENT_CHECK.md`,
> last changed at 0d36323). It records the check of O6b's verdict against two blind derivations of the rule. Paths refer to that repository: the files it names under `results/` and
> `followup/` are in this package's `results/` and `rules/`; the check's own scripts and outputs are not included.
> The changes made for this copy (this header, and phrases naming the private repository or a temporary folder) are
> listed in README.md.

# O6b confirmatory run: independent check of the verdict (main session, 2026-10-31)

The run finished under filing 7e9c6fa with no stop; the `verdict` output is at af15140
(results/blt_copy_latch_fresh_verdict.json), from the job outputs J1-J7 committed at 717ca5e, 8bd0b96, ceba4dc,
1f30951, fe67ebc, 609199d and b60f319.

**The two frozen blind derivations, run on the real job outputs.** d1
(analysis/2026-10-30_copyfix/fourth_check/derive/my_verdict3.py, sha256 01145fb4...) and d2
(analysis/2026-10-30_copyfix/fourth_check/derive_d2_interrupted_run/my_verdict3.py, sha256 331dda25...) were
written from the rule text before the code was read; both still match their recorded hashes. On
results/blt_copy_latch_fresh_j1.json ... _j7.json they give the code's decisions exactly:

| | code (`verdict`) | d1 | d2 |
|---|---|---|---|
| stops | none | none | none |
| kills | none (K8 reported) | none (K8 "bound-limited", not a kill) | none (K8 "bound-limited") |
| fix claim | holds, both rates | holds, both rates | holds, both rates |
| generality | holds: 6-digit and mixed IDs | holds: digits6, mixed | holds: 6-digit, mixed |
| windowed paired verdict | SIV+PO better | SIV+PO better | SIV+PO better |
| released paired verdict | L8 better | L8 better | L8 better |

**Run record.** J1's first start was refused by the job guard within 6 s (the driver's `caffeinate` command line
named the script), before anything was read or written; the rule records the same guard behaviour from the smoke
run, and this was not a crash. No code or rule line changed during the run (the filing commit and job-output commits
only). Nothing went to #35, and paper2v2 was not touched.
