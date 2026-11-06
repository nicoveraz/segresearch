> **Public copy of an internal check report** from the author's private working repository (`analysis/2026-10-31_p2copy/table_check/CHECK.md`,
> last changed at 5376f76). It describes results/blt_table_copy_onsets.json (exploratory, after the data). Paths refer to that repository: the files it names under `results/` and
> `followup/` are in this package's `results/` and `rules/`; the check's own scripts and outputs are not included.
> The changes made for this copy (this header, and phrases naming the private repository or a temporary folder) are
> listed in README.md.

# Novelty table at copy onsets: bytes-only check (2026-10-31)

Exploratory, after the data; dev texts only (O6's 293 F1 test texts, GSM8K test idx 0-299, and 295 fit texts, train idx 0-299). No model was run. Written by a workflow agent of the main session; the text below is its report, with file paths changed to the committed copies.

Files: `measure_table_copy.py` (this folder, unchanged; sha256 75853510158ec433ad1d85798e9a80dba31f4f5f05b601a32ab322447087ef09). It writes `results_table_copy.json` and `.log` next to itself; the committed outputs are those files renamed to `results/blt_table_copy_onsets.json` (sha256 1cef151e3494cc81dff0ba070e9c8b1fb146442d9c1a4f615a276b17b4471e53) and `results/blt_table_copy_onsets.log` (sha256 5e303ed3eba4e2e92272f19cef26f08f75133fe2108a9e7ff2d4a30291fa1878). In the tables, log:N is line N of the log. A first run (v1) was superseded: same F1 numbers, a float32/float64 comparison bug in one tie count; it is not committed.

**Verdict:** table avoids copy onsets

## Summary

The novelty table never starts a patch at the first byte of the copied ticket (c1). On O6's 293 dev F1 texts it does so 0 times at BLT-1B's default rate and 0 times at 10% (exact 95% upper bound 1.3%). Entropy starts there in 98.6% [97.3, 99.7] of copies at the default rate and 31.7% [26.3, 36.9] at 10%. The table also starts at c1 in 0 of 293 texts with 6-digit IDs and 0 of 293 with mixed IDs, built on the same dev problems.

This is not copy detection:
- The table gives the copy exactly the same scores as the source on 293 of 293 texts, because both sit after the same two bytes. So it never starts at the new ticket's first byte either (0%, against entropy's 100%).
- The byte pair before the ID ('t ' after 'Ticket ') scores -3.00, below the table's thresholds (-2.92 at the default rate, -2.32 at 10%). In a rate sweep the table would start at every source and copy from a rate of about 36%.

The table does start inside the copied ID, exactly as often as inside the source:
- 4-digit IDs: 20.8% of copies at the default rate, 9.2% at 10% (at 10% that is more than entropy, +8.9 [+5.5, +12.3]);
- 6-digit IDs: 20.8% and 11.6%;
- mixed IDs: 100% and 66.9%.

Share of all starts that fall on the copy: the table 0.2% at both rates; entropy 1.8% at the default rate and 0.6% at 10%.

The hybrids keep entropy's starts at the copy onset:
- entropy + '=' starts: 98.6% at the default rate, 29.0% at 10%;
- the label-free R8b: 97.6% at the default rate, 11.9% at 10% (-19.8 [-24.2, -15.4] against entropy). The drop comes from its higher entropy threshold; it still starts at the source onset in 100% of texts.

Verdict: on these templates the table avoids copy onsets, because of the bytes before the ID, not because it recognises copies. For paper 2 this means O6b's copy hazard belongs to entropy, and the recommended hybrids keep it. Whether the table's layouts actually copy better needs a model run.

## Method

Read-only on both repositories; no model loaded and nothing scored. The BLT tokenizer was loaded only for O6's text filter.

Texts:
- O6's dev F1 texts were rebuilt with the repository's own builder (realblt_quadrant.py:48 `build`, rng default_rng(0), train tickets drawn first, as realblt_copy_latch.py:164 `build_texts` does) and filter (realblt_copy_latch_fresh.py:189 `keeper`).
- GSM8K rows were read with itertools.islice: train idx 0-299 (fit), test idx 0-299 (test). This replaces the repository's whole-file `list(open(...))[:300]`.
- The rebuilt texts match results/blt_copy_latch_patcher.npz byte for byte, units included (295/295 train, 293/293 test). That file supplies the stored patcher entropy and BLT-1B's default layout D.
- c1..cL are the id_copy tokens and s1..sL the id_new tokens (token t = byte t-1).

Signals:
- Entropy: realblt_copy_latch.py:139 `sig_of`.
- Table: the stored ~/.cache/segresearch-blt/novtab_gsm.npz, looked up with segresearch scale/rules.py:45 `table_lookup` and aligned as realblt_waste.py:39-50 and realblt_cues.py:89-102 do. This is the 'nov' layout of R4 (realblt_cues.py:182) and R8 (realblt_waste.py:33-34). Table 5's script, realblt_quadrant.py, builds entropy layouts only.

Layouts (segresearch blt_layouts.py:77 `Fit` and :132 `layout`, train mode):
- Thresholds were fitted on the 295 fit texts at D's pooled fit rate r_d = 0.26036, as O6's D.X layouts are, and at 0.10.
- entropy: D itself at the default rate; E10 at 10%.
- table: kind 'nov'.
- entropy + '=' starts: 'forced[eq]:entropy', with realblt_budget.problem's start, the byte after '= '.
- R8b: 'forced[q30]:entropy'. The top-3% table value was recomputed from realblt_hybrid's own training texts (realblt_hybrid.py:42-48) and equals results/blt_hybrid.json:thresholds. The 1.5% and 5% variants are reported as secondary.
- Hybrid fill thresholds were fitted on the ticket texts.

Readouts per text:
- a start at c1, at c2..cL, and at any byte, and the same at the source;
- starts on the copy as a share of all starts;
- starts on L8 and V_8 positions (realblt_copy_latch.py:244 `add_masks`);
- coverage by unit kind.

Intervals: problem-level bootstrap with realblt_copy_veto.py:60 `boot_ratio` (2,000 resamples, seed 6). Paired differences are against entropy at the same rate. Exact bounds are given where the bootstrap interval has zero width.

Extras:
- a rate sweep of the table;
- table thresholds fitted R4's way, on plain text without the ticket;
- the table on 6-digit and mixed IDs (cell F3's template, realblt_copy_latch_fresh.py:148 `make`, seed 0), plus letters, doses, F2a and F2b, on the same dev problems with thresholds from their own dev fit texts;
- D at c1 from the dev smoke patcher outputs (GSM8K train idx < 128).

Reproduction checks:
- D at c1: 289, and E10 at c1: 93. Both equal results/blt_copy_latch_step1.json:a.n_D_starts_c1 and a.n_E10_starts_c1.
- The entropy thresholds equal step1.json:thresholds_nats.
- Entropy refitted to D's rate gives exactly D on 293/293 texts.

## Results

Citations: log:N is line N of results/blt_table_copy_onsets.log; full records are in results/blt_table_copy_onsets.json (keys F1, templates, table_sweep, smoke_dev_D).

A. O6's dev F1 texts: 293 GSM8K test problems with a 4-digit ticket; thresholds fitted on 295 train problems (idx 0-299). Values are % of copies (one per problem) with a problem-bootstrap 95% interval. * marks a zero-width bootstrap interval; the exact 95% bound is shown instead.

| layout @ rate | test rate | copy c1 | copy c2-c4 | copy, any byte | source s1 | source, any byte | % of all starts on the copy | % of starts on L8 positions | cite |
|---|---|---|---|---|---|---|---|---|---|
| entropy @D (BLT-1B's own layout) | 0.274 | 98.6 [97.3, 99.7] | 92.2 [89.1, 94.9] | 99.7 [99.0, 100] | 100 [98.7, 100]* | 100 [98.7, 100]* | 1.8 [1.8, 1.9] | 2.0 [1.9, 2.1] | log:10, log:69 |
| table @D (fit rate 0.2604) | 0.261 | 0.0 [0, 1.3]* | 20.8 [16.0, 25.3] | 20.8 [16.0, 25.3] | 0.0 [0, 1.3]* | 20.8 [16.0, 25.3] | 0.2 [0.1, 0.2] | 0.3 [0.3, 0.4] | log:11, log:70 |
| entropy + '=' starts @D | 0.273 | 98.6 [97.3, 99.7] | 91.5 [88.4, 94.5] | 99.7 [99.0, 100] | 100* | 100* | 1.8 [1.7, 1.9] | 2.0 [1.9, 2.1] | log:12, log:71 |
| R8b (entropy + table top 3%) @D | 0.272 | 97.6 [95.9, 99.3] | 88.7 [85.0, 92.2] | 99.7 [99.0, 100] | 100* | 100* | 1.7 [1.6, 1.8] | 1.9 [1.8, 2.0] | log:13, log:72 |
| entropy @10% (E10) | 0.107 | 31.7 [26.3, 36.9] | 0.3 [0.0, 1.4] | 32.1 [26.6, 37.2] | 100* | 100* | 0.6 [0.5, 0.7] | 0.8 [0.7, 0.9] | log:16, log:75 |
| table @10% | 0.103 | 0.0 [0, 1.3]* | 9.2 [6.1, 12.6] | 9.2 [6.1, 12.6] | 0.0 [0, 1.3]* | 10.9 [7.5, 14.7] | 0.2 [0.1, 0.2] | 0.3 [0.2, 0.4] | log:17, log:76 |
| entropy + '=' starts @10% | 0.107 | 29.0 [23.9, 34.1] | 0.3 [0.0, 1.4] | 29.4 [24.2, 34.5] | 100* | 100* | 0.6 [0.5, 0.7] | 0.8 [0.6, 0.9] | log:18, log:77 |
| R8b @10% | 0.105 | 11.9 [8.5, 16.0] | 0.0 [0, 1.3]* | 11.9 [8.5, 16.0] | 100* | 100* | 0.2 [0.2, 0.3] | 0.4 [0.3, 0.5] | log:19, log:78 |

Paired against entropy at the same rate (points, problem bootstrap):
- table @D: c1 -98.6 [-99.7, -97.3]; any byte -78.8 [-83.6, -74.1]; source s1 -100.0; share of starts on the copy -1.7 [-1.8, -1.6] (log:40).
- table @10%: c1 -31.7 [-36.9, -26.3]; c2-c4 +8.9 [+5.5, +12.3]; any byte -22.9 [-29.4, -16.0]; source s1 -100.0 (log:45).
- entropy + '=' starts: c1 +0.0 @D (log:41); -2.7 [-4.8, -1.0] @10% (log:46).
- R8b: c1 -1.0 [-2.4, +0.0] @D (log:42); -19.8 [-24.2, -15.4] @10% (log:47).
- Starts on V_8 positions (after any recurring 8-byte context): the table is above entropy, +1.3 [+0.7, +1.9] @D and +1.4 [+0.7, +2.2] @10% (log:40, log:45).
- Coverage by unit kind: the table starts at 100% of computed-value starts and 0% of final answers at both rates; entropy @10% covers 13.5% of computed-value starts (log:55, log:60, log:61).

B. Why the table behaves this way (bytes only):
- It gives c_k and s_k identical scores on 293/293 texts (log:8).
- Source and copy decisions differ at only 7 positions, all at 10% and all ties at the threshold (log:51).
- The onset context 't ' scores -3.0008. The thresholds are -2.920 (D rate) and -2.317 (10%). 34.5% of fit tokens score higher (log:5).
- Rate sweep: c1 coverage is 0% at rates 0.05-0.34, 18.1% at 0.35, 95.2% at 0.36 and 100% at 0.40 (log:83).
- With thresholds fitted R4's way on plain text, c1 coverage is 0% at both rates (log:84).

C. Other ID kinds and templates on the same dev problems (table only, bytes). c1 coverage is 0 of 293 in every case (0 of 292 for F2b), exact bound [0, 1.3].

| template | copy c2..cL @D | copy c2..cL @10% | cite |
|---|---|---|---|
| 6-digit ID (F3) | 20.8 [16.0, 25.3] | 11.6 [7.8, 15.0] | log:87-88 |
| mixed ID, e.g. QA3-7490 (F3) | 100 [98.7, 100] | 66.9 [61.4, 72.4] | log:91-92 |
| letters (secondary) | 50.5 [45.1, 56.3] | 8.2 [5.5, 11.6] | log:95-96 |
| doses (secondary) | 62.1 [56.7, 67.6] | 26.6 [21.5, 32.1] | log:99-100 |
| F2a (secondary) | 11.9 [8.2, 15.7] | 11.9 [8.2, 15.7] | log:103-104 |
| F2b (secondary) | 20.9 [16.1, 25.7] | 12.0 [8.6, 15.8] | log:107-108 |

Entropy for these kinds exists only in the dev smoke outputs (log:110): D starts at c1 in 6/6 6-digit and 3/4 mixed texts, and in 20/20 F2a and 20/20 F2b texts. The table starts at c1 in none of them.

D. Checks:
- The rebuilt texts are identical to the stored ones (log:1).
- The R8b top-3% value equals results/blt_hybrid.json:thresholds["0.03"] = -1.8733 (log:2).
- The entropy thresholds equal step1.json:thresholds_nats (log:4).
- D at c1: 289/293 and E10 at c1: 93/293, equal to step1.json:a (log:6).
- Entropy refitted to D's rate gives exactly D on 293/293 texts (log:8).

## Caveats

- Placement only: no model was run, so nothing here shows the exact-copy or bits cost of any layout. Table 5's -1.33 bits (paper2v2/manuscript.md:362) is the value of a start at the copy's first byte inside the entropy layout (realblt_quadrant.py:100, :130). The table's layout differs everywhere else as well, and BLT-1B was trained under entropy patching.
- It is not copy detection. The table scores the copy and its source identically (log:8, log:51), so it never starts at the new ticket's first byte either (0% vs entropy's 100%). Table 5 also prices a start there as a cost (-0.72 bits, manuscript.md:361), but whether skipping both helps under the table's own layout is untested.
- The result depends on the template and the rate. It rests on the byte pair before the ID: 't ', ': ', 'e ' and ' #' all score below the thresholds (log:5, log:86-106). From a table rate of about 36% the table starts at every source and copy (log:83). An ID that follows a high-novelty pair (for example '= ', which scores -1.13) would get starts at both the source and the copy.
- Starts inside the copied ID remain: at c2 in 20.8% (default rate) and 9.2% (10%) of 4-digit copies, and in 67-100% of mixed-ID copies (log:11, log:17, log:91-92). Their effect on exact copying is unmeasured. O6b's X-all and L8 treat a start at any byte c1..cL as a copy start.
- Dev only and exploratory. These are O6's dev texts, the texts L8 and SIV+PO were designed on, and nothing here was pre-registered. Entropy for 6-digit and mixed IDs exists only in 6 and 4 smoke texts (log:110). The extra templates were built here with seed 0, not the O6b cells' seeds.
- Rates are matched as O6 matches them: the threshold is fitted at D's pooled fit rate, 0.2604. On the test texts the table runs at 0.261 against D's 0.274 (log:10-11). The sweep shows c1 coverage stays at 0% up to a rate of 0.34, so the mismatch does not change the result.
- The stored table novtab_gsm.npz is the one R4, R8 and R8b used. It was fitted on all 7,473 GSM8K train solutions, GSM8K only (segresearch/realblt_combo.py:52-63; analysis/2026-10-29/review_findings.md:714), so its fit includes train rows >= 1800 and the O6c-reserved rows >= 6253. I used it as stored and did not refit it, because refitting would read those rows. The manuscript says the table was fitted on GSM8K and MATH (manuscript.md:130).
- Bootstrap intervals at 0% and 100% have zero width, so exact bounds are reported there (0 of 293 gives [0, 1.3%]).
- Repositories: this check wrote nothing to either repository. No new files or __pycache__ entries appeared, and segresearch's git status is unchanged. While it ran, another session committed O6c files to the private working repository (ae0266a, 5841bf7, 99f2550, 069c10b, plus results/blt_copy_latch_cutoff_*). None of the files this check imports or reads changed: git diff 3af1410..HEAD touches only O6c files.
- The first run (the files ending _v1) is superseded. Its F1 numbers are identical, but its tie count at the 't ' context was wrong because it compared a float64 value with float32 values; v2 fixes this.
