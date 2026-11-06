# O6b: a gated copy latch (L8) against patch starts inside verbatim copies. Decision rules

Written 2026-10-30, before any L8 mask, count or layout was computed on any text. Exploratory follow-up to O6
(followup/O6_COPY_VETO_RULES.md: rules 4c7808d, results 097ff1d and 2bf8da0); tracking issue #35. The design is
analysis/2026-10-30_copyfix/DESIGN.md at commit **4b76f51**, with the main session's review notes in its header.
Nothing here goes into paper2v2.

**Note (2026-10-30, after the independent check at 6e71573).** The first sentence above is false as written. Before
this file was written, the design had already computed L8 on the same dev texts, and this session had read those
results:
- [S1] (`analysis/2026-10-30_copyfix/synth/s1_gate.py`) evaluated conditions 4-6 at the 293 dev copies from O6's
  stored patcher outputs: 288 of 293, 92 of E10's 93 c1 starts, L8c 118 of 293.
- [S2] computed L8 masks on the 660 xdomain documents.
- [S4] and [S5] computed byte-level latches on the dev GSM8K texts.

What was written before any run is the rule, the Implementation section and the kill rules: realblt_copy_latch.py
had not run. So dev step 1 (a) reproduces [S1] copy by copy; it checks the implementation and is not evidence. Steps
1 (b) and (c) are new measurements ([S4] and [S5] used byte proxies and let condition 6 pass). The confirmatory rule
proposed at the end of this file replaces the DRAFT sections below.

**What is copied.** These parts of DESIGN.md (4b76f51) are copied without change; only heading levels differ:
- section 1's rule text, with the pre-registered L8c fallback;
- section 2, the comparators;
- section 3, the dev diagnostics, go criteria and stop rules;
- K1-K4;
- section 4 and K5-K9, under **DRAFT, pending PI approval**.

The "Implementation" section fixes what the design leaves open. It was also written before any run.

**Dev only until the PI approves a confirmatory rule.**
- Everything before the draft line runs on O6's existing texts and is exploratory.
- No fresh text is built, and no GSM8K train problem from idx 1800 on is read. No confirmatory job runs before the PI
  approves a confirmatory rule.
  - *Dated note (2026-10-30, after the second check at 67667ac):* dev smoke runs of the confirmatory code, on GSM8K
    train idx 0-299, are excepted. The one smoke run (ecca81e) is disclosed at the end of the confirmatory section.
- Nothing changes after dev data is seen, except the pre-registered L8c switch. Any other change would be dated and
  disclosed here.
- If a kill rule fires, the work stops and is reported. There is no redesign.

## 1. Rule L8 (DESIGN.md section 1, rule text)

**Rule L8: a gated copy latch.** Written 2026-10-30, before any L8 layout was run. This is an exploratory follow-up to O6 (followup/O6_COPY_VETO_RULES.md). Nothing here goes into paper2v2.

**Positions.** These follow O6:
- Token 0 is BOS, token t is byte t-1, and tokens 0 and 1 always start a patch.
- A start at token t is decided from the patcher's output at token t-1, which has read bytes 0..t-2 (realblt_copy_veto.py:83).
- For byte j, h(j) is the patcher's next-byte entropy in bits and a(j) is its argmax. Both come from its output at token j.

**Onset.** Token t is an onset when all six of these hold:
1. *Boundary.* Byte t-2 is ASCII and is not a letter or digit.
2. *Recurrence.* Bytes t-9..t-2 also occur ending at byte t-3 or earlier. This is V_8's test (realblt_copy_veto.py:43-48). Let e be the end of the latest such occurrence.
3. *Source run.* T is the longest run that starts at byte e+1 and is made of ASCII letters and digits. A single `.`, `,`, `-`, `/` or `:` may stand between two of them. T must not be empty, and it must end before byte t-9, so it lies wholly before the current match.
4. *Unpredictable source.* The sum of h over T's bytes is at least 10 bits.
5. *Agreement.* a(t-1) is T's first byte. Compare token ids, as realblt_copy_gate.py:79 does.
6. *Induction evidence.* h(t-1) ≥ log2 10 bits, or h(e+1) - h(t-1) ≥ 1 bit.

**Latch.**
- An onset token may not start a patch.
- Nor may token t+m, for m = 1..|T|-1, as long as bytes t-1..t+m-2 equal the first m bytes of T.
- The latch ends at the first mismatch, or after T's last byte. The byte after the copy is decided by entropy as usual.
- Latches from several onsets combine.

**Layouts.**
- The score is the patcher's entropy, set to -inf at latched tokens.
- The threshold is refitted on the train texts (blt_layouts.Fit, train mode). The pooled train rate then equals the unlatched layout's: D's own train rate, and 0.10.
- Verdict pairs: D.L8 vs D, and E10.L8 vs E10.

**What it does not use.**
- No learned probe, unit label, template string or answer marker.
- No distance limit.
- No added, moved or deferred start.
- Everything used for token t is a function of bytes 0..t-2 and the patcher's outputs at tokens up to t-1.

**Constants, fixed now.**
- k = 8, from O6's V_8 (O6:63).
- S0 = 10 bits, about three random decimal digits (3·log2 10 = 9.97). The 293 dev tickets have a minimum of 12.49 bits, so S0 is not set at that minimum.
- log2 10 and 1 bit in condition 6.
- Nothing is refitted after the dev diagnostics, except through the fallback below.

**Pre-registered fallback, L8c.**
- Trigger: the dev patcher pass finds L8 removing new-value computed-value starts from D at more than 0.25% of D's computed-value starts in the 294 dev equation texts (more than 2 of 799).
- Change: condition 6 becomes h(t-1) ≥ log2 10 alone.
- The switch is made on dev data, before any fresh text is built.

**Property.** Every token L8 latches is also vetoed by V_8. The 8 bytes before token t+m equal the 8 bytes before source byte e+1+m, which occur earlier. So L8's removals are a subset of V_8's at both rates.

## 2. Comparators (DESIGN.md section 2)

1. **D.V8 and E10.V8**: O6's veto, with matched refill. This is the like-for-like baseline and the positive control.
   - L8 is a subset of V8, so the pair isolates what the gate removes.
   - V8's dev costs at D were: computed -5.6, final -4.4, copied -4.6, bits per byte +0.0196 (pairs['D.V8 vs D']).
   - If those costs do not come back on the fresh texts, the cost checks are too weak for "no cost" to mean anything (K6).
2. **D.SIV and E10.SIV**: the bytes designer's design 1, with kmin = 5 (siv() in `analysis/2026-10-30_copyfix/bytes/final_rules.py`). This is the strongest rule that reads no model state.
   - On dev it covers c1..c4 in 293 of 293 texts and 0 of 977 computed tokens (`analysis/2026-10-30_copyfix/bytes/an15_out.json`, entry 'D1 SIV').
   - It should tie with L8 on the ticket template.
   - It should lose where only reading the model helps:
     - mid-line copies (cell F2b), where it cannot fire;
     - code and traces, where it vetoes 13.8% and 4.0% of positions (`analysis/2026-10-30_copyfix/judge_cost/out_xd_code_all.json` and `analysis/2026-10-30_copyfix/judge_cost/out_xd_traces_all.json`).
   - It runs in dev step 2 and in F2b only, to save Mac time.
   - *Note (2026-10-30, after the check at 6e71573): F2b is not a place where only reading the model helps. PO and
     SIV+PO, which read bytes only, fire at c1 in 293 of 293 dev problems rewritten into F2b's template; only SIV
     cannot. The confirmatory rule at the end of this file scores SIV+PO beside L8 in every cell.*

## 3. Dev diagnostics on the existing texts (DESIGN.md section 3; exploratory)

These use O6's 295 + 293 quadrant texts, the 300 + 294 equation texts and the xdomain documents. They decide whether fresh data is spent; they are not evidence for the claim.

**Mac time.** One BLT-1B pass takes 0.32-0.57 s: 275 problems × 10 layouts in 891 s (blt_copy_veto.log), and 275 × 6 in 946 s (blt_copy_gate.log). The patcher alone takes about 0.09 s per text: 915 problems in about 1.3 min (blt_cue_budget.log). Only one job runs at a time.

**Step 0: done (CPU, no Mac time).** The results are in section 1: [S1] for recall and the predicted gains, [S4] and [S5] for byte exposure, [S2] for natural text.

**Step 1: patcher only, about 3-5 min, one job.**
- Run the patcher on the 1,182 dev GSM8K texts, rebuilt as in realblt_copy_veto.py:94-104 and :170-176, and on the 360 xdomain test documents.
- Store, per position: the bf16 entropy, the argmax and the surprisal of the actual byte.
- Then, on the CPU in under 1 minute: build the L8 and L8c masks, refit both thresholds, and count.
- Go criteria (all needed):
  - (a) L8 fires at c1 in at least 95% of the 293 copies, and at at least 95% of E10's 93 starts at c1. Predicted: 98.3% and 98.9%.
  - (b) New-value computed-value starts removed in the 294 equation texts: 0 from E10 (predicted 0, a subset of V8's 0) and at most 2 of D's 799. Predicted 0-2; the 4 cases in [S5] are the ones to watch.
  - (c) At most 1.0 D start removed per quadrant text outside the copy. Predicted 0.2-0.8; V8 removed about 12.
- Reported:
  - which condition stops each of V8's 50 computed cases;
  - the summed source entropy at each ticket;
  - the threshold shift (predicted at most 2 bf16 steps);
  - same-value removals;
  - the E10 cost in xdomain with the real argmax (predicted between the upper and oracle rows of [S2]).
- Stop rules:
  - (a) fails: K1.
  - (b) fails at E10: K2.
  - (b) fails at D: switch to L8c and recheck (a) at E10 and (b). If L8c also fails: K2.
  - (c) fails: K3.

**Step 2: BLT-1B on the 293 dev texts, about 14-25 min, one job.**
- Layouts: D, E10, D.L8, E10.L8, D-L8 and E10-L8 (no refill), D.SIV and E10.SIV.
- Also D-fit: plain entropy at the refitted D-rate threshold. It costs nothing if identical to D, and it would expose any artifact of the threshold fit.
- D and E10 must reproduce the stored rows: id_copy exact equal per problem, and bits within 0.05 on at least 95% (O6's G0). Otherwise stop.
- Go criteria: D.L8 - D at least +60, E10.L8 - E10 at least +12, and every cost bar of section 4 met at both rates.
- Predictions: +70 to +78; +12 to +19; computed 0 ± 0.5; final 0 ± 2; copied 0 to -0.5; bits per byte without the copy 0 to +0.002.
- Otherwise: K4.

**Step 3: optional, about 2-3 min, outside L8.**
- Layout: E10.V8 with one start forced at the `k` of the copy's `\nTicket `.
- It tests the anchored design's premise. That is the only lever past the E10 ceiling: under V8, copies without an E10 onset start went only from 66.0% to 68.0% [S3].
- It is reported only and never changes L8.
- *Note (2026-10-30): "ceiling" here, and in the DRAFT sections below, means the onset-removal bound without refill
  (E10 without its starts at the copy). It is not a ceiling at 10%: under V6, the same copies went from 66.0% to
  72.0% [S3], and O6's matched E10.V6 gained +21.2 [+15.4, +27.0] on these texts.*

## Kill criteria for dev (DESIGN.md section 5: K1-K4)

- **K1:** in dev step 1, L8 fires at c1 in under 95% of dev copies. No BLT-1B run.
- **K2:**
  - any new-value removal from E10, in dev or in F6;
  - or new-value removals at D above 0.25% for both L8 and L8c.
  - Reading: the patcher does not separate copies from new values after recurring contexts. Drop L8.
- **K3:** more than 1.0 D start per dev text is removed outside the copy.
- **K4:** in dev step 2, any bar fails, or D.L8 - D is below +60, or E10.L8 - E10 is below +12. Do not spend fresh data.

## Implementation (written 2026-10-30, before any run)

Script: realblt_copy_latch.py, derived from realblt_copy_veto.py. It uses BLT-1B with the 512-byte windows, loaded
with blt_light.py as in O6.

    ../segresearch/.venv-blt/bin/python realblt_copy_latch.py step1|step2|step3

Only one BLT-1B job runs at a time; `pgrep -fl realblt` is checked before each step.

**Texts.** O6's texts, rebuilt in the same order:
- quadrant: 295 train and 293 test texts (realblt_copy_veto.py:94-104);
- equation: 300 train and 294 test texts (realblt_copy_veto.py:170-176);
- xdomain: the 660 documents stored in results/xdomain_gains_*.npz. That is 110 per domain (50 fit and 60 test), the
  set [S2] used. Results are reported on all 660 and on the 360 test documents.

**Patcher outputs.** Step 1 stores them in results/blt_copy_latch_patcher.npz (git-ignored) for steps 2-3.
- h(j) is the patcher's entropy at output token j, as the library returns it (bf16, nats), divided by ln 2 in
  float64.
- a(j) is the argmax of its logits at token j (realblt_copy_gate.py:79). The token id of byte x is x + 4; this is
  checked on every text.
- The surprisal of the actual byte is stored and reported, but L8 does not use it.
- D is the patcher's own patch starts, as in O6.

**Mask details.**
- Condition 3's run is the regex `[A-Za-z0-9]+(?:[.,\-/:][A-Za-z0-9]+)*`, matched at byte e+1. The scan stops at byte
  t-2, and T must end at byte t-10 or earlier. Stopping the scan there never changes the outcome, because a run that
  reaches byte t-9 fails anyway.
- Onsets are tested at every token, including tokens already latched. The mask is the union of all latches.
- L8c differs only in condition 6, which becomes h(t-1) ≥ log2 10.
- A latched layout's signal is the entropy, with latched tokens set to -inf. Its thresholds come from blt_layouts.Fit
  (train mode) on the 295 quadrant train texts, at D's pooled train rate and at 0.10.
- SIV is siv() from analysis/2026-10-30_copyfix/bytes/final_rules.py with kmin = 5, imported unchanged.

**CPU checks.** These run on every dev text (1,182 GSM8K texts and 660 xdomain documents), before any BLT-1B
scoring.
- *Causality.* For every cut p = 1..n-2, replace bytes p.. and the patcher outputs at tokens p+1... This is done
  twice: once with draws from the same text, once with the reversed suffix. The L8 and L8c decisions at tokens
  0..p+1 must not change. Bytes p.. include byte t-1 for every token t ≤ p+1, so every decision is tested against
  changes at and after its own byte.
- *Subset.* On every text, L8 ⊆ V_8 and L8c ⊆ L8.
- A failure is an implementation bug, not a design change. It is fixed before step 2, and the fix is disclosed here.

**Step 1 definitions.**
- (a) c1 is token a+1, where a is the copy's first byte. E10 is entropy@10% fitted on the 295 quadrant train texts,
  O6's E10 (93 copies have an E10 start at c1).
- (b) This follows O6's V4.
  - Computed-value tokens are {m.start()+3, m.start(1)+1} for each `= \$?(\d[\d,]*(?:\.\d+)?)` match in the 294
    equation test texts.
  - D is the patcher's own layout. E10 is entropy@10% fitted on the 300 equation train texts.
  - "Removed" means the token is in the layout and masked, before any refill.
  - A removal is same-value by O6's definition: an earlier occurrence of the 8 bytes before t continues with the same
    bytes through the value's end. Otherwise it is new-value.
  - The trigger and K2 count new-value removals. The strict count (all removals) is reported too.
- (c) D's starts that are masked at tokens other than c1..c4, averaged over the 293 quadrant test texts.
- Threshold shift: entl8 against entropy, fitted at D's pooled train rate and at 0.10. It is reported in nats and in
  bf16 steps; the step at x is 2^(floor(log2 x) - 7).
- xdomain E10 cost uses [S2]'s method (judge_cost/xd.py).
  - The cost is the stored gain at masked E10 base starts, minus the gain of the refill.
  - The refill is the top K non-base, unmasked positions by entropy, pooled per domain.
  - It is reported for L8 with the real argmax, and for the upper row (agreement always passes) and the oracle row
    (actual byte = T[0]), all recomputed with this implementation on the same documents.
  - Also reported: whether this run's patcher entropy equals the stored entropy.
- V8's computed cases are V_8's removals of D starts at computed-value tokens in the 294 equation test texts (O6:
  50). For each case, the report gives the first L8 condition that fails, or "latched".

**Step 2 definitions.**
- Layouts: D, E10, D-fit, D.L8, E10.L8, D-L8, E10-L8, D.SIV and E10.SIV. If step 1 switched, L8c replaces L8.
- Readouts and pairs follow O6 step 2 (realblt_copy_veto.py summarize: a paired bootstrap over problems with 2,000
  resamples and seed 6).
- Stop: D and E10 must reproduce the stored rows of results/blt_copy_veto.json. id_copy exact must be equal on every
  problem, and bits must be within 0.05 on at least 95%.
- K4's bars at dev step 2 (all are needed):
  - id_copy exact: D.L8 - D ≥ +60 and E10.L8 - E10 ≥ +12 points;
  - C2: bits per byte without the copy's digits ≤ +0.01;
  - C3: computed exact ≥ -2 points;
  - C3f: final exact ≥ -2 points;
  - C3c: copied exact ≥ -2 points;
  - rate: the test patch rate is within 0.005 of the unlatched layout's.
- C2-C3c and the rate bar apply to both pairs, on point estimates as in section 4. C4 is step 1's (b). C4s and the G
  bars need fresh cells and do not apply at dev.
- D-fit is reported against D: on how many problems the layouts are identical, and any difference in scores.

**Step 3.** It runs only if steps 1 and 2 pass. It is reported only and never changes L8.
- E10.V8 is V_8 at a matched 10%, from the same fit.
- E10.V8+k adds one start at token a-3, the `k` of the copy's `\nTicket `. So it has one more start per text.
- Both are scored on id_copy, split by whether E10 has a start at c1, and on every other kind.

**Outputs.** results/blt_copy_latch_step1.json, _step2.json and _step3.json, with their logs. All are exploratory
(dev). Smoke runs go to *_smoke.json and are not used.

---

*Superseded (2026-10-30). The two DRAFT sections below are DESIGN.md's text as copied before dev, kept as history.
The section "Confirmatory rule, proposed for filing" at the end of this file replaces them; its change log lists the
differences from DESIGN.md, with their reasons.*

## DRAFT, pending PI approval: confirmatory test on fresh data (DESIGN.md section 4)

**Data.**
- The pool is GSM8K train problems from idx 1800 on (~/.cache/segresearch-gsm8k/train.jsonl, realblt_budget.py:44).
- No GSM8K test problem is left fresh:
  - 0-299 were used by O6, the quadrant and R9 (realblt_quadrant.py:103-104; data/prose_rewrites/test.jsonl idx 0-299);
  - 300-1318 were used by R20 and R22 (data/prose_rewrites_fresh/test.jsonl idx 300-1318).
- On the train side:
  - 0-1499 is R9's (data/prose_rewrites/train.jsonl idx 0-1499);
  - the xdomain math documents come from the start of the file (xdomain_data.py:64-72, :146-153);
  - the designers read 300-1799 (`analysis/2026-10-30_copyfix/bytes/an10.py` line 11, `analysis/2026-10-30_copyfix/bytes/an16.py` line 7).
- From idx 1800 on, only whole-corpus reads that select nothing touch these problems: realblt_compress.py:44-52, and fine-tuning of other models in segresearch.
- That leaves 5,673 problems; about 5,056 pass O6's filters by a byte-level estimate [S3].
- Blocks, in file order (each block is the next N problems that pass the filters):

| block | fit texts | test texts |
|---|---|---|
| F1 (F6 uses the same problems) | 300 | 600 |
| F2a | 300 | 300 |
| F2b | 300 | 300 |
| F3 | 300 | 400 |
| F4, from problems whose distance is 550-950 bytes (1,537 such [S3]) | 300 | 200 |

- Each cell gets its own seed.
- The rule, bars and predictions are filed before any text is built. The texts, and F6's byte-level count of V4 candidates, are fixed before any model run.

**Cells.** Exact means every byte of the copied string is right under teacher-forced argmax. Bits is their summed -log2 p (O6:24-25).

| cell | text | layouts |
|---|---|---|
| F1 | O6's template unchanged (realblt_quadrant.py:57), fresh problems, new tickets | D, D-all, D.L8, D.V8, E10, E10-all, E10.L8, E10.V8 |
| F6 | F1's problems in realblt_budget format, no ticket. Counts: L8, L8c, V8. | scored: D, D.L8, E10, E10.L8 |
| F2a | context does not recur: the copy line is `\nIt was logged as #NNNN and closed.` | D, D-all, D.L8, E10, E10.L8 |
| F2b | recurring context in other words, mid-line: source `Order reference NNNN was opened. `, copy `\nWe shipped order reference NNNN today.` (shared context `rder reference `) | D, D-all, D.L8, D.SIV, E10, E10-all, E10.L8 |
| F3 | F1's template, 100 problems each: 6-digit IDs, 4 upper-case letters, mixed (`QX7-4821`). Plus 100 doses: source `Dose ordered: X mg. `, copy `\nDose ordered: X mg given.`, with X drawn from 12 common doses. | D, D-all, D.L8, E10, E10-all, E10.L8 |
| F4 (secondary) | far: `Notes:\nTicket NNNN opened.\n` placed before the question | as F3 |
| F5 (secondary) | F1's test texts with SEGR_BLT_WINDOW=0; thresholds and L8 recomputed from that model's own patcher | D, D-all, D.L8 |

**Verdicts.** Each compares a layout with its unlatched layout at the matched rate, with a paired bootstrap over problems (2,000 resamples).

| id | readout | bar | prediction |
|---|---|---|---|
| C0 (stop) | F1: D-all - D, id_copy exact | ≥ +20, lower bound ≥ +10 | +76 (+70 to +81) |
| C1 | F1: D.L8 - D | ≥ +15, lower bound > 0 | +74 (+68 to +79) |
| C1e | F1: E10.L8 - E10 | ≥ +15 (PI target), lower bound > 0 | +15.5 (+12 to +19); P(point ≥ 15) ≈ 0.6. Also reported: share of the measured ceiling (E10-all - E10), predicted ≥ 0.9 |
| C2 | bits per byte without the copy, both rates | ≤ +0.01 (O6's V2) | D 0 to +0.002; E10 ±0.001 |
| C3 | computed exact, both rates | ≥ -2 (O6's V3) | 0 ± 0.5 |
| C3f | final answers, both rates | ≥ -2 | 0 ± 2 |
| C3c | copied numbers, both rates | ≥ -2 | 0 to -0.5 |
| C4 | F6: new-value computed-value starts removed | 0 from E10; ≤ 0.25% of D's. O6's strict count reported. | D 0-3 of about 1,600; E10 0 |
| C4s | F6 scored: computed exact | ≥ -2 at both rates | 0 ± 0.3 |
| rate | test patch rate | within 0.005 | within 0.002 |
| control | D.V8 - D on F1 and F6 | V8's cost reappears: computed ≤ -2 or bits per byte ≥ +0.01 | id_copy +65 to +76; computed -3 to -8; final -1 to -8; copied -3 to -6; bits per byte +0.012 to +0.025; new-value removals 3-8% of D's |
| G1 | F2b: D.L8 - D | ≥ +15, lower bound > 0 | +55 to +78 |
| G1e | F2b: E10.L8 - E10 | lower bound > 0 | +6 to +18 |
| G1s | F2b: D.SIV - D | reported | 0 ± 3 |
| G2 (null) | F2a: share of copies where L8 fires; D.L8 - D and E10.L8 - E10 | ≤ 5%; within ±3; cost bars as C2-C3c | ≤ 2%; ±2. D-all - D reported, predicted +10 to +60 |
| G3 | F3, each ID type with D-all - D ≥ +20 | D.L8 - D ≥ +15, lower bound > 0, and ≥ 0.7 of D-all - D | fires on ≥ 90%; gain ≥ 0.8 of D-all - D |

Reported with predictions, without verdicts:
- **F3 doses.** L8 fires on at most 30%, because most common doses sum below 10 bits. D.L8 - D within ±5.
- **F4.** D-all - D at least +50 (O6:123). L8 fires on at least 80% if the argmax at c1 holds beyond 619 bytes. Otherwise D.L8 ≈ D.
- **F5.** D.L8 at least D + 20 on exact, with bits per copy no higher than D's. L8 fires on at most 10% of copies in texts over 700 tokens [JW0].

What each claim needs:
- **Fix claim:** C0 met, the control shows a cost, and C1, C1e, C2, C3, C3f, C3c, C4 and C4s all hold.
- **Generality claim:** G1, G2, and G3 for each type it names.

**Jobs, one at a time.**

| job | content | time |
|---|---|---|
| J1 | patcher only: all fit texts, plus F6 counts | 5-8 min |
| J2 | F1, 4,800 passes | 26-46 min |
| J3 | F6, 2,400 passes | 13-23 min |
| J4 | F2a + F2b, 3,600 passes | 19-34 min |
| J5 | F3, 2,400 passes | 13-23 min |
| J6 | F4, 1,200 passes | 6-11 min |
| J7 | F5, 1,800 passes | 10-17 min |

- The core (J1-J5) takes about 1.3-2.2 h; with J6-J7, about 1.5-2.7 h. Cost: $0.
- Stop after J2 if C0 fails.
- Code takes about 1 day: realblt_copy_latch.py, copied from realblt_copy_veto.py, plus the L8 mask, siv(), the template builders, the V4 split and per-type exact.

## DRAFT, pending PI approval: kill criteria K5-K9 (DESIGN.md section 5)

- **K5:** C0 fails. The hazard does not replicate on fresh problems; report it and stop.
- **K6:** the control fails: D.V8 shows computed ≥ -1, bits per byte ≤ +0.005 and final ≥ -1. No "no cost" claim from this test.
- **K7:** any of C1, C2, C3, C3f, C3c, C4 or C4s fails. No fix claim.
- **K8:** C1e is below +15. The PI's E10 target is not met.
  - If the result is at least 90% of the measured ceiling, record "ceiling-limited".
  - Otherwise the gate misses E10's onset starts.
- **K9:** G1 fails, so the fix is specific to `\nTicket `. If G3 fails for letters or mixed IDs, the claim is limited to digits.

## Dev step 1 results (2026-10-30; results/blt_copy_latch_step1.json, .log; exploratory)

The patcher ran on every dev text in 201 s. The stored outputs are in results/blt_copy_latch_patcher.npz, which is
git-ignored.

Texts: 295 + 293 quadrant, 297 + 294 equation and 660 xdomain documents. The equation train set has 297 texts, not
300: they are the first 300 train problems after the filter, exactly as O6 used them. So the GSM8K dev texts number
1,179, not 1,182.

**Checks** (before any BLT-1B scoring):
- *Causality:* 0 of 1.33 billion compared decisions changed. That covers every cut of every text, two perturbations
  each, for L8 and L8c.
- *Subset:* L8 ⊆ V_8 and L8c ⊆ L8 on every text.

**Reproduction of O6:**
- D has 799 starts at computed values and E10 has 129.
- V_8 removes 50 of D's 799, and 12.3 D starts per text outside the copy.
- E10 starts at c1 in 93 copies, and D in 289.
- The patcher's xdomain entropy equals the stored entropy on 660 of 660 documents.

**Go criteria. All pass; no kill rule fires, and L8 is kept (no switch to L8c).**

| criterion | result | prediction |
|---|---|---|
| (a) L8 fires at c1 | 288 of 293 copies, 98.3%; all 288 latched through c4 | 98.3% |
| (a) ... at E10's starts at c1 | 92 of 93, 98.9% | 98.9% |
| (b) new-value removals from E10 | 0 of 129 | 0 |
| (b) new-value removals from D | 1 of 799 (allowed: 2) | 0-2 |
| (c) D starts removed outside the copy | 0.26 per text (max 4) | 0.2-0.8 |

**Reported:**
- *(a) misses:* 4 copies fail agreement (the patcher's argmax at c1 is wrong) and 1 fails condition 6.
- *(b) the one removal from D:* `10560` after `15840`. This is one of the four [S5] cases: `... * (5280 feet/1 mile) = `
  follows `= 15840`. O6's strict count is also 1 at D and 0 at E10. L8 masks 1 of the 977 computed-value positions in
  all; L8c masks none.
- *V_8's 50 computed cases:*
  - 43 are stopped by condition 4 (the earlier value was predictable when first read);
  - 4 by condition 6, including `720` after `7200` and `1500` after `1000/month`;
  - 2 by condition 5;
  - 1 is removed (`10560`).
- *(c) where the 76 removed D starts fall:* 46 other, 16 repeated-word starts, 9 copied numbers, 3 new-word starts,
  2 computed values.
- *Ticket source entropy:* T is the ticket in 293 of 293 copies. Summed over it: minimum 12.49 bits, 1st percentile
  14.16, median 16.55.
- *Threshold shift at D's rate:* entl8's threshold is 1.3125 nats, 4 bf16 steps below the entropy's 1.34375. The
  prediction (at most 2 steps) was wrong. At 10% the shift is 0 steps; only the tie fraction changes (0.90 vs 0.55).
  For L8c the shift is 2 steps.
- *L8c:* it fires at c1 in 40.3% of copies but at 98.9% of E10's c1 starts. 119 copies have h(c1) ≥ log2 10.
- *xdomain E10 cost* (net bits per byte, [S2]'s method, all 660 documents; [S2]'s upper and oracle rows reproduce
  exactly with this code):

| domain | L8, real argmax | upper | oracle | L8 E10 starts masked per kB |
|---|---|---|---|---|
| math | +0.00003 | +0.00016 | -0.00032 | 0.25 |
| algebra | +0.00014 | +0.00053 | -0.00006 | 0.12 |
| code | +0.00032 | +0.00086 | -0.00030 | 1.57 |
| prose | -0.00030 | +0.00106 | +0.00009 | 0.53 |
| clinical | +0.00041 | +0.00099 | -0.00018 | 0.36 |
| traces | -0.00002 | -0.00003 | -0.00005 | 0.04 |

- The real mask's footprint lies between the upper and oracle masks' in all six domains. The net cost lies between
  them in four.
  - In prose it is below both (-0.00030).
  - In traces it is 0.00001 above the upper row; every traces row is within 0.0001 of zero.
- On the 360 test documents alone, the largest net cost is +0.00078 (math).
- In the math domain, L8 masks 0 of 425 computed-value positions and 0 of 110 final answers. It masks 6 of 971
  copied-number starts.

**Corrections and additions (2026-10-30, after the independent check at 6e71573).** The numbers are from
results/blt_copy_latch_devcheck.json (realblt_copy_latch_fresh.py devcheck at d84d097; stored outputs only, no model;
exploratory).
- *Causality, restated.* 0 changes. The 1,327,362,096 comparisons re-check each decision at every later cut. There
  are 2,098,222 distinct L8 and L8c decisions, and 4,181,732 comparisons at the boundary that matters: the cut at the
  byte just before the decided token, two perturbations, both rules. A planted read of the output at token t is
  caught (25 changed decisions on 20 texts). The check takes the stored outputs as given; their alignment holds
  separately (the argmax is the actual byte at all 318,406 positions under 0.01 bits; fidelity audit).
- *(a) reproduces [S1].* The predicted 98.3% and 98.9% were [S1]'s values on the same stored outputs (see the note
  under the header), and E10-L8's +15.7 [+11.6, +19.8] equals [S1]'s E10 figure.
- *Equation train texts.* The Implementation section's "300 train" should read 297: the first 300 train problems
  after realblt_budget.problem's filter, exactly O6's set (E10's 129 computed-value starts match O6's).
- *Smoke runs.* Before the full step 1, smoke runs on dev subsets ran at 00:53:01 -0300 (step 1: quad_train 30,
  quad_test 6, eq_train 30, eq_test 6, xd 12) and at 00:54:12 (steps 2-3 on 6 test texts). Their outputs were deleted
  before the full runs, and their logs survive only in a scratch folder. The only later code edit was a NaN guard in
  step 3's summary, 2 s before the full step 1 started. Nothing was tuned on them.
- *Whole-number recount.* The confirmatory rule judges "new value" on whole numbers, so `720` after `7200` is new
  (O6's test, used above, only checks that the earlier number starts with the value). Under it every L8 count above
  is unchanged: D 1 of 799 (`10560`), E10 0 of 129; on the 297 equation train texts, 0 of 890 and 0 of 123. V_8's 50
  removals at D split 46 new and 4 same, not 45 and 5. No go criterion, stop rule or decision changes, and the L8c
  trigger stays unmet. The dev rule's "0.25% (more than 2 of 799)" disagrees with itself at exactly 2 (2/799 =
  0.2503%); the code used the integer, 2, and that governed.
- *xdomain.* The real mask's footprint lies between the upper and oracle masks' in five of six domains, not six: in
  algebra it is 0.00247, below the oracle's 0.00254. E10 starts masked per kB lie between them in all six. D starts
  masked per kB, counted with the stored D starts rather than a float threshold: math 0.56, algebra 0.40, code 2.29,
  prose 0.64, clinical 0.54, traces 0.06. In code that is about 1.8 D starts per document. The 0.26 per text in (c)
  refers to the quadrant texts.
- *A mislabelled field.* In step1.json, the "how" field of b["D-V8"].cases says "latched" for V_8 removals that L8 did
  not mask. Only D-L8's cases are labelled correctly; V8_computed_cases gives each case's L8 outcome.
- *The fallback.* L8c could not have passed K4's D bar: it fires at c1 in 40.3% of copies, and DESIGN.md predicted
  D +29.7 for it. So a switch would in effect have ended at K4. The confirmatory rule has no fallback.

## Dev step 2 results (2026-10-30; results/blt_copy_latch_step2.json, .log; exploratory)

BLT-1B on the 293 quadrant test texts, about 9.5 min of scoring.

**Checks.**
- The patcher, rerun, equals the stored outputs on 293 of 293 texts. The thresholds equal step 1's; for the entropy
  and V_8 they also equal O6's.
- G0 is met: D and E10 reproduce O6's stored rows. id_copy exact is equal on 293 of 293, and bits are within 0.05 on
  100%.
- D-fit equals D on 293 of 293 problems. The refitted D-rate threshold, 1.34375 nats, reproduces the patcher's own
  layout.

**Exact match by unit kind** (patch rate; bits per byte without the copy's digits; %):

| layout | rate | bpb | id_copy | computed | copied | word_new | word_rep | final |
|---|---|---|---|---|---|---|---|---|
| D (default) | 0.274 | 0.8140 | 14.0 | 78.0 | 64.4 | 34.5 | 74.5 | 72.4 |
| D.L8 (matched) | 0.273 | 0.8147 | **89.4** | 78.3 | 64.9 | 34.8 | 74.6 | 72.7 |
| D-L8 (no refill) | 0.268 | 0.8161 | 88.4 | 78.0 | 64.4 | 34.7 | 74.5 | 73.0 |
| D.SIV (matched) | 0.273 | 0.8148 | 90.8 | 77.9 | 64.8 | 34.7 | 74.4 | 73.4 |
| E10 | 0.107 | 1.0237 | 54.6 | 18.2 | 35.4 | 25.3 | 66.8 | 32.8 |
| E10.L8 (matched) | 0.107 | 1.0227 | **69.6** | 18.2 | 35.5 | 25.4 | 66.8 | 33.8 |
| E10-L8 (no refill) | 0.106 | 1.0247 | 70.3 | 18.2 | 35.4 | 25.3 | 66.8 | 33.1 |
| E10.SIV (matched) | 0.107 | 1.0224 | 70.0 | 18.3 | 35.5 | 24.9 | 66.9 | 34.1 |

**K4's bars all hold, so no kill rule fires.** Differences are against the unlatched layout at the matched rate,
with 95% intervals:

| bar | D.L8 - D | E10.L8 - E10 | prediction (section 3) |
|---|---|---|---|
| id_copy exact (≥ +60; ≥ +12) | **+75.4 [+70.3, +80.2]** | **+15.0 [+10.9, +19.5]** | +70 to +78; +12 to +19 |
| C2, bits per byte without the copy (≤ +0.01) | +0.0006 [+0.0000, +0.0012] | -0.0010 [-0.0018, -0.0003] | 0 to +0.002 |
| C3, computed (≥ -2) | +0.3 [-0.7, +1.1] | +0.0 [-0.4, +0.4] | 0 ± 0.5 |
| C3f, final (≥ -2) | +0.3 [-1.0, +1.7] | +1.0 [-1.4, +3.4] | 0 ± 2 |
| C3c, copied (≥ -2) | +0.5 [+0.0, +1.0] | +0.1 [-0.3, +0.6] | 0 to -0.5 |
| rate (within 0.005) | -0.0008 | +0.0001 | within 0.002 |

- **Predictions.** The id_copy gains, computed values, final answers and D's bits per byte fell inside their
  predicted ranges. Three results fell outside, each on the side of no cost:
  - copied numbers were slightly better at both rates;
  - E10's bits per byte were slightly lower.
- **Reported without a verdict** (wording corrected 2026-10-30 after the check at 6e71573: "ceiling" became
  "onset-removal bound (no refill)", and "no cost" means no cost beyond the bars):
  - *The E10 onset-removal bound (no refill).* E10.L8, at the matched rate, reaches 0.96 of it (44 of 46 copies).
    - Without refill, E10-L8 gains +15.7 [+11.6, +19.8]. That equals the quadrant's E10 without its onset start.
      It is not a ceiling at 10%: O6's matched E10.V6 gained +21.2 [+15.4, +27.0] on the same texts.
    - The refill at the matched rate gives back 0.7 points net (2 copies: it fixed 2 and broke 4). (At 6e71573 this
      read "0.7 points (2 copies)"; "net" and ": it fixed 2 and broke 4" were added at 7be1185. Corrected after the
      third check; this note said the whole parenthesis was added.)
    - Against the PI's +15 target, E10.L8 sits exactly on it (44 copies, +15.02).
  - *Without refill,* D-L8 gains +74.4 and costs +0.0020 bits per byte, at a rate 0.006 lower.
  - *SIV ties L8 on this template,* as section 2 predicted: D.SIV +76.8, E10.SIV +15.4, with no cost beyond the
    bars. Paired, L8 minus SIV is -1.37 [-3.07, 0.00] at D and -0.34 [-1.71, +0.68] at E10 on id_copy exact, and
    within 0.7 points on every cost row (devcheck, part I). (The paired numbers were added 2026-10-30, after 6e71573.)
  - *Removed starts:*
    - D-L8 removes 284 onset and 448 inside starts at the copy, and 76 elsewhere: 46 other, 16 repeated words,
      9 copied numbers, 3 new words, 2 computed values.
    - E10-L8 removes 92 onset starts and 1 inside start, and 33 elsewhere.
  - *Against O6's V8 at the same rate* (O6 step 2): D.V8 gained +70.6 but cost computed -5.6, final -4.4, copied
    -4.6 and bits per byte +0.0196. L8 keeps the copy gain and shows none of those costs on these texts.
- **Reading.** On the dev texts, L8 restores the copy at a matched rate at both BLT-1B's default rate and
  entropy@10%. Every cost stays far inside its bar; the largest is +0.0006 bits per byte at D, whose interval
  [+0.00001, +0.00124] excludes zero. Dev passes. (Corrected 2026-10-30, after 6e71573; this sentence read "at D
  (interval [+0.0000, +0.0012])".)
  - The claim still needs fresh problems: L8 was designed on these texts.
  - The confirmatory test is drafted in followup/O6b_CONFIRMATORY_DRAFT.md and waits for the PI.

## Dev step 3 results (2026-10-30; results/blt_copy_latch_step3.json, .log; exploratory, reported only)

Steps 1 and 2 passed, so step 3 ran. It does not change L8.

E10.V8 reproduces O6's stored rows: id_copy exact is equal on 293 of 293, and bits are within 0.05 on all. (step3()
does not compute this; realblt_copy_latch_fresh.py devcheck, part F, computes and stores it. This parenthesis was
added 2026-10-30, after 6e71573.)

E10.V8+k is E10.V8 with one more start, at token a-3, the `k` of the copy's `\nTicket `:

| layout | rate | id_copy exact | with an E10 start at c1 (93) | without one (200) | final |
|---|---|---|---|---|---|
| E10 | 0.107 | 54.6 | 30.1 | 66.0 | 32.8 |
| E10.V8 | 0.108 | 71.3 | 78.5 | 68.0 | 34.5 |
| E10.V8+k | 0.110 | 47.8 | 53.8 | 45.0 | 36.9 |

- E10.V8+k - E10.V8 = **-23.5 [-29.4, -17.7]** on id_copy exact. Computed values, copied numbers and words are
  unchanged; final answers move +2.4 [-0.7, +5.5].
- **Reading.** The anchored design's premise fails here. A start at the `k` before the copy breaks copies, in copies
  with and without an E10 onset start alike.
- *Corrected 2026-10-30:* this does not show that nothing goes past the onset-removal bound (no refill) at 10%. O6's
  matched E10.V6 went past it by +5.5 [+0.7, +10.2], through its refill.
  - The reading first said (4bd2a7a): "A start at the `k` before the copy does not lift copies past the E10 ceiling;
    it breaks them, in copies with and without an E10 onset start alike. That leaves no lever past E10's ceiling among
    those tested, so choice 1 in the confirmatory draft matters." Both claims about the ceiling were removed at
    7be1185; they are quoted here so the correction keeps the original (added 2026-10-30, after the second check).

---

## Confirmatory rule, filed (2026-10-31)

**The PI's approval** (2026-10-31). The PI approved the rule for filing after a summary of the rule: what runs, the
bars, what each result means, the chances, and every place the rule departs from the options the PI had chosen. The approved text is this section at 4d295a0, from its heading to the end of
the file, with the code pinned at 4026bec.

**Status: filed and approved on 2026-10-31.** Written 2026-10-30, after dev steps 1-3 and the independent check at
6e71573 (analysis/2026-10-30_copyfix/verify/VERDICT.md and audits.json). Amended the same day after the dev smoke run
(ecca81e) and a second independent check at 67667ac (analysis/2026-10-30_copyfix/final_check/VERDICT.md and
audits.json); part C of the change log lists those post-smoke amendments. A third check at 0646d55
(analysis/2026-10-30_copyfix/recheck/VERDICT.md and audits.json) led to the amendments in part E, a fourth check at
9fe860b (analysis/2026-10-30_copyfix/fourth_check/VERDICT.md and audits.json) to those in part F, a fifth at 274bafb
(analysis/2026-10-30_copyfix/fifth_check/VERDICT.md and audits.json), on 2026-10-31, to those in part G, and a final
confirmation at 920af7e (analysis/2026-10-30_copyfix/final_confirm/VERDICT.md and audits.json) to those in part H. The PI's choices are applied here:
- 10% bar: the PI chose a share of the bound. L8 must recover at least 90% of what removing every copy start would gain
  (both measured without refill), and still show a gain above zero at the matched rate.
  - Since then (2026-10-31, part H): if E10's own hazard (E10-all - E10) has a lower bound ≤ 0, the 90% bar cannot be
    tested, and the fix claim is made at the default rate only.
- New values: the PI chose at most 4 of about 1,700 (0.25%), counted as whole numbers, and none at the 10% budget.
  - Since then (2026-10-31, part H): the earlier chance estimate of about 0.93 is C4's chance at the default rate
    alone, and "none at the 10% budget" is K2, a stop before any scoring with its own risk (0.7-5%; 14% or 30%
    pessimistic).
- Scope: the PI chose all cells (other wording, letter and mixed IDs, distant copies and the released model), with the
  byte-only rule SIV+PO scored beside L8 under a paired verdict fixed in advance.
  - Since then (2026-10-31, part G): no outcome settles whether reading the model is needed ("No outcome of this test
    shows that reading the model is needed", Claims), and the Mac time is 2.3-3.9 h, with every job rerun after a
    re-filing (Jobs).
- Approval: the PI chose to see the amended rule and approve its final text before any fresh text was built.
  - Since then (2026-10-31, part H): every re-filing needs the PI's approval too, and after filing this section changes
    only by dated notes and re-filing commits (Filing).

The second check left two parts to the PI (VERDICT items 2 and 4). The PI decided both later on 2026-10-30 (applied in
part D of the change log):
- F2b: the PI chose to keep F2b, gated: if there is no hazard, F2b reports "not testable" instead of a false "fix is
  template-specific", and the broader claim then covers only the ID kinds (6-digit, letters, mixed). F2b's wording is
  unchanged, with G1's hazard gate, and no dev BLT-1B check was run.
  - Since then (2026-10-31, part H): every ID kind is gated as F2b is, and the broader claim most likely covers 6-digit
    and mixed IDs only (Who is likely testable).
- Paired verdict: the PI chose two verdicts, one for the windowed model and one for the released model, each with its
  meaning fixed in advance: a tie means a byte rule suffices here; a loss means L8 dropped copies; F5's win reads as
  "knowing when the model can't copy helps", not proof. No extra runs.
  - Since then (2026-10-31, part G): a tie shows only that, on the copy rows with the hazard, no difference reached
    the class rule (Claims), and the "SIV+PO better" readings split the lost copies into copies L8 missed and side
    effects of the refit.

None of these was approval to file or run. The PI approved this text, which was filed and approved on 2026-10-31
(above). Until then no fresh text was built, no GSM8K train problem from idx 1800 on was read, and no job J0-J7 ran on
fresh problems (dev smoke runs excepted; see the end of this section).

**Filing.** The O6 session makes the filing commit, and only after the main session relays the PI's approval in the
PI's words.
- The filing commit changes only this heading line and the Status lines, which then say filed and approved, with the
  date. In it this heading becomes `## Confirmatory rule, filed (YYYY-MM-DD)`, a line of its own, and the PI's words
  are quoted under it.
- Before the first job, the main session checks that `git diff <approved commit> <filing commit> --
  followup/O6b_COPY_LATCH_RULES.md` shows nothing else.
- After filing, this section changes only by dated notes appended at its end, apart from a re-filing commit's changes
  (below). None changes a bar, class rule, prediction, reading or claim, and results are read under the filed text.
  `verdict` prints the rules file's changes since the filing commit (`git diff --stat`), and after a re-filing since
  the earlier filing commit too; every output records the rules file's sha256.
- A re-filing commit (see J1) changes only: the Code section's pin line, edited in place to name the fix commit; the
  Status lines; and one new line `## Confirmatory rule, filed (YYYY-MM-DD)`, appended at the end of this section with
  the disclosure and the PI's words quoted under it. Before the first rerun job, the main session checks that `git
  diff <previous filing commit> <re-filing commit> -- followup/O6b_COPY_LATCH_RULES.md` shows only those and the
  dated notes appended since the previous filing. (Part G; it said "A re-filing (see J1) follows the same steps.")

(Added 2026-10-30, after the fourth check; the filing commit had changed only the heading, so the Status would still
have read "proposed".)

**What the code checks** (filed_check, provenance and gate; reworded 2026-10-30 after the third check and again after
the fourth to say exactly this). Every command that reads or scores the fresh pool refuses unless O6B_FILED names a
commit that:
- is an ancestor of HEAD and has one parent;
- introduced a filing heading line: a line `## Confirmatory rule, filed (YYYY-MM-DD)` outside code fences and HTML
  comments, with a valid date no earlier than 2026-10-30, that its parent's rules file lacks. So only a filing or
  re-filing commit qualifies, and a later commit does not;
- changes no covered code file and no submodule against its parent: it changes the text only (since part F);
- holds the covered code unchanged from the code commit its Code section pins, and unchanged since in the working
  tree (since part F). The covered code is the script and every repository module it imports. Modules in the
  segresearch submodule are compared with the submodule commit that commit records.

The code does not check the approval quote; the filing procedure above does. Nor does it tell a re-filing from a
later text-only commit that repeats or re-adds a heading line: such a commit counts as a re-filing, which needs the
PI's approval, and the failure count still spans every filing (part F, guard G6). Only implementation-check
failures are counted across filings in code; that K10, K2 and K5 also stop the test under any later filing is kept by
procedure (part G, guard GATE-3). filed_headings approximates
Markdown's fences, comments and line breaks; the main session's check of the filing diff covers what it could
misread (part F, guard G5). Every output records:
- the filing commit, HEAD, each code file's sha256 and the submodule's commit;
- the rules file's sha256;
- the model snapshot, the weights' blob name, the transformers, torch and numpy versions, modeling_blt.py's sha256,
  the window, the threshold mode (blt_layouts.MODE, which must be "train"; since part G) and the device.

The job gate is described under Jobs.

This section is self-contained apart from section 1's definition of L8 and the Implementation section's mask details,
both above in this file ("apart from" added after the fourth check, the Implementation section after the fifth). [JW0]
is analysis/2026-10-30_copyfix/judge_cost/out_w0check.txt, the cost judge's check of the released conversion on dev
texts (DESIGN.md:23). "VERDICT item N" in change-log part A, and in Dev numbers, is the first check's
(analysis/2026-10-30_copyfix/verify/VERDICT.md). It replaces the DRAFT sections above,
followup/O6b_CONFIRMATORY_DRAFT.md (both stay as history) and DESIGN.md section 6, both its "It would show" and its "It
would not show" (see Claims). The change log at the end lists the differences from DESIGN.md section 4 and K5-K9, with
their reasons. All dev numbers quoted here are exploratory, and both L8 and SIV+PO were designed on the dev texts.
(Until part E this paragraph limited the replacement of DESIGN.md section 6 to the two claims.)

### The rule and the comparators (frozen)

- **L8**, exactly as in section 1 of this file, with the Implementation section's mask details. Constants: k = 8,
  S0 = 10 bits, log2 10 and 1 bit. There is no fallback. L8c is counted in F6 but never switched to.
- **Matched layouts.** X.R is the entropy with rule R's tokens set to -inf. Its threshold is refitted on the cell's
  own fit texts (blt_layouts.Fit, train mode): at D's pooled train rate on those texts (D.R) and at 0.10 (E10.R). Each
  is paired with its unlatched layout, D or E10.
- **No-refill layouts.** X-all is X without any start at a byte of the copied string (tokens c1..cL). E10-L8 is E10
  without the starts L8 masks. Neither is refilled.
- **Positive control: D.V8**, O6's V_8, refitted the same way; F1 only.
- **Byte-only rival: SIV+PO**, the bytes designer's design 2: siv(kmin = 5) OR po(), from
  analysis/2026-10-30_copyfix/bytes/final_rules.py, unchanged. D.SIVPO and E10.SIVPO are refitted the same way as L8,
  in every cell and at every rate where L8 is scored (F5: D only; this clause was "in every cell" until 2026-10-30,
  after the second check). PO, like L8, was designed on the dev texts; DESIGN.md:123 dropped SIV+PO for that reason,
  which applies equally to L8.
- **D.SIV** (design 1) stays in F2b only, for G1s.
- **L8b** is L8 without its patcher conditions 4-6: conditions 1-3 and the same latch. It is an ablation: its D starts
  removed are counted in F1 and F6, and its firing at c1 is reported in every cell. (This scope sentence was "counted
  in F6 and F1 only" until 2026-10-30, after the second check.)

### Data

**Pool.** GSM8K train problems from idx 1800 on, in file order: ~/.cache/segresearch-gsm8k/train.jsonl, 7,473 lines,
so 5,673 problems.

**Who has touched the pool before this test** (corrected audit, 2026-10-30):
- *Problems kept for scoring, fitting or few-shot use.* Every reader parses and keeps only problems at idx 1799 or
  below. Some first load the whole file into memory and slice it (`list(open(path))[:n]`: realblt_quadrant.py:103,
  realblt_copy_latch.py:171 and :177, realblt_copy_veto.py:99 and :172, realblt_copy_gate.py:50, realblt_router.py:51);
  they parse nothing past n (reworded 2026-10-30 after the second check, which found these whole-file loads):
  - O6, the quadrant and the copy scripts read 0-299: realblt_quadrant, realblt_router, quadrant_analysis,
    realblt_copy_gate, realblt_copy_veto, realblt_copy_latch, and realblt_copy_latch_fresh's selftest, devcheck and
    smoke runs;
  - the other BLT scripts read their first n_train or N problems, 300 by default: in this repository
    realblt_operand_mask, realblt_window_sweep, realblt_compress (fit), realblt_waste, realblt_hybrid and
    realblt_cues; in segresearch realblt_budget, realblt_typed, realblt_gain, realblt_general, realblt_combo (fit) and
    blt_fig1_dump;
  - segresearch's blt_screens read 0-399 (dependence_screen) and 0-499 (difficulty_screen); q3_branching and
    q3_branching_mlx read the first 4;
  - R9 reads 0-1499, through data/prose_rewrites/train.jsonl. So do the scripts that read R9's files: realblt_learned,
    realblt_cue_budget, realblt_matched_share, realblt_autolabel, realblt_forced_share, realblt_probe_reads,
    realblt_profiles, realblt_second_digit and patcher_variants;
  - the design's scratch scripts read 0-299, 300-899 and 300-1799;
  - xdomain math keeps fit documents up to idx 66 and its pool up to idx 432.
- *The L8 work's one read of idx 1800 on:* analysis/2026-10-30_copyfix/synth/s3_fresh.py ([S3]: lines 6 and 12-20).
  - It ran once, before 4b76f51. It built O6's template with a fixed ticket 4821 on every fresh problem and printed
    counts and source-to-copy distances: 5,056 kept, and 1,537 at 550-950 bytes.
  - It ran no model, saw no outcome and selected nothing.
  - Its counts are rough. It used an ASCII-only filter instead of O6's tokenizer filter (about 10% low on dev), and a
    template that differs from F4's.
  - F4's far window predates it: 550-900 bytes in analysis/2026-10-29/opportunities.md:48, :55 and :57, and 550-950
    in s3_fresh.py:23 as written before its single run. s3's output did not set the bound. J0 records each block's
    visited and kept problems, not the pool-wide counts s3 estimated ("the exact counts" until part F). (Corrected
    2026-10-30 after the second check; this line said "s3's single run moved the upper bound to 950".)
- *Whole-file reads that select nothing:*
  - realblt_compress.py:44-52 and segresearch realblt_combo.py:52-60, the novelty tables;
  - xdomain_data.py:69, which formats every problem and keeps only the first ones (above).
- *LoRA fine-tuning of BLT-1B itself* (segresearch realblt_finetune.py:59, :78-84; audit FR1).
  - 18 runs of 1,500 steps (3 rules × seeds 0-2 × windows 0 and 512, 2026-10-02 to 10-08) drew training problems at
    random from train idx 300 on. The runs share three draw sequences, one per seed.
  - Replaying the draws from indices alone gives about 2,650 distinct fresh problems, 47% of the pool.
  - This does not threaten validity. The test loads the base weights (blt_light.load of itazap/blt-1b-hf) and never
    the adapters. The patcher was never trained, and those runs scored only GSM8K test problems.
  - realblt_e2e.py reads train[0] and train[:n_train] and loads those adapters.
- *realblt_fullft.py (Study B)* fully fine-tunes BLT-1B, except the patcher, on draws from the same pool (lines 40, 92,
  101-110).
  - No full run was found: there is no output directory (~/.cache/segresearch-blt/fullft, its OUT), no registry file,
    and no job script that calls it. It needs a GPU with about 25 GB.
  - Its 2026-10-13 commit mentions a toy dry run on the Mac (one global layer, a few updates of 2 problems). That run
    may have drawn a few dozen problems; nothing of it is on disk.
  - Either way, the test loads the base weights.
- *Derived caches* (audit FR3; the grep on the jsonl path could not find them).
  - realtext.prepare_cache writes all 7,473 train problems, in file order, to ~/.cache/segresearch-gsm8k/gsm8k.npz.
  - mathexp.prepare_cache writes a seed-1 interleave of all of them with MATH train to
    ~/.cache/segresearch-math/math.npz. segresearch-reason/math.npz holds no GSM8K.
  - Models and tables trained or fitted on these streams include:
    - paper 1's small models;
    - Study A's 12M-145M models: their corpus file math_train.u8 is built from math.npz. Study A's results were not
      opened for this audit;
    - R2 and R13;
    - NCM;
    - deptrigger, novtab and the neural patcher;
    - novtab_gsm.
  - Some of these select by position in the shuffled stream: a last-2 MB slice, a first 2 MB or 4 MB, or random
    windows. So their selections include problems from idx 1800 on.
  - test_causal.py checks rule masks on random 1,024-byte windows over the whole train stream: gsm8k.npz in file
    order, and math.npz shuffled. Its thresholds come from the first 400 kB (idx 0-759) and the first 300 kB. It
    checks masks only and runs no model. This corrects the draft's row, "the first 400 kB of train bytes".
  - diag_repetition.py scores 145M checkpoints on the first bytes of the shuffled math_train.u8.
  - None of these enters this test (realblt_copy_latch_fresh.py's imports), and none selects problems for scoring by
    GSM8K index.
- *GSM8K test.* Nothing there is fresh:
  - idx 0-299 went to O6, the quadrant and R9;
  - idx 300-1318 went to R20 and R22;
  - reducible2.py ran BLT-1B on 1,292 of the 1,319 test problems;
  - paper 1 used test[:660] for validation.
- **So "fresh" means** never selected by GSM8K index for scoring, threshold fitting or design in this project, and
  never scored by BLT-1B in an evaluation. (Corrected 2026-10-30 after the second check. It said "never scored by any
  model in this project", which the rows above contradict: some fresh problems lie inside the shuffled streams of other
  models. "Threshold" was added after the third check, since the LoRA runs drew training examples at random.)
  - About half of the pool was sampled as LoRA training examples for BLT-1B, which this test does not load.
  - BLT-1B's own pretraining exposure to GSM8K cannot be audited. Dev scored GSM8K test-split problems, and the fresh
    pool is the train split. The stored dev patcher outputs show no split difference: D starts at c1 in 98.3% of
    train-split F1 texts and 98.6% of test-split ones; L8 fires at c1 in 99.0% and 98.3%; the entropy model's bits per
    byte on solution bytes are 1.141 and 1.164, a difference of -0.023 [-0.072, +0.025] (third check, critic). The main
    model cannot be compared without scoring. (This line said "it applies to dev and fresh problems alike" until part
    E.)
  - This corrects DESIGN.md:286, which said "about 600 problems that no earlier run, designer or judge used".

**Blocks (J0, before any model run).**
- Cells take blocks in the order F1, F2a, F2b, F3, F4. Each block starts after the last problem the previous block
  visited.
- A block is the next fit + test problems, in file order, whose text in the cell's own template passes O6's filter: at
  most 1,000 tokens, one token per byte plus BOS. Rejected problems are skipped and keep their draws.
- F4 also needs a distance of 550-950 bytes, from the source ID's first byte to the copy's, measured on F4's own
  template.
- Within a block, the first problems kept are the fit texts; the rest are the test texts.

| cell | fit | test | seed |
|---|---|---|---|
| F1 (F6 uses the same problems) | 300 | 600 | 6101 |
| F2a | 300 | 300 | 6102 |
| F2b | 300 | 300 | 6103 |
| F3 | 300 | 400 | 6104 |
| F4 | 300 | 200 | 6105 |

- **IDs** are drawn with the cell's seed, in problem order, and redrawn if they occur in the problem.
  - F1, F2a, F2b and F4 use 4 digits, 1000-9999. F1's texts are realblt_quadrant.build's.
  - F3 cycles four kinds over its fit texts (75 each) and over its test texts (100 each): 6 digits (100000-999999);
    4 upper-case letters; mixed, two letters, a digit, a dash and four digits (like `QX7-4821`); and doses.
- **The 12 doses (mg):** 5, 10, 20, 25, 40, 50, 100, 200, 250, 325, 500, 1000. A dose whose digits occur in the
  problem, not flanked by other digits, is not drawn; so `5.5` excludes 5, and `1,000` does not exclude 1000. If all
  12 are excluded, the problem is skipped. (Reworded 2026-10-30 to the code's test, after the second check; it said
  "a dose that occurs in the problem as a whole number".) The question, solution and answer are tested apart. Joined,
  a solution ending `$25` and the answer `25` read `$2525` and let dose 25 through (train idx 71; 15 of train idx
  0-1799 hide a dose this way). Fixed in the code on 2026-10-30 after the third check; the smoke's dose problems were
  not affected. So about 0.15 of the 175 fresh dose texts would have drawn a dose that occurs in the problem. Because
  the shorter list maps the same draw to another dose, the fix changes the dose in about 0.6 expected fresh dose
  texts, all in such problems; no other text, draw, generator state or kept set changes (added after the fourth
  check, builders B1 and numbers N4).
- The rule, bars and predictions are filed before any confirmatory text is built.
- J0 writes results/blt_copy_latch_fresh_blocks.json: every block's indices, IDs, kinds, distances and text hashes,
  plus F6's byte-level counts (computed-value positions, and V_8, SIV, PO, SIV+PO and L8b at them). It is committed
  before J1, so the texts and those counts are fixed before any model run. J1 rebuilds the texts and checks them
  against it.
- **Pool size.** F1-F3 keep 2,800 problems. In F4's template, 29% of train problems idx 0-1799 fall in its window
  (526 of 1,800; third check, critic), so F4's 500 should need about 1,700 problems, and J0 should visit about 4,600 of
  the 5,673. If the pool runs out, J0 stops and nothing runs. (Until part E this said 36% of dev test problems, 105 of
  293, and about 1,400.)

### Cells and layouts

| cell | text | layouts |
|---|---|---|
| F1 | O6's template unchanged, fresh problems, new tickets | D, D-all, D.L8, D.V8, D.SIVPO, E10, E10-all, E10.L8, E10-L8, E10.V8, E10.SIVPO |
| F6 | F1's problems in realblt_budget.problem's format, no ticket. Counts (J1): L8, L8c, V_8, SIV, PO, SIV+PO, L8b | scored: D, D.L8, D.SIVPO, E10, E10.L8, E10.SIVPO |
| F2a | the context does not recur: source `Ticket NNNN. `, copy line `\nIt was logged as #NNNN and closed.` | D, D-all, D.L8, D.SIVPO, E10, E10.L8, E10.SIVPO |
| F2b | the context recurs in other words, mid-line: source `Order reference NNNN was opened. `, copy `\nWe shipped order reference NNNN today.` | D, D-all, D.L8, D.SIV, D.SIVPO, E10, E10-all, E10.L8, E10.SIVPO |
| F3 | F1's template with 6-digit, letter and mixed IDs; doses: source `Dose ordered: X mg. `, copy `\nDose ordered: X mg given.` | D, D-all, D.L8, D.SIVPO, E10, E10-all, E10.L8, E10.SIVPO |
| F4 (secondary) | far: `Notes:\nTicket NNNN opened.\n` before the question; copy line as in F1 | as F3 |
| F5 (secondary) | F1's 600 test texts under the conversion as released (window 0). The patcher outputs, L8, D and the thresholds are recomputed from that model, fitted on F1's fit texts | D, D-all, D.L8, D.SIVPO |

- **Exact** means every byte of the copied string is right under teacher-forced argmax. **Bits** is their summed
  -log2 p (O6:24-25). Both are reported per ID kind in F3.
- Other unit kinds are labelled as realblt_quadrant.build labels them. Bytes of an ID belong to no other unit.
- Readouts and pairs follow O6 step 2: a paired bootstrap over problems, with 2,000 resamples and seed 6
  (realblt_copy_veto.boot_ratio). Every interval is the 2.5th-97.5th percentile of those resamples, and every "lower
  bound" is the 2.5th percentile.
- Bars are read on point estimates unless they name a lower bound.

### Before any scoring (J1, patcher only)

On every fresh text, fit and test, in every cell:
- **Implementation checks.**
  - Causality of L8 and L8c: realblt_copy_latch.causal_changes, at every cut, with two perturbations.
  - L8 ⊆ V_8, L8c ⊆ L8 and L8 ⊆ L8b.
  - Alignment: wherever the actual byte costs under 0.01 bits, the patcher's argmax is that byte.
  - G2 (a), on F2a's test texts: L8 fires at c1 on at most 5% of copies. It is decided by construction (L8 ⊆ V_8, and
    V_8 cannot fire at F2a's c1), so a failure is a bug. (Moved here from J4 on 2026-10-30, after the third check.)
  - A failure is a bug, not a result. J1 then writes only its checks and exits, before R1, R2, the F6 counts, C4 or K2
    are computed. The failed output is committed and pushed before the fix commit and the re-filing commit; committed
    after them, it would count as a run under the re-filing and block the redo (part G, guard GATE-4). A failed subset
    assert ends J1 earlier, with no output; that counts as a crash (below). By procedure it also counts toward the
    two-failure stop, which the gate cannot see for it: the crashes it causes under one filing count as one failure
    of J1's checks (part H). A second failure of the same job's checks stops the test, whether or not a re-filing came
    between; the gate counts the failed outputs in the history (since part F).
  - **The redo** (defined 2026-10-30 after the third check, when it said "the fix is disclosed, and J1 is redone
    once"; made two commits after the fourth check). It takes two commits, as every re-filing does:
    - the fix, in a commit of its own;
    - then a re-filing commit that leaves the code alone. It moves the Code section's pin to the fix (the pin line
      edited in place) and carries a new dated heading line `## Confirmatory rule, filed (YYYY-MM-DD)` (appended at the
      end of this section), with the disclosure and the PI's approval quoted under it. Every re-filing needs the PI's
      approval, relayed by the main session in the PI's words, and the O6 session makes the commit, as for the filing.

    A re-filing changes only the code its disclosure names: no bar, class rule, prediction, constant or reading.
    Under it J0 is rerun and committed, and then every job runs again once, in order.
    - (a) The earlier filing's block file and its J1-J7 and `verdict` outputs stay in the repository's history as
      committed and pushed. None is amended, rebased, force-pushed or deleted. The reruns commit new outputs at the
      same paths. The patcher outputs are git-ignored; only their sha256, in the earlier J1 output, survives.
    - (b) The verdict reports, beside the result, every bar, claim, class, kill or stop that differs from the
      earlier filing's output of the same job. It reads that output from the re-filing commit's own tree, keeping only
      the outputs written under the filing the earlier J1 records (since part G).
    - The code reads only outputs written under the current filing, so no output from an earlier filing is mixed
      in: J3-J7 wait for the re-filed J2, and `verdict` needs J2-J7 under the current filing unless a stop applies.
    - If the PI declines a re-filing: after a J1 failure, the result is "The implementation failed; nothing was
      tested." (`verdict` prints that sentence itself only after a second failure; after a first one it prints "not
      reached: the implementation check failed; a redo needs the PI's approval"). After a J7 failure or any other
      bug, the filed result stands with no deviation beside it; after a J7 failure F5 is unscored there.
  - **Which result counts** (added after the fourth check; rewritten after the fifth, part G; completed after the
    final confirmation, part H).
    - After a J1 check failure (the defined redo: J1 exits before any decision-relevant number), the re-filed chain
      is the result, with the differences beside it. A bug found before J1 has written any output (in J0, or J1
      crashing twice) is handled the same way.
    - From the moment J1 has written an output that passed its checks (R1, R2, C4 and K2 then exist), a bug counts as
      found once scoring has started. After a J7 check failure, or any other bug found once scoring has started, the
      filed code's outputs remain the confirmatory result, and the filed chain is first run to its end under the filed
      code, as far as it runs. After a J7 failure, F5 stays unscored there and the released verdict is "not reached".
      A corrected rerun needs a re-filing the PI approves. It is reported beside the filed result as a deviation,
      never in its place, and `verdict` prints it under that label.
    - There is at most one corrected rerun, and it is never itself re-filed. After a check failure, a second crash or
      a bug inside it, it is reported as far as it ran, beside the filed result, which stands. No redo follows, though
      its own lines may say one needs the PI's approval. (`verdict` compares only with the filing just before, so it
      would label a second re-filing wrongly. This is kept by procedure, since every re-filing needs the PI's
      approval.)
    - `verdict` prints "Confirmatory result (filing <commit>)" above the result's sentences. After a re-filing that is
      not a J1 redo, those are the earlier filing's, and the re-filed ones follow under "Deviation: corrected rerun
      (re-filing <commit>)"; after a J1 redo it prints "Result: re-filed chain (J1 redo)" instead. Its JSON's
      top-level claims are always the current chain's; `result` says which counts.
    - `verdict` reads the earlier chain with the re-filed code. That reading equals the earlier filing's own committed
      `verdict` output unless the fix changed the code that reads the outputs (verdict_from, paired_verdict, sentences
      and the functions they call). Before reporting, the main session compares the two. Where they differ, the
      earlier filing's committed `verdict` output is the confirmatory result, and the re-filed reading is reported
      with the deviation. If the earlier filing has no `verdict` output, the re-filed reading is the only reading of
      its outputs and is reported as that. This is kept by procedure.
    - J7's exit message says that the filing's outputs remain the confirmatory result. That holds under the filing;
      under a corrected rerun, `verdict`'s label governs (part H).
    - (Until part G a J7 check failure also earned a redo whose re-filed chain counted, though J7 runs after F1's
      rows are scored and read.)
    - A job that crashes before writing its output reruns once under the same filing; a second crash is a bug. The
      gate cannot see a crash that wrote nothing, so this is kept by procedure, as the approval quote is.
    - A lost patcher file is recomputed by hand with the pinned code, since no command recomputes it, and the
      recompute is logged in a dated note. It is accepted only if its sha256 equals J1's record; otherwise the test
      stops (part G).
  - J1 refuses a block file that is not committed and unmodified, or that was written under another filing commit.
- **R1, recall** (F1 test texts): L8 fires at c1 in at least 95% of copies, and in at least 95% of the copies where
  E10 starts at c1. If no F1 copy has an E10 start at c1, that half is "not testable" and does not fail R1 (added
  2026-10-30, after the third check; P ≈ 0).
- **R2, footprint** (F1 test texts): L8 removes at most 1.0 D start per text outside c1..c4.
- **C4 and K2** (F6 test texts): the counts below.
- **Stops.** A failed check stops the test, apart from the one redo above. R1 or R2 failing is K10. Any new-value
  removal from E10 is K2. No BLT-1B scoring follows a stop: J2-J7 refuse to start after one.

### Verdicts

Each row names its cell and its metric; "exact" rows are in points, and every lower bound is the bootstrap's 2.5th
percentile (above). Cells, metrics and the G1 gate were written out on 2026-10-30, after the second check (part C).

| id | cell: readout (metric) | bar | prediction (fresh) |
|---|---|---|---|
| C0 (stop) | F1 test texts: D-all - D (id_copy exact) | ≥ +20, lower bound ≥ +10 | +76 (+70 to +81) |
| C1 | F1: D.L8 - D (id_copy exact) | ≥ +15, lower bound > 0 | +75 (+68 to +79) |
| C1e | F1 (id_copy exact): (i) E10-L8 - E10 ≥ 0.9 × (E10-all - E10), point estimates, both without refill; (ii) E10.L8 - E10 at the matched rate | (i) and (ii): (ii)'s lower bound > 0. Untestable if E10-all - E10 has lower bound ≤ 0. (i) is exact on counts: an exact 90% share (10 k1 = 9 k2) passes | (i) ratio 1.0 (0.9 to 1.1); (ii) +15 (+11 to +19); E10-all - E10 +15.7 (+11 to +20) |
| C2 | F1: D.L8 - D and E10.L8 - E10 (bits per byte without the copy) | ≤ +0.01 | D 0 to +0.002; E10 -0.002 to +0.002 |
| C3 | F1: D.L8 - D and E10.L8 - E10 (computed values exact) | ≥ -2 | 0 ± 1 (a fresh estimate spreads about ±0.7 around dev's +0.3 with dev as truth, about ±1.2 with dev's uncertainty; part F) |
| C3f | F1: both rates (final answers exact) | ≥ -2 | D 0 ± 1.5; E10 +1 ± 2 |
| C3c | F1: both rates (copied numbers exact) | ≥ -2 | -0.5 to +1 |
| rate | F1 and F6: D.L8 - D and E10.L8 - E10 (test patch rate) | within 0.005 | within 0.002 |
| R1 | J1, F1 test texts; see above | at least 95% and 95% | 97-100%; 96-100% |
| R2 | J1, F1 test texts; see above | ≤ 1.0 | 0.2-0.4 |
| C4 | F6 test texts: L8's new-value removals of computed-value starts (whole-number test) | at most 4 from D; 0 from E10 (K2). O6's strict count reported | D 0-2 (about 1 expected); E10 0 |
| C4s | F6 scored: D.L8 - D and E10.L8 - E10 (computed values exact) | ≥ -2 at both rates (at D only when C1e is untestable; stated in part F) | 0 ± 0.7 with dev as truth, about 0 ± 1.2 with dev's uncertainty (0 ± 0.5 until part E) |
| control | F1: D.V8 - D (computed values exact; bits per byte without the copy) | holds if computed ≤ -2 or bits per byte without the copy ≥ +0.01 | computed -3 to -8; bits per byte +0.012 to +0.025 |
| G1 | F2b: D.L8 - D (id_copy exact). Tested only if F2b's D-all - D ≥ +20 (point estimate); otherwise "not testable: no hazard in F2b", with no K9 | ≥ +15, lower bound > 0 | not testable (P ≈ 0.91-0.97: in the smoke, D copied 10 of 10 F2b IDs exactly). If testable: +15 or more, near D-all - D (L8 fired at c1 on 10 of 10 smoke copies) |
| G1e (reported only) | F2b: E10.L8 - E10 (id_copy exact). Read only if F2b's E10-all - E10 has lower bound > 0 | lower bound > 0 | not testable (in the smoke, E10 never started at c1 in F2b) |
| G1s | F2b: D.SIV - D (id_copy exact) | reported; decided by construction, since SIV cannot fire mid-line | 0 ± 3 |
| G2 | F2a. (a) Implementation check, in J1 since the third check, decided by construction (L8 ⊆ V_8, and V_8 cannot fire at F2a's c1): L8 fires at c1 on at most 5% of F2a's test copies; a failure is a bug, handled as J1's checks are. (b) Claim conditions, at the fix claim's rates: D.L8 - D and E10.L8 - E10 (id_copy exact); F2a's cost rows as C2-C3c | (a) ≤ 5%; (b) within ±3, and as C2-C3c | (a) 0%; (b) ±2 |
| G3 | F3, each ID kind (6-digit, letter, mixed; doses are reported only) whose D-all - D ≥ +20 (point estimate): D.L8 - D (id_copy exact). A kind without that hazard is "not testable", with no K9 | ≥ +15, lower bound > 0, and ≥ 0.7 of D-all - D, exact on counts: an exact 70% share passes (since part F) | Testable: 6-digit P ≈ 0.95-0.97, letters ≈ 0.2-0.4, mixed ≈ 0.83-0.89 (in the smoke, letters had no hazard: D exact on 3 of 3; third check, coldread M3). Where testable: L8 fires on ≥ 90%; gain ≥ 0.8 of D-all - D |

- **Who is likely testable** (added 2026-10-30 after the third check). P(neither F2b nor letters testable) ≈ 0.55-0.78
  (0.53-0.77 until part G, a rounding slip). So the generality claim most likely covers 6-digit and mixed IDs only. The
  PI's choice for F2b (the broader claim then covers only the ID kinds: 6-digit, letters, mixed) did not foresee
  that for letters.
- When G1, G1e or a G3 kind is not testable, its "holds" (for G1e, "lower bound > 0") is empty (None) and its status
  says why; until part E the code printed a computed "holds" beside "testable: false". (G1e's field named after the
  fourth check.)

**New values (C4, K2).**
- Computed-value tokens are O6's V4 tokens: for each `= \$?(\d[\d,]*(?:\.\d+)?)` match, the byte after `= ` and the
  value's first digit.
- A masked start there is **same-value** when an earlier occurrence of the 8 bytes before it is followed by exactly the
  value, read as a whole number with the same pattern. Otherwise it is **new-value**. So `720` after `7200` is new.
- A trailing comma is punctuation, not part of a value: `150,` and `150` are the same value. (Added 2026-10-30 after
  the second check. The pattern had read `= 12, where` as the value `12,`. Recounted on the stored dev outputs, no
  count in the dev recount changes.)
- C4's bar is the integer 4 (`C4_MAX_D = 4`), compared with `<=`.
- O6's strict count (every removal) and O6's prefix test are reported beside it.

**C1e and the +15 target.**
- E10.L8 - E10 is also read against the PI's original +15 target, without a verdict.
- Below +15, K8 records "target not met", with the first reading that applies (four readings since 2026-10-30, after
  the second check; there were two):
  - "bound untestable", if C1e cannot be tested;
  - "refill", if E10-L8 - E10 reaches +15, so the matched refill gave the gain back;
  - "bound-limited", if C1e (i) holds;
  - otherwise "misses onset starts".
- R1's recall at E10's c1 starts is reported with it.
- If E10-all - E10 has a lower bound ≤ 0, C1e cannot be tested. That is recorded, and the fix claim is limited to the
  default rate.
- This asymmetry is intended. Where E10 shows the hazard, L8 must fix it there too; where it shows none, the claim is
  limited to D (stated 2026-10-30, after the third check).
- K8's first matching reading can name the lesser cause when two apply. K8 is reported only, and both numbers are
  printed beside it (added after the third check).

### The paired verdict: L8 against SIV+PO

*Split by the PI's choice of two verdicts, each with its meaning fixed in advance (post-smoke amendment, 2026-10-30; part D). Since the
third check, only copy rows with the hazard set a class (part E).* The second check found two things (VERDICT item 4):
- Within the window cells, SIV+PO masks the whole copy, so L8 can only tie or lose there.
- F5's "L8 better" rows come from L8 not firing in texts of 700 or more tokens, which a byte rule with a length cutoff
  could match.

The third check then found that any of the windowed verdict's 92 rows could set its class, dose and cost rows
included, whose readings would then be false (its VERDICT item 1). So only copy rows with the hazard set a class now,
which implements the meaning the PI chose: a loss means L8 dropped copies. There are two verdicts, each read as
fixed under Claims. Neither can show that reading the model is needed.

**Rows.** In each cell, at each rate (F5: D only), X.L8 - X.SIVPO is paired over problems on:
- id_copy exact;
- id_copy bits per copy;
- the cost rows: C2 (bits per byte without the copy), C3 (computed), C3f (final) and C3c (copied).

F6 has no copy rows. F3 reports each ID kind and the doses separately.

**A row's class.** Rows are oriented so that a positive difference means L8 is better (fewer bits, more exact).
- **L8 better:** the point estimate is at least the margin, and the 95% interval lies above 0.
- **SIV+PO better:** the point estimate is at most minus the margin, and the 95% interval lies below 0. ("Excludes 0"
  named no side until part F.)
- **Tie:** anything else.
- Margins: 2 points on exact rows; 0.002 bits per byte on C2; 0.5 bits per copy.

**The rows that set each verdict's class** (part E):
- **Windowed:** the id_copy exact rows of F1, F2b and F3's 6-digit, letter and mixed IDs, at D and at E10. This is
  BLT-1B with the 512-byte windows the fix claim is about. A row counts only where its cell shows the hazard at that
  rate:
  - at D, the cell's D-all - D ≥ +20 (point estimate; G1's and G3's gate);
  - at E10, the cell's E10-all - E10 has a lower bound > 0 (C1e's and G1e's gate).
- **Released:** F5's id_copy exact row at D, under the conversion as released (window 0), with no gate. Its reading is
  about the reverse of the windowed hazard. Under the released conversion, where the patcher's argmax at c1 is wrong
  (long texts), removing the copy's starts lowers exact copies: 65.5% with every copy start removed, against 76.5%
  when D keeps its starts there ([JW0]). F5's D.L8 - D, on the same rows and with the same classes, is reported beside
  it. ("That hazard" was reworded after the fourth check.)

**Each verdict's class**, from its class rows:
- **tie:** every class row ties;
- **SIV+PO better:** at least one class row is SIV+PO better and none is L8 better;
- **L8 better:** the same with the directions swapped;
- **mixed:** class rows in both directions;
- **not testable** (windowed only): no row passes its gate. Through the verdict code this needs K5, since F1's gate
  at D is C0's point condition; it is defined for completeness.

A verdict is "not reached" after any stop, or while its jobs are missing: J2-J5 for the windowed verdict, a scored J7
for the released one.

**Every other row is reported beside the classes,** row by row, with its class. When it does not tie, it carries this
pre-registered sentence (amended after the fourth check, as part F lists):
- a copy row of F1, F2b, an F3 ID kind or the doses at a rate where its cell shows no hazard: "No hazard here
  (<gate>), so a difference here is not a fixed or missed copy; it is reported only." <gate> is "D-all - D below +20"
  at D and "E10-all - E10 has no lower bound above 0" at E10;
- F2a's id_copy rows: "Neither rule touches the copy; a difference is a side effect of the refit.";
- id_copy bits per copy, the doses' included: "L8's copies cost more (fewer) bits than the byte rule's on <row>.";
- the cost rows (C2, C3, C3f and C3c, F2a's and all of F6's included): "L8 cost more (less) than the byte rule on
  <row>.";
- **the doses, reported apart as F4 is:**
  - their id_copy exact row, where the dose cell shows the hazard at that rate: "L8 lost dose copies the byte rule
    kept: in <a> of the <n> lost copies L8 did not fire at c1, as its S0 condition skips short doses by design, and
    this is that design's cost; in <b> it fired, and the loss is a side effect of the refit." If L8 is the better
    one: "L8 kept dose copies the byte rule lost."
  - their cost rows (C2-C3c): "The byte rule also masks the recurring line prefix before the dose; a difference
    there is neither a copy loss nor an L8 cost." Both dose lines start "Dose ordered: ", and SIV vetoes the recurring
    "ordered: " nine tokens before c1 on every dose copy line; L8 does not.
- F4's rows are reported apart with their classes. If the patcher's argmax at c1 fails at 550-950 bytes, SIV+PO should
  win F4's copy row; that would bound L8 to copies the patcher can read.

**Discordant texts.** Each id_copy exact row that does not tie, in any group, lists the texts (their problem indices)
where X.L8 and X.SIVPO differ. They are counted by direction, by whether L8 fires at c1 and by whether X starts a patch
at c1, as totals per direction and crossed (the crossed counts restored and the indices added in part G). So a loss at
copies L8 missed can be told from a side effect of the refit, and the "SIV+PO better" readings give that split (since
part F). On dev's F1 rows (L8 against SIV), 2 of the 5 copies L8 lost at D, and all 3 discordant texts at E10, were side
effects; L8's firing at c1 separated misses from side effects in all 9 texts (fourth check).

**Under K6** (the control shows no cost), a tie on a cost row carries no information, and the verdict output says so.

**SIV+PO's own matched rate**, D.SIVPO - D and E10.SIVPO - E10 in each cell, is reported beside the verdicts against
L8's tolerance (within 0.005), without a verdict.

**Chance under a true difference of zero** (copy_latch_paired_odds.py; results/blt_copy_latch_fresh_paired_odds.log),
given that the windowed verdict is reached (no stop, and J2-J5 present; since part G). No correction is made for the
number of class rows: up to ten, about 4.9-6.0 expected to pass their gates (the six of part E, and since part F the
6-digit and mixed IDs at E10).
- If side effects of the refit flip 0.5% of copies each way, P(the windowed class leaves "tie") ≈ 0.006-0.008. At 1%
  it is ≈ 0.042-0.057. Dev's F1 rows suggest about this range: L8 alone kept 1 copy of 293 at each rate.
- If they flip 3-10% each way, it is ≈ 0.20-0.25. That is an upper range: F2a's unprotected copy flipped once in 10
  smoke texts, which suggests at most about 5% each way (fourth check, numbers N3).
- (Restated after the fourth check, exactly and with the E10 rows; part E gave 0.006-0.03 and 0.17-0.20 for about four
  rows. Part G wrote 0.20-0.25 for "0.21-0.25", whose low end cut the log's 0.205 off, and dropped F2a's smoke flip as
  support for 3-10%.)

**Expected** (dev, exploratory), given that the windowed verdict is reached (since part G). These come from the odds
script, exact over the counts since part F: each (lost, kept) count's class comes from one seeded bootstrap draw (2,000
resamples, seed 6), and the counts' distribution is summed exactly ("exact" explained in part G). SIV stands in for
SIV+PO on dev's 293 F1 texts (both mask c1..c4 in 293 of 293). The gate chances are the rule's predicted testabilities:
the smoke-based ones under Verdicts for D, and for the 6-digit and mixed IDs at E10 0.5-0.9 and 0.4-0.7. That basis is
F1's dev E10 hazard, with D breaking 2 of 3 6-digit and 1 of 2 mixed smoke copies (fourth check). Each row is given its
own truth.
- **Windowed.**
  - "Tie" with P ≈ 0.69-0.72 and "SIV+PO better" with P ≈ 0.28-0.31, with dev as truth.
  - "Tie" ≈ 0.52-0.55 and "SIV+PO better" ≈ 0.44-0.47 with dev's uncertainty (Jeffreys, exact; without the E10 rows
    it was 0.535-0.560, fourth check).
  - "SIV+PO better" ≈ 0.91-0.93 if L8's recall at c1 is 97% and every copy it misses is lost, and ≈ 1.00 at 95%
    recall, R1's floor (added in part G).
  - "L8 better" ≈ 0-0.01, and "mixed" ≈ 0-0.01.
  - **A tie can include a class row 2 points or more from zero** (either way, any row size): ≈ 0.47-0.51 of all
    outcomes with dev as truth, ≈ 0.37-0.39 with dev's uncertainty (0.44-0.46 without the E10 rows; fourth check). At
    100 texts, 2 or 3 copies lost and none kept still tie (-3 [-7, 0], for one), and 4 lost and none kept does not (-4
    [-8, -1]). With copies moving both ways, a 100-text row can tie further behind (6 lost and 1 kept, -5 points).
    (Corrected in part G: it labelled this figure "a 100-text class row sits 2-5 points behind", and said "2, 3 or 4
    copies lost and none kept still ties".)
  - The class rows are expected to be F1's at D and E10, the 6-digit and mixed IDs' at D, and likely their rows at
    E10. F2b (testable with P ≈ 0.03-0.09) and the letters (≈ 0.2-0.4) are likely not testable; no E10 row is
    counted for them, since D broke none of their smoke copies.
  - The second check's "≈ 0.3-0.65" for at least one SIV+PO-better copy row held at dev's recall (98.3%) and counted
    F2b and the letters as testable. It is replaced by the figures above.
- **Released: "L8 better"**, with P ≈ 1 on dev ([JW0], dev texts under the released conversion). 10.2 of its 10.9
  dev points come from texts of 700 or more tokens, where the patcher's argmax at c1 was never right and L8 does not
  fire.

| cell | id_copy exact, L8 - SIV+PO (D; E10) | other rows | basis (dev, exploratory) |
|---|---|---|---|
| F1 (class rows) | -1.4 (-4 to +1); -0.3 (-3 to +2): tie or SIV+PO better. At D, SIV+PO better with P ≈ 0.17 with dev as truth, ≈ 0.28 with dev's uncertainty, and ≈ 0.84 at 97% recall if every miss is lost (the second check's 0.6-0.8 assumed fewer losses) | tie | Dev L8 - SIV: -1.37 [-3.07, 0.00] and -0.34 [-1.71, +0.68]. SIV+PO covers c1..c4 in 293 of 293 dev texts, L8 in 288 |
| F6 (beside) | none | tie | L8 and SIV+PO each removed 1 of 1,689 D computed-value starts |
| F2a (beside) | 0 ± 2: tie. A non-tie is a side effect of the refit (P ≈ 0.045 for a 300-text row if 3-10% of copies flip each way; the odds log) | tie | No rule fires at the copy (V_8, SIV, PO: 0 of 293 dev rewrites). Smoke idx 40 flipped: D.SIVPO exact, D and D.L8 not |
| F2b (a class row only where it shows the hazard) | -2 (-15 to +2): tie or SIV+PO better (SIV+PO better with P ≈ 0.25-0.30 if testable). Without the hazard it is reported beside | tie | PO covers c1 in 293 of 293 dev rewrites, SIV in 0. In the smoke L8 fired at c1 on 10 of 10 copies, and no layout broke a copy |
| F3 ID kinds (class rows where the kind shows the hazard) | -1 (-6 to +2): tie or SIV+PO better. Per kind, SIV+PO better with P ≈ 0.07 with dev as truth, ≈ 0.10 with dev's uncertainty; ≈ 0.26 at 97% recall, 0.59 at 95%, 0.95 at 90% (G3's predicted firing), if every miss is lost | tie | SIV and PO cover c1 in 293 of 293 dev rewrites of each kind |
| F3 doses (apart) | -5 (-25 to +2): tie or SIV+PO better where the dose cell shows the hazard | tie, or SIV+PO better on rows that the prefix veto touches | SIV covers c1 in 293 of 293, PO in 34. L8 fires on at most about 60% (S0) |
| F4 (apart) | Tie only if L8 fires on nearly every copy: SIV+PO better with P ≈ 0.7-1.0 at 90-95% firing (third check) | tie | The patcher's argmax at c1 held in 8 of 8 dev copies at 534-619 bytes |
| F5 (released) | +11 (+4 to +18); bits per copy +2.6 (+1.0 to +3.5): L8 better | tie | [JW0], dev texts under the released conversion: D-all, 65.5% exact and 5.30 bits per copy, against 76.5% and 2.67 when D keeps the copy wherever the argmax at c1 is wrong |

Part D relabelled this table "tie or SIV+PO better" for F1, F2b and F3 (F2b's said "tie unless L8 misses mid-line
onsets"), and replaced the doses' basis "L8 should fire on ≤ 30% (S0)". Part E changed the rest:
- each cell's group;
- F1's chance with dev's uncertainty;
- F2a's "decided by construction";
- F2b's "L8 was not measured there", replaced by the smoke fact;
- F3's per-kind odds at lower recall;
- the doses' firing ("about 50-60%");
- F4's "tie if L8 fires on ≥ 80%".

Part F gave F1's row one figure per scenario, F2a's the odds log's exact figure (it said 0.03-0.04; the simulated
log had 0.046-0.048), and F3's 90% its name, "G3's predicted firing" (it said "G3's floor").

The ranges that reach +2 (F1 at E10, F2a and F2b) touch the L8-better margin: at 600 texts, +2 points with no copy
lost is "L8 better", and at 300 texts 6 kept and none lost is. They are dev's spreads, and the labels name the likely
classes (part F, coherence L3).

### Reported without verdicts (with predictions)

- **C1's share of the hazard**, (D.L8 - D)/(D-all - D). Dev 0.99 (75.4 of 76.1). Predicted ≥ 0.9.
- **F3 doses.** L8 fires on at most about 60% of dose copies: those of three or more digits (59% of the dev draws)
  can sum above 10 bits, and shorter ones mostly do not. D.L8 - D within ±5. (Restated 2026-10-30, after the smoke run,
  where L8 fired on 2 of 2 doses, and the second check; it said "at most 30%, because most common doses sum below 10
  bits". "About 50-60%" became "at most about 60%" after the third check, numbers N3.)
- **F2a.** D-all - D on id_copy exact, predicted +10 to +60 (DESIGN.md; no dev basis). This readout was dropped at
  7be1185 without a reason and is restored here (2026-10-30, after the second check).
- **E10.V8 in F1**, V_8 at the matched 10%. It shows whether masking more than the onset, with refill, goes past the
  onset-removal bound at 10%. Dev step 3 had E10.V8 at +16.7 over E10 (71.3% against 54.6% exact), beside the bound's
  +15.7. (Its purpose was unstated until 2026-10-30.)
- **F4.** D-all - D at least +50 (O6:123). L8 fires on at least 80% if the patcher's argmax at c1 holds beyond 619
  bytes (dev: 8 of 8 at 534-619); otherwise D.L8 ≈ D.
- **F5.** D.L8 at least D + 20 on exact, with bits per copy no higher than D's. L8 fires on at most 10% of copies in
  texts over 700 tokens [JW0]. Exact by text length: under 600, 600-700 and over 700 tokens.
- **F6 counts** at D and E10, new-value and strict: L8c, V_8, SIV, PO, SIV+PO and L8b.
  - V_8's new-value removals: 3-8% of D's starts (dev, whole-number test: 46 of 799 and 57 of 890).
  - L8b: about V_8's (dev: L8b removes the same 50 of 799 as V_8). This attributes L8's separation to its patcher
    conditions.
  - SIV+PO: 0-3.
- **D starts removed outside the copy in F1**, by rule (R2's count for SIV, SIV+PO, V_8 and L8b). Predicted: SIV
  0.5-0.8 and SIV+PO 0.5-0.9 per text (audit counts on dev: 0.64 and 0.68).

### Claims

- **Fix claim.** It needs all of:
  - C0 met;
  - the control showing a cost;
  - R1 and R2;
  - C1, C1e, C2, C3, C3f, C3c, C4, C4s and the rate bar;
  - no K2.

  If C1e is untestable, the fix claim is made at the default rate only. It then needs the bars at D (C2-C3c, C4s and
  the rate bar, so C4s too is read at D only), with C4 and K2 unchanged. When it does not hold, its scope says why
  (since the third check; amended after the fourth):
  - "failed" after K2, K10 or K7 (with K6 as well: "failed: K7 (...); and void: the control showed no cost (K6)");
  - "void: the control showed no cost (K6)" after K6 alone;
  - "not reached" after K5 or a failed implementation check;
  - "not reached: J1 has not run", "not reached: J2 has not run", or "not reached: J3 has not run" while that job's
    output is missing and nothing else failed (J3-J7 may run in any order).
- **Generality claim** (one AND, rewritten 2026-10-30 after the second check). It holds when all of these hold:
  - the fix claim;
  - G2 (a), a J1 check since the third check, and G2 (b) at the fix claim's rates;
  - G1, if testable;
  - G3, for every ID kind where it applies.

  At least one of G1 and G3 must be testable; otherwise the claim is "not testable". Its scope names what was tested:
  F2b's reworded context (if G1 was testable) and each ID kind where G3 applied, all read at the default rate. G1e is
  reported only. After any stop, or with J3, J4 or J5 missing, the claim is "not reached" (J3 added after the third
  check). K9 is read whenever no stop applies, as before; since part F its sentence prints only while the fix claim
  holds.
- **Paired verdicts.** The windowed and the released verdict above, reported beside the claims. Each is "not reached"
  after any stop, or while its jobs are missing.
- **Scope of the claims** (written after the fourth check; the checkpoint named after the fifth). The claims cover
  BLT-1B's base weights (itazap/blt-1b-hf) with 512-byte windows; the conversion as released (window 0) enters only
  the released paired verdict.
- **F4 and F5 enter no claim** (added after the third check). F4 is reported apart, and F5 enters only the released
  paired verdict. DESIGN.md section 6's F5 line, "that the agreement gate keeps the fix from harming the released
  conversion", has no successor. F5's L8 against D is reported beside the released verdict, with exact copies by text
  length, and is not tested as a claim.

**What each outcome licenses** (pre-registered 2026-10-30 after the second check, the paired verdicts' readings after
the PI's choice, and amended after the third, fourth and fifth checks). This replaces DESIGN.md section 6's reading.
Each reading is quoted whole, with its qualifiers, wherever the result is reported; `verdict` prints each one with its
numbers filled in (since part F), under the label that says which result counts (since part G), in this order: the stops
(a failed implementation check, K10, K2, K5); the fix claim, or K7 and then K6; the generality claim; the two paired
verdicts; and last, that no outcome shows reading the model is needed.
- *Fix claim at both rates:* "On fresh GSM8K problems in O6's `\nTicket NNNN` template, in BLT-1B with its 512-byte
  local windows, scored teacher-forced, L8 restores verbatim copies at BLT-1B's default patch rate and at entropy@10%,
  each compared at a matched rate. At the default rate it gains <C1, with its interval> points of exact copies, <C1's
  share> of what removing every copy start gains; at 10% it gains <E10.L8 - E10, with its interval>. Its costs stay
  within the bars, read on point estimates: at most 2 points on computed values, final answers and copied numbers; at
  most +0.01 bits per byte outside the copy; a patch rate within 0.005; at most 4 new-value removals of about 1,700
  computed-value starts at the default rate, and none at 10%."
- *Fix claim at the default rate only:* the same sentence for the default rate, adding: "At 10% the onset-removal
  bound could not be told from zero, so the fix was not tested at that rate." (It said "so that rate was not tested"
  until part F: K2 and R1's E10 half are still read at 10%.)
- *A failed implementation check, a second time or once with no re-filing:* "The implementation failed; nothing was
  tested." After a first failure, while the PI decides on a re-filing, `verdict` prints "not reached: the
  implementation check failed; a redo needs the PI's approval"; if the PI declines, the main session reports the
  sentence above (part G).
- *K10:* "L8 failed its recall or footprint check before scoring."
- *K2:* "L8 removed a new value at 10%; L8 is dropped."
- *K5:* "In O6's `\nTicket` template, scored teacher-forced with 512-byte windows (<n> texts), the hazard did not
  replicate on fresh problems; L8's copy fix was not scored." (It said "L8 was not tested" until part F, though J1 had
  read R1, R2, C4 and K2, and "on point estimates" until part G, though C0 also has a lower bound. The stops are
  listed in the order `verdict` prints them since part G.)
- *K7* (no stop), in two parts, each printed only if it has a bar (since part G):
  - F1's bars: "In O6's `\nTicket` template (F1), scored teacher-forced with 512-byte windows (<n> texts), L8 failed
    on fresh problems:" followed by them;
  - F6's bars: "On the same problems without the ticket (F6), L8 failed:" followed by them. C4 reads "C4 (<k>
    new-value removals by the patcher at the default rate; at most 4 allowed)".
  - The rate bar is listed under the cell or cells whose rate failed, with its rates, for example "rate (D)".

  There is no claim of a fix or of no cost. K7's reading governs; if K6 also fired, K6's follows it. (Until part G one
  sentence put every bar under the template and "on point estimates", F6's C4 and C4s included.)
- *K6:* "The cost bars could not see V_8's known cost; no claim either way."
- *Generality claim:*
  - if it holds: "In BLT-1B with its 512-byte local windows, the fix also holds for" followed by the parts tested,
    joined by "; and ": "the context reworded mid-line", and the ID kinds tested with "in the same `\nTicket`
    template" (for example "6-digit IDs and mixed IDs in the same `\nTicket` template"); then "at the default rate".
  - if not testable: "No other wording or ID kind showed the hazard, so generality was not tested."
  - if any K9 (only while the fix claim holds): "In O6's `\nTicket` template, scored teacher-forced with 512-byte
    windows (<n> texts), at the default rate, the fix does not extend to" followed by the ID kind. For F2b: "Scored
    teacher-forced with 512-byte windows (<n> texts), at the default rate, the fix does not extend to the context
    reworded mid-line (F2b's template)." ("On point estimates" dropped in part G: G1 and G3 have lower bounds.)
  - if the fix claim failed: "No generality claim: the fix claim failed."; after K6 alone, "No generality claim: the
    fix claim is void (K6)." K9 is still read and listed beside it, but its sentence is not printed.
  - if it fails otherwise: "No generality claim:" followed by the failing part (G2 (b) at a rate). G2 (a) cannot fail
    here: a G2 (a) failure is a bug, and J1 handles it as it handles its other checks.
- *Not reached, without a stop* (status lines, not readings; smoke runs print them, and so does an earlier filing's
  chain that stopped short, printed as the confirmatory result beside a corrected rerun): "The fix claim was not
  reached: <job> has not run." and "The generality claim was not reached: <why>." (Printed since part F, listed in
  part G, corrected in part H.)
- *Windowed paired verdict* (BLT-1B with its 512-byte windows). Its class rows are id_copy exact in F1, F2b and F3's
  ID kinds, where the cell shows the hazard at that rate.
  - tie: "On the cells and rates with the hazard, no copy row is classed as differing: none differs by 2 points or
    more with a 95% interval that excludes 0 (largest difference <x> points, <row>, <interval>). A 100-text row can
    still sit a few points apart. This is not an equivalence test."
  - SIV+PO better: "L8 lost copies the byte rule kept, in <cells and rates>: in <a> of the <n> lost copies L8 did not
    fire at c1 (copies L8 missed); in <b> it fired, and the loss is a side effect of the refit."
  - L8 better: "L8 kept copies the byte rule lost, in <cells and rates>. This is not expected by construction: within
    the window SIV+PO masks the whole copy."
  - mixed: "Read cell by cell; it settles nothing."
  - not testable: "No cell and rate showed the hazard, so no copy row could set a class."
  - The rows beside the class carry their own sentences (above). None of them changes the class.
- *Released paired verdict* (F5: the conversion as released, window 0, D only; its class row is id_copy exact).
  - L8 better: "Relative to SIV+PO, a rule that stops where the model cannot copy helps under the released conversion;
    F5's exact copies by text length show where. A byte rule with a length cutoff might match it; that is untested."
    F5's L8 against D is reported beside it.
  - tie: "Under the released conversion the copy row is not classed as differing (<x> points, <interval>): a
    difference of 2 points or more would need a 95% interval that excludes 0. This is not an equivalence test."
  - SIV+PO better: "Under the released conversion, L8 lost copies the byte rule kept: in <a> of the <n> lost copies L8
    did not fire at c1 (copies L8 missed); in <b> it fired, and the loss is a side effect of the refit."
- The tie readings above replace "no copy row differs by 2 points or more ... within 2 points" (part E), which a tied
  100-text row 3-5 points apart contradicted; the "SIV+PO better" readings gained the split (part F).
- **No outcome of this test shows that reading the model is needed.** A tie shows only that, on the copy rows with the
  hazard, no difference reached the class rule; the cost rows and the cells reported apart are read beside it. A win
  for L8 shows only that this byte rule does worse; another byte rule might not, and the released cell's win comes from
  where L8 stops firing, which a length cutoff might reproduce. This replaces the scope option's promise of a verdict
  "which settles whether reading the model is needed". (Until part F the second sentence read "A tie shows that this
  byte rule suffices on these templates".)

**What it cannot show** (added 2026-10-30, after the third check, and completed after the fourth; it carries DESIGN.md
section 6's "It would not show" forward):
- free generation: every readout is teacher-forced on gold text;
- real identifiers: the IDs are synthetic;
- other models: only BLT-1B's base weights (itazap/blt-1b-hf at snapshot 91aa6b8e), windowed and as released; no
  other model, and not the official BLT weights. The claims cover the windowed conversion only (Scope of the claims;
  this pointer added in part G);
- other templates: these templates only;
- contexts that do not recur over 8 bytes: shorter keys are untested, and F2a measures the gap;
- ID lists and near-duplicate IDs: several strings after one context, and near-duplicates, are untested;
- doses and other short or common strings, which S0 excludes (added in part F);
- gains past the onset-removal bound at E10 (added in part F);
- natural text: counted, not scored;
- a length-cutoff rival: none is scored, so F5's win is not separated from a rule that stops in long texts;
- mechanism and optimality: why a start inside the copy breaks it, and whether k, S0, log2 10 and 1 bit are optimal;
- pretraining exposure: BLT-1B's exposure to GSM8K cannot be audited;
- that reading the model is needed (above).

### Kill and stop criteria

- **K2.** Any new-value removal from E10 in F6. Drop L8 and stop. C4 replaces K2's former D clause.
  - The risk is about 0.7-5%: V_8 itself removed 0 of 252 E10 computed-value starts at dev, and L8 lies within V_8
    (choices audit; second check, numbers N4). Two pessimistic scenarios give more. It is 14% if E10's rate equalled
    D's pooled dev rate (1 of 1,689, about 256 E10 starts), and a Jeffreys bound on dev's 0 of 252 allows 30%.
    (Reworded 2026-10-30 after the third check; it said "The risk is not small. The choices audit estimated 1-5%".)
- **K5.** C0 fails: the hazard does not replicate on fresh problems. Report and stop. J3-J7 refuse to start after
  K5, and without J2's output (enforced in code since 2026-10-30, after the second check).
- **K6.** The control fails: on F1, D.V8 - D shows computed > -2 and bits per byte without the copy < +0.01, the exact
  complement of the control bar. No fix claim, and no "no cost" claim: the cost bars could not see V_8's known cost.
- **K7.** Any of C1, C1e (when testable), C2, C3, C3f, C3c, C4, C4s or the rate bar fails. When C1e is untestable,
  C2-C3c, C4s and the rate bar are read at D only; C4 and K2 are unchanged. No fix claim.
- **K8.** E10.L8 - E10 is below +15: the PI's original target is not met. Read as above. This is not a kill. Its
  chance is about even (P ≈ 0.5: dev landed at +15.02; change-log item 27; placed here after the fourth check).
- **K9.** G1 fails where it is testable: the fix does not extend to the context reworded mid-line (F2b). G3 fails for
  an ID kind where it applies: the fix does not extend to that kind, and K9 names each one. Any K9 means no generality
  claim. It does not touch the fix claim. Since part F its sentence ("does not extend to") prints only while the fix
  claim holds; when the fix claim fails, the generality claim reads "No generality claim: the fix claim failed." (or
  "... is void (K6)." after K6 alone), and K9 is listed without its sentence.
  (Rewritten 2026-10-30 after the second check; it said "G1 fails: the fix is specific to `\nTicket `. G3 fails for
  letters or mixed IDs: the claim is limited to digits." After the third check, "the fix is specific to the
  `\nTicket` wording" became "does not extend to", as the licensed sentence says.)
- **K6-K9 are read only when no stop applies** (since the third check). After a stop, the stop's own reading applies.
- **K10.** R1 or R2 fails in J1. Stop before any scoring.
- If the redone J1 fails an implementation check again, stop and report. The gate counts failed J1 and J7 outputs
  across re-filings and refuses the job after two (since part F).
- **A failed J7 check** (causality or alignment under window 0) is a bug, as in J1. J7 writes only its checks and
  exits before scoring. F5 stays unscored in the confirmatory result and the released verdict is "not reached". A
  re-filed chain, if the PI approves one, is a deviation reported beside it (Which result counts): its fix goes in its
  own commit, then a re-filing commit, as for J1, and under it J0 and every job run again once. A second failure of
  J7's checks leaves that chain's F5 unscored too. The fix claim is unaffected, and the windowed verdict does not need
  F5. (Defined 2026-10-30, after the second check; "released" was added after the PI's choice, part D, the re-filing
  after the third check, part E, and its two commits after the fourth; it said "it is fixed, disclosed and redone
  once". Until part G the re-filed chain was the result.)

### Jobs

One BLT-1B job runs at a time. `pgrep -fl realblt` is checked before each job, and the script refuses to start while
another one runs. That guard matches any process whose command line contains "realblt", so nothing else that mentions
it may run during a job; the smoke run hit this twice (see the note at the end of this section).

| job | content | Mac time |
|---|---|---|
| J0 | blocks and byte counts, committed before J1 | CPU, 3-5 min |
| J1 | the patcher on every fresh text (about 4,200); checks; R1, R2; C4, K2; thresholds | 12-15 min |
| J2 | F1: 11 layouts × 600 = 6,600 passes | 35-63 min |
| J3 | F6: 6 × 600 = 3,600 | 19-34 min |
| J4 | F2a and F2b: 7 × 300 + 9 × 300 = 4,800 | 26-46 min |
| J5 | F3: 8 × 400 = 3,200 | 17-30 min |
| J6 | F4: 8 × 200 = 1,600 | 9-15 min |
| J7 | F5: the window-0 patcher on F1's 900 texts, checks, then 4 × 600 = 2,400 | 15-26 min |

- In all, about 2.3-3.9 h of Mac time. Cost: $0.
- Order: J0; J1 (stops: checks, K10, K2); J2 (stop: K5); then J3-J7; then `verdict`.
- **The job gate** (gate and blocked in the code). Outside smoke runs a command first runs `git fetch`, refusing if it
  fails, and then refuses unless:
  - it has not run under the current filing commit: its output does not exist under that filing, and no commit
    after the filing commit, on HEAD or on its upstream, touched its output file. So an output that was committed and
    pushed still counts after rm, a reset or `git rm`. The gate cannot see an output deleted before it was committed,
    or a commit removed before or after it was pushed. Each job's output is therefore committed and pushed as soon as
    the job ends, before anything else is done in the repository, and no output or its commit is deleted, reset away
    or force-pushed away. This is kept by procedure, as the crash rule is. (Part G stated these limits; it said "So an
    output removed with rm, reset away or removed with `git rm` still counts.") A redo needs a re-filing commit, under
    which J0 and every job run again, once each, in order;
  - for J1 and J7, their checks have not failed twice, counted over the history;
  - every job output that exists is committed and unmodified, no tracked output is missing from disk, and HEAD has an
    upstream, is pushed and is not behind it. So each job's output is committed and pushed before the next job starts;
  - J1 finds J0's block file;
  - for J2-J7 and `verdict`, the filing commit, the script's hash, every covered file's hash, the model snapshot, the
    weights' blob name, the transformers, torch and numpy versions and modeling_blt.py's hash all equal J1's record.
    The versions are matched to J1's record, not pinned;
  - for J2-J7, the patcher outputs (results/blt_copy_latch_fresh_patcher.npz, git-ignored) have the sha256 J1
    recorded. A lost patcher file is recomputed by hand with the pinned code, logged in a dated note, and accepted
    only if its sha256 equals that record; otherwise the test stops;
  - J1's stops do not apply (they block J2-J7), and for J3-J7 J2's output exists under this filing and K5 does not
    apply;
  - for `verdict`, J2-J7 all exist under this filing, unless a stop applies.

  The second check added the stops (2026-10-30), the third check the one-run rule, the committed-and-pushed rule and
  J1's record (2026-10-30), the fourth the fetch, the history and upstream checks, the failure count, the numpy and
  blob records and `verdict`'s condition, and the fifth the stated limits (part G).
- Each scoring job J2-J6 reruns the patcher on its cell's test texts and counts the texts whose outputs differ from
  J1's. Every mask and layout is built from J1's stored outputs, so a nonzero count is reported as a limitation, not
  a stop (stated 2026-10-30, after the second check). J7 does not rerun the patcher check: it recomputes the patcher
  outputs under window 0 and runs the implementation checks on them. (Corrected after the third check; this line
  said "each scoring job".)
- Fresh problems kept: 3,300. J0 should visit about 4,600 of the 5,673, since F4's window keeps about 29%. (Until part
  E: "3,300 of the 5,673, plus rejects".) J0 records each block's visited and kept problems, not the pool-wide counts
  s3 estimated, and, after filing, train.jsonl's sha256 and line count (since part F).
- The Mac time, 2.3-3.9 h, is above the scope option's "About 2-3 h of Mac time", and a re-filing reruns every job.

### Code

- realblt_copy_latch_fresh.py, at commit 4026bec. It moved on 2026-10-30 from d84d097 after the second check, from
  15ab486 after the PI's choices, from 6f0318c after the third check, and from 264fc6c after the fourth, through
  56b2d00, and on 2026-10-31 from 3d19131 after the fifth; parts C-G of the change log list the changes. It holds:
  - the builders: make, make_f6 and take_blocks;
  - the whole-number test (same_value) and C4's integer (C4_MAX_D = 4);
  - the layouts, including E10-L8, the -all layouts and the SIV+PO layouts;
  - a scorer for copies of any length;
  - verdict_from and paired_verdict, which compute every bar, stop, kill and claim above, and both paired verdicts
    with their class rows, readings and sentences, from the job outputs;
  - sentences, which fills in every licensed sentence of Claims for `verdict` to print, with display names ("F3
    6-digit IDs", not "F3 digits6");
  - k7_sentences, which splits K7's reading into F1's bars and F6's (since part G);
  - for a re-filing, earlier_filing and differences, which list what differs from the earlier filing's outputs, and
    result_of, which says which result counts and labels the printed result (since part G);
  - the guard (filed_headings, filed_check, provenance, model_record) and the job gate (gate, failures, blocked).
- It imports the rule, its causality check, V_8, SIV, PO and the bootstrap unchanged.
- **The model tested** (named after the third check): BLT-1B, itazap/blt-1b-hf at snapshot
  91aa6b8e168046ad91e517d2191e1e974f50cc01, its base weights, loaded offline (the script sets HF_HUB_OFFLINE=1 before
  transformers loads). It runs through transformers 5.18.0 with the 512-byte-window patch (modeling_blt.py sha256
  df8ecdd4...) and torch 2.14.1, on the Mac's MPS device. The weights' file is blob fdc3e6f1... (its LFS sha256).
  The script refuses any other snapshot, a hub not in offline mode, and windows switched off outside J7. Every output
  records the snapshot, the blob, the versions (numpy's included), the patch file's hash, the window and the device.
  J2-J7 and `verdict` refuse any of these that differ from J1's record, apart from the device and the window. The
  versions are matched to J1's record, not pinned. (Blob, numpy, window and the offline check were added after the
  fourth check.) The script also refuses python -O (or PYTHONOPTIMIZE), which would drop the rule's asserts, and a
  threshold mode other than train, and every output records the mode (part G).
- F5 sets blt_load.WINDOW = 0 in the process before loading the model. That is what SEGR_BLT_WINDOW=0 sets, since
  load() reads it at call time. The script checks that the loaded model has no windows, with an explicit exit that
  python -O keeps (an assert until part F).
- **Self-test, on bytes and dev problems only** (results/blt_copy_latch_fresh_selftest.json):
  - the generic builder reproduces realblt_quadrant.build on O6's 588 dev texts;
  - F6 equals realblt_budget.problem's bytes on 294;
  - each template on 60 dev problems: its ID occurs exactly twice, units do not overlap it, and V_8 fires at c1 in
    every template but F2a;
  - the whole-number cases (six, two of them with a trailing comma);
  - the control and K6 partition the outcomes;
  - the dev blocks are deterministic;
  - C1e (i) agrees with 10 k1 ≥ 9 k2 on all 360,600 count pairs at n = 600 (without the tolerance, 14 of the 60
    exact ties failed);
  - filing heading lines are not found inside a sentence, a code fence or an HTML comment, without a date, or with a
    date before 2026-10-30 or an invalid one, and the rules file has none yet;
  - the guard refuses, here: an empty name, HEAD, a tree, the index, a commit without a filing heading and an unknown
    name;
  - in a scratch repository with a submodule, the guard accepts three commits:
    - the filing commit;
    - the same filing commit after a later commit that leaves the code alone;
    - a two-commit re-filing: the fix, then a re-filing commit that pins it.
  - In that repository it refuses sixteen cases:
    - no heading; a heading quoted in a sentence; a heading inside a code fence; a heading inside an HTML comment; a
      heading dated before 2026-10-30;
    - a side branch; the index; a merge; a filing commit that also edits code;
    - O6B_FILED=HEAD after a later commit; a later commit with changed code; a re-filing commit whose pin is not its
      code;
    - the filing commit with code changed after it, committed or not; code changed in the submodule;
    - an imported file the commit lacks.
  - the job gate, in a scratch repository with a remote and the smoke outputs as stand-ins:
    - it lets J2 run under the filing;
    - it refuses J2 again, after its output was removed with rm, after a reset to before it (pushed, then reset;
      "not pushed" until part G, which mislabelled it), and after `git rm` committed and pushed;
    - after a two-commit re-filing and the reruns of J0 and J1, it refuses J3 while J2 is the earlier filing's, and
      `verdict` while J2-J7 are;
    - the differences from the earlier filing's outputs are listed (one changed bar);
    - which result counts after a re-filing: a J1 redo gives the re-filed chain; a J7 check failure and another bug
      give the earlier filing's chain, with the re-filed one printed beside it as a deviation (since part G);
  - the guard covers 11 files: 7 in this repository (the script, _seg, blt_light, realblt_copy_latch, realblt_copy_veto,
    realblt_quadrant and bytes/final_rules.py) and 4 in the segresearch submodule (blt_layouts, blt_load,
    realblt_budget, registry), which are compared with the submodule commit the filing commit records;
  - the model record (snapshot, blob, versions, patch hash, device);
  - draw() never returns a dose hidden by joining the problem's parts (train idx 71; 400 seeds), which the joined test
    lets through. (Until part F the case checked doses_allowed only.)
- The self-test JSON was written by the code as committed at 4026bec and names that commit (part F's order of work;
  until then it was written from the working tree before each code commit).
- **Branch test** (copy_latch_verdict_test.py; results/blt_copy_latch_fresh_verdict_test.log). It prints the number
  of hand-set cases it evaluated: 54 at 4026bec (50 at 3d19131, 44 at 264fc6c). They cover:
  - every bar, stop, kill and claim label, and the sentences for most of them;
  - each class of both paired verdicts, with the tie reading's largest difference and the "SIV+PO better" split;
  - the side rows that must not change the windowed class (a dose cost row, a dose bits row, an F2a flip, an F2a cost
    row, an F1 bits row, an F2b copy row without the hazard, an L8-better cost row), and the case where no windowed row
    passes its gate;
  - J3 missing, with and without a failed bar; K6 with K7; K9 when the fix claim fails;
  - K7's two parts: C4 at 5, a failing C4s, a failing F6-only rate, a C1 that fails only on its lower bound, and F1's
    rate (since part G); a first and a second J1 check failure;
  - the stop part of the job gate (the rest is in the self-test).

  (Part D gave 38 evaluations; the test made 37.) Both tests were rerun on 4026bec.
- The BLT-1B smoke run on dev problems is disclosed in the dated note at the end of this section.
- **Odds** (copy_latch_paired_odds.py; results/blt_copy_latch_fresh_paired_odds.log): the paired-verdict chances
  above, exact over the counts since part F (one seeded bootstrap draw per count). The discordance comes from dev's
  stored F1 rows. The gate chances are the rule's predicted
  testabilities, not dev rows: the smoke-based ones under Verdicts, and the E10 ones in the paired subsection. (Added
  after the third check; until part F this line said "from dev's stored F1 rows only".)

### Dev numbers behind the predictions (exploratory; results/blt_copy_latch_devcheck.json, d84d097)

**Whole-number recount.** No L8 count or dev decision changes (see the step 1 corrections, under Dev step 1 results
above). V_8 at D splits 46/4.
Recounted 2026-10-30 with the trailing comma stripped: all 28 rule-and-layout rows of devcheck part A, four counts
each, are unchanged (copy_latch_recount_check.py; results/blt_copy_latch_fresh_recount_check.log).

**C4 and K2, from the stored outputs of both equation sets** (the first check's VERDICT item 8: use the pooled rate):
- L8 removed 1 of 1,689 D computed-value starts (0.059%): 1 of 799 on eq_test, 0 of 890 on eq_train. L8c removed
  none.
- L8 masked 2 of 2,111 computed-value positions:
  - `10560` after `15840`, a D start;
  - `2210` after `2125`, not a D start: the patcher's entropy there, 1.85 bits, was below D's 1.93-bit threshold.
  So the 0 of 890 is partly luck.
- **F6's expected D computed-value starts:** about 1,715 at the pooled density. It is 1,798 at the equation train
  density (F6's problems are train problems) and 1,631 at the test density.
- **P(C4 holds)**, at most 4 at D: 0.93 with a Jeffreys posterior on the pooled rate; 0.996 at the plug-in rate. From
  eq_test alone, 0.74 (Jeffreys).
- For reference, P(no removal at all), O6's former bar: 0.35-0.36.
- **K2:** 0 removals of 252 E10 computed-value starts at dev; the risk is under K2 above.
- **P(fix claim) ≈ 0.80-0.99**, or about 0.47-0.86 under K2's pessimistic scenarios (added after the fourth check).
  - It adds F1's cost bars (C2, C3, C3f, C3c and F1's half of the rate bar, at D and E10). From dev's F1 rows they all
    pass at n = 600 with P ≈ 1.00 with dev as truth and ≈ 0.98 with dev's uncertainty, the misses mostly C3f.
  - It is the product of the joint figure below (0.85-0.99; 0.50-0.86 under K2's pessimistic scenarios), F1's cost
    bars (≈ 0.98-1.00) and C1e's chance of passing, about 0.96-1.0 by the PI's option (Status): 0.85 × 0.98 × 0.96
    ≈ 0.80, and 0.50 × 0.98 × 0.96 ≈ 0.47 (fourth check, critic gap 6, critic/costbars_joint.out; C1e's factor named
    in part G).
- **Joint P(fix claim)**, from C4, K2 and R1 only (the F1 bars and C1e taken to pass, as dev suggests; labelled so
  after the third check): 0.88-0.99, taking K2's risk at 0.7-5% (V_8 itself removed 0 of 252 E10
  computed-value starts at dev, and L8 lies within V_8). R1's risk, about 0-4.4% (Jeffreys uncertainty on dev's
  288/293 and 92/93), lowers the floor to about 0.85. Pessimistic: 0.66 (C4 under Jeffreys, 0.93, times no K2 under
  a Jeffreys bound on 0 of 252, 0.71). K2's 14% and 30% scenarios above are pessimistic too; under them the joint
  figure is about 0.50-0.86 (second check, numbers N4).
- **P(stop at J1)**: about 1-9% (K2 0.7-5%; R1 0-4.4%; R2 and the checks about 0); 14-33% under K2's pessimistic
  scenarios. A stop at J1 means no scoring: no claim and no paired verdict.
- (Restated 2026-10-30, after the second check. This line said "0.92-0.99, or 0.66 pessimistic (choices audit)",
  which assumed K2 at about 1% and left out R1.)
- **P(generality claim | fix claim)** (added after the fourth check).
  - At the flip rates dev and the smoke suggest (0.5-1% each way), P ≈ 0.9-1.0. At 3% each way it is ≈ 0.9, and at 10%
    ≈ 0.6.
  - Basis: G2 (b)'s copy condition (±3 points on 300 texts) fails with P ≈ 0 at dev-like flips, ≈ 0.05 at 3% and
    ≈ 0.39 at 10% each way (fourth check). The smoke saw no D.L8-against-D flip in 10 F2a texts.
  - It assumes G3 holds wherever testable, as it does when L8 fires on 90% or more of a kind's copies; L8 fired on all
    10 F3 smoke copies.
  - G2 (b)'s cost conditions on F2a (C2-C3c at 300 texts) fail with P ≈ 0-0.03, from F1's dev spreads, and the
    testabilities are taken as independent (both added in part G).
- **P(generality not testable)** ≈ 0.002-0.007: no hazard in F2b or in any F3 ID kind, from the testabilities under
  Verdicts.
- **Its most likely scope:** 6-digit and mixed IDs only, at the default rate, with P ≈ 0.43-0.67. That is both kinds
  testable, and neither F2b nor the letters.

**Byte rules on dev rewrites** (the 293 quad_test problems; their dev tickets, except F3's seeded IDs).
- Copies masked at c1:
  - F1: V_8, SIV, PO, SIV+PO and L8b, 293 each;
  - F2a: 0 for every rule;
  - F2b: SIV 0; PO, SIV+PO, V_8 and L8b 293 each (L8b masks the whole copy in 290);
  - F3's 6-digit, letter and mixed IDs: every rule 293;
  - F3's doses: SIV 293, PO 34, SIV+PO 293;
  - F4: every rule 293.
- Masked tokens outside the copy, per text: SIV+PO 3.9-16.9, L8b 45-47, V_8 87-98.

**Natural text** (660 xdomain documents, stored outputs; E10 cost by [S2]'s method). Each cell gives the share of
positions masked, then D starts masked per kB, then the net E10 bits per byte:

| domain | L8 | SIV+PO |
|---|---|---|
| math | 0.20%, 0.56, +0.00003 | 1.25%, 1.30, -0.00032 |
| algebra | 0.25%, 0.40, +0.00014 | 0.68%, 0.47, -0.00010 |
| code | 1.21%, 2.29, +0.00032 | 14.14%, 8.65, +0.00273 |
| prose | 0.13%, 0.64, -0.00030 | 0.15%, 0.20, +0.00009 |
| clinical | 0.42%, 0.54, +0.00041 | 0.05%, 0.09, +0.00011 |
| traces | 0.01%, 0.06, -0.00002 | 4.04%, 14.77, -0.00110 |

- In code and traces SIV+PO masks 14% and 4% of positions, and 4-240 times as many D starts per kB as L8 (3.8 and
  237 times; "4-250" corrected 2026-10-30).
- Elsewhere both rules mask little, and in prose and clinical text SIV+PO masks fewer D starts than L8.
- E10's net cost depends on how the refill orders tied entropies; the table uses np.argsort's default order. Under
  200 random tie orders it stays within about ±0.0005 bits per byte, except code SIV+PO (+0.0024 to +0.0031) and
  traces SIV+PO (-0.0009 to -0.0014) (second check, numbers N6; this line said "within ±0.003 bits per byte for both"
  until 2026-10-30).
- These are counts and an E10 estimate. No BLT-1B scoring of natural text is planned, so the paired verdict cannot
  weigh them.

**Far copies.** In the 8 dev copies at 534-619 bytes, the patcher's argmax at c1 was right in 8 of 8. D-all was exact
in 8 of 8, and D in 1.

### Change log (2026-10-30)

**A. Differences from DESIGN.md section 4 and K5-K9**, each with its reason:
1. C1e has two parts: (i) E10-L8 - E10 ≥ 0.9 × (E10-all - E10), both without refill; (ii) E10.L8 - E10 with lower
   bound > 0. It replaces "≥ +15, lower bound > 0". Reason: the PI chose a share of the bound; VERDICT items
   1-2.
2. E10-L8 is added to F1 (600 passes). Reason: VERDICT item 2, a like-for-like numerator.
3. E10-all is defined: no start at any byte of the copied string, and no refill. Reason: VERDICT item 2.
4. If E10-all - E10 has lower bound ≤ 0, C1e is untestable and the fix claim is limited to the default rate. Reason:
   PI choice; VERDICT item 2.
5. C1e is in the fix claim and in K7. +15 is only a readout, with K8's two readings ("bound-limited", "misses onset
   starts"; four readings since part C). Reason: PI choice; VERDICT item 1.
6. "Ceiling" became "onset-removal bound (no refill)". Reason: VERDICT disclosures; O6's matched E10.V6 reached +21.2.
7. C4 at D is the integer 4, coded exactly; E10 stays at 0. Reason: the PI chose at most 4; VERDICT
   item 4.
8. New values are judged on whole numbers. Reason: VERDICT item 5.
9. O6's strict count is reported, not a bar. Reason: PI choice (the option counts new values); VERDICT item 5.
10. K2 applies to E10 in F6; its D clause is replaced by C4, and its risk is disclosed. Reason: choices audit L3.
11. The scope is all cells, F1-F6, F4 and F5 included. Reason: the PI chose all cells with the byte rival.
12. SIV+PO (D.SIVPO, E10.SIVPO) is added with matched refill in every cell and at every rate where L8 is scored (all
    seven cells; F5 at D only), with a pre-registered paired verdict, its rows, classes, margins, patterns and
    per-cell predictions (two verdicts since part D, and set by copy rows with the hazard since part E). Reason: PI
    choice; VERDICT item 3. (Corrected 2026-10-30: it said "in every cell where it
    can be built (all seven)", but F5 has no E10 layouts.)
13. G1s is labelled as decided by construction. D.SIV stays only in F2b. Reason: VERDICT item 3; choices audit L4.
14. G2 is relabelled as an implementation check, and the generality claim rests on G1 and G3 (since part C, G2 is
    split into (a), an implementation check, and (b), claim conditions). Reason: choices audit L4.
15. The control is on F1 only. It names its metrics (computed; bits per byte without the copy), and K6 is its exact
    complement (final answers left out). Reason: VERDICT item 6.
16. The rate bar applies in F1 and F6, at both rates (at D only when C1e is untestable, as K7 says), and is in the fix
    claim and K7. Reason: VERDICT item 6; choices audit L1. (The rate clause was added 2026-10-30, after the second
    check.)
17. R1 and R2 are new rows from F1's patcher outputs, checked in J1 before scoring; K10 is new. Reason: VERDICT
    item 7.
18. J1 runs the implementation checks on every fresh text, alignment included, and a failure stops the test, apart
    from one redo. Reason: the draft's added stop rule, now listed (timeline audit F7); fidelity audit L1. The redo
    was in the section from 7be1185 but not in this item (corrected 2026-10-30, after the second check). Reason for
    the redo: a failed check is a bug, not a result. Since 2026-10-30 J1 exits after a failed check, before R1, R2,
    the F6 counts, C4 or K2 exist, so the redo follows no decision-relevant number.
19. There is no L8c fallback in the confirmatory test. Reason: the rule is frozen after dev (timeline audit F8).
20. The data paragraph is corrected: s3_fresh.py, realblt_combo.py, xdomain_data.py:69, the LoRA runs, fullft, the
    derived caches and test_causal, plus the definition of "fresh". DESIGN.md:286's "no earlier run ... used" is
    corrected. Reason: VERDICT disclosures; freshness audit FR1-FR6.
21. Blocks are fixed: their order, each cell's own filter, F4's distance on F4's own template, the seeds, the 12
    doses and F3's kind cycle. Reason: timeline audit F9; freshness audit FR5.
22. "The rule, bars and predictions are filed before any text is built" now reads "before any confirmatory text is
    built". Reason: timeline audit F2 (s3 built throwaway texts in memory).
23. Predictions are restated from dev and the devcheck where dev fell outside DESIGN's ranges: C2 at E10 (dev -0.0010,
    at the edge of ±0.001), C3c (dev +0.5 and +0.1, outside 0 to -0.5) and C4 (now 0-2, from the pooled rate). C3's
    range was widened from 0 ± 0.5 to 0 ± 1 for another reason: a fresh point estimate at n = 600 spreads about ±0.7
    around dev's +0.3 with dev as truth, and about ±1.2 with dev's uncertainty (dev interval [-0.7, +1.1] at n = 293;
    "±0.6" until part E, and the second figure added in part F). R1, R2, C1e and the paired-verdict predictions are
    new. F4's line cites the dev far copies. Reason: dev results and VERDICT item 8.
    (Corrected 2026-10-30, after the second check: this item gave "dev fell outside DESIGN's ranges" as C3's reason
    too, but dev's C3, +0.3 and +0.0, was inside 0 ± 0.5.)
24. J1 runs the patcher on every fresh text, test texts included (for R1, R2 and the checks). The jobs' layouts and
    times are updated. Reason: items 2, 12 and 17.
25. F5 sets the window to 0 in the process; F5 adds D.SIVPO. Reason: the rule modules assert the 512-byte windows at
    import; item 12.
26. C1's share of D-all - D is reported. Reason: choices audit L5.
27. The PI chose 1 after dev had landed at +15.02, where a fixed +15 had P ≈ 0.5. Recorded at the timeline audit's
    request.

Items 28-36 were missing from this list until 2026-10-30; the second check found them (integrity M3):

28. C1's prediction moved from +74 to +75 (+68 to +79). Reason: dev step 2's +75.4.
29. C4s's prediction moved from 0 ± 0.3 to 0 ± 0.5. No reason was recorded at 7be1185. It is kept because F6 was
    never scored at dev, and a difference in computed values at n = 600 spreads about as C3's does (item 23). The bar
    (≥ -2) is unchanged. After the third check, C3's spread was restated as about ±0.7, so C4s's prediction is
    widened to 0 ± 0.7 to match its own basis. After the fourth check (numbers N5) that is labelled "with dev as
    truth", beside about 0 ± 1.2 with dev's uncertainty.
30. G2's prediction moved from "≤ 2%" to "0%". Reason: L8 lies within V_8, and V_8 cannot fire at F2a's c1 (self-test:
    0 of 60). F2a's D-all - D readout, predicted +10 to +60, was dropped at 7be1185 without a reason; it is restored
    under "Reported without verdicts".
31. The control lost its predictions for id_copy (+65 to +76), final answers (-1 to -8) and copied numbers (-3 to -6).
    Reason: its bar was narrowed to its two metrics (item 15). J2 still reports those rows.
32. L8b, an ablation, is added: its D starts removed in F1 and F6, J0's byte counts, its firing at c1 in every cell,
    and the subset check L8 ⊆ L8b in J1. Reason: it tests whether L8's separation from V_8 comes from the patcher
    conditions 4-6 (dev: L8b removes the same 50 of 799 as V_8).
33. "Bytes of an ID belong to no other unit." Reason: F3's letter and mixed IDs would otherwise also be labelled as
    words or numbers, so their bytes would count twice.
34. F5 reports exact copies by text length (under 600, 600-700 and over 700 tokens). Reason: [JW0] traced F5's dev
    gain to texts over 700 tokens.
35. "If the pool runs out, J0 stops and nothing runs." Reason: the pool is finite (5,673 problems), and F4's window
    rejects about two thirds of them.
36. J1 may be redone once after a failed implementation check. Reason: item 18.

**B. Low findings in audits.json, one line each:**
- fidelity L1 (causality wording, alignment): restated (2.10 million distinct decisions, 4.18 million boundary
  comparisons); J1 asserts alignment; a planted-peek test is caught (devcheck G).
- fidelity L2 (step 3's reproduction not computed by its script): computed and stored (devcheck F; 293 of 293 for
  E10 and E10.V8).
- fidelity L3 (D proxy from a float threshold): the new code counts stored D starts. The dev table's proxy column was
  descriptive; corrected values are in the step 1 corrections.
- numbers F4 ("no cost"): now "no cost beyond the bars". D's bits-per-byte interval [+0.00001, +0.00124] is printed;
  SIV against L8 is given by its paired difference.
- numbers F5 ("1.33 billion"): restated, in the step 1 corrections and the #35 follow-up.
- numbers F6 (algebra footprint): corrected in the step 1 corrections.
- numbers F7 (0.25% against 2 of 799): C4 is now the integer 4. At dev the integer gloss, 2, governed.
- numbers F8 (0.26 holds for GSM8K only): the D-start column is added (code, about 1.8 per document), and 0.26 is
  scoped to the quadrant texts.
- numbers F9 ([S1] values shown as predictions): labelled as reproductions; out-of-range results flagged in the #35
  follow-up.
- numbers F10 (the "how" field): disclosed. The new code's case lists name the rule; the dev JSON is left as written.
- timeline F4 (0.25% two ways): as numbers F7.
- timeline F5 (297, not 300): dated note in the step 1 corrections.
- timeline F6 (smoke runs unrecorded): disclosed in the step 1 corrections. This time the smoke outputs are kept and
  committed.
- timeline F7 (the draft's extra stop rule and K8 rewording): listed here as items 5 and 18.
- timeline F8 (L8c could not pass K4): disclosed; no fallback here (item 19).
- timeline F9 (doses and seeds): fixed (item 21).
- freshness FR4 (test_causal's row): corrected in the data paragraph.
- freshness FR5 (s3's filter and template): labelled as estimates; J0 counts exactly.
- freshness FR6 (table gaps): heading now "stop at idx 1799 or below"; the other R9 readers, reducible2 and paper 1
  are added.
- choices L1 (the rate bar has no consequence): item 16.
- choices L2 (the control on F6, and its gap with K6): item 15.
- choices L3 (K2's single-event risk): disclosed under K2; the D clause is replaced (item 10).
- choices L4 (G2 and G1s decided by construction): items 13-14.
- choices L5 (C1 cannot tell a partial fix from a full one): the share is reported (item 26), and R1 bars recall.
- VERDICT's process note (bytecode files): this work runs Python with -B, and the new script sets
  sys.dont_write_bytecode before importing the rule modules.

**C. Post-smoke amendments (2026-10-30, after the second check at 67667ac).** They were made with the smoke outputs
(see the next note) and the second check's VERDICT.md and audits.json in hand. The code moved from d84d097 to
15ab486, and the self-test and the branch test were rerun on it. No model ran, and no fresh problem was read. The
section as it stood before these amendments is kept at
analysis/2026-10-30_copyfix/final_check/integrity/rules/RULES_ecca81e.md; the interim notes are at c03bd14 (pointers
added after the third check). The section as it stood before part E is kept at
analysis/2026-10-30_copyfix/recheck/integrity/RULES_eeb487b.md, and before part F at 628da50 (pointers added after the
fourth check).

*Must-fix items (VERDICT section 2, and the J1 redo):*
1. **C1e tie** (code F1). C1e (i) compared floats, so an exact 90% share failed: 14 of the 60 exact ties at n = 600.
   It now allows 1e-9, which agrees with 10 k1 ≥ 9 k2 on every count pair at n = 600 (self-test). A branch-test case
   covers the tie.
2. **G1 gate** (numbers N1, coherence M2).
   - G1 is tested only if F2b's D-all - D ≥ +20. G1e is reported only, and read only if F2b's E10-all - E10 has
     lower bound > 0. Otherwise each is "not testable", with no K9.
   - K9 names each failing part.
   - The generality claim is one AND with a holds and a scope, and it needs the fix claim. G2 is split into (a) its
     implementation check and (b) claim conditions read at the fix claim's rates.
   - G1's and G1e's predictions are restated from the smoke; they were "+55 to +78" and "+6 to +18". Reason: the smoke
     found no hazard in F2b, so the ungated G1 would likely have fired a false K9 (P(G1) ≈ 0.07-0.17).
   - F2b's wording waits for the PI. (The PI decided later the same day to keep it, gated; part D.)
3. **K7** (four audits). Reworded as the main session gave it, and item 16 matched. The code already read it this way.
4. **Paired verdict.** Waits for the PI. Only "not reached" was added, with item 5. (The PI decided it later the
   same day; part D applies the choice.)
5. **K5** (code F3, coherence M3).
   - run_job and J7 refuse J3-J7 after K5 or without J2's output, and J2-J7 after a J1 stop.
   - After any stop, the fix claim, the generality claim and the paired pattern are "not reached". With J4 or J5
     missing, the generality claim is "not reached"; with J2-J5 or a scored J7 missing, so is the pattern.
     (Reconciled in part E: after K2 or K10 the fix claim reads "failed", since K2 drops L8 and R1 and R2 are L8's own
     bars; it reads "not reached" after K5 or a failed implementation check. With no stop and no J3 output, the fix
     and generality claims are "not reached: J3 has not run"; if K6 or K7 fired, the fix claim reads "failed" or
     "void" instead, and the generality claim stays "not reached" (part F's "unless K6 or K7 fired", reworded in part
     G).)
   - A failed J7 check is defined (Kill and stop criteria).
   - The branch test covers K5, a J1 check failure, a J7 check failure and missing outputs.
6. **Guard** (code F2).
   - O6B_FILED must resolve to a commit (`git rev-parse --verify <x>^{commit}`) that is an ancestor of HEAD.
   - That commit's rules file must have `## Confirmatory rule, filed (YYYY-MM-DD)` as a whole line.
   - The script and every repository module it imports must be tracked in that commit and unchanged since. Modules in
     the segresearch submodule are compared with the submodule commit that the filed commit records. The list
     includes final_rules.py, which realblt_copy_latch loads outside sys.modules.
   - Every output records the commit, HEAD, the submodule's commit, the script's hash and the code files.
   - The guard now covers every fresh command (J0-J7 and `verdict`), not only the pool read.
7. **J1 redo** (integrity H1). J1 exits after a failed check, before R1, R2, the F6 counts, C4 or K2. Item 18 lists
   the redo.

*Record items:*
- **Smoke record** (integrity M1 and L2; compliance M2). The next note lists the edits made during the smoke run and the
  smoke's facts that bear on the rule. "Dev smoke runs excepted" is added to the file's header and to the Status.
- **Joint P** (numbers N4): restated with K2 at 0.7-5%, the pessimistic scenarios named, and P(stop at J1) added.
- **Change log** (integrity M3). Items 28-36 are added, and items 12, 16, 18 and 23 corrected. "Every difference" is
  dropped here, in the note above the DRAFT sections and in the draft's pointer.
- **"Fresh"** (coherence M6): corrected in the data paragraph.
- **Bar rows** (coherence M5): every row names its cell and metric, and the bootstrap percentile is stated.
- **F4's window line** (compliance L6, over integrity L5): corrected. s3_fresh.py:23 used 550-950 before its only run.
- **The Study A/B pre-registration** (integrity L6).
  - What happened, while this session checked whether realblt_fullft had run (times UTC, all before 67667ac; the
    full list since the third check):
    - 10:59:44: a grep of segresearch/notes.md for "fullft", "full fine-tun" and "study b", and a listing of
      segresearch/scale/cloud;
    - 10:59:55: a grep of the shared Study A/B pre-registration (segresearch/prereg/2026-10-13_scale150_bltft.md) for
      "ran", "result", "status", "cancel", "not run" and "deferred", case-insensitive; a grep of notes.md for "study b",
      "fullft", "scale150_bltft" and "bltft"; a grep of registry.py for "fullft";
    - 11:00:07: a grep for "fullft" over segresearch's scale/, *.py and *.sh and over segresearch/scale (nothing under
      scale/ matched), and a search of the git log for "fullft", "study b" and "bltft".
    - Added after the fourth check, which found them missing:
      - 10:59:50: a recursive grep -rl "fullft" over segresearch's md, json, jsonl, log and txt files and over this
        repository; a git log for realblt_fullft.py; and its first 50 lines;
      - 11:00:12: grep -rln "fullft" over segresearch's py, sh, ipynb and json files;
      - the listing filters: at 10:59:44, segresearch/ for "fullft", "kaggle" and "study", and segresearch/results
        for "fullft", "full_ft", "studyb" and "study_b"; at 10:59:55, prereg/ for "bltft", and
        ~/.cache/segresearch-blt/fullft.
      - The greps printed file names only: notes.md, files of this repository and realblt_fullft.py, and nothing
        under scale/.
  - What it showed: the file's status line and its 50M gate lines, and no Study A outcome.
  - The main session judged it no breach: the file is a plan, not a Study A results file.
- **Dev readings** (integrity M2, optional): the removed step 3 sentences and the old step 2 interval are quoted where
  they were corrected.

*Other findings, one line each (fixed, or no change and why):*
- compliance L1 (E10.SIVPO in F5): fixed in wording, "every cell and rate where L8 is scored (F5: D only)".
- compliance L2 (K8 when C1e is untestable): fixed; K8 reads "bound untestable".
- compliance L3 (change-log gaps): fixed (items 28-36; 12, 16, 18 and 23).
- compliance L4 (NOTES.md:386 and :405; #35): fixed. NOTES.md has dated corrections, and the #35 note scopes 0.26 per
  text to the quadrant texts.
- compliance L5 (DESIGN.md:286 and :288): no change to DESIGN.md, because it is the main session's design record at
  4b76f51. This section corrects both lines (the data paragraph, and "What each outcome licenses" in place of its
  section 6 for the two claims). A dated note in DESIGN.md is left to the main session.
- compliance L6 (F4's upper bound): fixed.
- compliance L7 (joint P): fixed.
- compliance L8 (K5 not enforced): fixed (item 5).
- compliance L9 (a trailing comma): fixed. The comma is stripped from both values, two self-test cases cover it, and
  no dev count changes.
- coherence L1 (K8's readings): fixed; "bound untestable" and "refill" are added.
- coherence L2 (change log): as compliance L3.
- coherence L3 (joint P and K2): as compliance L7.
- coherence L4 (E10.V8's purpose; L8b's scope): fixed. E10.V8 is a stated readout, and L8b's scope is reworded.
- coherence L5 (paired verdict: SIV+PO's rate, multiplicity, F2a's decided rows, ties under K6): this line said "no
  change now, because the paired verdict waits for the PI's choice". Addressed in part D, items 4 and 7.
- coherence L6 (provenance; the block file; rerun mismatches): fixed.
  - Every output records the code.
  - J1 refuses a block file that is uncommitted, modified or written under another filed commit.
  - A rerun mismatch is defined (Jobs).
- coherence M3 (K5, a failed J7, an incomplete pattern): fixed (item 5).
- coherence M4 (what the claims assert): fixed for the fix and generality claims ("What each outcome licenses"). The
  paired verdicts' sentences were added after the PI's choice (part D, item 5).
- coherence M5 (cell, metric, percentile): fixed (Verdicts).
- code F5 (doses as an ID kind; "limited to digits"): fixed. G3 covers 6-digit, letter and mixed IDs, doses are
  reported only, and K9 names each kind.
- code F6 (block file not checked): fixed (J1).
- code F7 (dose exclusion): fixed in wording; the rule now states the code's test. No code change, because the two
  differ only where a dose's digits sit inside a decimal (`5.5` excludes 5, which is conservative) or a thousand is
  written with a comma (`1,000` does not exclude 1000: 1 of 600 dev problems contains `1,000`, and it changes no dev
  draw; this last clause was added after the third check, numbers N9).
- code F8 (unscoped readouts; G1e): fixed. Rows name F1, and G1e is reported only.
- code F9 ("every reader stops at idx 1799"): fixed in wording; the whole-file loads are listed.
- integrity L1 (undated additions): fixed; dated in the dev record and NOTES.md.
- integrity L2 ("no job J0-J7 runs"): fixed; it now says "on fresh problems", dev smoke runs excepted.
- integrity L3 (the PI's choices by label only): fixed; the choices are stated in full.
- integrity L4 (commit checks; a stale self-test output): fixed by the guard and J1's checks. The self-test was rerun
  on the amended code.
- integrity L5 (s3 "selected nothing"): no change to that sentence, because the corrected F4 line shows the bound
  predates s3's run.
- integrity L6: recorded above (no breach).
- numbers N5 (doses fire on about 50-60%): restated and dated, with the old wording quoted.
- numbers N6 (refill tie order; "4-250 times"): fixed in wording.
- numbers N7 (C3f off-centre at E10): restated from dev step 2, to D 0 ± 1.5 and E10 +1 ± 2 (it was 0 ± 2).
- numbers N8 (a trailing comma): as compliance L9.

**D. Amendments after the PI's choices on F2b and the paired verdict (2026-10-30, post-smoke).** The PI's words and
the option texts are quoted in the Status. These were made with the smoke outputs, both checks and the PI's choices in
hand. No model ran, and no fresh problem was read. The code moved from 15ab486 to 6f0318c.
1. **F2b** (the PI chose to keep it, gated): F2b's wording and G1's gate (part C, item 2) are unchanged. No dev BLT-1B
   check was run.
2. **The paired verdict is split** (the PI's choice), following VERDICT item 4.
   - There are two verdicts: windowed (F1, F2a, F2b, F3 with each ID kind and the doses, and F6, at both rates) and
     released (F5 at D, with L8 against D beside it).
   - Each has a class (tie, SIV+PO better, L8 better or mixed, with "cost rows only" where it applies) and a reading
     fixed under Claims.
   - They replace the single pattern, which read:
     - "Reading the model adds something": at least one primary row is L8 better, and no within-window id_copy row is
       SIV+PO better. The primary rows were F5's two id_copy rows and the cost rows of F1 and F6. The within-window
       id_copy rows were F1's, F2b's and F3's ID kinds' (not the doses').
     - "A byte rule suffices on these templates": no row of F1, F2a, F2b, F3, F5 or F6 is L8 better.
     - "Mixed": anything else, reported row by row.
     - It was expected to read "reading the model adds something", through F5 alone.
   - Reason: within the window SIV+PO masks the whole copy, so L8 can only tie or lose there. F5's win comes from L8
     not firing in long texts, which a length cutoff might match (second check: coherence H3, numbers N2-N3,
     compliance M3). So the old labels would have read F5's win as "reading the model adds something" while every
     windowed row tied.
3. **The doses count in the windowed verdict**, as the rest of F3 does; the old pattern left them out of the
   within-window rows. Reason: the windowed verdict reads every row of its cells, and "L8 lost copies the byte rule
   kept" is as true of doses as of any other kind. (Superseded by part E: the doses are reported apart, as F4 is,
   because the byte rule also vetoes the recurring prefix before the dose.)
4. **The cost rows are no longer "primary".** They still move a verdict if one does not tie, but they cannot decide in
   practice (coherence H3), and the subsection says so. Under K6 the output notes that a tie on a cost row carries no
   information (coherence L5). (Superseded by part E: no cost row sets a class, and "they cannot decide in practice"
   was dropped; it rested on F1's and F6's rows only, and a dose cost row set the smoke's class.)
5. **Readings.** Each class of both verdicts has its reading pre-registered under Claims, with the statement that no
   outcome shows that reading the model is needed. This replaces DESIGN.md section 6's reading and the scope option's
   "which settles whether reading the model is needed".
6. **Predictions.**
   - F1, F2b and F3 are relabelled "tie or SIV+PO better". P(F1 at D SIV+PO better) ≈ 0.17 with dev as truth
     (≈ 0.28 with dev's uncertainty, added in part E), or 0.6-0.8 at 97% recall. F3's per-kind ≈ 0.06-0.10 was
     left out here (added in part E). P(at least one windowed id_copy row SIV+PO better) ≈ 0.3-0.65; that was the
     second check's P("mixed") under the old pattern, at dev's recall (98.3%), with F2b and the letters counted as
     testable. Part E replaces it with the gated classes' odds.
   - F2a is labelled decided by construction. (Dropped in part E: no rule touches F2a's copy, yet smoke idx 40
     flipped, so a non-tie there is a side effect of the refit.)
   - The doses' "≤ 30%" basis is replaced; part C (numbers N5) had already superseded it.
7. **Also reported** (coherence L5):
   - SIV+PO's own matched rate in each cell.
   - No correction for the number of rows is applied: the margins keep a true difference of zero out of both
     classes, and every row that does not tie is listed. (Dropped in part E: under a true zero, P(some windowed row
     not tied) was ≈ 0.11-0.14 over the 92 rows. Part E states the null rate of the class rows instead.)
8. **Code and tests.**
   - paired_verdict returns the two verdicts, each with its class, its reading and the rows that did not tie.
     classify names the other side (SIV+PO or D).
   - The branch test covers each class of both verdicts, the doses, F4 apart, a missing J5 or J7 and the K6 note.
   - The self-test and the branch test were rerun.
   - verdict_from also ran without error on the raw smoke outputs, at 13:57:48 UTC. This line said "its classes, on
     10 texts per cell, are not read"; corrected after the third check, they were printed and are recorded here:
     - windowed "SIV+PO better (cost rows only)", read "L8 cost more than the byte rule". It came from one row, F3
       dose, D, C2: -0.0052 [-0.0093, -0.0010] on 2 texts;
     - released "L8 better", from F5, D, id_copy bits per copy.
     - No bar, class rule or prediction changed after that print until part E, which changes the class rules because
       of it (see the smoke note).
     - Corrected after the fourth check (integrity I4): the print at 13:57:48 showed the classes and their readings.
       The deciding rows named above were found later, by the third check.

**E. Amendments after the third check (2026-10-30, post-smoke; 0646d55,
analysis/2026-10-30_copyfix/recheck/VERDICT.md and audits.json).**
- These were made with the smoke outputs, three checks and the PI's choices in hand. They involve no new PI decision:
  they follow the main session's calls under the PI's earlier choices.
- No model ran, no fresh problem was read, and nothing was posted to #35.
- The code moved from 6f0318c to 264fc6c. The self-test, the branch test (44 cases) and the odds script ran on it.

*Must-fix items (the third check's VERDICT section 2):*
1. **Copy rows with the hazard set each paired class** (coherence H1, numbers N1, coherence M2, coldread M4, compliance
   L1; critic gap 5).
   - The windowed class rows are id_copy exact, L8 - SIV+PO, in F1, F2b and F3's 6-digit, letter and mixed IDs. Each
     counts at D where its cell's D-all - D ≥ +20, and at E10 where its E10-all - E10 has a lower bound > 0.
   - The released class row is F5's id_copy exact at D, with no gate.
   - "Not testable" applies if no windowed row passes its gate. The "(cost rows only)" classes are dropped.
   - Every other row is reported beside the classes with a pre-registered sentence, the doses and F4 apart. Each
     id_copy row that does not tie lists its discordant texts.
   - Readings: tie says "within 2 points (not an equivalence test)"; SIV+PO better names its cells and rates; the
     released reading says "relative to SIV+PO" and points to the by-length readout. DESIGN.md:295's F5 line has no
     successor.
   - Dropped: F2a's "decided by construction", "the cost rows cannot decide in practice" and "the margins keep a true
     difference of zero out". The class rows' null rate is stated instead.
   - Reason: any of the 92 windowed rows could set the class, dose and cost rows included, whose readings would then
     be false, and the smoke's class came from one dose cost row. The change implements "a loss means L8 dropped
     copies".
   - Branch test: each of these leaves the class the copy rows set: a dose C2-only row, a dose bits-only row, an F2a
     flip, an F1 bits-only row, an F2b copy row without the hazard, and an L8-better cost row beside a copy loss. One
     more case has no row passing its gate.
2. **Smoke record** (integrity I1). The classes printed at 13:57:48 UTC, with their deciding rows, are recorded in part
   D, item 8, and the dose-row and letters facts are added to the smoke note. (The deciding rows were identified
   later, by the third check; see part D, item 8. Noted in part G.)
3. **Readings for stops, K6 and a failed generality claim** (coherence M1, coldread M2; compliance L3, coherence L2,
   code L2).
   - Six sentences are added under "What each outcome licenses".
   - The fix claim's scope reads "failed" after K2, K10, K6 or K7, and "not reached" after K5 or a failed check.
   - K6-K9 are read only without a stop.
   - Part C, item 5, is reconciled.
   - G2 (a) moved into J1's implementation checks, since J1's patcher outputs cover every F2a text.
4. **J3 missing** (code M2, compliance L2). With no stop and no J3 output, the fix and generality claims are "not
   reached: J3 has not run". If a bar has already failed (K6 or K7), the fix claim reads "failed" instead, since one
   failed bar voids it. A branch-test case covers J3 missing.
5. **Guard, provenance and job order** (code M1, coldread M1; compliance L6, coherence L6, coldread L1, coldread L6;
   critic gaps 1 and 3; the code lows).
   - The filing check: filed_check requires that the named commit introduced its heading line, outside code fences and
     HTML comments, with a valid date no earlier than 2026-10-30.
   - Re-filing is defined (J1; Kill and stop criteria): a new dated heading, the disclosure and the PI's approval
     quoted under it, J0 rerun, and every job again once.
   - The model: HF_HUB_OFFLINE=1 is set before transformers loads, and snapshot 91aa6b8e is checked. Provenance records
     the snapshot, the library versions, modeling_blt.py's sha256 and the device, and Code names the model.
   - Hashes: each covered file's sha256 is recorded, and J1 records the patcher outputs' sha256.
   - The job gate:
     - each job runs once per filing;
     - earlier outputs must be committed, unmodified and pushed;
     - J2-J7 and `verdict` must match J1's record;
     - J2-J7 check the patcher outputs' hash.
   - Code lows:
     - verdict_from without J1 gives "not reached";
     - R1's E10 half is "not testable" when no F1 copy has an E10 start at c1;
     - provenance checks the filing once per process and records nothing unchecked.
   - Wording: the filing paragraph and the self-test list say exactly what is checked, and Jobs no longer says J7
     reruns the patcher check.
   - Self-test cases:
     - refused: a later commit with changed code; HEAD after a later commit; a heading in a code fence, in an HTML
       comment or dated before 2026-10-30;
     - passes: a re-filing commit.
6. **Letters** (coldread M3). The smoke fact is added, with G3's testability per kind and P(neither F2b nor letters
   testable) ≈ 0.53-0.77 (see part G). F3's paired row covers kinds without the hazard.
7. **What it cannot show** (coldread M5; coherence L7, coldread L2; critic gap 4).
   - The list is added under Claims.
   - The fix sentence says "teacher-forced" and "read on point estimates", and it carries the measured gain and C1's
     share.
   - The generality sentence says "in the same `\nTicket` template".
   - Each reading is quoted whole.
   - F4 and F5 enter no claim.
8. **Predictions** (numbers N2, compliance L7).
   - F1 at D: ≈ 0.17 with dev as truth, ≈ 0.28 with dev's uncertainty.
   - The recall behind the old 0.3-0.65 is stated (98.3%).
   - The gated classes' odds are restated (copy_latch_paired_odds.py).

*Also in this round:*
- **Doses drawn although the problem contains them** (critic gap 6). draw() now tests the question, solution and
  answer apart, and a self-test case uses train idx 71's pattern.
  - 15 of train idx 0-1799 hide a dose this way, so about 0.15 of the 175 fresh dose texts would have drawn such a
    dose ("about 0.1" until part F). The fix changes the dose in about 0.6 expected texts, all in such problems (Data,
    part F).
  - The smoke's dose problems were not affected, and non-dose draws are unchanged.
- **Dev-basis lines** (critic gap 7).
  - The split difference is stated in the data paragraph.
  - F4's window keeps 29%, and J0 should visit about 4,600 problems (Blocks; Jobs).
- **Filing** (critic gap 8): the O6 session makes the filing commit only after the main session relays the PI's words,
  quoted under the heading.
- **Option texts** (compliance L4): the three shortened choices are stated in full.
- **C1e's asymmetry is intended** (coldread L3): one line under "C1e and the +15 target".
- **The third check's section 3:**
  - G1, G1e and G3 labels: when untested, holds is None and a status says why (code; branch-test asserts).
  - Branch-test count: the test prints it (44), and Code and the smoke note are corrected (part D's "38" was 37).
  - Stale or conflicting lines, now corrected:
    - DESIGN.md section 6's replacement (Status);
    - K8's readings, "patterns" and G2's split (change-log items 5, 12 and 14);
    - K2's risk wording (K2);
    - "L8 was not measured there" and the doses' "never measured" (paired table; the smoke saw 2 dose texts);
    - part D, item 6's missing per-kind odds;
    - the PI's labels in full (paired subsection, part D, code);
    - "threshold fitting" in "fresh".
  - Undated edits in the frozen subsection: dated (SIV+PO's rate clause, L8b's scope).
  - The dated note on the refill (dev step 2): corrected.
  - The Study A/B note: every term and both later greps are listed.
  - Old-wording pointers: part C points to RULES_ecca81e.md and to c03bd14.
  - Wording: K9 says "does not extend to" (rule and code), and the joint P is labelled "C4, K2 and R1 only".
  - #35: no more comments. The existing one is left as it is.

*Unverified items, one line each:*
- coherence L1 (F4's "tie if L8 fires on ≥ 80%"; F3 at 90%; ranges reaching +2): fixed in the paired table. F4 is
  SIV+PO better with P ≈ 0.7-1.0 at 90-95% firing, and F3 0.95 at 90%. The ranges reaching +2 stay, under "tie or
  SIV+PO better".
- coherence L4 (K8's first-match order): no change, because K8 is reported only and both numbers are printed beside it;
  a line says so.
- coldread L7 (J0 visits about 4,500): fixed; about 4,600, the critic's count.
- numbers N3 (the dose rows disagree): fixed; "at most about 60%" in both places.
- numbers N5 (F6's C2 nearer -0.0003): no change, because the sentence that cited -0.001 is dropped and no cost row
  sets a class now.
- numbers N6 (C3 spreads about ±0.7): fixed in item 23. C4s's prediction, which cites it, is widened to 0 ± 0.7
  (item 29).
- numbers N9 (the `1,000` case affects no dev draw): fixed in code F7's line.

**F. Amendments after the fourth check (2026-10-30, post-smoke; finished after midnight, 2026-10-31 -0300; 9fe860b,
analysis/2026-10-30_copyfix/fourth_check/VERDICT.md and audits.json).**
- These were made with the smoke outputs, four checks and the PI's choices in hand. They involve no new PI decision:
  they follow the main session's calls.
- No model ran, no fresh problem was read, and nothing was posted to #35.
- The code moved from 264fc6c to 56b2d00, and then to 3d19131 (below, under precedence). Each code commit was
  followed by the self-test on the committed code, whose JSON names it (0f4f303, then 8ab1fce). The branch test (50
  hand-set cases) and the odds script ran on 3d19131.
- The section as it stood before part F is at 628da50; before part E, at
  analysis/2026-10-30_copyfix/recheck/integrity/RULES_eeb487b.md.
- No class rule, bar, stop or claim logic changed beyond the main session's list. The new refusals, the labels and
  sentences below, and two report-only additions (equivalence L6; coherence L6, item 5) are the code's differences
  from the frozen derivations.

*Must-fix items (the fourth check's VERDICT section 2):*
1. **Paired readings** (equivalence M1, compliance M3, coherence H1-H2 and M2-M3, coldread CR-H1 and CR-L5).
   - Both tie readings now say what the class rule shows: no copy row is classed as differing, with the largest
     difference, its row and its interval filled in, and "This is not an equivalence test." They said that no copy row
     differs by 2 points or more, yet a tied 100-text row can sit 3-5 points apart: 3 lost and none kept is -3 [-7, 0].
   - Expected gives how often a tie includes a 100-text class row 2-5 points behind: ≈ 0.47-0.51 of all outcomes with
     dev as truth, ≈ 0.37-0.39 with dev's uncertainty (0.44-0.46 without the E10 rows). (Corrected in part G: the
     figure is for a tie with any class row 2 points or more from zero; see Expected.)
   - "A tie shows that this byte rule suffices on these templates" became "A tie shows only that, on the copy rows
     with the hazard, no difference reached the class rule; the cost rows and the cells reported apart are read beside
     it." So "tie" covers only the gated copy rows, narrower than the option's "a byte rule suffices here"; the main
     session tells the PI.
   - Both "SIV+PO better" readings give the discordant split: copies L8 missed (it did not fire at c1) against side
     effects of the refit (it fired). The class rule is unchanged, and there is no "unresolved" class.
   - Rows beside the class: a copy row without the hazard reads "No hazard here (<gate>), so a difference here is not
     a fixed or missed copy; it is reported only."; F2a's cost rows take the cost sentence; the doses' cost rows keep
     the prefix sentence, their bits row takes the bits sentence, and their copy sentence gives the split.
   - Branch test: a tied F3 class row at -3 [-7, 0] gives the new tie reading with those numbers.
2. **Re-filing** (compliance M1-M2, coherence M1, coldread CR-M3 and CR-L10; guard G1, G7 and G10; critic gap 3).
   - Rule (J1): the earlier filing's outputs stay in the history as committed and pushed; the verdict lists what
     differs from them; a re-filing changes only the code its disclosure names; what applies if the PI declines; which
     result counts; crashes; a lost patcher file.
   - Code: outside smoke runs read_out reads only outputs written under the current filing, so J3-J7 wait for the
     re-filed J2. `verdict` refuses unless a stop applies or J2-J7 all exist under the current filing; this also
     closes guard G7, an early `verdict` using up its one run. cmd_verdict reads the earlier filing's outputs from the
     re-filing commit's own tree and lists every bar, claim, class, kill or stop that differs.
   - A second failure of the same job's checks stops the test across re-filings: the gate counts the failed J1 and J7
     outputs in the history (guard G10).
   - Self-test: earlier-filing J2-J7 beside a re-filed J1; J3 and `verdict` refused; the differences listed.
3. **Claim sentences** (coldread CR-M1 and CR-M2; critic gaps 4 and 6).
   - "in BLT-1B with its 512-byte local windows" is in the fix sentence, its default-rate form and the generality
     sentence. A scope line under Claims says the claims cover the base weights with 512-byte windows, and the
     conversion as released enters only the released paired verdict.
   - K5's, K7's and K9's sentences carry the template, the windows, point estimates and the number of texts; K9's also
     "at the default rate".
   - Under Dev numbers: P(fix claim) ≈ 0.80-0.99 (about 0.47-0.86 under K2's pessimistic scenarios), P(generality
     claim | fix claim), P(generality not testable) ≈ 0.002-0.007 and the most likely scope (6-digit and mixed IDs at
     D, P ≈ 0.43-0.67), each with its basis. K8's P ≈ 0.5 stands beside K8.
4. **Odds** (numbers N1-N2; compliance L1, coherence L2, coldread CR-L2, integrity I1 and I7).
   - The script is exact: each row's class is read from a table of the class rule over every (lost, kept) count, and
     the counts' distribution is summed exactly, so the Jeffreys figures carry no simulation noise (see part G).
   - The 6-digit and mixed IDs' E10 rows are added, with gate chances 0.5-0.9 and 0.4-0.7; the basis is F1's dev E10
     hazard, with D breaking 2 of 3 6-digit and 1 of 2 mixed smoke copies. About 4.9-6.0 class rows are expected.
   - Restated: the null rates (≈ 0.006-0.008 at 0.5% each way, 0.042-0.057 at 1%, 0.21-0.25 at 3-10% (see part G); they
     were 0.006-0.03 and 0.17-0.20); P(tie) ≈ 0.69-0.72 with dev as truth and 0.52-0.55 with dev's uncertainty
     (0.535-0.560 exact without the E10 rows); F2a's 300-text row ≈ 0.045. The main session asked for 0.046-0.048, the
     simulated log's figure; the exact rerun gives 0.045 at both 3% and 10%.
   - The Code line says where the gate chances come from.
5. **Job gate and filing guard** (guard G2-G4; compliance L6).
   - gate() runs `git fetch` first and refuses if it fails. It refuses when HEAD has no upstream or is behind it, when
     a tracked output is missing from disk, and when any commit after the filing commit, on HEAD or on its upstream,
     touched the job's output file. Self-test: the rm, reset and git-rm routes are refused.
   - filed_check requires one parent, a text-only change against it, and code equal to the code commit its Code
     section pins. A re-filing takes two commits, the fix and then a re-filing commit that pins it (J1; the J7 stop).
     Self-test: a filing commit that also edits code and a merge are refused, as is a re-filing commit whose pin is not
     its code; a two-commit re-filing passes.

*Safeguards (the critic's blocking gaps):*
6. **The filed text is frozen** (critic gap 2; coldread CR-L7). Filing says what the filing commit may change (the
   heading and the Status lines, which then say filed and approved) and that the main session checks its diff; after
   filing, only dated notes are appended. Every output records the rules file's sha256, and `verdict` prints `git diff
   --stat` of the rules file since the filing commit.
7. **An independent check of part F** (critic gap 1): the main session reruns the fourth check's harnesses at 3d19131
   before the PI sees the text. Nothing was needed here.

*Also in this round:*
- **Call 1** (compliance L3): part C's reconciliation (item 5) adds "unless K6 or K7 fired", and a branch-test case
  gives "failed: K7 (C1)" with J3 missing and C1 failing.
- **Precedence of readings** (coherence L1, compliance L4, coldread CR-L1).
  - K7's reading governs, and K6's follows it. K6 alone labels the fix claim "void: the control showed no cost (K6)"
    (it was "failed"); with K7, "failed: K7 (...); and void: ...". With J3 missing, "failed" if K7 fired, "void" if
    only K6.
  - K9's "does not extend to" sentence prints only while the fix claim holds; otherwise the generality claim reads "No
    generality claim: the fix claim failed." After K6 alone it reads "... is void (K6).", so as not to call a void
    claim failed; that variant is this session's.
  - 56b2d00 had also dropped K9 from the kills list when the fix claim did not hold, which would have changed a kill
    record beyond the list. 3d19131 restores it: K9 is read whenever no stop applies, as at 264fc6c, and only its
    sentence waits on the fix claim. The kill strings, the G3 status and the generality parts keep their internal kind
    names, as the derivations read them; the printed sentences use display names.
- **A row's class** (equivalence L3): "the 95% interval lies above 0 (below 0 for SIV+PO better)".
- **Printed sentences** (equivalence L7-L8, coldread CR-L11): `verdict` prints every licensed sentence with its
  numbers filled in, under display names ("F3 6-digit IDs", not "F3 digits6"), in the order Claims gives.
- **Stated as the code does it** (equivalence L9): C4s is read at D only when C1e is untestable; with J2 missing,
  "not reached: J2 has not run".
- **What it cannot show** (compliance L5, coherence L5): gains past the onset-removal bound at E10, and doses and other
  short or common strings, which S0 excludes.
- **Smoke record** (integrity I4 and I5; "coldread CR-L12" until part G): the 13:57:48 print showed classes and
  readings, and the deciding rows came from the third check; and why the released smoke class went from "L8 better" to
  "tie".
- **Wording** (coldread CR-L4 and CR-L7-CR-L9; integrity I3): the released class row names the reverse effect;
  "self-contained apart from section 1's definition of L8"; the library versions are matched to J1's record, not
  pinned; the pointers to the pre-part-E and pre-part-F text (above, and in part C's header).
- **Records** (critic gap 5): J0 records train.jsonl's sha256 and line count, after filing; provenance adds numpy's
  version and the weights' blob name, and J2-J7 and `verdict` compare both with J1's record.
- **Guard G8**: the window is checked with an explicit exit that python -O keeps, the hub's offline mode is read from
  huggingface_hub as well as from the environment, and the blob name is recorded and compared.
- **The self-test JSON** (section 3): written by the committed code and naming it (Code).
- **For the note to the PI** (coldread CR-L3, CR-L12 and CR-L13; integrity I5): the Mac time against the option's
  "About 2-3 h", and that a re-filing reruns every job (Jobs); the likely generality scope (Dev numbers); the class
  rules' revision after a smoke verdict was printed (smoke note).
- **#35**: no comments.

*Dispositions of the other findings, one line each:*
- equivalence L1 (branch case 15): fixed; the case sets one copy L8 missed and R1's recall at 0.9, as J1 writes them.
- equivalence L2 (G3's 0.7 share): fixed; it allows 1e-9, as C1e (i) does, and G3's row says it is exact on counts. A
  branch-test case (70 of 136 against 100 of 136) covers it.
- equivalence L4 (G2 (a) read from J4 when J1 lacks it): no change, because only a J1 written before part E lacks G2
  (a); at the pin J1 always writes it and exits on its failure.
- equivalence L5 (J7 scored after a failed check): no change, because only smoke runs score F5 after a failed check;
  cmd_j7 exits otherwise.
- equivalence L6 (class rows dropped while a job is missing): fixed, report only; the verdict output keeps the class
  rows found so far, with their discordant texts, when the windowed class is not reached.
- equivalence L10 (outputs the gate refuses): no change, because the gate and blocked refuse those jobs, so no such
  output can exist.
- guard G5 (filed_headings against Markdown): no change in code, because each case needs unusual Markdown or a control
  character, the rules file has none, and the main session's check of the filing diff would show one. The
  approximation is stated under "What the code checks".
- guard G6 (a later commit that repeats a heading): no change in code, because such a commit must now have one parent,
  change text only and hold the code its pin names, so it can only file the pinned code. It counts as a re-filing,
  which needs the PI's approval, and the failure count spans every filing. Stated under "What the code checks".
- guard G9 (assume-unchanged, skip-worktree, clean filters): no change, because each needs a deliberate local git
  setting, and every output's file hashes, read from disk, would show such an edit against the pinned commit's files.
- guard G11 (two misleading messages): fixed; the gate names a missing upstream, and a missing block file ("run J0
  first").
- builders B1 and numbers N4 (the dose fix): fixed in wording. About 0.15 of the 175 fresh dose texts would have drawn
  a dose the problem contains ("about 0.1" before), and the fix changes the dose in about 0.6 expected texts, all in
  such problems (Data; part E's line).
- builders B2 (the dose self-test case): fixed; it calls draw() over 400 seeds on train idx 71 and checks that the
  joined test would let 25 through.
- coherence L3 (ranges reaching +2; F2a's figure): fixed. A line under the paired table says the +2 ends touch the
  L8-better margin, and F2a's row cites the odds log.
- coherence L6 (redo edge cases): fixed. A J1 that ends with no output counts as a crash (one rerun under the same
  filing, kept by procedure, since the gate cannot see it); failures count per job across re-filings; K5 reads "L8's
  copy fix was not scored"; the default-rate sentence reads "so the fix was not tested at that rate"; each job's
  patcher-rerun mismatch count is carried into the verdict output.
- coldread CR-L6 (paired table labels): fixed; one figure per scenario for F1, and "G3's predicted firing" for F3.
- compliance L7: fixed; part A states the PI's choices, G1e's field is named, and the branch test
  asserts its status.
- integrity I2 (the Study A/B note): fixed; the two greps and the listing filters are added.
- integrity I6 (small edits): logged here. Unlogged until now: part E deleted the released "mixed" reading
  (unreachable with one class row), changed "filed commit" to "filing commit" in the clean text, and changed K2's
  stop label in the code to "K2: a new-value removal from E10 in F6". Dated where they stand now: the K8 first-match
  line, "F4 and F5 enter no claim", the Odds bullet and code F7's last clause.
- numbers N5 (C3's and C4s's spreads): fixed in wording; about ±0.7 with dev as truth and about ±1.2 with dev's
  uncertainty, in C3's and C4s's rows and items 23 and 29. No bar is involved, and the predictions keep their ranges.

**G. Amendments after the fifth check (2026-10-31, post-smoke; 274bafb,
analysis/2026-10-30_copyfix/fifth_check/VERDICT.md and audits.json).**
- These were made with the smoke outputs, five checks and the PI's choices in hand. They involve no new PI decision:
  they follow the main session's calls, chosen to keep the rule simple.
- No model ran, no fresh problem was read, and nothing was posted to #35.
- The code moved from 3d19131 to 4026bec, and the self-test then ran on the committed code (its JSON, committed at
  5dd8d76, names 4026bec). The branch test (54 hand-set cases) and the odds script ran on 4026bec; the odds log is
  unchanged.
- The section as it stood before part G is at 18dc460.
- No class rule, bar, stop or claim logic changed. The code's new outputs are labels and records: which result
  counts, K7's two printed parts, the sentence after a first J1 check failure, the discordant texts' crossed counts
  and indices, and the refusals of python -O and of a threshold mode other than train.

*Must-fix items (the fifth check's VERDICT section 2):*
1. **Which result counts** (compliance C-M1, coldread M2 and L4).
   - Only a J1 check failure earns a redo whose re-filed chain is the result, with the differences beside it, since
     J1 exits before any decision-relevant number.
   - After a J7 check failure, or any other bug found once scoring has started, the filed code's outputs remain the
     confirmatory result, and the filed chain is first run to its end under the filed code. After a J7 failure F5
     stays unscored there and the released verdict is "not reached". A corrected rerun needs a re-filing the PI
     approves, and it is a deviation beside the result, never in its place.
   - Code: result_of, called by cmd_verdict, reads the redo from the earlier filing's J1 stop. It prints
     "Confirmatory result (filing <commit>)" and then, unless the re-filing was a J1 redo, the re-filed sentences under
     "Deviation: corrected rerun (re-filing <commit>)"; after a J1 redo it prints "Result: re-filed chain (J1 redo)"
     instead ("instead" added in part H). The output records which result counts (`result`), and J7's exit message
     says the same.
   - Self-test: one case per branch (a J1 redo, a J7 check failure, another bug).
   - Until part G a J7 check failure also earned a redo whose re-filed chain counted, though J7 runs after F1's rows
     are scored and read.
2. **K7's sentence** (equivalence M1 and L1, coldread L5).
   - Two parts, each printed only if it has a bar: F1's bars under the template, now marked "(F1)"; F6's bars as "On
     the same problems without the ticket (F6), L8 failed: ...", with C4's count. The rate bar goes under the cell
     whose rate failed.
   - "On point estimates" is dropped from K5, K7 and K9 (the rule, TEMPLATE and F2b's lead): C0, C1, C1e (ii), G1 and
     G3 also have lower bounds.
   - Branch test: C4 at 5, a failing C4s, a failing F6-only rate, a C1 that fails only on its lower bound, and F1's
     rate.
3. **The gate's limits** (guard GATE-1 and GATE-2), by text only.
   - An output committed and pushed still counts after rm, a reset or `git rm`. The gate cannot see an output deleted
     before it was committed, or a commit removed before or after it was pushed. So each output is committed and
     pushed as soon as its job ends, and no output or its commit is deleted, reset away or force-pushed away: kept by
     procedure.
   - The self-test's reset case, whose output had been pushed, is relabelled "(pushed, then reset)".
   - The failed J1 output is committed and pushed before the fix (guard GATE-4). Only check failures are counted
     across filings, and K10, K2 and K5 are kept by procedure (guard GATE-3).
4. **The re-filing commit** (coldread M3; compliance C-L1, guard GUARD-2 and VERDICT-1). Filing lists what a re-filing
   commit may change: the pin line in place, the Status lines, and one heading appended at the end with the disclosure
   and the PI's words. The main session diffs it against the previous filing commit. `verdict` also prints the rules
   file's diff stat from the earlier filing commit.
5. **The tie paragraph** (numbers N5-1 and N5-2, integrity INT-L1, coldread L1). Expected's paragraph takes the
   verdict's text.
   - The figure is for a tie with a class row 2 points or more from zero, either way and of any size.
   - 2 or 3 lost and none kept tie, and 4 lost and none kept does not (-4 [-8, -1]). With copies moving both ways, a
     100-text row can tie further behind.
   - Part F's copy of the old label carries a dated correction.

*Also in this round (the fifth check's VERDICT section 3):*
- **Records.**
  - The discordant-text record (equivalence M2, integrity INT-M1 (b), coldread L8). Since 56b2d00 it held only totals
    per direction, which part F did not log. Part G keeps the totals, which the split reads, and restores the crossed
    counts (direction by L8's firing by X's start). It also adds the texts' problem indices, so each non-tied row lists
    its texts as the rule says. Report only.
  - The second-failure labels (integrity INT-M1 (a)). Since 56b2d00, J1 and J7 write "implementation check failed
    again (second failure)" ("(J7) again" for J7) when an earlier committed output of the same job failed its checks,
    and their exit messages say what follows. Part F did not log this.
  - Changes that part F made or kept without logging them (compliance C-L7 and C-L8; integrity INT-L3, INT-L4 and
    INT-L5), dated here (2026-10-31):
    - K9's F2b lead names F2b's template, not O6's. The G8 checks went past the window check the main session asked for:
      the offline mode is read from huggingface_hub, windows are refused outside J7 and recorded, and the blob is
      compared. K7 comes before K6 in the kills list. sentences() prints two status lines for a claim not reached
      without a stop; they are now listed under Claims, and only smoke runs reach them (corrected in part H). The
      no-hazard sentence's gate has an E10 form. The scope sentence sits under Claims, and "What it cannot show" now
      points to it.
    - Dated in place but missing from part F: J0's counts named (coldread CR-L10 of the fourth check); dev's
      discordance figures (coherence M2); Expected's 0.91-0.93 and "mixed ≈ 0-0.01" (numbers N1 and N2); "not
      reached: J1 has not run" among the fix claim's labels (equivalence L9); the window record and its refusal
      outside J7 (guard G8); the smoke note's entry on part F's verdict run.
    - Fourth-check findings with no named disposition in part F:
      - equivalence L11 and integrity I8: fixed under part F, item 2;
      - compliance L2: fixed under item 4 and coherence L3;
      - coherence L4: fixed under item 1 (F2a's cost rows take the cost sentence);
      - builders B3: fixed under "The self-test JSON";
      - numbers N3: its second point is fixed now. F2a's row says "each way", and the null no longer cites F2a's
        smoke flip as support for 3-10%.
    - Slips corrected:
      - part F's smoke-record line cites integrity I4 and I5 (it said coldread CR-L12);
      - part E, item 2 notes that the deciding rows came later;
      - the third check's attribution in Jobs again names the committed-and-pushed rule and its date;
      - "What the code checks" again carries its date;
      - the stops' sentences are listed in the order `verdict` prints them.
- **Status quotes** (coldread M1). A dated "Since then:" line stands under the paired-verdict option (what a tie
  shows; the split) and under the scope option (nothing settles whether reading the model is needed; 2.3-3.9 h).
- **Procedure edges.**
  - After a first J1 check failure, `verdict` prints "not reached: the implementation check failed; a redo needs the
    PI's approval". "The implementation failed; nothing was tested." needs a second failure, or the PI declining,
    which the main session reports (compliance C-L10, integrity INT-L2).
  - A lost patcher file is recomputed by hand with the pinned code and logged in a dated note. It is accepted only on
    a matching sha256; otherwise the test stops (compliance C-L3, coldread L3, guard GATE-5).
  - provenance refuses python -O (or PYTHONOPTIMIZE), which would drop the subset asserts and rebuild's comparison.
    It also refuses a threshold mode other than train, and every output records the mode (guard GUARD-1, compliance
    C-L4, coldread L9).
  - earlier_filing keeps only the outputs written under the earlier J1's filing, so two earlier filings are never
    combined (guard VERDICT-1).
  - Part C's "unless K6 or K7 fired" now says what each claim reads (compliance C-L2).
- **Numbers and wording.**
  - The 3-10% null reads ≈ 0.20-0.25: the log gives 0.205-0.250, and a model faithful to the code's bootstrap
    0.20-0.24 (numbers N5-3). "Exact" is explained: exact over the counts, with one seeded bootstrap draw per count.
  - F2a's row: "if 3-10% of copies flip each way" (compliance C-L5).
  - P(neither F2b nor letters testable) ≈ 0.55-0.78 (numbers N5-9, coldread L2).
  - P(fix claim)'s basis names C1e's 0.96-1.0 and shows the product (numbers N5-5, compliance C-L6).
  - P(generality | fix)'s basis adds G2 (b)'s F2a cost conditions (fail P ≈ 0-0.03) and says the testabilities are
    taken as independent (numbers N5-6).
  - Expected and the null are given that the windowed verdict is reached (numbers N5-8).
  - Expected adds "SIV+PO better" ≈ 1.00 at 95% recall, R1's floor (coldread L6).
  - [JW0] is defined, the Implementation section is named, and "VERDICT item 8" is tied to the first check (coldread
    L7).
  - The scope line names the checkpoint, itazap/blt-1b-hf (coldread L10).
- **Not changed** (the main session's calls):
  - no reflog code in the gate (M-3 is fixed by text only);
  - no command that recomputes the patcher file;
  - the E10 gate chances stay as stated, a conservative judgment (numbers N5-4: every stated probability holds within
    about 0.01);
  - the Jeffreys figures keep one truth per row (numbers N5-7: one shared truth per rate gives a tie ≈ 0.58-0.60);
  - the compare harness's parsers (compliance C-L9): the main session uses the fifth check's patched copy.
- **#35**: no comments.

**H. Amendments after the final confirmation (2026-10-31, post-smoke; 920af7e,
analysis/2026-10-30_copyfix/final_confirm/VERDICT.md and audits.json).**
- Text only. No code file changed: the pin stays at 4026bec, and the self-test JSON (5dd8d76), the branch-test log
  and the odds log stand as they are.
- These involve no new PI decision: they follow the main session's calls, which take the confirmation's text fixes,
  not its code alternatives.
- No model ran, no fresh problem was read, and nothing was posted to #35.
- The section as it stood before part H is at 2e9a1d4.

*Must-fix items (the confirmation's VERDICT section 2), in its words, under Which result counts:*
1. **A bug between J1 and J2** (final PI-M1; equivalence F-3 and final PI-L3).
   - A bug found before J1 has written any output (in J0, or J1 crashing twice) is handled as a J1 check failure.
   - From the moment J1 has written an output that passed its checks, a bug counts as found once scoring has
     started.
   - The status lines for a claim not reached are printed by smoke runs and by an earlier filing's chain that stopped
     short, printed as the confirmatory result beside a corrected rerun.
   - Part G said "only smoke runs reach them" (part G, Records). That was wrong: a second J2 crash, for one, prints
     them too. Corrected here, with a pointer where part G says it.
2. **A second re-filing** (equivalence F-1 and guard G6-1; compliance L-1 and final PI-L2). There is at most one
   corrected rerun, and it is never itself re-filed. `verdict` compares only with the filing just before, so this is
   kept by procedure.
3. **The reading code** (compliance M-1). `verdict` reads the earlier chain with the re-filed code. Where a fix
   changed the code that reads the outputs, the earlier filing's committed `verdict` output is the confirmatory
   result. The main session compares the two before reporting: kept by procedure.

*Same pass (the confirmation's VERDICT section 3, text only):*
- J7's exit message: its clause holds under the filing, and under a corrected rerun `verdict`'s label governs
  (equivalence F-2, guard G6-2).
- The verdict JSON: its top-level claims are always the current chain's, and `result` says which counts (compliance
  L-2). This sentence replaces "Its output records which result counts (`result`)."
- After a J1 redo, `verdict` prints "Result: re-filed chain (J1 redo)" instead of "Confirmatory result"; "instead" is
  written into Which result counts and into part G, item 1 (compliance L-3).
- Changes since 18dc460 that part G did not list (compliance L-5), dated here (2026-10-31): the smoke note's entry on
  part G's verdict run; ", exact" dropped from F2a's row in the paired table; the pointer from Dev numbers to the step
  1 corrections; "as every re-filing does" in the redo's paragraph.
- Superseded figures in older change-log parts now point to part G (compliance L-6): part E's "≈ 0.53-0.77", and
  part F's "The script is exact ..." and "0.21-0.25".
- Discordant texts: "the crossed counts restored and the indices added in part G" (compliance L-6). It said "the
  indices and the crossed counts restored", but part G added the indices.
- Subset-assert crashes (final PI-L1, second half). By procedure, a failed subset assert in J1 counts toward the
  two-failure stop, which the gate cannot see for it. The crashes it causes under one filing count as one failure of
  J1's checks, so the crash rule's rerun and the redo of a bug found before J1 wrote anything still apply once.
- Status (final PI-L6): a dated "Since then:" line under each of the four choices that lacked one (the 10% bar,
  new values, approval and F2b), from the confirmation's list of departures.

*Left as they are (the main session's calls):*
- K5's reading ("L8's copy fix was not scored") and K7's F6 part printed alone, both in the code;
- the odds script's docstring.

*#35*: no comments.

### Smoke run on dev problems (dated note, 2026-10-30; a code check, not a result)

- **When.** After the predictions above were committed (7be1185), from 08:20 to 08:28 -0300.
  `O6_SMOKE=1 realblt_copy_latch_fresh.py` ran blocks, j1, j2, j3, j4, j5, j6, j7 and verdict, one job at a time.
- **What.** GSM8K train idx 0-299, the dev problems, with the confirmatory seeds and builders. Each template took 10 fit
  and 10 test texts: F1 idx 0-20, F2a 21-40, F2b 41-60, F3 61-80 and F4 81-127. F6 used F1's 20 problems, and F5 F1's
  20 under window 0.
- **Two refused starts, both from the guard.** j3 first refused to start because a monitor of mine had "realblt" in its
  command line. Later the driver stopped before j5 because a check process of mine had it too. Neither job had begun;
  each was rerun unchanged.
- **Result.** Every command ran to the end (exit 0) on the code at d84d097, and no code changed.
  - The implementation checks passed on all 120 texts: 0 changed decisions and 0 misaligned positions. Under window
    0 the same held on F1's 20 texts.
  - In every scored cell, the patcher rerun matched the stored outputs.
  - Every cell wrote its layouts, pairs and readouts. F3 reported each kind, with copies of 6, 4, 8 and 3 bytes.
  - F5 loaded the model without windows, as asserted.
  - `verdict` computed every bar, kill rule and the paired pattern.
- **Verdict branches.** copy_latch_verdict_test.py takes the smoke outputs' structure and sets values by hand. In all
  15 cases, `verdict` gives the outcome this rule names (results/blt_copy_latch_fresh_verdict_test.log; at ecca81e.
  Part C took the test to 26 cases and the job gate, part D to 37 evaluations, and part E to 44 hand-set cases, a
  count the test now prints):
  - all bars pass;
  - C1e untestable, alone and with an E10 cost failing;
  - C1e (i) failing;
  - +15 missed while C1e holds;
  - K6;
  - C4 at 4 and at 5;
  - K2;
  - the rate bar failing;
  - K10;
  - each paired pattern, and F4 counting toward neither;
  - K9.
- **Outputs.** results/blt_copy_latch_fresh_{blocks,j1,j2,j3,j4,j5,j6,j7,verdict}_smoke.json and
  results/blt_copy_latch_fresh_smoke.log are committed; the smoke patcher outputs (*_smoke.npz) are git-ignored.
  - Their numbers rest on 10 test texts per cell and are not read as results.
  - No decision rule, bar, prediction or constant was changed after them, until the post-smoke amendments (part C of
    the change log). (Corrected 2026-10-30, after the second check. It said "No rule, bar, prediction or constant",
    but the procedural edits below came after the smoke had started.)
- **Edits made during the smoke run** (listed 2026-10-30, after the second check; integrity M1).
  - What had been read before them: the J1 checks, each cell's layout rates and patcher-rerun counts, and the block
    indices and byte counts. No id_copy or cost pair had been read, and no verdict (first read at 08:27:50).
  - 08:25:51. The Blocks bullet "The rule, bars and predictions are filed before any confirmatory text is built." was
    inserted. ", so the texts and those counts are fixed before any model run" was added to the J0 bullet. Change-log
    item 22 had quoted the first sentence at 7be1185, where it was missing.
  - 08:27:14. In Code, "a dated note below this section" became "the dated note at the end of this section".
  - 08:27:16. The guard sentence in Jobs was added ("nothing else that mentions it may run during a job; the smoke run
    hit this twice").
- **What the smoke showed that bears on the rule** (noted 2026-10-30, after the second check). These are 10 texts per
  cell, but part C used the first two facts.
  - F2b had no hazard. D copied 10 of 10 IDs exactly, and every layout was exact on 10 of 10. Both D-all - D and
    E10-all - E10 were 0.0 [0, 0]; D started a patch at c1 in 10 of 10 texts, E10 in none. The rule then had no
    hazard gate for G1, so the smoke verdict printed "K9: G1 fails; the fix is specific to the \nTicket template".
    Part C adds the gate and restates G1's predictions from this.
  - L8 fired at c1 on 2 of 2 dose copies, against "at most 30%" in the readout and "≤ 30%" in the paired table.
  - F2a's D-all - D was -30 points [-60, 0], against DESIGN's +10 to +60, which part C restores unchanged as a readout.
  - The smoke verdict also printed K7 (C3, C3c) and K9 for G3 (digits6, mixed). On 10 texts these carry no
    information, and nothing was changed for them.
  - Added after the third check:
    - **A dose cost row set the windowed class.** When part D's verdict code ran on the raw smoke outputs (13:57:48
      UTC), the windowed class was "SIV+PO better (cost rows only)". It came from one row, F3 dose, D, C2 (-0.0052
      [-0.0093, -0.0010] on 2 texts); without the doses it was "tie". Both dose lines start "Dose ordered: ", so SIV
      vetoes the recurring "ordered: " before every dose copy and L8 does not. Where D starts on that prefix (2 of 4
      smoke dose texts), a dose cost row moves with no copy lost. The rule's claim that "the cost rows cannot decide
      in practice" rested on F1's and F6's rows only. Part E's change to the paired verdicts follows from this: only
      copy rows with the hazard set a class, and the doses are reported apart.
    - **The letters had no hazard.** In F3, letter IDs had D exact on 3 of 3 and D-all - D 0.0 [0, 0], while the
      6-digit and mixed IDs showed the hazard; the smoke verdict marked letters "applies": false. Part E adds G3's
      testability predictions per kind.
    - **Part E's verdict code on the raw smoke outputs** (a code check, run once part E's code was written):
      windowed "tie", from F1, 6-digit and mixed IDs at D; released "tie"; the dose C2 row is reported apart with its
      sentence. Nothing in the rule was changed for it.
    - **Why the released smoke class went from "L8 better" to "tie"** (added after the fourth check). Part D set it
      from every F5 row, and F5's id_copy bits per copy row was L8 better. Since part E only F5's id_copy exact row
      sets it, and that row tied on the smoke's 10 texts.
    - **Part F's verdict code on the raw smoke outputs** (a code check, run once part F's code was written): windowed
      and released "tie", with their new readings; "failed: K7 (C3, C3c)"; K9 for 6-digit and mixed IDs listed as
      before, without its sentence, since the fix claim failed, and "No generality claim: the fix claim failed."
      Nothing in the rule was changed for it.
    - **Part G's verdict code on the raw smoke outputs** (a code check, run once part G's code was written): the same
      classes, kills and labels; K7's sentence now reads "In O6's `\nTicket` template (F1), scored teacher-forced
      with 512-byte windows (10 texts), L8 failed on fresh problems: C3, C3c."; the result is labelled "Confirmatory
      result (smoke run, dev problems)". Nothing in the rule was changed for it.
