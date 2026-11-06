# O6: do entropy patch starts break verbatim copies, and does a suffix-match veto fix it? Decision rules

Written 2026-10-29, before any layout in this test was run. Exploratory gate for a follow-up paper (ANALYSIS_2026-10-29
section 5, O6 = analysis/2026-10-29/opportunities.md C6); nothing here goes into paper2v2/manuscript.md.

**Why.** In the quadrant test (results/blt_quadrant.json, 293 GSM8K test problems with a random 4-digit ticket inserted
twice), BLT-1B copies the second ticket exactly 70.3% of the time without a patch start at its first byte and 26.3%
with one (entropy@10% layout otherwise fixed; -1.33 bits [-1.44, -1.24]). Under BLT-1B's own default layout (~25% of
bytes) it copies it exactly only 14.0% of the time. Before building anything on that, step 1 checks that the 14% is
caused by default starts at or inside the copy, and not by the conversion, the template or the window.

**Texts.** The 293 quadrant problems and the 295 train problems, rebuilt with realblt_quadrant.build in the same seeded
order (train first, then test). Checked before writing this rule: the unit counts per problem equal the saved rows.
Source-to-copy distance (first digit to first digit) is 59-619 bytes, median 222; 285 copies lie within BLT-1B's
512-byte local window, 8 beyond it.

**Template note.** Both tickets are preceded by the same 8 bytes, `\nTicket `, so the copy is an induction setup: the
context before the copy repeats the context before the source. This should make copying easier, not harder. It also
means the veto below fires at the copy's first byte only because the preceding context repeats. A copy with fresh
context would be protected only from its (k+1)-th byte. The synthetic copy set (same vs different context, near vs
far) tests this; it is not built until the author approves it after the gate.

**Positions.** Token t = byte t-1. The copy's digits are tokens c1..c4. Onset = a patch start at c1; inside = a start at
any of c2..c4. A start at token t is decided from the patcher's entropy at t-1, so it may use bytes up to t-2. Exact =
all four digits right under teacher-forced argmax. Bits = the copy's summed -log2 p.

## Step 1: the gate (BLT-1B with the 512-byte windows, blt_load)

Layouts, per problem:
- **FULL**: patch_lengths=None, BLT-1B's own path.
- **D**: the default layout given explicitly, read from the patcher's own patch lengths (threshold 1.3354 nats).
- **D-all**: D without any start at c1..c4.
- **D-onset**: D without the start at c1.
- **D-inside**: D without starts at c2..c4.
- **D+onset**: D with a start at c1 and none at c2..c4.

Dumped per problem: D's starts in the copy and in the 8 bytes before it; the patcher entropy (bits) deciding each of
c1..c4 and the byte after; the patcher's own argmax at c1..c4; the distance; exact and bits under every layout.

- **G0, reproduction (a stop condition, not a verdict).** D's starts equal the default path's on every problem
  (ignoring a start past the last byte); FULL and D give the same id_copy result per problem; and FULL reproduces the
  saved rows (id_copy exact 0.140 within 1 point; per-problem bits within 0.05 on at least 95% of problems). If D and
  FULL differ, stop and find out why before reading
  anything else.
- **G1, cause.** exact(D-all) - exact(D) on id_copy, paired bootstrap over problems (2,000 resamples). **Holds** if at
  least +20 points with the 95% interval's lower bound at least +10.
- **G2, not the window.** The same difference on the 285 copies within the window. **Holds** if at least +15 points with
  the lower bound above 0.
- **The gate passes** if G0 is met and G1 and G2 hold. Otherwise the hazard is dropped (C6's kill criterion: "the 14%
  is not caused by starts at or inside the copy"), step 2 is not run, and paper 2's quadrant paragraph gets at most one
  sentence.
- **Conversion check (reported; reading fixed now).** FULL, D and D-all again with SEGR_BLT_WINDOW=0, the conversion as
  released (unbounded local attention, its own default layout). If D-all - D is also at least +10 points there, the
  hazard does not depend on the unmerged window fix; otherwise any write-up says it does. This is a caveat, not a kill:
  the windowed model is the architecture as trained. The official BLT code and weights are not on this Mac and are not
  checked.
- **Reported without a verdict.** Share of copies with an onset start and with an inside start under D; exact under D
  by those cells; D+onset - D-all (the onset start alone); D-onset and D-inside; which digit fails first; the patcher's
  entropy profile and its own copy accuracy; exact within vs beyond the window.

## Step 2: the veto (only if the gate passes)

**Veto V_k, k = 6 and 8.** Token t may not start a patch if the k bytes ending at byte t-2 (the last k bytes known when
the start at t is decided) also occur earlier, ending at byte t-3 or before (overlap allowed). The match is still
extending while each new byte keeps the last k bytes recurring; the first byte that breaks it lifts the veto at the
next decision, unless the new k bytes recur elsewhere. It is causal: it reads only what the entropy decision reads.

**Layouts** on the 293 quadrant problems (thresholds fitted on the 295 train problems with the same insertion,
blt_layouts.Fit, train mode, as in the quadrant script):
- **D** and **D.V_k**: the veto applied to the default, with the entropy threshold lowered so that the pooled train rate
  equals D's (the matched rate, refilled by entropy);
- **E10** and **E10.V_k**: entropy@10% (the quadrant's layout) and the veto at a matched 10%;
- reported without a verdict: D-V_k and E10-V_k, the veto without refill (same thresholds, fewer starts).

**Readouts.** id_copy exact and bits; every other unit kind (computed, id_new, copied, word_new, word_rep, final) exact
and bits; bits per byte over the whole text and over the text without the copy's digits; patch rate on test; vetoed
starts per text and the kinds they fall on. Paired bootstrap over problems, 2,000 resamples.

**Verdicts, for each k, against the unvetoed layout at the matched rate:**
- **V1, copies restored (default):** D.V_k - D on id_copy exact. **Holds** if at least +15 points (C6's fix threshold)
  with the lower bound above 0.
- **V1e, copies restored (entropy@10%):** E10.V_k - E10 on id_copy exact. **Holds** if the lower bound is above 0.
- **V2, no cost to the text:** bits per byte without the copy's digits, D.V_k - D and E10.V_k - E10. **Holds** if both
  point estimates are at most +0.01.
- **V3, no cost to computed values:** exact on computed units, both pairs. **Holds** if both point estimates are at
  least -2 points.
- **V4, no harm on equation text:** on 300 GSM8K equation texts (test problems 0-299 in realblt_budget's stripped format,
  no ticket; entropy@10% fitted on 300 train texts), the veto removes no start at a computed-value start (the byte after
  `= ` and the value's first digit) from D or from E10. **Holds** if that count is 0. Reported alongside: the count and
  share of such starts, and how many vetoed values repeat an earlier computation with the same value (a true copy).

**The fix claim** (for the follow-up paper) needs V1, V1e, V2, V3 and V4 for the same k; if both k pass, k = 8 is
reported first (it vetoes less).

**Predictions (written now).** G0 is met. Under D most copies have an onset start (about 60-70%) and some an inside
start (30-40%). D-all restores exact copy to about 60%, so G1 and G2 hold, and the conversion check agrees. V1 and V1e
hold for both k; V2 and V3 hold; V4 fails for k = 6 (about 1-3% of computed starts follow a repeated 6-byte context)
and narrowly for k = 8, with most vetoed values repeating an earlier computation.

**Not in this test.** The synthetic copy set (record numbers, doses, dates; same vs different context; near vs far
from the 512-byte window) waits for the author's go after the gate. Teacher forcing only; one model; one template.

Scripts: realblt_copy_gate.py -> results/blt_copy_gate.json (SEGR_BLT_WINDOW=0 -> results/blt_copy_gate_w0.json);
realblt_copy_veto.py -> results/blt_copy_veto.json. Smoke runs (a few problems) go to *_smoke.json and are not used.

## Step 1 results (2026-10-29; results/blt_copy_gate.json, .log)

Run with blt_light.py (the hash-embedding table memory-mapped on the CPU; the standard loader thrashed the 16 GB Mac).
G0 shows the outputs are unchanged.

| layout | id_copy exact | bits per copy |
|---|---|---|
| FULL / D (default, 27% of bytes) | 14.0% | 3.06 |
| D-all (no start at c1..c4) | **90.1%** | 1.66 |
| D-onset | 28.3% | 2.83 |
| D-inside | 10.2% | 3.40 |
| D+onset (onset only) | 9.2% | 3.43 |

- **G0 met:** D equals FULL on 293 of 293; FULL reproduces the saved rows (exact 0.140; bits within 0.05 on 100%).
  Note: 193 starts differ from a float comparison of entropy with the threshold, because the library compares in bf16.
  D is therefore read from the patcher's own lengths, as the rule says.
- **G1 HOLDS:** D-all - D = **+76.1 points [+71.0, +80.9]** (1.40 bits saved per copy [1.32, 1.49]).
- **G2 HOLDS:** within the window (285 copies) +75.8 [+70.9, +81.1]. Beyond it (8 copies), D-all is 8 of 8 exact.
- **The gate passes.**
- Descriptive:
  - D starts at the copy's onset in 98.6% of problems and inside it in 92.2%. By digit: c1 98.6%, c2 21.5%, c3 50.2%,
    c4 85.3%. D also always starts at the `T` of `\nTicket `.
  - The onset start alone (D+onset - D-all) costs -80.9 points [-85.0, -76.1].
  - Failures fall on the last digit: 227 of 252 first errors under D are at c4.
  - The patcher's entropy deciding c1..c4 is 3.16, 1.56, 1.93 and 2.30 bits. The patcher's own argmax copies the
    ticket 30.7% of the time; by digit, 0.99, 0.98, 0.84 and 0.40.
- Unit test of the veto, without the model, before step 2 ran: it fires at c1..c4 and the byte after in 100% of texts
  for both k. It also vetoes 17% (k = 8) and 24% (k = 6) of all positions, because GSM8K text repeats phrases.

## Conversion check (2026-10-29; results/blt_copy_gate_w0.json, .log)

SEGR_BLT_WINDOW=0 is the conversion as released: unbounded local attention, and its own default layout at 31% of
bytes.
- The explicit default equals BLT-1B's own path on 292 of 293 problems.
- id_copy exact: D 23.2%, D-all 65.5%. **D-all - D = +42.3 points [+34.1, +50.9]**, at least +10, so by the rule's
  reading the hazard does not depend on the unmerged window fix.
- Unlike with the windows, removing the starts here *costs* bits per copy: -1.72 [-2.35, -1.10].
  - Without the window, some copies fail at their first digits when no start is placed. Under D-all, the first error
    falls at c1 24 times and at c2 32 times; with the windows it never does.
  - Among the 121 copies with a start at all four digits, D-all is worse than D (31.4% vs 37.2%).
  - The 8 copies beyond 512 bytes go from 5 exact under D to 0 under D-all.
  - So in the released conversion the harm shows in exact match but not in bits. The windowed model is the
    architecture as trained.

## Step 2 results (2026-10-29; results/blt_copy_veto.json, .log)

**Checks.**
- E10 reproduces the quadrant's entropy@10% layout: id_copy exact 0.546, and bits within 0.05 on 100% of problems.
- Train rates are matched: the default is 0.260, and D.V_k is refilled to the same rate (test rates 0.274 and 0.276).

**Exact match by unit kind**, 293 problems (patch rate; bits per byte without the copy's digits; %):

| layout | rate | bpb | id_copy | computed | copied | word_new | word_rep | final |
|---|---|---|---|---|---|---|---|---|
| D (default) | 0.274 | 0.814 | 14.0 | 78.0 | 64.4 | 34.5 | 74.5 | 72.4 |
| D.V8 (matched) | 0.276 | 0.834 | 84.6 | 72.5 | 59.8 | 33.5 | 72.8 | 67.9 |
| D.V6 (matched) | 0.276 | 0.844 | 84.3 | 69.8 | 58.3 | 32.1 | 72.3 | 68.3 |
| E10 | 0.107 | 1.024 | 54.6 | 18.2 | 35.4 | 25.3 | 66.8 | 32.8 |
| E10.V8 (matched) | 0.108 | 1.025 | 71.3 | 20.1 | 34.8 | 25.3 | 66.3 | 34.5 |
| E10.V6 (matched) | 0.108 | 1.029 | 75.8 | 18.9 | 35.2 | 24.6 | 66.2 | 31.4 |

**Verdicts** (differences against the unvetoed layout at the matched rate, with 95% intervals):

| | k = 8 | k = 6 |
|---|---|---|
| V1, D.V - D on id_copy | **+70.6 [+65.2, +75.8] HOLDS** | +70.3 [+64.5, +75.8] HOLDS |
| V1e, E10.V - E10 on id_copy | **+16.7 [+10.6, +22.5] HOLDS** | +21.2 [+15.4, +27.0] HOLDS |
| V2, bits/byte without copy (D; E10) | +0.0196; +0.0009: **FAILS** (D) | +0.0296; +0.0053: FAILS (D) |
| V3, computed exact (D; E10) | -5.6 [-7.8, -3.1]; +1.9: **FAILS** (D) | -8.2 [-10.5, -5.7]; +0.8: FAILS (D) |
| V4, computed starts removed (D; E10) | 50 of 799; 0 of 129: **FAILS** (D) | 76 of 799; 0 of 129: FAILS (D) |
| **fix claim** | **no** | **no** |

- **The fix claim fails for both k.** The veto restores copies at a matched rate. At BLT-1B's default rate, though,
  it costs computed values, bits per byte and, as reported, final answers (D.V8 -4.4 [-7.5, -1.4]) and copied
  numbers (-4.6 [-6.2, -3.1]).
- **V4:** the veto removes 50 (k = 8) and 76 (k = 6) of the default layout's 799 starts at computed values in the 294
  equation texts (796 values). Only 5 and 8 of those repeat an earlier computation with the same value. The rest
  follow a recurring context with a new value, for example `... will pay 8 * 5 = $40` or `80 sheep = 260`. Entropy@10%
  has only 129 such starts, and the veto removes none.
- **Where the vetoed starts fall** (k = 8): 10.7% of the default layout's starts. Of these, 1,111 fall at repeated word
  starts, 469 at copied-number starts, 749 at the ticket copy, 51 at computed-value starts and 1,568 elsewhere. At
  entropy@10% it vetoes 6.9% of starts.
- **Reported without a verdict:**
  - At entropy@10% with k = 8, no cost was detected: id_copy +16.7, bits per byte +0.0009 [-0.0023, +0.0040],
    computed +1.9 [+0.1, +3.7], final +1.7 [-2.7, +6.1]. Every check other than V4 on D would pass there.
  - This is a post hoc observation, not a result. A veto for the entropy@10% budget only, or one that also requires
    the match to have extended into the copied string, would need a new dated rule before any run.
  - Without refill (D-V_k, E10-V_k), every cost is larger.
- **Reading:** the hazard is real, and in this template a suffix-match veto removes it. But "the last k bytes recur"
  marks recurring contexts as well as copies: GSM8K's formulaic lines repeat contexts such as `* 5 = $` before new
  values. At BLT-1B's own rate, that removes starts the model needs.
