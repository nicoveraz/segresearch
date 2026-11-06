# R15: a general label-free rule? "A new word or symbol of any kind starts here", across six domains. Decision rules

Written 2026-10-28, before any probe in this test was fitted.

**Why.** R12/R14: a label-free probe on BLT-1B's patcher state, trained on "a number not seen earlier in the text",
finds computed values in GSM8K solutions; it reads the role of a number from the phrase before it. The cross-domain
test asked whether one rule predicts where an extra patch start pays in six domains: no two-byte map (X3) and no
gain-trained state probe (R11) transferred. Untested: the label-free recipe widened from numbers to any kind of
content. Asked by the PI.

**Data.** Unchanged from the cross-domain test (results/xdomain_gains_{domain}.npz: math, algebra, code, prose,
clinical, traces; 50 FIT + 60 TEST documents each; per-position gains of an extra start against entropy@10%, bits on
the next 16 bytes and exact match of the next unit) and R11's cached patcher states (last layer, normed). The state at
token t-1 scores a start at token t.

**Labels (computed from the bytes alone; training only).** At token t (byte t-1):
- **new_any**: a word or number (a maximal run of ASCII letters and digits) starts at this byte, and the same run does
  not occur earlier in the document (exact match);
- **new_num** (reference, R12's label generalized to these documents): the same with the run starting with a digit.

**Probes.** Logistic regression on PCA-32 of the standardized state (as R12; lambda 10), all positives and a seeded
25% of negatives, at most 60,000 training positions (seeded subsample), every position t >= 2 of the training
documents:
- **N_dom**: new_any, fitted on the domain's own FIT documents (label-free, so in-domain fitting needs no
  supervision: one recipe for every domain);
- **N_lodo**: new_any, fitted on the other five domains' FIT documents (one detector for every domain);
- **NUM_dom**: new_num, fitted on the domain's own FIT documents.

**Comparison.** Exactly the cross-domain analysis (xdomain_analysis.py selection and bootstrap; R11's code): per TEST
document the top k = round(0.03 n) positions without a start in B (unit objective: unit starts only), capture =
summed gain / the oracle's, 1,000 bootstrap resamples of TEST documents, differences paired, against **E**
(entropy). Reported for reference: R11's gain-trained P_dom and the two-byte map G_dom, recomputed.

**Verdicts (each objective separately).**
- **V1, a general recipe:** N_dom - E above 0 (interval) in >= 5 of 6 domains: **holds**; <= 2: **fails**; otherwise
  **mixed**, with the domains listed.
- **V2, a general detector:** N_lodo - E above 0 in >= 5 of 6: **holds**; <= 2: **fails**; otherwise **mixed**.

**Predictions (written now).** V1 bits mixed (above entropy in math, algebra and traces, where new values are what a
start buys; not in prose or clinical text, where new words are rare words that entropy already flags); V1 unit
fails; V2 fails on both (R11's gain probes did not transfer; new_any means different things in code and prose).

**Reported without a verdict.** N_dom - NUM_dom per domain (does widening beyond numbers help outside math?);
N_dom - P_dom (label-free vs gain-trained); detection AUC of each probe for its own label on TEST documents.

Script: xdomain_newtok.py -> results/xdomain_newtok.json (XN_SMOKE=1: 10 FIT / 10 TEST documents per domain,
results not used).

## Smoke test (2026-10-28, 10 FIT / 10 TEST documents per domain; results not used)

The pipeline ran end to end; no change.

## Results (2026-10-28; results/xdomain_newtok.json, .log)

Capture of the oracle gain for an extra 3% of starts, minus entropy (95% interval over TEST documents):

| domain | bits: N_dom | bits: N_lodo | bits: NUM_dom | unit: N_dom | unit: N_lodo |
|---|---|---|---|---|---|
| math | -0.02 [-0.05, +0.01] | -0.03 [-0.06, +0.00] | +0.09 [+0.05, +0.12] | -0.13 | -0.15 |
| algebra | +0.01 [-0.01, +0.04] | +0.02 [-0.01, +0.04] | +0.03 [+0.00, +0.06] | -0.02 | -0.03 |
| code | **+0.06 [+0.03, +0.10]** | **+0.05 [+0.01, +0.08]** | -0.12 | +0.00 | -0.03 |
| prose | -0.01 [-0.04, +0.01] | -0.02 [-0.06, +0.00] | -0.07 | -0.16 | -0.15 |
| clinical | -0.01 [-0.04, +0.01] | -0.01 [-0.03, +0.01] | -0.07 | +0.00 | -0.10 |
| traces | +0.02 [-0.01, +0.05] | **+0.07 [+0.04, +0.11]** | +0.06 [+0.02, +0.09] | -0.05 | +0.10 [+0.00, +0.20] |

- **V1 bits FAILS** (above entropy only in code). **V2 bits FAILS** (code, traces). **V1 unit FAILS**, **V2 unit
  FAILS** (no domain). Predictions: V1 unit and V2 right; V1 bits predicted mixed (math, algebra, traces), failed.
- Widening from numbers to any new word (N_dom - NUM_dom, bits): worse in math (-0.11) and traces (-0.04), no
  different in algebra, better in code (+0.18), prose (+0.05) and clinical text (+0.06), where new numbers are rare.
  Against the gain-trained probe (N_dom - P_dom, bits): below or tied in every domain.
- The probes detect their own labels well (AUC 0.92-0.99), so the failure is not detection: new words are frequent
  (6-12% of positions) and most are not where a start pays.

**Reading.** "Something new starts here" is not a general rule for where an extra patch start pays in BLT-1B. New
numbers pay in math-like text, new identifiers slightly in code; new words in prose and clinical text do not beat
entropy, and adding them to math dilutes the number signal. This agrees with X1-X4 and R11: entropy remains the best
general rule; the label-free probe's value is specific to computed values.
