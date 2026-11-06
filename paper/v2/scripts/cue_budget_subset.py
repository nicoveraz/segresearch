"""Write the released subset of a BLT-1B measurement made in the author's private working repository (paper 1 v2,
section 5): BLT-1B's entropy patcher at the computed values of 915 further GSM8K test problems.

    python paper/v2/scripts/cue_budget_subset.py FULL_RESULTS_JSON     -> paper/v2/results/blt_cue_budget_entropy.json

The measurement (BLT-1B with its 512-byte window restored, transformers 5.18, at inference; no training) ran the
entropy patcher over GSM8K test problems 300-1,318 in their stripped form (question, solution without calculator
annotations, "The final answer is N", as realblt_budget.problem() builds it). 915 of the 1,019 problems were used:
those that also had a valid LLM rewrite as prose for another part of the same measurement (not released) and fit BLT-1B's
1,000-token context. At the first digit of every computed value (the number after '= ', with any '$' between them) and
at the final answer it saved the patcher's next-byte entropy H (bits), the probability it gives to a digit (Pd), and
whether entropy starts a patch there at budgets of 5-25% of bytes, with thresholds fitted on 1,382 GSM8K training
problems. It did the same on a copy of the text with each '= ' before a value replaced by 'is ' ("orig_is").

This script keeps exactly that and nothing else: the rows of the 'fresh' problems for the original and the 'is '
writings, their columns problem, j, kind, cue, H, Pd and coverage at the seven common budgets, the thresholds at those
budgets, the counts, the run's metadata, and three summaries the measurement stored for cross-checking (they are
recomputed from the rows by cue_budget_numbers.py). The measurement script itself is not released (see
paper/v2/README.md).
"""
import json
import os
import sys

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # paper/v2
OUT = os.path.join(V2, "results", "blt_cue_budget_entropy.json")
WRITINGS = ("orig", "orig_is")
RATES = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25)
KEEP = ("problem", "j", "kind", "cue", "H", "Pd")


def main(full):
    C = json.load(open(full))
    assert not C["meta"]["smoke"]
    for w in WRITINGS:
        assert tuple(C["rates_patcher"][w][:len(RATES)]) == RATES, "unexpected budget order"
    cols = C["patcher_rows"]["cols"]
    rows = []
    for r in C["patcher_rows"]["data"]:
        v = dict(zip(cols, r))
        if v["split"] == "fresh" and v["w"] in WRITINGS:
            rows.append([v["w"], *(v[k] for k in KEEP), v["covered_at_rates"][:len(RATES)]])
    out = {
        "about": "BLT-1B entropy patcher at computed values and final answers of 915 GSM8K test problems (300-1,318); "
                 "released subset of a measurement in the author's private working repository; see paper/v2/README.md",
        "meta": {k: C["meta"][k] for k in ("device", "torch", "transformers", "window", "model")},
        "rates": list(RATES),
        "thresholds_bits": {w: {str(r): C["thresholds_bits"][w][str(r)] for r in RATES} for w in WRITINGS},
        "n": {"train_problems": C["n"]["orig"]["train"], "problems": C["n"]["orig"]["fresh"],
              "values": C["n"]["orig"]["values"],
              "train_rate_entropy10": {w: C["n"][w]["train_rate_E10"] for w in WRITINGS}},
        "rows": {"cols": ["w", *KEEP, "covered_at_rates"], "data": rows},
        "stored": {"S1_is_minus_eq_coverage_at_10": {k: C["verdicts"]["S1"][k] for k in ("value", "ci", "n")},
                   "Q1_orig_coverage_after_dollar_at_10": C["verdicts"]["Q1"]["orig"],
                   "covered_res": {lay: C["layouts"][f"orig|{lay}"]["covered_res"] for lay in ("E", "E@15", "E@20")}},
    }
    json.dump(out, open(OUT, "w"))
    print(f"{len(rows)} rows -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main(sys.argv[1])
