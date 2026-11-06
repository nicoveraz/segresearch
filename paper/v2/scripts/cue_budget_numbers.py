"""The cue and budget numbers of paper 1 v2 (section 5), recomputed from the released rows; numpy only, no model.

    python paper/v2/scripts/cue_budget_numbers.py      -> paper/v2/results/cue_budget_numbers.json

Input: paper/v2/results/blt_cue_budget_entropy.json (cue_budget_subset.py): for 915 GSM8K test problems (from 300-1,318)
and each computed value (the number after '= ', with any '$' between them; 2,448 values) and final answer, BLT-1B's
next-byte entropy H in bits and the probability Pd it gives to a digit, both at the value's first digit, and whether
entropy starts a patch there at budgets of 5, 7.5, 10, 12.5, 15, 20 and 25% of bytes (thresholds fitted on 1,382 GSM8K
training problems); the same for a copy of the text with each '= ' before a value replaced by 'is '.

cue: '= ' when the value follows '= ' directly, '$' when it follows '= $' (the patch start counted is then at the digit
after the '$'). Intervals: 2,000 bootstrap resamples of the 915 problems, drawn as in the original measurement
(numpy default_rng(1), multinomial weights over the sorted problems; ratio of resampled sums), so the summaries stored
with the rows are reproduced exactly (checked below).
"""
import json
import os

import numpy as np

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # paper/v2
SRC = os.path.join(V2, "results", "blt_cue_budget_entropy.json")
OUT = os.path.join(V2, "results", "cue_budget_numbers.json")
NBOOT = 2000


def load():
    C = json.load(open(SRC))
    cols = C["rows"]["cols"]
    rows = [dict(zip(cols, r)) for r in C["rows"]["data"]]
    return C, rows


def compute():
    C, rows = load()
    rates = C["rates"]
    i = {r: k for k, r in enumerate(rates)}
    probs = sorted({v["problem"] for v in rows if v["kind"] == "fin" and v["w"] == "orig"})
    pi = {p: k for k, p in enumerate(probs)}
    Wm = np.random.default_rng(1).multinomial(len(probs), np.full(len(probs), 1 / len(probs)), size=NBOOT).astype(float)

    def mean_ci(pairs):
        S, N = np.zeros(len(probs)), np.zeros(len(probs))
        for p, x in pairs:
            S[pi[p]] += x; N[pi[p]] += 1
        bs = (Wm @ S) / np.maximum(Wm @ N, 1e-12)
        return {"value": float(S.sum() / N.sum()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n": int(N.sum())}

    cov = lambda v, r: v["covered_at_rates"][i[r]] == "1"
    O = [v for v in rows if v["w"] == "orig" and v["kind"] == "res"]
    F = [v for v in rows if v["w"] == "orig" and v["kind"] == "fin"]
    by = {"$": [v for v in O if v["cue"] == "$"], "= ": [v for v in O if v["cue"] == "= "], "all": O, "final": F}
    out = {"problems": len(probs), "values": len(O), "n": {k: len(v) for k, v in by.items()},
           "log2_10": float(np.log2(10)), "threshold_bits": C["thresholds_bits"]["orig"],
           "coverage": {k: {str(r): float(np.mean([cov(v, r) for v in g])) for r in rates} for k, g in by.items()}}
    for k in ("$", "= "):
        H = np.array([v["H"] for v in by[k]]); Pd = np.array([v["Pd"] for v in by[k]])
        out[f"after {k.strip()}"] = {"H_median": float(np.median(H)), "H_p99": float(np.percentile(H, 99)),
                                     "Pd_median": float(np.median(Pd)), "Pd_ge_0.9": float((Pd >= 0.9).mean())}
    missed = [v for v in O if not cov(v, 0.1)]
    out["missed_at_10_Pd_ge_0.9"] = mean_ci((v["problem"], float(v["Pd"] >= 0.9)) for v in missed)
    # 'is ' instead of '= ': the 1,862 values written '= N' in the original, in both writings
    A = {(v["problem"], v["j"]): v for v in rows if v["w"] == "orig_is" and v["kind"] == "res"}
    B = {(v["problem"], v["j"]): v for v in by["= "]}
    ks = sorted(B)
    out["is_instead_of_eq"] = {
        "n": len(ks),
        "covered_eq": float(np.mean([cov(B[k], 0.1) for k in ks])), "covered_is": float(np.mean([cov(A[k], 0.1) for k in ks])),
        "difference": mean_ci((k[0], float(cov(A[k], 0.1)) - float(cov(B[k], 0.1))) for k in ks),
        "Pd_median_eq": float(np.median([B[k]["Pd"] for k in ks])), "Pd_median_is": float(np.median([A[k]["Pd"] for k in ks])),
        "threshold_bits_10_is": C["thresholds_bits"]["orig_is"]["0.1"]}
    # the summaries stored with the rows must come out exactly
    st = C["stored"]
    s1 = st["S1_is_minus_eq_coverage_at_10"]; d = out["is_instead_of_eq"]["difference"]
    assert s1["n"] == d["n"] and abs(s1["value"] - d["value"]) < 1e-12 and all(abs(a - b) < 1e-12 for a, b in zip(s1["ci"], d["ci"]))
    assert st["Q1_orig_coverage_after_dollar_at_10"]["value"] == out["coverage"]["$"]["0.1"] == 0.0
    for lay, r in (("E", 0.1), ("E@15", 0.15), ("E@20", 0.2)):
        assert abs(st["covered_res"][lay] - out["coverage"]["all"][str(r)]) < 1e-12, lay
    return out


def main():
    out = compute()
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{out['problems']} problems, {out['values']} computed values ({out['n']['$']} after '= $', {out['n']['= ']} after '= ')")
    print("budget:     " + " ".join(f"{100 * float(r):>6.1f}%" for r in out["coverage"]["all"]))
    print("threshold:  " + " ".join(f"{t:6.2f}b" for t in out["threshold_bits"].values()), f"(log2 10 = {out['log2_10']:.2f})")
    for k, v in out["coverage"].items():
        print(f"{k:10s}  " + " ".join(f"{100 * x:6.1f}%" for x in v.values()))
    for k in ("after $", "after ="):
        print(k, {a: round(b, 4) for a, b in out[k].items()})
    m = out["missed_at_10_Pd_ge_0.9"]
    print(f"missed at 10% with P(digit) >= 0.9: {100 * m['value']:.1f}% [{100 * m['ci'][0]:.1f}, {100 * m['ci'][1]:.1f}] of {m['n']}")
    s = out["is_instead_of_eq"]; d = s["difference"]
    print(f"'is ' for '= ' ({s['n']} values): covered {100 * s['covered_eq']:.1f}% -> {100 * s['covered_is']:.1f}% "
          f"({100 * d['value']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]); median P(digit) "
          f"{s['Pd_median_eq']:.3f} -> {s['Pd_median_is']:.3f}")
    print("-> paper/v2/results/cue_budget_numbers.json")


if __name__ == "__main__":
    main()
