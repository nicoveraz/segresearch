"""The prose-rewrite numbers of paper 1 v2 (sections 5 and 13), from the rows realblt_prose.py saved; no model is run.

    python paper/v2/scripts/r9_numbers.py          -> paper/v2/results/r9_numbers.json

Inputs: paper/v2/results/blt_prose.json (realblt_prose.py: BLT-1B at a 10% budget, the same 221 GSM8K test problems
and 711 computed values in two writings, the original solution and an LLM rewrite as prose without '=';
per value and layout, whether entropy starts a patch at its first byte and whether it is exactly right) and
paper/v2/data/prose_rewrites/test.jsonl (the texts). Intervals: 2,000 resamples of problems with the bootstrap of
realblt_prose.py (its boot(), copied below; seed 0 per call), so the verdicts stored in blt_prose.json are reproduced
exactly; this script adds the gap left between the entropy layout and the layout with a start at every value
('with - entropy') and the bytes per computed value in each writing.
"""
import json
import os

import numpy as np

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # paper/v2
ROWS = os.path.join(V2, "results", "blt_prose.json")
TEXTS = os.path.join(V2, "data", "prose_rewrites", "test.jsonl")
OUT = os.path.join(V2, "results", "r9_numbers.json")
FORMATS = ("orig", "prose")


def boot(keys, d, n=2000, seed=0):
    """Mean of d with a 95% bootstrap interval over problems (keys[i][0] is the problem); as realblt_prose.boot."""
    rng = np.random.default_rng(seed)
    P = np.array([k[0] for k in keys]); d = np.asarray(d, float)
    probs = np.unique(P); by = {p: d[P == p] for p in probs}
    bs = [np.concatenate([by[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(n)]
    return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d))}


def compute():
    J = json.load(open(ROWS))
    rows = J["rows"]
    get = lambda f, lay, kind="res": {(x["problem"], x["j"]): x for x in rows
                                      if x["format"] == f and x["layout"] == lay and x["kind"] == kind}
    eo, ep = get("orig", "entropy"), get("prose", "entropy")
    keys = sorted(set(eo) & set(ep))
    E = {f: get(f, "entropy") for f in FORMATS}
    W = {f: get(f, "with") for f in FORMATS}
    WO = {f: get(f, "without") for f in FORMATS}
    DF = {f: get(f, "default") for f in FORMATS}
    out = {"n_values": len(keys), "n_problems": len({k[0] for k in keys})}
    out["covered_entropy"] = {f: float(np.mean([E[f][k]["covered"] for k in keys])) for f in FORMATS}
    out["V1_coverage_prose_minus_orig"] = boot(keys, [float(ep[k]["covered"]) - float(eo[k]["covered"]) for k in keys])
    gain = {f: [float(W[f][k]["ok"]) - float(WO[f][k]["ok"]) for k in keys] for f in FORMATS}
    out["V2_gain_orig"] = boot(keys, gain["orig"])
    out["V2_gain_prose"] = boot(keys, gain["prose"])
    out["V2_gain_prose_minus_orig"] = boot(keys, [p - o for p, o in zip(gain["prose"], gain["orig"])])
    out["with_minus_entropy"] = {f: boot(keys, [float(W[f][k]["ok"]) - float(E[f][k]["ok"]) for k in keys]) for f in FORMATS}
    out["exact"] = {f: {lay: float(np.mean([g[k]["ok"] for k in keys]))
                        for lay, g in (("default", DF[f]), ("entropy", E[f]), ("with", W[f]), ("without", WO[f]))}
                    for f in FORMATS}
    # the stored verdicts must be reproduced exactly (same rows, same bootstrap)
    for name in ("V1_coverage_prose_minus_orig", "V2_gain_orig", "V2_gain_prose", "V2_gain_prose_minus_orig"):
        stored = J["verdicts"][name]
        assert abs(stored["value"] - out[name]["value"]) < 1e-12 and \
            all(abs(a - b) < 1e-12 for a, b in zip(stored["ci"], out[name]["ci"])), f"{name} does not reproduce"
    # bytes per computed value: the 10% budget is a share of the whole text (question, solution, final line)
    recs = {r["idx"]: r for r in map(json.loads, open(TEXTS))}
    probs = sorted({k[0] for k in keys})
    nval = sum(len(recs[p]["values"]) for p in probs)
    assert nval == len(keys), "every value of these problems is scored"
    out["bytes_per_value"] = {f: sum(len(recs[p][f].encode()) for p in probs) / nval for f in FORMATS}
    out["bytes_per_value"]["prose_over_orig"] = out["bytes_per_value"]["prose"] / out["bytes_per_value"]["orig"]
    out["rates"] = {k: v for k, v in J["rates"].items() if k.endswith(("|entropy", "|with", "|without"))}
    return out


def main():
    out = compute()
    json.dump(out, open(OUT, "w"), indent=1)
    pct = lambda v: f"{100 * v['value']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]"
    print(f"{out['n_problems']} problems, {out['n_values']} computed values")
    print("entropy covers: " + ", ".join(f"{f} {100 * v:.1f}%" for f, v in out["covered_entropy"].items()),
          "| prose - orig", pct(out["V1_coverage_prose_minus_orig"]))
    print("a start at every value (with - without): orig", pct(out["V2_gain_orig"]), "prose", pct(out["V2_gain_prose"]),
          "prose - orig", pct(out["V2_gain_prose_minus_orig"]))
    print("with - entropy: " + ", ".join(f"{f} {pct(v)}" for f, v in out["with_minus_entropy"].items()))
    print("exact: " + "; ".join(f"{f} " + ", ".join(f"{l} {100 * x:.1f}%" for l, x in v.items()) for f, v in out["exact"].items()))
    b = out["bytes_per_value"]
    print(f"bytes per value: orig {b['orig']:.1f}, prose {b['prose']:.1f} (x{b['prose_over_orig']:.2f})")
    print("-> paper/v2/results/r9_numbers.json")


if __name__ == "__main__":
    main()
