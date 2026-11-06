"""Outside-math numbers for paper 2 v2 (Appendix C), from segresearch's public run files only (CPU, no model; new in v2).

Reads <segresearch>/results/generic/{prose,code,trace,trace_novnet,code_acc}/<rule>_s<seed>.json, the 12M models
trained from scratch outside math for version 1 (2 seeds per rule, Kaggle T4 GPUs in fp16), and writes
paper2/v2/results/v2_outside_math.json:

  arms   per domain and rule: seeds, devices, evaluation patch rate, and the mean and sample SD of bits per byte (on
         all, new and copied bytes) and, where measured, exact match (%)
  pairs  novelty minus entropy, with seed-level Welch 95% intervals in scale.analyze.welch's convention (t from its T95
         table at the floored Welch-Satterthwaite df; with 2 seeds per arm the df is 1 or 2, so t is 12.71 or 4.30)

Version 1 gave these as means and SDs only (paper2/tables/paper_outside.md), and its code exact-match rows with
binomial standard errors over tokens; here every difference gets a seed-level interval.

    python paper2/v2/run.py v2_outside_math          (or python paper2/v2/scripts/v2_outside_math.py)
"""
import glob
import json
import math
import os

import numpy as np

import _seg
from scale.analyze import T95, welch

GEN = os.path.join(_seg.SEG, "results", "generic")
OUT = os.path.join(_seg.HERE, "results", "v2_outside_math.json")

# (domain folder, rule a, rule b, measures): b - a
PAIRS = [("prose", "entropy10", "novelty", ("bpb", "bpb_new", "bpb_copy")),
         ("code", "entropy10", "novelty", ("bpb",)),                         # thresholds at a 10% training budget
         ("code", "entropy_t17", "novelty_t10", ("bpb",)),                   # the higher-budget pair (rates differ)
         ("trace", "entropy10", "novelty", ("bpb", "acc_computed", "acc_copied")),
         ("trace_novnet", None, "novelty_net", ()),                          # reported as an arm only
         ("code_acc", "entropy10", "novelty", ("acc_id_new", "acc_rhs_copy", "bpb"))]
PCT = {"acc_computed", "acc_copied", "acc_id_new", "acc_id_copy", "acc_num_new", "acc_rhs_copy", "acc_rhs_new"}


def load(folder):
    out = {}
    for f in sorted(glob.glob(os.path.join(GEN, folder, "*_s*.json"))):
        o = json.load(open(f))
        out.setdefault(o["rule"], []).append(o)
    return {k: sorted(v, key=lambda o: o["seed"]) for k, v in out.items()}


def vals(rs, m):
    return [o[m] * (100.0 if m in PCT else 1.0) for o in rs]


def arm(rs):
    res = {"seeds": [o["seed"] for o in rs], "devices": sorted({o["device"] for o in rs}),
           "rate": float(np.mean([o["val_rate"] for o in rs]))}
    for m in ("bpb", "bpb_new", "bpb_copy") + tuple(sorted(PCT)):
        if all(m in o for o in rs):
            v = np.array(vals(rs, m), float)
            res[m] = {"values": [float(x) for x in v], "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    return res


def pair(ra, rb, m):
    x, y = np.array(vals(ra, m), float), np.array(vals(rb, m), float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else 1.0
    d, h = welch(x, y)
    assert abs(h - T95.get(int(df), 1.96) * se) < 1e-12
    return {"diff": float(d), "ci": [float(d - h), float(d + h)], "df": float(df), "t": T95.get(int(df), 1.96)}


def main():
    res = {"source": "segresearch results/generic/*/<rule>_s<seed>.json (12M, 2 seeds per rule)",
           "convention": "b - a; bits per byte, or exact match in points; seed-level Welch 95% interval "
                         "(scale.analyze.welch: t from T95 at the floored Welch-Satterthwaite df)",
           "arms": {}, "pairs": {}}
    for folder, a, b, measures in PAIRS:
        R = load(folder)
        for rule, rs in R.items():
            res["arms"][f"{folder} {rule}"] = arm(rs)
        if a is None:
            continue
        for m in measures:
            res["pairs"][f"{folder}: {b} - {a} | {m}"] = pair(R[a], R[b], m)
    res["devices"] = sorted({d for v in res["arms"].values() for d in v["devices"]})
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", os.path.relpath(OUT, _seg.SEG))
    for k, v in res["pairs"].items():
        print(f"  {k}: {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] (df {v['df']:.2f})")
    print("  devices:", res["devices"])


if __name__ == "__main__":
    main()
