"""Training-side numbers for paper 2 v2, from segresearch's public run files only (CPU, no model; new in v2).

Reads <segresearch>/results/scale/runs/*/result.json (the 1M, 12M and 50M runs of papers 1 and 2; no other size is
read), <segresearch>/results/scale/cross/12m_nov10_s{0,1}/ (two same-seed T4 re-runs of the 12M table runs) and
paper2/v2/results/r13_pilot/result.json, and writes paper2/v2/results/v2_training_numbers.json:

  runs       every run: arm, size, seed, clipping, device, teacher-forced exact match on computed results and final
             answers, bits per byte, evaluation patch rate, end-to-end GSM8K and MATH \\boxed{} accuracy
  failed     runs whose training failed, by the pilot's rule: bits per byte above the median of their size + 1 (or not
             finite). They are left out of every mean and interval below.
  arms       per size, arm and recipe (clip / noclip): seeds, devices, means and sample SDs
  pairs      differences of arm means (b - a) with seed-level Welch 95% intervals, the published convention of
             scale.analyze.welch: t from its T95 table at the floored Welch-Satterthwaite df. Each pair records whether
             both arms share a recipe and a device.
  t4_reruns  the two re-runs against the Table 1 runs of the same seeds, and their switched-layout evaluations
  floors     end-to-end GSM8K (problems solved of 100) and MATH \\boxed{} accuracy, over the Table 1 runs and over all
  r13_pilot  the one-seed MPS pilot beside the 12M seeds of the arms it is compared with
  check      the pairs that paper2v2_intervals.json also holds (intervals.py training()), with the largest difference

    python paper2/v2/run.py v2_training_numbers          (or python paper2/v2/scripts/v2_training_numbers.py)
"""
import glob
import json
import math
import os

import numpy as np

import _seg
from scale.analyze import T95, welch

SCALE = os.path.join(_seg.SEG, "results", "scale")
OUT = os.path.join(_seg.HERE, "results", "v2_training_numbers.json")
SIZES = ("1m", "12m", "50m")
MULT = {"computed_acc": 100.0, "final_acc": 100.0, "bpb": 1.0}
NAME = {"computed_acc": "computed", "final_acc": "final", "bpb": "bpb"}

# (size, recipe, a, b): b - a, both arms with the same recipe
SAME = [("1m", "noclip", "entropy10", "nov10"), ("1m", "noclip", "entropy10", "syntax+entropy10"),
        ("1m", "noclip", "syntax+entropy10", "nov10"), ("1m", "noclip", "entropy10", "dep10"), ("1m", "noclip", "dep10", "nov10"),
        ("12m", "noclip", "entropy10", "nov10"), ("12m", "noclip", "entropy10", "syntax+entropy10"),
        ("12m", "noclip", "syntax+entropy10", "nov10"), ("12m", "noclip", "entropy10", "dep10"), ("12m", "noclip", "dep10", "nov10"),
        ("12m", "noclip", "nov10", "novc13"), ("12m", "noclip", "entropy10", "novc13"), ("12m", "noclip", "syntax+entropy10", "novc13"),
        ("12m", "noclip", "entropy10", "typed14"),
        ("50m", "clip", "entropy10", "nov10"), ("50m", "clip", "dep10", "nov10"), ("50m", "clip", "nov10", "novc13"),
        ("50m", "clip", "entropy10", "novc13"), ("50m", "clip", "entropy10", "dep10"), ("50m", "clip", "entropy10", "typed14"),
        ("50m", "noclip", "entropy10", "syntax+entropy10"), ("50m", "noclip", "dep10", "syntax+entropy10"),
        ("50m", "noclip", "entropy10", "dep10")]
# (size, (arm a, recipe a), (arm b, recipe b)): b - a across recipes (clipped against unclipped)
ACROSS = [("50m", ("entropy10", "noclip"), ("entropy10", "clip")),           # the effect of clipping on entropy
          ("50m", ("dep10", "noclip"), ("dep10", "clip")),                   # ... and on dependence
          ("50m", ("syntax+entropy10", "noclip"), ("nov10", "clip"))]        # clipped table - unclipped hand rule
# the arms of paper 2's Table 1 (1M and 12M unclipped, 50M clipped), and the 50M hand rule (unclipped, 2 seeds)
TABLE1 = {("1m", "noclip"): ("entropy10", "dep10", "syntax+entropy10", "nov10"),
          ("12m", "noclip"): ("entropy10", "dep10", "syntax+entropy10", "nov10", "novc13"),
          ("50m", "clip"): ("entropy10", "dep10", "nov10", "novc13")}
HAND50 = ("50m", "noclip", "syntax+entropy10")


def welch_df(x, y):
    """scale.analyze.welch, plus the Welch-Satterthwaite df it floors (asserted equal to welch's own output)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else 1.0
    d, h = welch(x, y)
    assert abs(h - T95.get(int(df), 1.96) * se) < 1e-12
    return float(d), float(h), float(df)


def load():
    runs = []
    for f in sorted(glob.glob(os.path.join(SCALE, "runs", "*", "result.json"))):
        o = json.load(open(f))
        if o.get("toy") or o["size"] not in SIZES:
            continue
        runs.append({"run": os.path.basename(os.path.dirname(f)), "size": o["size"], "arm": o["arm"], "seed": o["seed"],
                     "recipe": "clip" if (o.get("cfg") or {}).get("clip") else "noclip", "device": o.get("device"),
                     **{k: o.get(k) for k in ("computed_acc", "computed_n", "final_acc", "final_n", "bpb", "rate",
                                              "e2e_acc", "e2e_n", "boxed_acc", "boxed_n", "params")},
                     "cfg": o.get("cfg")})
    return runs


def failed(runs):
    bad = set()
    for size in SIZES:
        b = [r["bpb"] for r in runs if r["size"] == size and np.isfinite(r["bpb"])]
        med = float(np.median(b))
        bad |= {r["run"] for r in runs if r["size"] == size and (not np.isfinite(r["bpb"]) or r["bpb"] > med + 1)}
    return bad


def groups(runs, bad):
    g = {}
    for r in runs:
        if r["run"] not in bad:
            g.setdefault((r["size"], r["arm"], r["recipe"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["seed"]) for k, v in g.items()}


def summary(rs):
    out = {"seeds": [r["seed"] for r in rs], "devices": sorted({r["device"] for r in rs})}
    for m in ("computed_acc", "final_acc", "bpb", "rate"):
        v = np.array([r[m] for r in rs], float) * MULT.get(m, 1.0)
        out[NAME.get(m, m)] = {"values": [float(x) for x in v], "mean": float(v.mean()),
                               "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
    return out


def pair(ra, rb):
    out = {"n": [len(ra), len(rb)], "devices": [sorted({r["device"] for r in ra}), sorted({r["device"] for r in rb})]}
    out["same_device"] = len(set(out["devices"][0]) | set(out["devices"][1])) == 1
    for m, mult in MULT.items():
        d, h, df = welch_df([r[m] * mult for r in ra], [r[m] * mult for r in rb])
        out[NAME[m]] = {"diff": d, "ci": [d - h, d + h], "df": df, "df_floored": int(df)}
    return out


def main():
    runs = load()
    bad = failed(runs)
    G = groups(runs, bad)
    res = {"source": "segresearch results/scale/runs (1M, 12M, 50M) and results/scale/cross; scale.analyze.welch",
           "convention": "differences are b - a in points (accuracy) or bits per byte; 95% Welch interval over seeds, "
                         "t from scale.analyze.T95 at the floored Welch-Satterthwaite df (seed variance only)",
           "runs": [{k: v for k, v in r.items() if k != "cfg"} | {"cfg": r["cfg"], "failed": r["run"] in bad} for r in runs],
           "failed": sorted(bad)}
    res["arms"] = {f"{s} {a} {c}": summary(v) for (s, a, c), v in sorted(G.items())}
    # the 1M dependence arm with its failed seed, for the table's footnote
    dep1 = [r for r in runs if r["size"] == "1m" and r["arm"] == "dep10"]
    res["arms_with_failed_seeds"] = {"1m dep10 noclip": summary(sorted(dep1, key=lambda r: r["seed"]))}

    P = {}
    for s, c, a, b in SAME:
        P[f"{s} {c}: {b} - {a}"] = pair(G[(s, a, c)], G[(s, b, c)]) | {"recipe": "same", "a": a, "b": b}
    for s, (a, ca), (b, cb) in ACROSS:
        P[f"{s}: {b} {cb} - {a} {ca}"] = pair(G[(s, a, ca)], G[(s, b, cb)]) | {"recipe": "across (clipped vs unclipped)", "a": f"{a} {ca}", "b": f"{b} {cb}"}
    res["pairs"] = P

    # T4 re-runs of 12M nov10 seeds 0 and 1 (separate trainings, same recipe and device as Table 1's nov10)
    T = {}
    for sd in (0, 1):
        d = os.path.join(SCALE, "cross", f"12m_nov10_s{sd}")
        o = json.load(open(os.path.join(d, "result.json")))
        x = json.load(open(os.path.join(d, "cross_eval.json")))
        t1 = next(r for r in runs if r["run"] == f"12m_nov10_s{sd}")
        T[f"seed {sd}"] = {"device": o["device"], "table1_device": t1["device"],
                           "final_rerun": 100 * o["final_acc"], "final_table1": 100 * t1["final_acc"],
                           "final_rerun_minus_table1": 100 * (o["final_acc"] - t1["final_acc"]),
                           "computed_rerun": 100 * o["computed_acc"], "computed_table1": 100 * t1["computed_acc"],
                           "bpb_rerun": o["bpb"], "bpb_table1": t1["bpb"],
                           "switched_layout": {k: {"final": 100 * v["final_acc"], "computed": 100 * v["computed_acc"],
                                                   "bpb": v["bpb"], "rate": v["rate"]} for k, v in x.items()}}
    T["mean over the two re-runs"] = {k: float(np.mean([T[f"seed {s}"]["switched_layout"][k]["final"] for s in (0, 1)]))
                                     for k in T["seed 0"]["switched_layout"]}
    res["t4_reruns"] = T

    # floors: end-to-end GSM8K (of e2e_n problems) and MATH \boxed{} accuracy
    def floor(rs):
        e = [(round(r["e2e_acc"] * r["e2e_n"]), r["e2e_n"], r["run"]) for r in rs if r.get("e2e_n")]
        bx = [(100 * r["boxed_acc"], r["run"]) for r in rs if r.get("boxed_n")]
        return {"n_runs": len(rs), "e2e_max_solved": max(e)[0], "e2e_n": sorted({x[1] for x in e}),
                "e2e_runs_at_max": sorted(x[2] for x in e if x[0] == max(e)[0]),
                "boxed_min": min(bx)[0], "boxed_min_run": min(bx)[1], "boxed_max": max(bx)[0], "boxed_max_run": max(bx)[1]}
    t1 = [r for (s, c), arms in TABLE1.items() for r in runs if r["size"] == s and r["recipe"] == c and r["arm"] in arms]
    hand = [r for r in runs if (r["size"], r["recipe"], r["arm"]) == HAND50]
    res["floors"] = {"table1_runs_ok": floor([r for r in t1 if r["run"] not in bad]),
                     "table1_runs_failed": [{"run": r["run"], "boxed": 100 * r["boxed_acc"], "e2e_solved": round(r["e2e_acc"] * r["e2e_n"])}
                                            for r in t1 if r["run"] in bad],
                     "hand_rule_50m_unclipped": floor(hand),
                     "all_runs_ok": floor([r for r in runs if r["run"] not in bad])}

    # R13's one-seed pilot (MPS) beside the 12M seeds of the arms it is read against
    p = json.load(open(os.path.join(_seg.HERE, "results", "r13_pilot", "result.json")))
    res["r13_pilot"] = {"pilot": {"arm": p.get("arm"), "device": p.get("device"), "final": 100 * p["final_acc"],
                                  "computed": 100 * p["computed_acc"], "bpb": p["bpb"], "rate": p["rate"]},
                        "12m": {a: {k: res["arms"][f"12m {a} noclip"][k]["values"] for k in ("final", "computed", "bpb", "rate")}
                                | {"devices": res["arms"][f"12m {a} noclip"]["devices"]}
                                for a in ("entropy10", "nov10", "syntax+entropy10", "dep10")}}

    # check against paper2v2_intervals.json (intervals.py training(), the private copy's run folders)
    iv = os.path.join(_seg.HERE, "results", "paper2v2_intervals.json")
    worst, n = 0.0, 0
    for k, v in json.load(open(iv))["training"]["pairs"].items():
        if k in P:
            for m in ("computed_acc", "final_acc", "bpb"):
                mine = P[k][NAME[m]]
                worst = max(worst, abs(mine["diff"] - v[m]["diff"]), *(abs(a - b) for a, b in zip(mine["ci"], v[m]["ci"])))
            n += 1
    res["check"] = {"pairs_compared_with_paper2v2_intervals": n, "largest_abs_difference": worst}
    assert n >= 15 and worst < 1e-9, res["check"]

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"runs {len(runs)}; failed {sorted(bad)}; pairs {len(P)}; check {res['check']}")
    for k, v in P.items():
        print(f"  {k:52s} " + "  ".join(f"{m} {v[m]['diff']:+7.3f} [{v[m]['ci'][0]:+7.3f}, {v[m]['ci'][1]:+7.3f}]"
                                        for m in ("computed", "final", "bpb")) + ("" if v["same_device"] else "  (cross-device)"))
    print("  T4 re-runs, final answers (re-run - Table 1):",
          [round(T[f"seed {s}"]["final_rerun_minus_table1"], 2) for s in (0, 1)])
    print("  floors:", json.dumps(res["floors"]))
    print("wrote", os.path.relpath(OUT, _seg.SEG))


if __name__ == "__main__":
    main()
