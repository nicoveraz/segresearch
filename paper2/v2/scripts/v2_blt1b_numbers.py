"""BLT-1B inference numbers for paper 2 v2 that no earlier output holds, from shipped result files (CPU, no model;
new in v2). Every BLT-1B number is teacher-forced, at inference, on a model trained under entropy patching.

Reads paper2/v2/results/blt_waste.json (R8: 294 stripped GSM8K test problems), blt_hybrid.json (R8b, the same
problems), blt_compress.json, paper2v2_intervals.json and blt_prose.json (with R9's rewrites), and writes paper2/v2/results/v2_blt1b_numbers.json:

  bpb_vs_entropy  bits per byte of the `=` rule, the novelty table, the table plus the answer marker and the hybrid
                  (entropy plus the table's top 3%) against entropy at a 10% budget, on the whole text, the question
                  and the solution: pooled bits / pooled bytes, paired problem bootstrap (2,000 resamples)
  zero_value      where the patch starts that buy no bits (value <= 0 in R8 part A) fall, by position type:
                  share of starts x share with value <= 0 / the layout's share with value <= 0
  k_bracket       the measured entropy points that bracket the table's computed-result accuracy at 10%, and the
                  range of the rate ratio they allow (R8 part B interpolates linearly between them)
  compress_pairs  computed-result exact match between the compressor layouts of blt_compress.json, paired by problem
                  (2,000 resamples); the problem grouping comes from blt_waste.json's per-problem counts and is
                  checked by reproducing paper2v2_intervals.json's 'blt1b res <k>-entropy@<R>' intervals exactly
  prose_by_cue    R9's prose test (blt_prose.json rows, data/prose_rewrites/test.jsonl): coverage and exact match of
                  entropy and the novelty table on computed values, split by the bytes before each value ('= ', '$',
                  plain space), as r9_cue_split.py splits entropy's rows (after the data; descriptive, no interval)

    python paper2/v2/run.py v2_blt1b_numbers          (or python paper2/v2/scripts/v2_blt1b_numbers.py)
"""
import json
import os

import numpy as np

import _seg

R = os.path.join(_seg.HERE, "results")
OUT = os.path.join(R, "v2_blt1b_numbers.json")
B = 2000


def boot_pooled(rows_a, rows_b, region, rng):
    """b - a of pooled bits per byte over a region, paired by problem; rows_*: {problem: row}."""
    P = np.array(sorted(set(rows_a) & set(rows_b)))
    get = lambda rows, k: np.array([rows[p][k] for p in P], float)
    ba, na, bb, nb = (get(rows_a, f"bits_{region}"), get(rows_a, f"n_{region}"),
                      get(rows_b, f"bits_{region}"), get(rows_b, f"n_{region}"))
    d = bb.sum() / nb.sum() - ba.sum() / na.sum()
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(P), len(P))
        bs.append(bb[i].sum() / nb[i].sum() - ba[i].sum() / na[i].sum())
    return {"diff": float(d), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "a": float(ba.sum() / na.sum()), "b": float(bb.sum() / nb.sum()), "n_problems": int(len(P))}


def main():
    w = json.load(open(os.path.join(R, "blt_waste.json")))
    h = json.load(open(os.path.join(R, "blt_hybrid.json")))
    by = {}
    for r in w["rowsB"] + h["rows"]:
        by.setdefault(r["layout"], {})[r["problem"]] = r
    res = {"source": "results/blt_waste.json, blt_hybrid.json, blt_compress.json, paper2v2_intervals.json",
           "convention": f"b - a; paired problem bootstrap, {B} resamples, 2.5th and 97.5th percentiles"}

    # 1. bits per byte against entropy at 10%
    rng = np.random.default_rng(0)
    bpb = {}
    for lay in ("eq@0.1", "table@0.1", "table+marker@0.1", "hybrid(0.03)@0.1"):
        for region in ("all", "q", "sol"):
            bpb[f"{lay} - entropy@0.1 | {region}"] = boot_pooled(by["entropy@0.1"], by[lay], region, rng)
    for k, v in bpb.items():                      # the pooled levels reproduce the stored curves
        lay, region = k.split(" - ")[0], k.split("| ")[1]
        cur = (w["curves"].get(lay) or h["curves"][lay])[f"bpb_{region}"]
        assert abs(v["b"] - cur) < 1e-9, (k, v["b"], cur)
    res["bpb_vs_entropy"] = bpb

    # 2. where the starts that buy nothing fall (R8 part A, 150 problems)
    zv = {}
    for lay in ("table@0.1", "entropy@0.1"):
        A = w["A"][lay]
        tot = A["share_value_le_0"]
        zv[lay] = {"n_starts": A["n_starts"], "share_value_le_0": tot,
                   "by_type": {t: {"share_of_starts": v["share_of_starts"], "share_value_le_0": v["share_value_le_0"],
                                   "share_of_zero_value_starts": v["share_of_starts"] * v["share_value_le_0"] / tot}
                               for t, v in A["by_type"].items()}}
        assert abs(sum(x["share_of_zero_value_starts"] for x in zv[lay]["by_type"].values()) - 1) < 1e-9
    res["zero_value"] = zv

    # 3. the entropy points bracketing the table's 10% computed-result accuracy (R8 part B)
    C = w["curves"]
    tab = C["table@0.1"]
    ent = sorted((v["rate"], v["acc_res"], k) for k, v in C.items() if k.startswith("entropy@"))
    lo = max((e for e in ent if e[1] <= tab["acc_res"]), key=lambda e: e[1])
    hi = min((e for e in ent if e[1] >= tab["acc_res"]), key=lambda e: e[1])
    k_lin = (lo[0] + (tab["acc_res"] - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1])) / tab["rate"]
    assert abs(k_lin - w["B"]["table@0.1 acc_res"]["k"]) < 1e-9
    res["k_bracket"] = {"table": {"rate": tab["rate"], "acc_res": tab["acc_res"]},
                        "entropy_below": {"layout": lo[2], "rate": lo[0], "acc_res": lo[1]},
                        "entropy_above": {"layout": hi[2], "rate": hi[0], "acc_res": hi[1]},
                        "k_linear": k_lin, "k_range_from_bracketing_points": [lo[0] / tab["rate"], hi[0] / tab["rate"]],
                        "k_bootstrap_ci_stored": w["B"]["table@0.1 acc_res"]["ci"]}

    # 4. compressor layouts paired by problem (R5); grouping checked against paper2v2_intervals.json extras (a)
    items = json.load(open(os.path.join(R, "blt_compress.json")))["items"]
    ent10 = sorted(by["entropy@0.1"].values(), key=lambda r: r["problem"])
    nres = [r["n_res"] for r in ent10]
    P = np.repeat(np.arange(len(nres)), nres)
    probs = np.unique(P)
    assert len(items["entropy@10"]["res_exact"]) == len(P)

    def paired(a, b, rng):
        d = np.array(items[b]["res_exact"], float) - np.array(items[a]["res_exact"], float)
        grp = {p: d[P == p] for p in probs}
        bs = [np.concatenate([grp[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(B)]
        return {"diff": float(100 * d.mean()), "ci": [float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))],
                "acc_a": float(100 * np.mean(items[a]["res_exact"])), "acc_b": float(100 * np.mean(items[b]["res_exact"]))}

    rng = np.random.default_rng(1)                          # intervals.py extras(a): same seed, order and B
    ext = json.load(open(os.path.join(R, "paper2v2_intervals.json")))["extras"]
    worst = 0.0
    for Rb in (10, 15):
        for k in ("typed", "nov", "novc", "lz", "ppm", "lz+nov"):
            got = paired(f"entropy@{Rb}", f"{k}@{Rb}", rng)
            ref = ext[f"blt1b res {k}-entropy@{Rb}"]
            worst = max(worst, abs(got["diff"] - ref["diff"]), *(abs(x - y) for x, y in zip(got["ci"], ref["ci"])))
    assert worst < 1e-9, worst
    rng = np.random.default_rng(2)
    res["compress_pairs"] = {"grouping_check": {"reproduces": "paper2v2_intervals.json extras 'blt1b res <k>-entropy@<R>'",
                                                "largest_abs_difference": worst, "n_problems_with_results": int(len(probs)),
                                                "n_computed_results": int(len(P))},
                             "lz+nov@10 - nov@10": paired("nov@10", "lz+nov@10", rng),
                             "lz+nov@15 - nov@15": paired("nov@15", "lz+nov@15", rng)}

    # 5. R9's prose test split by the cue before each value (r9_cue_split.py's cue(), for entropy and the table)
    from r9_cue_split import cue
    rows = json.load(open(os.path.join(R, "blt_prose.json")))["rows"]
    recs = {r["idx"]: r for r in map(json.loads, open(os.path.join(_seg.HERE, "data", "prose_rewrites", "test.jsonl")))}
    agg = {}
    for x in rows:
        if x["kind"] != "res" or x["layout"] not in ("entropy", "nov"):
            continue
        r, f = recs[x["problem"]], x["format"]
        a = agg.setdefault(f"{f}|{x['layout']}|{cue(r[f], r[f + '_spans'][x['j']])}", {"n": 0, "covered": 0, "exact": 0})
        a["n"] += 1; a["covered"] += bool(x["covered"]); a["exact"] += bool(x["ok"])
    split = {k: {"n": a["n"], "covered": a["covered"] / a["n"], "exact": a["exact"] / a["n"]} for k, a in sorted(agg.items())}
    split2 = json.load(open(os.path.join(R, "blt_prose_cue_split.json")))   # entropy's rows reproduce r9_cue_split.py
    for k, v in split2.items():
        f, c = k.split("|")
        e = split[f"{f}|entropy|{c}"]
        assert e["n"] == v["n"] and abs(e["covered"] - v["entropy_covered"]) < 1e-12 and abs(e["exact"] - v["exact_entropy"]) < 1e-12
    for f in ("orig", "prose"):                             # the table's pooled coverage reproduces blt_prose.json
        k = [v for key, v in split.items() if key.startswith(f"{f}|nov|")]
        cov = sum(v["covered"] * v["n"] for v in k) / sum(v["n"] for v in k)
        L = json.load(open(os.path.join(R, "blt_prose.json")))["by_layout"][f"{f}|nov|res"]
        assert abs(cov - L["covered"]) < 1e-12, (f, cov, L["covered"])
    res["prose_by_cue"] = {"source": "results/blt_prose.json rows; data/prose_rewrites/test.jsonl; cue() of r9_cue_split.py",
                           "note": "after the data, descriptive; '<format>|<layout>|<cue>', layout 'nov' is the novelty table",
                           "split": split}

    json.dump(res, open(OUT, "w"), indent=1)
    for k, v in bpb.items():
        print(f"  bpb {k:36s} {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  ({v['b']:.3f} vs {v['a']:.3f})")
    for t, v in sorted(zv["table@0.1"]["by_type"].items(), key=lambda x: -x[1]["share_of_zero_value_starts"]):
        print(f"  table@0.1 zero-value starts: {t:10s} {100 * v['share_of_zero_value_starts']:5.1f}%"
              f"  (of its starts {100 * v['share_of_starts']:4.1f}%, value <= 0 in {100 * v['share_value_le_0']:4.1f}%)")
    kb = res["k_bracket"]
    print(f"  k {kb['k_linear']:.3f}, bracketing points allow {kb['k_range_from_bracketing_points'][0]:.2f}-"
          f"{kb['k_range_from_bracketing_points'][1]:.2f}")
    for k in ("lz+nov@10 - nov@10", "lz+nov@15 - nov@15"):
        v = res["compress_pairs"][k]
        print(f"  {k}: {v['diff']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}] ({v['acc_b']:.1f} vs {v['acc_a']:.1f})")
    for k, v in res["prose_by_cue"]["split"].items():
        print(f"  R9 {k:22s} n {v['n']:4d} covered {100 * v['covered']:5.1f}% exact {100 * v['exact']:5.1f}%")
    print("wrote", os.path.relpath(OUT, _seg.SEG))


if __name__ == "__main__":
    main()
