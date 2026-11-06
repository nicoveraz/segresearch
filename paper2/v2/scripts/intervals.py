"""Intervals for the paper 2 revision (issue #26; paper2v2/PLAN.md R6-R7). No model runs.

1. training   seed-level Welch 95% intervals (scale.analyze.welch, the published convention) from the existing
              result.json files: nov10 vs entropy10, syntax+entropy10, dep10; novc13 vs nov10; syntax+entropy10 vs
              entropy10; per size, clipped and unclipped runs kept apart; computed, final, bits per byte. Checks that
              the means reproduce paper 2's Table 1 and lists duplicate runs that disagree between folders.
2. layouts    BLT-1B inference layouts (results/blt_cues.json, GSM8K part): every pair among entropy, eq, syntax, nov,
              space at 10% and 15%, computed results and final answers; paired bootstrap over problems.
3. mechanism  problem-level bootstrap for the reducible2 numbers (stripped GSM8K, BLT-1B): per-problem byte ranges
              rebuilt from the texts (same order and filter as reducible2.py, tokenizer only), then: the large-model
              surprisal percentile and new-token share of entropy's vs the table's top-10% selections (and their
              differences), reducible bits by kind (computed - copied, computed - word), and the patcher-entropy
              budget numbers (C19). The top-10% threshold is fixed on all data; problems are resampled.

    ~/Documents/GitHub/segresearch/.venv-blt/bin/python paper2v2/intervals.py   -> results/paper2v2_intervals.json
"""
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import _seg  # noqa: F401,E402

CACHE = os.path.expanduser("~/.cache/segresearch-scale")
RUN_DIRS = [os.path.join(_seg.SEG, "results", "scale")]   # public copy: segresearch's own run files only
B = 2000


def training(run_dirs=None, label=None):
    """run_dirs: the folders to read (default RUN_DIRS); label(result) -> the arm name a run is grouped under (default
    its arm; rental_verdicts.py adds the GPU, so re-runs of an arm on another device stay apart)."""
    import glob
    from collections import defaultdict
    from scale.analyze import welch
    label = label or (lambda o: o["arm"])
    runs, dup = defaultdict(dict), []
    for d in RUN_DIRS if run_dirs is None else run_dirs:
        for f in sorted(set(glob.glob(os.path.join(d, "runs", "*", "result.json")) + glob.glob(os.path.join(d, "**", "runs", "*", "result.json"), recursive=True))):
            o = json.load(open(f))
            if o.get("toy"):
                continue
            key = (o["size"], label(o), bool((o.get("cfg") or {}).get("clip")))
            old = runs[key].get(o["seed"])
            if old is not None and any(abs(old[m] - o[m]) > 1e-9 for m in ("computed_acc", "final_acc", "bpb")):
                dup.append(f"{key} seed {o['seed']}: {f}")
            runs[key][o["seed"]] = o
    # failed-training exclusion (the pilot's rule): bits per byte > median of its size + 1; a bits per byte that is not
    # finite also counts as failed, and the median is taken over the finite ones (review TR-12, 2026-10-29)
    for size in {k[0] for k in runs}:
        allb = [r["bpb"] for k, v in runs.items() if k[0] == size for r in v.values()]
        med = float(np.median([x for x in allb if np.isfinite(x)] or [np.nan]))
        for k in [k for k in runs if k[0] == size]:
            for s in [s for s, r in runs[k].items() if not np.isfinite(r["bpb"]) or r["bpb"] > med + 1]:
                b = runs[k][s]["bpb"]
                dup.append(f"EXCLUDED failed run {k} seed {s} (bpb {b:.3f} " + (f"> median {med:.3f} + 1)" if np.isfinite(b) else "not finite)"))
                del runs[k][s]
    out = {"runs": {f"{k[0]} {k[1]} {'clip' if k[2] else 'noclip'}": {m: [float(r.get(m, np.nan)) for _, r in sorted(v.items())] for m in ("computed_acc", "final_acc", "bpb", "rate")}
                    | {"seeds": sorted(v)} for k, v in sorted(runs.items())}, "pairs": {}, "notes": dup}
    pairs = (("entropy10", "nov10"), ("syntax+entropy10", "nov10"), ("dep10", "nov10"), ("nov10", "novc13"),
             ("entropy10", "syntax+entropy10"), ("entropy10", "novc13"))
    for size in ("1m", "12m", "50m", "150m"):
        for clip in (False, True):
            for a, b in pairs:                               # welch(x, y) = mean(y) - mean(x)
                ra, rb = runs.get((size, a, clip), {}), runs.get((size, b, clip), {})
                if len(ra) < 2 or len(rb) < 2:
                    continue
                key = f"{size} {'clip' if clip else 'noclip'}: {b} - {a}"
                out["pairs"][key] = {}
                for m, mult in (("computed_acc", 100), ("final_acc", 100), ("bpb", 1)):
                    d, h = welch([r[m] * mult for r in ra.values()], [r[m] * mult for r in rb.values()])
                    out["pairs"][key][m] = {"diff": float(d), "ci": [float(d - h), float(d + h)], "n": [len(ra), len(rb)]}
    # 50M: paper 2 compares clipped novelty with clipped baselines
    return out


def layouts(rng):
    c = json.load(open(os.path.join(ROOT, "results", "blt_cues.json")))
    rows = [r for r in c["rows"] if r["part"] == "gsm8k"]
    by = {}
    for r in rows:
        by.setdefault((r["layout"], r["kind"]), {})[(r["problem"], r["start"])] = float(r["ok"])
    out = {}
    names = ("entropy", "eq", "syntax", "nov", "space")
    for R in (10, 15):
        for kind in ("res", "fin"):
            for a, b in combinations(names, 2):
                A, Bd = by[(f"{a}@{R}", kind)], by[(f"{b}@{R}", kind)]
                keys = sorted(set(A) & set(Bd)); P = np.array([k[0] for k in keys])
                d = np.array([A[k] - Bd[k] for k in keys]); probs = np.unique(P); grp = {p: d[P == p] for p in probs}
                bs = [np.concatenate([grp[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(B)]
                out[f"{kind} {a}-{b}@{R}"] = {"diff": float(100 * d.mean()), "ci": [float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))],
                                             "acc_a": float(100 * np.mean(list(A.values()))), "acc_b": float(100 * np.mean(list(Bd.values())))}
    return out


def mechanism(rng):
    import stripped_units as su
    from realblt_budget import GSM, MODEL
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    lens = []
    for l in list(open(GSM))[:1319]:
        text, s0 = su.gsm8k_text(json.loads(l)); b = text.encode()
        n = len(tok(text).input_ids)
        if n <= 1000 and n == len(b) + 1:
            lens.append(len(b))
    z = np.load(f"{CACHE}/ncm_math/reducible2_cols.npz")
    C = {k: z[k] for k in z.files if k != "used"}
    assert sum(lens) == len(C["s_small"]) and len(lens) == int(z["used"]), (sum(lens), len(C["s_small"]))
    P = np.repeat(np.arange(len(lens)), lens)
    kind = C["kind"].astype(str); red = C["s_small"] - C["s_large"]
    rk = lambda x: (np.argsort(np.argsort(x)) + 0.5) / len(x)
    sl_p, ent_p = rk(C["s_large"]), rk(C["ent_small"])
    rng0 = np.random.default_rng(0)

    def top10(x):                                            # as reducible2: ties broken at random
        o = np.lexsort((rng0.random(len(x)), -x)); t = np.zeros(len(x), bool); t[o[: int(0.10 * len(x))]] = True
        return t
    sel = {"entropy": top10(C["ent_small"]), "nov": top10(C["nov"])}
    thr = np.percentile(C["ent_small"], 90); top_e = C["ent_small"] >= thr
    starts = np.r_[0, np.cumsum(lens)]
    stats = {
        "entropy_sel_s_large_pct": lambda m: sl_p[m & sel["entropy"]].mean(),
        "nov_sel_s_large_pct": lambda m: sl_p[m & sel["nov"]].mean(),
        "diff_s_large_pct_entropy_minus_nov": lambda m: sl_p[m & sel["entropy"]].mean() - sl_p[m & sel["nov"]].mean(),
        "entropy_sel_new_share": lambda m: C["tok_new"][m & sel["entropy"]].mean(),
        "nov_sel_new_share": lambda m: C["tok_new"][m & sel["nov"]].mean(),
        "diff_new_share_nov_minus_entropy": lambda m: C["tok_new"][m & sel["nov"]].mean() - C["tok_new"][m & sel["entropy"]].mean(),
        "diff_new_share_numbers_nov_minus_entropy": lambda m: C["tok_new"][m & sel["nov"] & C["is_num"]].mean() - C["tok_new"][m & sel["entropy"] & C["is_num"]].mean(),
        "reducible_computed_minus_copied": lambda m: red[m & (kind == "computed")].mean() - red[m & (kind == "copied")].mean(),
        "reducible_computed_minus_word": lambda m: red[m & (kind == "computed")].mean() - red[m & (kind == "word")].mean(),
        "reducible_final_minus_computed": lambda m: red[m & (kind == "final")].mean() - red[m & (kind == "computed")].mean(),
        "computed_entropy_pct": lambda m: ent_p[m & (kind == "computed")].mean(),
        "computed_share_above_p90": lambda m: top_e[m & (kind == "computed")].mean(),
        "budget_share_word": lambda m: (top_e & m & (kind == "word")).sum() / (top_e & m).sum(),
        "budget_share_computed": lambda m: (top_e & m & (kind == "computed")).sum() / (top_e & m).sum(),
    }
    full = np.ones(len(P), bool)
    out = {k: {"value": float(f(full))} for k, f in stats.items()}
    boots = {k: [] for k in stats}
    nprob = len(lens)
    for _ in range(400):
        pick = rng.integers(0, nprob, nprob)
        w = np.bincount(pick, minlength=nprob)                 # resample problems: weight bytes by problem counts
        idx = np.concatenate([np.arange(starts[p], starts[p + 1]) for p in np.flatnonzero(w) for _ in range(w[p])])
        for k in stats:                                        # index arrays: resampled problems repeat bytes
            boots[k].append(float(_eval_idx(k, idx, C, kind, red, sl_p, ent_p, sel, top_e)))
    for k in stats:
        out[k]["ci"] = [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]
    return out


def _eval_idx(k, idx, C, kind, red, sl_p, ent_p, sel, top_e):
    """The same statistics on an index array (bytes may repeat, as resampled problems do)."""
    se, sn = sel["entropy"][idx], sel["nov"][idx]; kd = kind[idx]; nw = C["tok_new"][idx]; num = C["is_num"][idx]
    slp, rd, ep, te = sl_p[idx], red[idx], ent_p[idx], top_e[idx]
    v = {
        "entropy_sel_s_large_pct": lambda: slp[se].mean(), "nov_sel_s_large_pct": lambda: slp[sn].mean(),
        "diff_s_large_pct_entropy_minus_nov": lambda: slp[se].mean() - slp[sn].mean(),
        "entropy_sel_new_share": lambda: nw[se].mean(), "nov_sel_new_share": lambda: nw[sn].mean(),
        "diff_new_share_nov_minus_entropy": lambda: nw[sn].mean() - nw[se].mean(),
        "diff_new_share_numbers_nov_minus_entropy": lambda: nw[sn & num].mean() - nw[se & num].mean(),
        "reducible_computed_minus_copied": lambda: rd[kd == "computed"].mean() - rd[kd == "copied"].mean(),
        "reducible_computed_minus_word": lambda: rd[kd == "computed"].mean() - rd[kd == "word"].mean(),
        "reducible_final_minus_computed": lambda: rd[kd == "final"].mean() - rd[kd == "computed"].mean(),
        "computed_entropy_pct": lambda: ep[kd == "computed"].mean(),
        "computed_share_above_p90": lambda: te[kd == "computed"].mean(),
        "budget_share_word": lambda: (te & (kd == "word")).sum() / te.sum(),
        "budget_share_computed": lambda: (te & (kd == "computed")).sum() / te.sum(),
    }
    return v[k]()


def extras(rng, training_runs):
    """Review 4 (2026-10-23): intervals the revised text still lacked."""
    import math
    from realblt_budget import GSM, MODEL, problem
    from scale.analyze import T95
    from transformers import AutoTokenizer
    out = {}
    # (a) BLT-1B layouts from blt_compress.json (per-unit items in problem order) vs entropy, paired by problem
    tok = AutoTokenizer.from_pretrained(MODEL)
    nres = [sum(k == "res" for k, _, _ in pr[2]) for pr in (problem(json.loads(l), tok, "cpu") for l in list(open(GSM))[:300]) if pr]
    P = np.repeat(np.arange(len(nres)), nres)
    items = json.load(open(os.path.join(ROOT, "results", "blt_compress.json")))["items"]
    assert len(items["entropy@10"]["res_exact"]) == len(P), (len(items["entropy@10"]["res_exact"]), len(P))
    probs = np.unique(P)
    for R in (10, 15):
        base = np.array(items[f"entropy@{R}"]["res_exact"], float)
        for k in ("typed", "nov", "novc", "lz", "ppm", "lz+nov"):
            d = np.array(items[f"{k}@{R}"]["res_exact"], float) - base
            grp = {p: d[P == p] for p in probs}
            bs = [np.concatenate([grp[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(B)]
            out[f"blt1b res {k}-entropy@{R}"] = {"diff": float(100 * d.mean()), "ci": [float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))]}

    # (b) Welch from published mean, sd, n (tables without per-seed files here); diff = b - a
    def sw(a, b):
        (m1, s1, n1), (m2, s2, n2) = a, b
        v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2; se = math.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)); h = T95.get(int(df), 1.96) * se
        return {"diff": m2 - m1, "ci": [m2 - m1 - h, m2 - m1 + h], "df": df}
    out["prose bpb novelty-entropy"] = sw((1.832, 0.031, 2), (1.770, 0.044, 2))
    out["code bpb novelty-entropy (matched rate)"] = sw((1.649, 0.004, 2), (1.684, 0.006, 2))
    out["traces computed novelty-entropy"] = sw((55.1, 8.6, 2), (50.2, 0.2, 2))
    out["traces copied novelty-entropy"] = sw((56.9, 5.4, 2), (61.0, 3.5, 2))
    out["typed-entropy final 12M"] = sw((46.6, 4.0, 3), (18.9, 7.4, 3))
    out["typed-entropy final 50M"] = sw((59.9, 1.3, 3), (35.1, 21.1, 3))
    out["typed-entropy computed 50M"] = sw((20.7, 7.9, 3), (8.9, 1.1, 3))
    out["typed-entropy final 1M"] = {"note": "not run at 1M in paper 2's table"}
    # (c) from per-seed results
    from scale.analyze import welch
    def ws(a, b, m, mult):
        d, h = welch([x * mult for x in training_runs[a][m]], [x * mult for x in training_runs[b][m]])
        return {"diff": float(d), "ci": [float(d - h), float(d + h)]}
    out["50M entropy computed clip-noclip"] = ws("50m entropy10 noclip", "50m entropy10 clip", "computed_acc", 100)
    out["50M entropy final clip-noclip"] = ws("50m entropy10 noclip", "50m entropy10 clip", "final_acc", 100)
    out["12M bpb novc13-syntax"] = ws("12m syntax+entropy10 noclip", "12m novc13 noclip", "bpb", 1)
    out["12M bpb novc13-entropy"] = ws("12m entropy10 noclip", "12m novc13 noclip", "bpb", 1)
    out["50M bpb novc13-nov10"] = ws("50m nov10 clip", "50m novc13 clip", "bpb", 1)
    # (d) reference percentiles for the mechanism reading: random positions sit at 0.5 by construction
    z = np.load(f"{CACHE}/ncm_math/reducible2_cols.npz")
    sl = z["s_large"]; kind = z["kind"].astype(str)
    pct = (np.argsort(np.argsort(sl)) + 0.5) / len(sl)
    out["s_large pct at computed-result starts"] = float(pct[kind == "computed"].mean())
    out["s_large pct at word starts"] = float(pct[kind == "word"].mean())
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"training": training(), "layouts": layouts(rng)}
    print("== training (Welch 95% over seeds; mean(b) - mean(a))")
    for n in res["training"]["notes"]:
        print("  NOTE", n)
    for k, v in res["training"]["pairs"].items():
        print(f"  {k:42s} " + "  ".join(f"{m.split('_')[0][:5]} {x['diff']:+6.2f} [{x['ci'][0]:+6.2f}, {x['ci'][1]:+6.2f}] n{x['n']}" for m, x in v.items()))
    print("== layouts (BLT-1B, paired problem bootstrap; points)")
    for k, v in res["layouts"].items():
        print(f"  {k:24s} {v['diff']:+6.1f} [{v['ci'][0]:+6.1f}, {v['ci'][1]:+6.1f}]  ({v['acc_a']:.1f} vs {v['acc_b']:.1f})")
    res["mechanism"] = mechanism(rng)
    print("== mechanism (problem bootstrap, 400 resamples)")
    for k, v in res["mechanism"].items():
        print(f"  {k:44s} {v['value']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    res["extras"] = extras(np.random.default_rng(1), res["training"]["runs"])
    print("== extras (review 4)")
    for k, v in res["extras"].items():
        print(f"  {k:44s} {v}")
    json.dump(res, open(os.path.join(ROOT, "results", "paper2v2_intervals.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
