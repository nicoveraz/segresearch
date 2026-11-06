"""Seed-level comparisons for the models trained from scratch (paper 1 v2, section 6), kept apart by training recipe.

    python paper/v2/scripts/scale_compare.py          -> paper/v2/results/scale_compare.json

results/scale/runs/<size>_<arm>_s<seed>[_clip]/result.json holds two recipes:
  unclipped  the 28 runs of Table 3 (no gradient clipping: cfg has no 'clip'; NVIDIA A40, bf16): entropy10, dep10 and
             syntax+entropy10 at 1M, 12M and 50M, jump10 and entropy20 at 50M;
  clipped    50M re-runs of entropy10 and dep10 with gradient clipping at norm 1.0 (cfg['clip'] = 1.0, folder suffix
             _clip; the same A40 and bf16, 3 seeds each; merged into this repository at commit 7ba188e).
The same folder also holds runs of other rules on other hardware (Tesla T4, fp16, at 1M and 12M); they are not used.
scale/analyze.load() keys runs by (size, arm, seed) only, so a clipped run and an unclipped run with the same arm and
seed overwrite each other there; this script keys runs by recipe as well.

Per comparison B - A: the difference of the seed means (points for accuracies, bits for bits per byte), a Welch 95%
interval over seeds (scale.analyze.welch: Welch-Satterthwaite df floored, t quantile from analyze.T95), and whether
every seed of B is above (or below) every seed of A. Failed trainings (bits per byte more than 1 bit above the median
of their size, scale.analyze.failed) are excluded: one, 1M dep10 seed 0.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from scale import analyze  # noqa: E402

RUNS = os.path.join(ROOT, "results", "scale", "runs")
OUT = os.path.join(ROOT, "paper", "v2", "results", "scale_compare.json")
PAPER1_ARMS = analyze.ARMS + analyze.EXTRA          # the rules of Table 3
CLIPPED_ARMS = ("entropy10", "dep10")
METRICS = ("final_acc", "computed_acc", "bpb")


def load(recipe):
    """{(size, arm): {seed: result}} for one recipe ('unclipped': Table 3; 'clipped': the 50M clipping re-runs)."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "result.json"))):
        o = json.load(open(f))
        clip = bool(o["cfg"].get("clip"))
        if recipe == "unclipped" and (clip or o["arm"] not in PAPER1_ARMS or o["device"] != "NVIDIA A40"):
            continue
        if recipe == "clipped" and (not clip or o["arm"] not in CLIPPED_ARMS):
            continue
        k = (o["size"], o["arm"])
        assert o["seed"] not in runs.setdefault(k, {}), f"two {recipe} runs for {k} seed {o['seed']}: {f}"
        runs[k][o["seed"]] = dict(o, dir=os.path.basename(os.path.dirname(f)))
    return runs


def runs_ok(recipe):
    runs = load(recipe)
    bad = analyze.failed(runs)
    return {k: {sd: o for sd, o in v.items() if (k[0], k[1], sd) not in bad} for k, v in runs.items()}, sorted(bad)


def vals(rs, k):
    return [o[k] if k == "bpb" else 100 * o[k] for _, o in sorted(rs.items())]


def compare(ok, size, a, b):
    out = {}
    for k in METRICS:
        x, y = vals(ok[(size, a)], k), vals(ok[(size, b)], k)
        d, h = analyze.welch(x, y)
        out[k] = {"diff": float(d), "lo": float(d - h), "hi": float(d + h), "A": x, "B": y,
                  "mean_A": float(np.mean(x)), "mean_B": float(np.mean(y)),
                  "separated": bool(min(y) > max(x) or max(y) < min(x)), "seeds": [len(x), len(y)]}
    return out


def main():
    res = {}
    for recipe in ("unclipped", "clipped"):
        ok, bad = runs_ok(recipe)
        r = {"runs": sum(len(v) for v in load(recipe).values()), "failed": [list(x) for x in bad],
             "devices": sorted({o["device"] for v in ok.values() for o in v.values()}),
             "clip": sorted({str(o["cfg"].get("clip")) for v in ok.values() for o in v.values()}),
             "e2e_max": max(o["e2e_acc"] for v in ok.values() for o in v.values()),
             "means": {f"{s}|{a}": {k: float(np.mean(vals(rs, k))) for k in METRICS + ("rate",)} | {"seeds": len(rs)}
                       for (s, a), rs in sorted(ok.items())},
             "compare": {}}
        sizes = ("1m", "12m", "50m") if recipe == "unclipped" else ("50m",)
        for size in sizes:
            r["compare"][f"{size}|dep10-entropy10"] = compare(ok, size, "entropy10", "dep10")
            if recipe == "unclipped":
                r["compare"][f"{size}|syntax+entropy10-entropy10"] = compare(ok, size, "entropy10", "syntax+entropy10")
        res[recipe] = r
    # the recipe change itself, same rule and seeds (50M)
    un, _ = runs_ok("unclipped"); cl, _ = runs_ok("clipped")
    res["clipped_minus_unclipped"] = {}
    for arm in CLIPPED_ARMS:
        res["clipped_minus_unclipped"][arm] = {}
        for k in METRICS:
            x, y = vals(un[("50m", arm)], k), vals(cl[("50m", arm)], k)
            d, h = analyze.welch(x, y)
            res["clipped_minus_unclipped"][arm][k] = {"diff": float(d), "lo": float(d - h), "hi": float(d + h)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for recipe in ("unclipped", "clipped"):
        r = res[recipe]
        print(f"== {recipe}: {r['runs']} runs, failed {r['failed']}, devices {r['devices']}, clip {r['clip']}, "
              f"end-to-end max {r['e2e_max']}")
        for name, c in r["compare"].items():
            for k, v in c.items():
                f = "{:+.3f} [{:+.3f}, {:+.3f}]" if k == "bpb" else "{:+.1f} [{:+.1f}, {:+.1f}]"
                print(f"  {name:28s} {k:12s} " + f.format(v["diff"], v["lo"], v["hi"]) +
                      f"  B {v['mean_B']:.3f} vs A {v['mean_A']:.3f}  seeds {v['seeds']}  separated={v['separated']}")
    for arm, c in res["clipped_minus_unclipped"].items():
        for k, v in c.items():
            f = "{:+.3f} [{:+.3f}, {:+.3f}]" if k == "bpb" else "{:+.1f} [{:+.1f}, {:+.1f}]"
            print(f"  clipped - unclipped 50m {arm:10s} {k:12s} " + f.format(v["diff"], v["lo"], v["hi"]))
    print(f"-> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
