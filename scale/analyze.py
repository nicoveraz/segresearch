"""Summarize the scaling pilot (#27) from a results directory (runs/<size>_<arm>_s<seed>/result.json).

    python -m scale.analyze RESULTS_DIR        # prints a table per metric and the gaps with seed separation

Per size and rule: mean and per-seed values; per comparison (dep - entropy, hand-written - entropy): the gap of the
means, a Welch 95% interval over seeds, whether every seed of one rule is above every seed of the other, and a
pooled two-proportion interval over all targets (each run scores the same targets). Runs whose training failed
(bits per byte more than 1 bit above the median of their size) are listed and excluded.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

SIZES = ("1m", "12m", "50m")
ARMS = ("entropy10", "dep10", "syntax+entropy10")
EXTRA = ("jump10", "entropy20")                  # 50M only: BLT's monotonic rule, and entropy at twice the budget
NAMES = {"entropy10": "entropy", "dep10": "dependence", "syntax+entropy10": "hand-written",
         "jump10": "jump (BLT monotonic)", "entropy20": "entropy at 20%"}
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "scale")
METRICS = (("final_acc", "final answers"), ("computed_acc", "computed results"), ("boxed_acc", "MATH boxed"),
           ("bpb", "bits per byte"), ("e2e_acc", "end-to-end GSM8K"), ("rate", "eval patch rate"))
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}


def load(d):
    runs = defaultdict(dict)
    for f in glob.glob(os.path.join(d, "runs", "*", "result.json")):
        o = json.load(open(f))
        runs[(o["size"], o["arm"])][o["seed"]] = o
    return runs


def failed(runs):
    out = set()
    for size in SIZES:
        b = [o["bpb"] for (s, a), seeds in runs.items() if s == size for o in seeds.values()]
        if not b:
            continue
        med = float(np.median(b))
        out |= {(s, a, sd) for (s, a), seeds in runs.items() if s == size for sd, o in seeds.items() if o["bpb"] > med + 1}
    return out


def runs_ok(d=RESULTS):
    """{(size, arm): {seed: result}} without failed trainings, and the set of failures (size, arm, seed)."""
    runs = load(d)
    bad = failed(runs)
    return {k: {sd: o for sd, o in v.items() if (k[0], k[1], sd) not in bad} for k, v in runs.items()}, bad


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y.mean() - x.mean()
    if len(x) < 2 or len(y) < 2:
        return d, float("nan")
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else 1
    return d, T95.get(int(df), 1.96) * se


def main(d):
    runs = load(d)
    bad = failed(runs)
    if bad:
        print("training failures excluded: " + ", ".join(f"{s} {NAMES[a]} s{sd}" for s, a, sd in sorted(bad)))
    ok = {k: {sd: o for sd, o in v.items() if (k[0], k[1], sd) not in bad} for k, v in runs.items()}
    for key, label in METRICS:
        print(f"\n## {label}")
        print(f"| size | " + " | ".join(NAMES[a] for a in ARMS) + " |")
        print("|---|" + "---|" * len(ARMS))
        for size in SIZES:
            cells = []
            for a in ARMS:
                v = [o[key] for o in ok.get((size, a), {}).values()]
                if not v:
                    cells.append("—")
                    continue
                pct = key != "bpb"
                f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
                cells.append(f"{f(np.mean(v))}{'%' if pct else ''} ({', '.join(f(x) for x in v)}; n={len(v)})")
            if any(c != "—" for c in cells):
                print(f"| {size} | " + " | ".join(cells) + " |")
    print("\n## gaps over entropy (points; Welch 95% over seeds; seeds separated?; pooled over targets)")
    for key in ("final_acc", "computed_acc"):
        for size in SIZES:
            base = ok.get((size, "entropy10"), {})
            for a in ("dep10", "syntax+entropy10"):
                other = ok.get((size, a), {})
                if len(base) < 1 or len(other) < 1:
                    continue
                x = [o[key] for o in base.values()]; y = [o[key] for o in other.values()]
                gap, hw = welch(x, y)
                sep = min(y) > max(x) if gap > 0 else max(y) < min(x)
                n = next(iter(other.values()))[key.replace("_acc", "_n")]
                p1, p2 = np.mean(x), np.mean(y)
                se = math.sqrt(p1 * (1 - p1) / (n * len(x)) + p2 * (1 - p2) / (n * len(y)))
                print(f"  {key[:-4]:8s} {size:>3} {NAMES[a]:12s} − entropy: {100 * gap:+.1f} "
                      f"[±{100 * hw:.1f}] separated={'yes' if sep else 'no'}  pooled ±{196 * se:.1f}  "
                      f"(seeds {len(x)} vs {len(y)}, {n} targets each)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
