"""Compare the new rules with the pilot's across sizes (local; second paper). Read-only; scale/analyze.py (the
published paper's tables) is not touched.

    .venv-scale/bin/python -m scale.compare_new [EXTRA_RESULTS_DIR ...]

Reads results/scale/runs/*/result.json plus any extra directories holding runs/*/result.json (e.g. the pod's
downloaded results_all/). Clipped runs (cfg["clip"] set; directory suffix _clip) are kept apart from unclipped ones.
Prints, per metric, every rule x size as mean [per-seed]; then gaps against entropy10 and dep10 at each size (Welch
95% interval over seeds, and whether the seeds separate); then the clipping check at 50M: clipped vs unclipped
entropy10 and dep10 (if clipping is neutral, the pilot's unclipped rules are a fair reference for clipped runs).
Runs whose training failed (bits per byte more than 1 bit above the median of their size) are excluded and listed.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from scale.analyze import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = ("1m", "12m", "50m", "150m")
RULES = ("entropy10", "dep10", "syntax+entropy10", "nov10", "novc10", "novc13", "typed14", "final+typed14")
NAMES = {"entropy10": "entropy", "dep10": "dependence", "syntax+entropy10": "hand-written", "nov10": "novelty", "novc10": "novelty (corpus-fitted, 10%)",
         "novc13": "novelty (corpus-fitted, 13%)",
         "typed14": "typed", "final+typed14": "typed + answer cue"}
METRICS = (("computed_acc", "computed results", 100), ("final_acc", "final answers", 100),
           ("boxed_acc", "MATH boxed", 100), ("bpb", "bits per byte", 1), ("rate", "eval patch rate", 100))


def load(dirs):
    runs = defaultdict(dict)                      # (size, rule, clipped) -> {seed: result}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "runs", "*", "result.json")) + glob.glob(os.path.join(d, "**", "runs", "*", "result.json"), recursive=True):
            o = json.load(open(f))
            if o.get("toy"):
                continue
            runs[(o["size"], o["arm"], bool((o.get("cfg") or {}).get("clip")))][o["seed"]] = o
    bad = []
    for size in SIZES:
        b = [o["bpb"] for (s, _, _), v in runs.items() if s == size for o in v.values()]
        if not b:
            continue
        med = float(np.median(b))
        for k, v in runs.items():
            if k[0] == size:
                for seed in [sd for sd, o in v.items() if o["bpb"] > med + 1]:
                    bad.append((k, seed, v[seed]["bpb"])); del v[seed]
    return runs, bad


def label(rule, clip):
    return NAMES.get(rule, rule) + (" (clipped)" if clip else "")


def main(extra):
    runs, bad = load([os.path.join(ROOT, "results", "scale"), *extra])
    rows = sorted({(r, c) for (_, r, c) in runs}, key=lambda rc: (RULES.index(rc[0]) if rc[0] in RULES else 99, rc[1]))
    for key, title, mult in METRICS:
        print(f"\n== {title}")
        print(f"  {'rule':28s}" + "".join(f"{s:>26s}" for s in SIZES))
        for rule, clip in rows:
            cells = []
            for s in SIZES:
                v = runs.get((s, rule, clip), {})
                xs = [mult * v[sd][key] for sd in sorted(v) if v[sd].get(key) is not None]
                cells.append(f"{np.mean(xs):6.1f} [{', '.join(f'{x:.1f}' for x in xs)}]" if xs and mult == 100 else
                             f"{np.mean(xs):6.3f} [{', '.join(f'{x:.3f}' for x in xs)}]" if xs else "-")
            print(f"  {label(rule, clip):28s}" + "".join(f"{c:>26s}" for c in cells))
    print("\n== gaps (points; Welch 95% over seeds; 'sep' = every seed above / below every seed of the reference)")
    for key, title, _ in METRICS[:2]:
        for s in SIZES:
            for ref in ("entropy10", "dep10"):
                for rule, clip in rows:
                    if rule in ("entropy10", "dep10", "syntax+entropy10") and not clip:
                        continue
                    a, b = runs.get((s, rule, clip), {}), runs.get((s, ref, False), {})
                    x = [100 * o[key] for o in a.values()]; y = [100 * o[key] for o in b.values()]
                    if len(x) < 2 or len(y) < 2:
                        continue
                    g, hw = welch(y, x)                       # scale.analyze.welch: (mean(x) - mean(y), 95% half-width)
                    lo, hi = g - hw, g + hw
                    sep = "sep+" if min(x) > max(y) else "sep-" if max(x) < min(y) else ""
                    print(f"  {title:17s} {s:4s} {label(rule, clip):28s} - {NAMES[ref]:11s} {g:+6.1f} [{lo:+.1f}, {hi:+.1f}] {sep}")
    print("\n== clipping check at 50M (clipped minus unclipped pilot, same rule; neutral if near 0)")
    for rule in ("entropy10", "dep10", "nov10", "typed14", "final+typed14"):
        a, b = runs.get(("50m", rule, True), {}), runs.get(("50m", rule, False), {})
        if a and b:
            print(f"  {NAMES[rule]:22s} " + "  ".join(
                f"{t}: {100 * (np.mean([o[k] for o in a.values()]) - np.mean([o[k] for o in b.values()])):+5.1f}"
                for k, t, _ in METRICS[:2]) + f"  bpb: {np.mean([o['bpb'] for o in a.values()]) - np.mean([o['bpb'] for o in b.values()]):+.3f}"
                  f"  (clipped n={len(a)}, unclipped n={len(b)})")
    if bad:
        print("\nexcluded (training failed): " + ", ".join(f"{k[0]} {label(k[1], k[2])} s{sd} (bpb {b:.2f})" for k, sd, b in bad))


if __name__ == "__main__":
    main(sys.argv[1:])
