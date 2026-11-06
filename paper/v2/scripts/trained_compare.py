"""Seed-level comparisons for the small models trained from scratch (paper 1 v2, sections 6, 7 and 12), from the results
registry.

    python paper/v2/scripts/trained_compare.py        -> paper/v2/results/trained_compare.json

Version 2 gives these comparisons with their intervals where version 1 stated a tie, or a difference, without one:
  section 6   Table 2's models (D = 128, 1.1M parameters, 3 seeds): jump against entropy at 10%; each rule from 10% to
              20% of bytes; bits per byte of the hand-written rule against entropy, jump and dependence, and of dependence
              against entropy; at the 25% budget, word starts plus math syntax against entropy;
  section 7   dependence alone against entropy plus dependence (2 seeds) in Table 2's models;
  section 12  per-digit and operand-aligned patches against their matched random controls (5 seeds each; notes.md:
              rand07 is the twin of +digits, rand22 the twin of +ops), and the learned patcher (ndep10) against the table
              (dep10).

Rows come from registry.load() through stats.mlx_rows() (MLX training runs; smoke tests and superseded runs dropped).
Each comparison A - B gives the difference of the seed means (points for accuracies, bits for bits per byte), the Welch
95% interval over seeds that stats.py uses for results/stats.md (Welch-Satterthwaite degrees of freedom rounded down, t
quantiles from stats.T975), and whether every seed of one rule is above every seed of the other. stats.compare() also
computes a pooled interval, which does not apply to bits per byte, so the Welch interval is computed here with the same
formula; for accuracies the script checks that it equals stats.compare()'s.
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import stats  # noqa: E402

OUT = os.path.join(ROOT, "paper", "v2", "results", "trained_compare.json")
TB, LG, NS = "math_tight_budget", "math_larger", "math_new_strategies"
COMPARISONS = [   # (section, experiment, rule A, rule B, metric): A - B
    ("6", TB, "jump10", "entropy10", "final_acc"),
    ("6", TB, "jump10", "entropy10", "computed_acc"),
    ("6", TB, "entropy20", "entropy10", "final_acc"),
    ("6", TB, "dep20", "dep10", "final_acc"),
    ("6", TB, "syntax+entropy20", "syntax+entropy10", "final_acc"),
    ("6", TB, "entropy20", "entropy10", "computed_acc"),
    ("6", TB, "dep20", "dep10", "computed_acc"),
    ("6", TB, "syntax+entropy20", "syntax+entropy10", "computed_acc"),
    ("6", TB, "syntax+entropy10", "entropy10", "bpb"),
    ("6", TB, "syntax+entropy10", "jump10", "bpb"),
    ("6", TB, "syntax+entropy15", "entropy15", "bpb"),
    ("6", TB, "syntax+entropy20", "entropy20", "bpb"),
    ("6", TB, "syntax+entropy10", "dep10", "bpb"),
    ("6", TB, "syntax+entropy15", "dep15", "bpb"),
    ("6", TB, "syntax+entropy20", "dep20", "bpb"),
    ("6", TB, "dep10", "entropy10", "bpb"),
    ("6", TB, "dep15", "entropy15", "bpb"),
    ("6", TB, "dep20", "entropy20", "bpb"),
    ("6", LG, "words+syntax", "entropy", "computed_acc"),
    ("7", TB, "dep10", "entdep10", "computed_acc"),
    ("7", TB, "dep10", "entdep10", "final_acc"),
    ("12", NS, "words+syntax+digits", "words+syntax+rand07", "computed_acc"),
    ("12", NS, "words+syntax+ops", "words+syntax+rand22", "final_acc"),
    ("12", TB, "ndep10", "dep10", "computed_acc"),
    ("12", TB, "ndep10", "dep10", "final_acc"),
]


def seeds(rows, exp, rule, metric):
    """One value per seed: points for accuracies, bits per byte as stored."""
    return [r[metric] if metric == "bpb" else 100 * r[metric] for r in rows if r.get("experiment") == exp and r["rule"] == rule]


def welch(va, vb):
    """mean(va) - mean(vb) with the Welch 95% interval of stats.compare(): (difference, low, high)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    d = va.mean() - vb.mean()
    sa, sb = va.var(ddof=1) / len(va), vb.var(ddof=1) / len(vb)
    se = math.sqrt(sa + sb)
    df = (sa + sb) ** 2 / (sa ** 2 / (len(va) - 1) + sb ** 2 / (len(vb) - 1)) if se > 0 else 1
    t = stats.T975.get(max(1, min(8, int(df))), 2.0)
    return float(d), float(d - t * se), float(d + t * se)


def compute():
    rows = stats.mlx_rows()
    out = []
    for sec, exp, a, b, m in COMPARISONS:
        va, vb = seeds(rows, exp, a, m), seeds(rows, exp, b, m)
        d, lo, hi = welch(va, vb)
        if m != "bpb":                                   # the same numbers as stats.compare(), which results/stats.md uses
            c = stats.compare(rows, a, b, m, experiment=(exp, None))
            assert abs(100 * c["d"] - d) < 1e-9 and abs(100 * c["welch"][0] - lo) < 1e-9 and abs(100 * c["welch"][1] - hi) < 1e-9
        out.append(dict(section=sec, experiment=exp, a=a, b=b, metric=m, seeds=[len(va), len(vb)], a_values=va, b_values=vb,
                        diff=d, lo=lo, hi=hi, includes_zero=lo < 0 < hi, separated=min(va) > max(vb) or min(vb) > max(va)))
    return out


def main():
    out = compute()
    with open(OUT, "w") as f:
        json.dump({"source": "results/registry via stats.mlx_rows(); Welch 95% interval over seeds as in stats.compare()",
                   "comparisons": out}, f, indent=1)
        f.write("\n")
    for o in out:
        nd = 3 if o["metric"] == "bpb" else 1
        print(f"§{o['section']} {o['experiment']}: {o['a']} - {o['b']} [{o['metric']}] seeds {o['seeds'][0]}/{o['seeds'][1]}: "
              f"{o['diff']:+.{nd}f} ({o['lo']:+.{nd}f} to {o['hi']:+.{nd}f}){' separated' if o['separated'] else ''}")


if __name__ == "__main__":
    main()
