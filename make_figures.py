"""Figures from the results registry (writes results/figures/).

acc_vs_compute: accuracy against forward compute per byte (flops.py) for the D=128 math models, one line per patch
rule across budgets; points are seed means, bars span the seeds.

    uv run make_figures.py
"""
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flops
import registry

OUT = os.path.join(registry.ROOT, "results", "figures")


def main():
    per_byte, per_patch, patcher = flops.harness()
    ref = 0.25 * per_patch + per_byte + patcher                      # BLT entropy at 25%: 100%
    rs = [r for r in registry.load() if r.get("D") == 128 and r.get("corpus", "math") == "math"
          and r.get("experiment") in ("math_tight_budget", "math_larger") and not r.get("smoke")]
    g = defaultdict(list)
    for r in rs:
        g[r["rule"]].append(r)
    lines = {"BLT entropy": ["entropy10", "entropy15", "entropy20", "entropy"],
             "dependence (no labels)": ["dep10", "dep15", "dep20"],
             "hand-written result boundaries": ["syntax+entropy10", "syntax+entropy15", "syntax+entropy20"],
             "word starts + math syntax": ["words+syntax"]}
    lookup = {"dependence (no labels)"}                               # a table lookup: no entropy model needed
    style = {"BLT entropy": ("#888888", "o"), "dependence (no labels)": ("#1f77b4", "s"),
             "hand-written result boundaries": ("#d62728", "^"), "word starts + math syntax": ("#ff7f0e", "D")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, metric, title in zip(axes, ("final_acc", "computed_acc"), ("Final answers (GSM8K)", "Computed results (in-line)")):
        for label, names in lines.items():
            xs, ys, lo, hi = [], [], [], []
            for n in names:
                if n not in g:
                    continue
                rate = sum(r["rate"] for r in g[n]) / len(g[n])
                cost = rate * per_patch + per_byte + (0 if label in lookup else patcher)
                v = [100 * r[metric] for r in g[n]]
                xs.append(100 * cost / ref); ys.append(sum(v) / len(v)); lo.append(ys[-1] - min(v)); hi.append(max(v) - ys[-1])
            c, m = style[label]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=m, capsize=3, lw=1.5, label=label,
                        ls="none" if len(xs) == 1 else "-")
        ax.set_title(title); ax.set_xlabel("forward compute per byte (% of BLT entropy at 25%)")
        ax.set_ylabel("exact match (%)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("D=128 math models trained at 10-25% patch budgets (points: seed means; bars: seed range)", fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"acc_vs_compute.{ext}"), dpi=150)
    print("wrote results/figures/acc_vs_compute.{svg,png}")


if __name__ == "__main__":
    main()
