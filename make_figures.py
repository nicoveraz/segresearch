"""Figures from the results registry (writes results/figures/).

acc_vs_compute: accuracy against forward compute per byte (flops.py) for the D=128 math models, one line per patch
rule across budgets; points are seed means, bars span the seeds.

    uv run make_figures.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flops
import registry

OUT = os.path.join(registry.ROOT, "results", "figures")


def blt_compute():
    """BLT-1B: computed-result and final-answer exact match against forward compute per byte, per layout family
    (train-fitted thresholds, issue #26 reruns; default layout at ~28%). Dependence alone is a lookup and needs no
    entropy model; layouts that use entropy pay for it."""
    per_byte, per_patch, patcher = flops.blt1b()
    rows = [r for r in registry.load() if r.get("experiment") == "blt_budget" and r.get("thresh") == "train" and r.get("rerun") == 26]
    by = {r["layout"]: r for r in rows}
    if "default" not in by:
        return
    ref = by["default"]["rate"] * per_patch + per_byte + patcher
    fams = {"BLT entropy": ("entropy", "#888888", "o", True), "boundary after '= ' (hand-written)": ("results", "#d62728", "^", True),
            "entropy + dependence (no labels)": ("entdep", "#1f77b4", "s", True), "dependence alone (lookup, no entropy model)": ("dep", "#2ca02c", "D", False)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, metric, title in zip(axes, ("results_exact", "final_exact"), ("In-line computed results (GSM8K)", "Final answers (GSM8K)")):
        d = by["default"]
        ax.scatter([100], [100 * d[metric]], color="black", marker="*", s=120, zorder=5, label="BLT-1B default (~28%)")
        for label, (k, c, m, ent) in fams.items():
            pts = sorted((by[f"{k}@{R}"]["rate"], by[f"{k}@{R}"][metric]) for R in (10, 15) if f"{k}@{R}" in by)
            xs = [100 * (r * per_patch + per_byte + (patcher if ent else 0)) / ref for r, _ in pts]
            ax.plot(xs, [100 * v for _, v in pts], color=c, marker=m, lw=1.5, label=label)
        ax.set_title(title); ax.set_xlabel("forward compute per byte (% of BLT-1B default)"); ax.set_ylabel("exact match (%)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("BLT-1B, patch layouts changed at inference at 10% and 15% of bytes (294 GSM8K test problems, 796 in-line results)", fontsize=10)
    fig.tight_layout(); os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"blt1b_acc_vs_compute.{ext}"), dpi=150)
    print("wrote results/figures/blt1b_acc_vs_compute.{svg,png}")


def fig1():
    """Where the patches start in one GSM8K solution: BLT-1B entropy vs entropy + dependence, both with 10% of bytes
    starting a patch (equal counts for this problem). Bars mark patch starts; shading marks in-line computed results."""
    path = os.path.join(registry.ROOT, "results", "fig1_layouts.json")
    if not os.path.exists(path):
        return
    d = json.load(open(path)); text = d["text"]
    lines, off = [], 0
    for ln in text.split("\n"):
        lines.append((off, ln)); off += len(ln) + 1
    lines = [(o, ln) for o, ln in lines if not ln.startswith("John buys")]            # show the worked solution
    res = [(a, b) for a, b in d["results"]]
    W = max(len(ln) for _, ln in lines)
    fig, axes = plt.subplots(2, 1, figsize=(W * 0.105 + 0.6, 2 * (len(lines) * 0.34 + 0.6)))
    for ax, key, title, col in zip(axes, ("entropy@10", "entdep@10"),
                                   ("BLT entropy: 0 of 6 computed results start a patch", "Entropy + dependence (no labels): 6 of 6"),
                                   ("#888888", "#1f77b4")):
        starts = set(d[key])
        for r, (o, ln) in enumerate(lines):
            y = len(lines) - r
            for a, b in res:
                if o <= a < o + len(ln):
                    ax.add_patch(plt.Rectangle((a - o - 0.5, y - 0.42), b - a, 0.84, color="#ffe9a8", zorder=0))
            for c, ch in enumerate(ln):
                ax.text(c, y, ch, family="monospace", fontsize=10, ha="center", va="center")
                if o + c in starts:
                    ax.plot([c - 0.5, c - 0.5], [y - 0.42, y + 0.42], color=col, lw=2)
        ax.set_xlim(-1, W); ax.set_ylim(0.3, len(lines) + 0.7); ax.axis("off")
        ax.set_title(title, fontsize=10, loc="left")
    fig.suptitle("BLT-1B patch starts at a 10% budget (bars); in-line computed results shaded", fontsize=10)
    fig.tight_layout(); os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_patch_starts.{ext}"), dpi=150)
    print("wrote results/figures/fig1_patch_starts.{svg,png}")


def main():
    fig1()
    blt_compute()
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
