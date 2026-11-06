"""R17 figure: entropy and probe score around the first digit of a number, by kind (results/blt_profiles.json).

    python plot_profiles.py      -> results/blt_profiles.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
KINDS = [("computed", "computed value", "#2a78d6"), ("other_new", "other new number", "#eb6834"),
         ("copy", "copy (seen earlier)", "#1baf7a"), ("word", "word (reference)", "#eda100")]
SIGS = [("entropy", "Patcher entropy (bits)"), ("probe", "Detector score (log-odds)")]
FMTS = [("prose", "Prose rewrites"), ("orig", "Equation-style originals")]

r = json.load(open(os.path.join(HERE, "results", "blt_profiles.json")))
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID})
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
for i, (sig, ylab) in enumerate(SIGS):
    for j, (fmt, title) in enumerate(FMTS):
        ax = axes[i, j]; ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ends = []
        for k, label, col in KINDS:
            rows = [x for x in r["profiles"][fmt][f"{k}|{sig}"] if "mean" in x]
            xs = [x["offset"] for x in rows]; ys = [x["mean"] for x in rows]
            ax.fill_between(xs, [x["ci"][0] for x in rows], [x["ci"][1] for x in rows], color=col, alpha=0.15, linewidth=0)
            ax.plot(xs, ys, color=col, linewidth=2, solid_capstyle="round")
            ends.append([ys[-1], xs[-1], label])
        lo, hi = ax.get_ylim(); gap = 0.075 * (hi - lo)                # end labels pushed apart, no collisions
        ends.sort()
        for a in range(1, len(ends)):
            ends[a][0] = max(ends[a][0], ends[a - 1][0] + gap)
        for y, x, label in ends:
            ax.annotate(label, (x + 0.25, y), va="center", fontsize=8.5, color=INK2)
        ax.set_xlim(-8.3, 7.4); ax.set_xticks(range(-8, 5, 2))
        if i == 0:
            ax.set_title(title, fontsize=11, color=INK, loc="left")
            ax.text(0.15, 0.97, "first digit", transform=ax.get_xaxis_transform(), fontsize=8.5, color=INK2, va="top")
        if i == 1:
            ax.set_xlabel("bytes from the number's first digit")
        if j == 0:
            ax.set_ylabel(ylab)
fig.legend([plt.Line2D([], [], color=c, linewidth=2) for _, _, c in KINDS], [l for _, l, _ in KINDS],
           loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("BLT-1B's patcher around a number: entropy separates results from copies only after the first digit",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
fig.savefig(os.path.join(HERE, "results", "blt_profiles.png"), dpi=150, facecolor=SURFACE)
print("saved results/blt_profiles.png")
