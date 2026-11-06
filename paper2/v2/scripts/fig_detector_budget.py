"""Figure: how many starts the detector should force (R19, R20). BLT-1B at 10%, change over entropy in exact match on
computed values (x) and on final answers (y), per layout; R20's fresh problems with 95% intervals.
    ~/anaconda3/bin/python paper2v2/fig_detector_budget.py      -> paper2v2/figures/fig_detector_budget.png
Palette: the dataviz skill's default categorical slots 1-3 (validated; green is below 3:1 contrast, so every point is
also direct-labelled and every series has its own marker shape)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r19 = json.load(open(f"{ROOT}/results/blt_forced_share.json"))["diff"]
r20 = json.load(open(f"{ROOT}/results/blt_matched_share.json"))["diff"]
ink, muted, grid, surf = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
FAM = [("detector's top q% forced, rest entropy (R19)", [("D0_5", "0.5%"), ("D1", "1%"), ("D1_5", "1.5%"), ("D2", "2%"), ("D3", "3%")], "#2a78d6", "o"),
       ("one ranking: entropy + w x detector (R19)", [("F0_5", "w 0.5"), ("F1", "w 1"), ("F2", "w 2"), ("F4", "w 4")], "#eb6834", "s")]
FRESH = ("fresh problems, 95% intervals (R20)", [("M", ""), ("D3", ""), ("F2", "")], "#1baf7a", "D")
OFF = {"prose": {"D0_5": (-22, -3), "D1": (-14, 6), "D1_5": (4, 6), "D2": (6, -3), "D3": (6, -3), "F0_5": (4, -11), "F1": (-8, 7),
                 "F2": (-6, 8), "F4": (-2, 7)},
       "orig": {"D0_5": (6, -9), "D1": (-10, 7), "D1_5": (6, -12), "D2": (-4, 7), "D3": (8, -12), "F0_5": (6, -9), "F1": (6, -10),
                "F2": (-26, -12), "F4": (10, -12)}}
FRESH_LAB = {"prose": {"F2": ("fresh, w 2", (1.0, -2.7)), "M": ("fresh, share of numbers", (1.0, -4.1)), "D3": ("fresh, 3%", (1.0, -5.5))},
             "orig": {"F2": ("fresh, w 2", (29.0, 7.6)), "M": ("fresh: 3% and share of numbers tie", (27.0, 3.2)), "D3": None}}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=surf)
for ax, (f, title) in zip(axes, (("prose", "GSM8K as natural prose"), ("orig", "GSM8K, equation-style originals"))):
    ax.set_facecolor(surf)
    ax.axhline(0, color=muted, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.plot([0], [0], marker="o", ms=8, color=ink, mec=surf, mew=1.5, zorder=4)
    ax.annotate("entropy (BLT's rule)", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=7.5, color=muted)
    for name, pts, col, mk in FAM:
        xs = [100 * r19[f"{f}|{k}-E|exact"]["value"] for k, _ in pts]; ys = [100 * r19[f"{f}|{k}-E|fin"]["value"] for k, _ in pts]
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=7, mec=surf, mew=1.5, zorder=3)
        for (k, lab), x, y in zip(pts, xs, ys):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=OFF[f][k], fontsize=7.5, color=muted)
    name, pts, col, mk = FRESH
    for k, lab in pts:
        dx, dy = r20[f"{f}|{k}-E|exact"], r20[f"{f}|{k}-E|fin"]
        x, y = 100 * dx["value"], 100 * dy["value"]
        ax.errorbar(x, y, xerr=[[x - 100 * dx["ci"][0]], [100 * dx["ci"][1] - x]], yerr=[[y - 100 * dy["ci"][0]], [100 * dy["ci"][1] - y]],
                    fmt=mk, ms=7, color=col, mec=surf, mew=1.5, elinewidth=1.2, capsize=0, zorder=5)
        if FRESH_LAB[f][k]:
            t, xy = FRESH_LAB[f][k]
            ax.annotate(t, (x, y), xytext=xy, textcoords="data", fontsize=7.5, color=muted, va="center",
                        arrowprops=dict(arrowstyle="-", color=muted, lw=0.7, shrinkA=2, shrinkB=5))
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_xlabel("change in exact match on computed values (points)", fontsize=9, color=muted)
    ax.set_ylabel("change in final answers (points)", fontsize=9, color=muted)
    ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.text(0.02, 0.97, "better: up and to the right", transform=ax.transAxes, fontsize=8, color=muted, ha="left", va="top")
axes[0].set_xlim(-1.5, 21); axes[0].set_ylim(-7.2, 3.6)
axes[1].set_xlim(-3, 62); axes[1].set_ylim(-2, 19)
handles = [plt.Line2D([], [], color=c, marker=m, lw=2 if i < 2 else 0, ms=7, label=n) for i, (n, _, c, m) in enumerate(FAM + [FRESH])]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=ink)
fig.suptitle("BLT-1B at 10%: forcing the detector's starts trades final answers for computed values in prose; one fused ranking does not",
             x=0.01, ha="left", fontsize=10.5, color=ink)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(f"{ROOT}/figures/fig_detector_budget.png", dpi=160, facecolor=surf)
print("saved paper2v2/figures/fig_detector_budget.png")
