"""Figure: answers vs bits per byte on BLT-1B at inference, by patching rule and budget (R8 / R8b data).
    ~/anaconda3/bin/python paper2v2/fig_tradeoff.py      -> paper2v2/figures/fig_tradeoff.png
Palette: the dataviz skill's default categorical slots 1-4 (validated; two below 3:1 contrast, so every series is
also direct-labelled by marker shape and legend)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
w = json.load(open(f"{ROOT}/results/blt_waste.json"))["curves"]; h = json.load(open(f"{ROOT}/results/blt_hybrid.json"))["curves"]
C = dict(w); C.update({k: v for k, v in h.items() if "forced" not in k})
fam = [("entropy (BLT's rule)", ["entropy@0.05", "entropy@0.1", "entropy@0.15", "entropy@0.2", "entropy@0.25", "entropy@0.3"], "#2a78d6", "o", (8, -4)),
       ("novelty table", ["table@0.05", "table@0.075", "table@0.1", "table@0.15"], "#eb6834", "s", (8, -4)),
       ("entropy + start after '=' (hand-written)", ["eq@0.075", "eq@0.1", "eq@0.15"], "#1baf7a", "^", (7, -13)),
       ("entropy + table's top 3% (label-free)", ["hybrid(0.03)@0.1", "hybrid(0.03)@0.15"], "#eda100", "D", (6, 7))]
ink, muted, grid, surf = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=surf)
for ax, (metric, title) in zip(axes, (("acc_res", "Computed results"), ("acc_fin", "Final answers"))):
    ax.set_facecolor(surf)
    for name, keys, col, mk, off in fam:
        xs = [C[k]["bpb_all"] for k in keys]; ys = [100 * C[k][metric] for k in keys]
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=8, mec=surf, mew=1.5, zorder=3)
        for k, x, y in zip(keys, xs, ys):
            r = float(k.split("@")[1]) * 100
            if r in (25, 30):                                  # crowded corner: label only the 30% end
                if r == 30:
                    ax.annotate("25-30%", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=7.5, color=muted)
                continue
            ax.annotate(f"{r:g}%", (x, y), textcoords="offset points", xytext=off, fontsize=7.5, color=muted)
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_xlabel("bits per byte, whole text (lower is better)", fontsize=9, color=muted)
    ax.set_ylabel("exact match (%)", fontsize=9, color=muted)
    ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.set_xlim(0.64, 1.18)
    ax.text(0.98, 0.97, "better: up and to the left", transform=ax.transAxes, fontsize=8, color=muted, ha="right", va="top")
handles = [plt.Line2D([], [], color=c, marker=m, lw=2, ms=7, label=n) for n, _, c, m, _ in fam]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=ink)
fig.suptitle("BLT-1B at inference: answers vs bits per byte, by patching rule and budget (labels = patch budget)", x=0.01, ha="left", fontsize=10.5, color=ink)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(f"{ROOT}/figures/fig_tradeoff.png", dpi=160, facecolor=surf)
