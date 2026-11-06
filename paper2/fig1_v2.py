"""Figure 1 of paper 2, version 2: teacher-forced exact match of the models trained from scratch, by size and rule.

    python3 paper2/fig1_v2.py            # -> paper2/figures/fig1_scale_v2.png

Reads results/scale/runs/<size>_<rule>_s<seed>[_clip]/result.json only (1M and 12M unclipped, 50M clipped, as in
Table 1); no model. Lines join the means over seeds; small markers are seeds, hollow for runs on a Kaggle T4 in fp16
and filled for runs on the A40 in bf16 (the `device` field of each result). The failed 1M dependence seed is left out,
as in Table 1 and every interval: a run is dropped when its bits per byte exceed the median over every plotted run of
its size plus 1, the rule of scale/analyze.py's failed(). Colours are
the first five slots of a categorical palette checked for colour-vision deficiency on a white surface.
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper2", "figures", "fig1_scale_v2.png")
SIZES = ["1m", "12m", "50m"]
RULES = [("nov10", "novelty table", "#2a78d6", "o"), ("entropy10", "entropy", "#eb6834", "s"),
         ("syntax+entropy10", "hand-written rule", "#1baf7a", "^"), ("dep10", "dependence", "#eda100", "D"),
         ("novc13", "corpus-fitted table", "#e87ba4", "v")]
INK, MUTED, GRID = "#1f1f1e", "#52514e", "#e4e3df"


def runs():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "results/scale/runs/*/result.json")):
        m = re.fullmatch(r"(\w+?)_(.+)_s(\d)(_clip)?", os.path.basename(os.path.dirname(f)))
        size, rule, clip = m.group(1), m.group(2), bool(m.group(4))
        if size in SIZES and clip == (size == "50m"):
            out.setdefault((size, rule), []).append(json.load(open(f)))
    return out


def ok(rs, size_runs):
    """Drop runs that failed by scale/analyze.py's rule: bits per byte above the median over every run of the size + 1."""
    if not rs:
        return []
    b = sorted(r["bpb"] for r in size_runs)
    med = (b[len(b) // 2] + b[(len(b) - 1) // 2]) / 2
    return [r for r in rs if r["bpb"] <= med + 1]


def main():
    R = runs()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for ax, (key, title) in zip(axes, (("computed_acc", "Computed results"), ("final_acc", "Final answers"))):
        for j, (rule, label, col, mk) in enumerate(RULES):
            xs, ys = [], []
            for i, s in enumerate(SIZES):
                rs = ok(R.get((s, rule), []), [x for (sz, _), v in R.items() if sz == s for x in v])
                if not rs:
                    continue
                v = [100 * r[key] for r in rs]
                xs.append(i); ys.append(sum(v) / len(v))
                off = (j - 2) * 0.045
                for r, y in zip(rs, v):
                    t4 = "T4" in r["device"]
                    ax.plot(i + off, y, mk, ms=4.5, mec=col, mew=1.2, mfc="white" if t4 else col, alpha=0.9, zorder=3)
            ax.plot(xs, ys, "-", color=col, lw=2, zorder=2, label=label)
        ax.set_title(title + " (%)", fontsize=10, color=INK, loc="left")
        ax.set_xticks(range(len(SIZES)), [s.upper() for s in SIZES])
        ax.set_xlim(-0.3, len(SIZES) - 0.7)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    handles = [plt.Line2D([], [], color=col, lw=2, marker=mk, ms=5, mfc=col, mec=col, label=label)
               for _, label, col, mk in RULES]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=5, mec=MUTED, mfc=MUTED, label="A40, bf16"),
                plt.Line2D([], [], ls="", marker="o", ms=5, mec=MUTED, mfc="white", label="T4, fp16")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8.5, handlelength=2.2,
               columnspacing=1.2)
    fig.text(0.5, 0.115, "1M and 12M unclipped; 50M with gradient clipping. Lines: means over seeds; markers: seeds.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT, dpi=200, facecolor="white")
    print("->", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
