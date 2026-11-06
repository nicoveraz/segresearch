"""Rebuild Table 3 (paper/tables/table6_scaling.md) and Figure 2 (paper/figures/scaling.pdf) of paper 1 v2 from the runs
without gradient clipping; no model is run.

    python paper/v2/scripts/scaling_unclipped.py

Since the merge of arXiv:2610.22100's experiments (commit 7ba188e), results/scale/runs/ also holds 50M re-runs of
entropy10 and dep10 with gradient clipping (cfg["clip"] = 1.0, folders ending in _clip) and runs of other rules, some on
a Tesla T4. scale/analyze.load() keys runs by size, rule and seed only, so at commit 1985984 a clipped run can replace the
unclipped run with the same key (which one wins depends on directory order), and make_tables.py and make_figures.py then
mix the two recipes in Table 3 and Figure 2. This script gives scale.analyze a loader that keeps only the runs of Table 3
(the rules of analyze.ARMS and analyze.EXTRA, without gradient clipping: 28 runs, all on the A40), then calls
make_tables.scale_table() and make_figures.scaling() unchanged; they write the table, the figure and, as make_figures.py
always does, results/figures/scaling.{svg,png}. tests/test_paper_numbers.py checks that the table is the published one,
byte for byte, and that the figure reads the same runs.
"""
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from scale import analyze  # noqa: E402


def load_unclipped(d=analyze.RESULTS):
    """As scale.analyze.load(d), {(size, rule): {seed: result}}, but only Table 3's runs: its rules, no gradient clipping."""
    runs = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "runs", "*", "result.json"))):
        o = json.load(open(f))
        if o["cfg"].get("clip") or o["arm"] not in analyze.ARMS + analyze.EXTRA:
            continue
        assert o["seed"] not in runs[(o["size"], o["arm"])], f"two runs for {o['size']} {o['arm']} seed {o['seed']}: {f}"
        runs[(o["size"], o["arm"])][o["seed"]] = o
    return runs


def main():
    analyze.load = load_unclipped          # analyze.runs_ok(), which both builders call, now reads Table 3's runs only
    import make_figures
    import make_tables
    make_tables.scale_table()
    make_figures.scaling()


if __name__ == "__main__":
    main()
