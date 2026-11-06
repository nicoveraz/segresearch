"""Where BLT-1B's entropy budget goes, by kind of position (review 2, W2). Reads reducible2's cached per-byte columns
(stripped GSM8K test text, BLT-1B's patcher entropy) and reports, per unit kind, the mean entropy and percentile at
unit starts, the share of starts above the 90th percentile (a 10% budget), and the share of that budget each kind
takes.

    python3 budget_by_kind.py      -> results/patcher_budget_by_kind.json
"""
import json
import os

import numpy as np

COLS = os.path.expanduser("~/.cache/segresearch-scale/ncm_math/reducible2_cols.npz")


def main():
    z = np.load(COLS)
    e, kind = z["ent_small"], z["kind"].astype(str)
    pct = (np.argsort(np.argsort(e)) + 0.5) / len(e)
    thr = float(np.percentile(e, 90))
    top = e >= thr
    res = {"n_bytes": int(len(e)), "mean_entropy_all": float(e.mean()), "threshold_p90": thr, "by_kind": {}}
    for k in ("computed", "copied", "final", "word", ""):
        m = kind == k
        res["by_kind"][k or "no unit start"] = {
            "n": int(m.sum()), "mean_entropy": float(e[m].mean()), "mean_percentile": float(pct[m].mean()),
            "share_above_p90": float(top[m].mean()), "share_of_budget": float((top & m).sum() / top.sum())}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "patcher_budget_by_kind.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
