"""BLT-1B results restricted to targets inside the first 512 bytes (writes results/blt_window512.json).

The Hugging Face conversion of BLT-1B we used (itazap/blt-1b-hf) lost BLT's 512-byte sliding window in the entropy
patcher and the local encoder/decoder (huggingface/transformers issue #49185; fix PR #49188 not merged as of
2026-10-07). Before byte 512 the window covers every earlier byte, so those positions are computed exactly as by
the original model; later positions are not. This recomputes the BLT-1B budget and fine-tune comparisons on the
targets that end within the first 512 bytes, from the per-target files in results/items/.

Target positions are rebuilt from the GSM8K test file with the same formatting as realblt_budget.problem() (kept in
sync by hand; the counts are asserted against the saved files).

    uv run blt_window.py
"""
import glob
import json
import os
import re
from math import comb

import numpy as np

import registry

GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
WINDOW = 512
OUT = os.path.join(registry.ROOT, "results", "blt_window512.json")


def positions():
    """End token index (token t = byte t-1) of every in-line result and final answer, in the saved order."""
    res, fin, long_ = [], [], 0
    for p in [json.loads(l) for l in open(GSM)][:300]:
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip(); ans = ans.strip()
        b = (p["question"] + "\n" + sol + "\nThe final answer is " + ans).encode()
        if len(b) + 1 > 1000:                                            # realblt_budget.MAX_TOKENS
            continue
        res += [m.end(1) + 1 for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b)]
        fin.append(len(b) + 1)
        long_ += len(b) > WINDOW
    return np.array(res), np.array(fin), long_


def mcnemar(a, b):
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum()); n = n10 + n01
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0


def main():
    res, fin, n_long = positions()
    budget = json.load(open(os.path.join(registry.ROOT, "results", "items", "blt_budget_train.json")))["data"]["layouts"]
    assert len(res) == len(budget["default"]["res_exact"]) and len(fin) == len(budget["default"]["fin_exact"])
    inside = {"res": res <= WINDOW + 1, "fin": fin <= WINDOW + 1}      # target ends within the first 512 bytes
    out = {"problems": len(fin), "problems_longer": int(n_long),
           "inside": {k: int(v.sum()) for k, v in inside.items()}, "total": {"res": len(res), "fin": len(fin)},
           "acc": {}, "diff": {}, "finetune": {}}
    for lay, d in budget.items():
        out["acc"][lay] = {k: {"all": float(np.mean(d[f"{k}_exact"])), "inside": float(np.mean(np.array(d[f"{k}_exact"])[inside[k]]))}
                           for k in ("res", "fin")}
    for a, b in (("results@15", "entropy@15"), ("results@10", "entropy@10"), ("entdep@15", "entropy@15"), ("entdep@10", "entropy@10")):
        for k in ("res", "fin"):
            x = np.array(budget[a][f"{k}_exact"], int)[inside[k]]; y = np.array(budget[b][f"{k}_exact"], int)[inside[k]]
            out["diff"][f"{a}-{b}:{k}"] = {"points": float(100 * (x.mean() - y.mean())), "p": mcnemar(x, y)}
    lay = {"entropy": "entropy@10", "results": "results@10", "entdep": "entdep@10"}
    for f in sorted(glob.glob(os.path.join(registry.ROOT, "results", "items", "blt_finetune_*.json"))):
        it = json.load(open(f)); r = it["trained_rule"]
        x = np.array(it["data"]["layouts"][lay[r]]["res_exact"], int)
        out["finetune"].setdefault(r, []).append({"all": float(x.mean()), "inside": float(x[inside["res"]].mean())})
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{n_long} of {len(fin)} problems longer than {WINDOW} bytes; inside: {out['inside']}")
    for k, v in out["diff"].items():
        print(f"  {k:28s} {v['points']:+.1f} points  p = {v['p']:.1e}")
    for r, v in out["finetune"].items():
        print(f"  fine-tuned {r:8s} all {100 * np.mean([x['all'] for x in v]):.1f}%  inside {100 * np.mean([x['inside'] for x in v]):.1f}%")


if __name__ == "__main__":
    main()
