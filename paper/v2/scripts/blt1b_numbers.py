"""BLT-1B numbers new in paper 1 v2 (sections 4, 5 and 11), from the per-target file of Table 1; no model is run.

    python paper/v2/scripts/blt1b_numbers.py          -> paper/v2/results/blt1b_numbers.json

Input: results/items/blt_budget_train_w512.json (realblt_budget.py, train-fitted thresholds, 512-byte window restored:
per layout, exact match of each of the 796 in-line computed results and 294 final answers, and whether a patch starts
at each result's first byte) and the GSM8K test set (~/.cache/segresearch-gsm8k/test.jsonl, as realtext.py downloads
it), from which the result spans are rebuilt exactly as realblt_budget.problem() builds them.

1. Split by cue (an analysis after the fact): a result written '= $N' starts with the '$' (realblt_budget.problem's span
   starts at the byte after '= '), one written '= N' with its first digit. Per layout: how many results of each kind, the
   share whose first byte starts a patch, and exact match.
2. Paired comparisons that the results/stats.md tables do not list: per pair of layouts, the difference in exact match
   with a 95% interval that resamples whole problems and an exact McNemar test (stats.boot, stats.mcnemar).
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from stats import boot, mcnemar  # noqa: E402

ITEMS = os.path.join(ROOT, "results", "items", "blt_budget_train_w512.json")
GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
OUT = os.path.join(ROOT, "paper", "v2", "results", "blt1b_numbers.json")
PAIRS = (("dep@15", "default"), ("dep@10", "default"), ("entdep@15", "default"))


def result_kinds(n_problems=300, max_tokens=1000):
    """('$' or 'digit', problem index) for every in-line result, in realblt_budget's order."""
    kinds, pids, used = [], [], 0
    for i, line in enumerate(open(GSM)):
        if i >= n_problems:
            break
        p = json.loads(line)
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip(); ans = ans.strip()
        b = (p["question"] + "\n" + sol + "\nThe final answer is " + ans).encode()
        if len(b) + 1 > max_tokens:                    # BLT-1B: one token per byte plus BOS
            continue
        for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b):
            kinds.append("$" if b[m.start() + 2:m.start() + 3] == b"$" else "digit")
            pids.append(used)
        used += 1
    return np.array(kinds), pids, used


def main():
    D = json.load(open(ITEMS))["data"]
    kinds, pids, used = result_kinds()
    assert used == 294 and pids == D["res_pid"], "result spans do not match the per-target file"
    res = {"problems": used, "results": len(kinds), "n": {k: int((kinds == k).sum()) for k in ("$", "digit")},
           "by_cue": {}, "pairs": {}}
    for lay in D["layouts"]:
        cov = np.array(D["covered"][lay], bool); ex = np.array(D["layouts"][lay]["res_exact"], bool)
        res["by_cue"][lay] = {k: {"covered": float(cov[kinds == k].mean()), "exact": float(ex[kinds == k].mean())}
                              for k in ("$", "digit")}
    L = D["layouts"]
    fpid = list(range(len(L["default"]["fin_exact"])))
    for a, b in PAIRS:
        for key, pid in (("res_exact", D["res_pid"]), ("fin_exact", fpid)):
            x, y = np.array(L[a][key], bool), np.array(L[b][key], bool)
            lo, hi = boot(x, y, pid); p, nb, nc = mcnemar(x, y)
            res["pairs"][f"{a} - {b}|{key}"] = {"diff": float(x.mean() - y.mean()), "lo": float(lo), "hi": float(hi),
                                                "a": float(x.mean()), "b": float(y.mean()), "mcnemar_p": p,
                                                "a_only": nb, "b_only": nc}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{used} problems, {len(kinds)} results: {res['n']['$']} written '= $N', {res['n']['digit']} written '= N'")
    print("layout      | covered $ / digit | exact $ / digit")
    for lay, v in res["by_cue"].items():
        print(f"{lay:11s} | {100 * v['$']['covered']:5.1f} / {100 * v['digit']['covered']:5.1f} | "
              f"{100 * v['$']['exact']:5.1f} / {100 * v['digit']['exact']:5.1f}")
    for k, v in res["pairs"].items():
        print(f"{k:32s} {100 * v['a']:.1f}% vs {100 * v['b']:.1f}%: {100 * v['diff']:+.1f} [{100 * v['lo']:+.1f}, "
              f"{100 * v['hi']:+.1f}]  McNemar p {v['mcnemar_p']:.2g} ({v['a_only']} / {v['b_only']})")
    print(f"-> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
