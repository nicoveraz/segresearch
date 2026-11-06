"""Exploratory, post hoc (2026-10-29, after the prose-rewrite rows had been used for the verdicts set before that run):
BLT-1B entropy coverage and exact match on computed values, split by the cue just before each value ('= ', '$', plain
space); paper 1 v2, section 5. (Docstring rewritten for the public copy.)

    python paper/v2/scripts/r9_cue_split.py      -> paper/v2/results/blt_prose_cue_split.json
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cue(text, s):
    pre = text[:s]
    if pre.endswith("= ") or pre.endswith("="):
        return "= "
    if pre.endswith("$"):
        return "$"
    return "space" if pre.endswith(" ") else "other"


def main():
    rows = json.load(open(os.path.join(HERE, "results", "blt_prose.json")))["rows"]
    recs = {r["idx"]: r for r in map(json.loads, open(os.path.join(HERE, "data", "prose_rewrites", "test.jsonl")))}
    agg = collections.defaultdict(collections.Counter)
    for x in rows:
        if x["kind"] != "res" or x["layout"] not in ("entropy", "with"):
            continue
        r, f = recs[x["problem"]], x["format"]
        a = agg[(f, cue(r[f], r[f + "_spans"][x["j"]]))]
        if x["layout"] == "entropy":
            a["n"] += 1; a["covered"] += bool(x["covered"]); a["exact_entropy"] += x["ok"]
        else:
            a["exact_with"] += x["ok"]
    out = {f"{f}|{c}": {"n": a["n"], "entropy_covered": a["covered"] / a["n"], "exact_entropy": a["exact_entropy"] / a["n"],
                        "exact_with": a["exact_with"] / a["n"]} for (f, c), a in sorted(agg.items())}
    json.dump(out, open(os.path.join(HERE, "results", "blt_prose_cue_split.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:12s} n {v['n']:4d} covered {v['entropy_covered']:.3f} exact {v['exact_entropy']:.3f} -> with {v['exact_with']:.3f}")


if __name__ == "__main__":
    main()
