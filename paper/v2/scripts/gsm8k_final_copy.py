"""How often is a GSM8K final answer a copy of the last number written before it? (paper 1 v2, sections 3 and 6)

    python paper/v2/scripts/gsm8k_final_copy.py [TEST_JSONL] --out paper/v2/results/gsm8k_final_copy.json

Reads the GSM8K test set (default ~/.cache/segresearch-gsm8k/test.jsonl, the file realtext.py downloads: 1,319
problems, MIT licence). Standard library only; no model. Every solution ends in one line '#### N'; N is the final
answer.

Two views of each solution, as the models in the paper read it:
  raw   the models trained from scratch (realtext._encode, mathexp.py, scale/): the solution as released, calculator
        annotations <<expr=R>> kept, final answer after '\\n#### '.
  blt   BLT-1B (realblt_budget.problem): annotations removed, solution stripped, final answer after
        '\\nThe final answer is '.
Subsets: all 1,319 problems; the first 660 (the evaluation split of the models trained from scratch,
mathexp.N_VAL_GSM); the BLT-1B test problems (the first 300 whose text fits 1,000 tokens, i.e. bytes + 1 <= 1000; 294).

Definitions
  final answer  the text after the solution's only '\\n#### ', stripped of whitespace.
  number        a match of NUM: a run of digits, optionally followed by thousands groups of exactly three digits
                (',ddd') and a decimal part ('.d+'). A sentence-final period is not part of a number. A '-' directly
                before the digits is a sign only when it does not follow a letter, a digit, ')' or '.', so the binary
                minus in '16-3' is not a sign (two final answers are negative: -10 and -3).
  last number   the last match of NUM in the solution before the final-answer line: the number whose end is
                closest to the final answer.
  match         three nested definitions:
                  identical    byte for byte the same string;
                  same_digits  equal once thousands separators are removed from both ('70,000' and '70000'): the
                               definition the paper reports;
                  same_value   numerically equal (also counts '16.00' and '16').
  distances     UTF-8 bytes, over the problems that match (same_digits):
                  tail            between the end of the last number and the line break before the final-answer line
                                  (the paper's "ends a median of 8 bytes before the final-answer line");
                  to_answer       from the end of the last number to the first byte of the final answer
                                  (tail + 6 for '\\n#### ', tail + 21 for '\\nThe final answer is ');
                  start_to_start  from the first byte of the last number to the first byte of the final answer.
                Percentiles are nearest-rank (the smallest value with at least that share at or below it).
  last_result   share whose final answer equals (same_digits) the last computed result: raw view, the R of the last
                calculator annotation <<expr=R>>; blt view, the last number right after '= ' (realblt_budget's
                in-line results, '= $' allowed).
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics

NUM = re.compile(r"(?:(?<![A-Za-z0-9).])-)?\d+(?:,\d{3})*(?:\.\d+)?")
ANNOT = re.compile(r"<<[^>]*>>")
BLT_RESULT = re.compile(r"= \$?(\d[\d,]*(?:\.\d+)?)")          # realblt_budget.problem's in-line results
MARK = "\n#### "
BLT_MARK = "\nThe final answer is "
DEFAULT = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")


def nb(s):
    return len(s.encode())


def digits(s):
    return s.replace(",", "")


def nearest_rank(xs, p):
    xs = sorted(xs)
    return xs[max(0, math.ceil(p * len(xs)) - 1)]


def view(p, name):
    """(solution text before the final-answer line, final answer, bytes of the final-answer marker, last result)."""
    body, final = p["answer"].rsplit(MARK, 1)
    final = final.strip()
    if name == "raw":
        res = [m.group(0)[2:-2].rpartition("=")[2] for m in ANNOT.finditer(body)]
        return body, final, nb(MARK), ([r for r in res if r] or [None])[-1]
    sol = ANNOT.sub("", p["answer"].rsplit("#### ", 1)[0]).strip()
    res = [m.group(1) for m in BLT_RESULT.finditer(sol)]
    return sol, final, nb(BLT_MARK), (res or [None])[-1]


def blt_fits(p):
    sol, ans = p["answer"].rsplit("#### ", 1)
    text = p["question"] + "\n" + ANNOT.sub("", sol).strip() + BLT_MARK + ans.strip()
    return nb(text) + 1 <= 1000                      # BLT-1B tokens = bytes + BOS; realblt_budget.MAX_TOKENS


def one(p, name):
    body, final, mark, last_res = view(p, name)
    out = {"final": final, "last": None, "identical": False, "same_digits": False, "same_value": False,
           "last_result": last_res is not None and digits(last_res) == digits(final)}
    nums = list(NUM.finditer(body))
    if nums:
        s, tail = nums[-1].group(), body[nums[-1].end():]
        out.update(last=s, identical=s == final, same_digits=digits(s) == digits(final),
                   same_value=float(digits(s)) == float(digits(final)), tail=nb(tail), to_answer=nb(tail) + mark,
                   start_to_start=nb(s) + nb(tail) + mark)
    return out


def summarize(rows, name):
    per = [one(p, name) for p in rows]
    n = len(per)
    s = {"n": n}
    for k in ("identical", "same_digits", "same_value", "last_result"):
        s[k] = sum(p[k] for p in per)
        s[k + "_share"] = s[k] / n
    hit = [p for p in per if p["same_digits"]]
    for k in ("tail", "to_answer", "start_to_start"):
        g = [p[k] for p in hit]
        s[k] = {"median": statistics.median(g), "p75": nearest_rank(g, .75), "p90": nearest_rank(g, .9),
                "min": min(g), "max": max(g)}
    s["no_match_examples"] = [[p["last"], p["final"]] for p in per if not p["same_value"]][:10]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--out")
    a = ap.parse_args()
    raw = open(a.path, "rb").read()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    assert len(rows) == 1319 and all(r["answer"].count(MARK) == 1 for r in rows), "expected the GSM8K test set"
    blt = [p for p in rows[:300] if blt_fits(p)]
    res = {"source": "GSM8K test set (openai/grade-school-math, test.jsonl), 1,319 problems",
           "sha256": hashlib.sha256(raw).hexdigest(), "regex": NUM.pattern}
    for sub, rs, views in ((("all", rows, ("raw", "blt")), ("first660", rows[:660], ("raw",)), ("blt294", blt, ("blt",)))):
        for v in views:
            res[f"{sub}_{v}"] = s = summarize(rs, v)
            print(f"{sub:8s} {v:3s} n={s['n']:4d}: same digits {s['same_digits']} ({100 * s['same_digits_share']:.1f}%), "
                  f"same value {100 * s['same_value_share']:.1f}%, identical {100 * s['identical_share']:.1f}%, "
                  f"= last computed result {100 * s['last_result_share']:.1f}%; tail median {s['tail']['median']} "
                  f"(p75 {s['tail']['p75']}, p90 {s['tail']['p90']}); to answer median {s['to_answer']['median']}; "
                  f"start to start median {s['start_to_start']['median']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
