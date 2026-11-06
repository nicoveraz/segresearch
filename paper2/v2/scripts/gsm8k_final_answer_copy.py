"""How often a GSM8K test solution's final answer repeats the last number written before its '####' line (CPU,
stdlib only; new in v2).

Definition (primary). The final answer is the text after the last '####', stripped, with commas removed. The last
number before the marker is the last match of  -?\\d[\\d,]*(?:\\.\\d+)?  in the solution text before '####', with
GSM8K's calculator annotations <<...>> left in place (they precede the result they compute); its commas are removed
and a trailing '.' is stripped. The two are compared as strings. Distances are UTF-8 bytes from the end of that number
to the end of its line, and to the '####' marker (a newline always precedes the marker). Quantiles: the median is
statistics.median; p75 and p90 are sorted(x)[int(q * n)].

Variants reported beside it: the unsigned pattern \\d[\\d,]*(?:\\.\\d+)?; numeric equality (so 5.00 = 5); the
annotations removed first. Each over all 1,319 test solutions and over the first 660, the GSM8K split on which the
models trained from scratch are scored (realtext.N_VAL, mathexp.N_VAL_GSM).

    python paper2/v2/scripts/gsm8k_final_answer_copy.py [TEST_JSONL]
        TEST_JSONL defaults to ~/.cache/segresearch-gsm8k/test.jsonl, the file realtext._load('test') downloads from
        openai/grade-school-math (MIT); its sha256 is recorded in the output.
"""
import hashlib
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # paper2/v2
OUT = os.path.join(HERE, "results", "gsm8k_final_answer_copy.json")
PATTERNS = {"signed": r"-?\d[\d,]*(?:\.\d+)?", "unsigned": r"\d[\d,]*(?:\.\d+)?"}
N_EVAL = 660


def number(s):
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def measure(rows, pattern, equality, strip_annotations):
    rx = re.compile(pattern)
    n_eq, to_line, to_marker = 0, [], []
    for r in rows:
        a = r["answer"]
        i = a.rfind("####")
        fin = a[i + 4:].strip().replace(",", "")
        body = a[:i]
        if strip_annotations:
            body = re.sub(r"<<[^>]*>>", "", body)
        ms = list(rx.finditer(body))
        if not ms:
            continue
        last = ms[-1].group()
        if equality == "string":
            ok = last.replace(",", "").rstrip(".") == fin
        else:
            ok = number(last) is not None and number(last) == number(fin)
        if ok:
            n_eq += 1
            end = len(body[:ms[-1].end()].encode())
            to_line.append(len(body.rstrip("\n").encode()) - end)
            to_marker.append(len(body.encode()) - end)
    q = lambda x, p: sorted(x)[int(p * len(x))]
    return {"n": len(rows), "n_equal": n_eq, "share": n_eq / len(rows),
            "bytes_to_end_of_line": {"median": statistics.median(to_line), "p75": q(to_line, .75), "p90": q(to_line, .9),
                                     "zero": sum(x == 0 for x in to_line)},
            "bytes_to_marker": {"median": statistics.median(to_marker), "p75": q(to_marker, .75), "p90": q(to_marker, .9)}}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
    raw = open(path, "rb").read()
    rows = [json.loads(l) for l in raw.decode("utf-8").split("\n") if l.strip()]
    assert all(r["answer"].count("####") >= 1 for r in rows)
    res = {"input": {"file": "GSM8K test.jsonl (openai/grade-school-math, MIT)", "sha256": hashlib.sha256(raw).hexdigest(),
                     "lines": len(rows)},
           "definition": __doc__.split("\n\n")[1].replace("\n", " "),
           "primary": {}, "variants": {}}
    for subset, rs in (("all", rows), (f"first_{N_EVAL}", rows[:N_EVAL])):
        res["primary"][subset] = measure(rs, PATTERNS["signed"], "string", False)
        for pat in PATTERNS:
            for eq in ("string", "numeric"):
                for strip in (False, True):
                    res["variants"][f"{subset} | {pat} | {eq} | {'annotations removed' if strip else 'annotations kept'}"] = \
                        measure(rs, PATTERNS[pat], eq, strip)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for subset, v in res["primary"].items():
        print(f"{subset}: {v['n_equal']} of {v['n']} ({100 * v['share']:.1f}%); bytes to end of line: median "
              f"{v['bytes_to_end_of_line']['median']}, p90 {v['bytes_to_end_of_line']['p90']}; to '####': median "
              f"{v['bytes_to_marker']['median']}, p90 {v['bytes_to_marker']['p90']}")
    for k, v in res["variants"].items():
        print(f"  {k:62s} {v['n_equal']:5d} ({100 * v['share']:.1f}%)")
    print("wrote", os.path.relpath(OUT, os.path.dirname(os.path.dirname(HERE))))


if __name__ == "__main__":
    main()
