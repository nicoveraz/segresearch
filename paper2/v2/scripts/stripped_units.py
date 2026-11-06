"""Unit labels for worked math in the STRIPPED format of the BLT-1B layout experiments (realblt_budget.problem):
calculator annotations <<...>> removed, final line 'The final answer is N'. Shared by reducible2.py and
samemeasure.py so both use the format of the BLT-1B results (paper3/REVIEW_2026-10-22.md, A1).

Kinds (byte spans in one problem's text):
    computed   a number right after '=' (optionally '= $'), in the solution
    final      the number after 'The final answer is '
    copied     another number in the solution whose string occurs earlier in the problem (question or solution)
    word       an alphabetic word in the solution (start right after a space)
"""
import re

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def gsm8k_text(p):
    """GSM8K jsonl row -> (text, solution start), formatted as realblt_budget.problem."""
    sol, ans = p["answer"].rsplit("#### ", 1)
    sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    text = p["question"] + "\n" + sol + "\nThe final answer is " + ans.strip()
    return text, len(p["question"]) + 1


def strip_annotated(problem_text):
    """A synthetic problem in GSM8K's annotated format (synth_math.py) -> stripped text, solution start."""
    q, rest = problem_text.split("\n", 1)
    rest = re.sub(r"<<[^>]*>>", "", rest)
    rest = rest.replace("\n#### ", "\nThe final answer is ")
    return q + "\n" + rest.strip(), len(q) + 1


def units(text, s0):
    """[(start, end, kind)] in byte offsets of text.encode() (ASCII problems: char == byte)."""
    out = []
    f0 = text.rfind("The final answer is ")
    for m in NUM.finditer(text):
        a, b = m.span()
        if a < s0:
            continue
        if f0 >= 0 and a == f0 + len("The final answer is "):
            out.append((a, b, "final")); continue
        if re.search(r"=\s*\$?\s*$", text[max(s0, a - 4):a]):
            out.append((a, b, "computed")); continue
        if m.group(0) in text[:a]:
            out.append((a, b, "copied"))
    for m in re.finditer(r"(?<= )[A-Za-z]+", text):
        if m.start() >= s0:
            out.append((m.start(), m.end(), "word"))
    return sorted(out)
