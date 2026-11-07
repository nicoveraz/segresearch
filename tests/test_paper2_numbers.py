"""Paper 2, version 2: the manuscript must not contradict the results it is derived from.

Each test recomputes a number of paper2/manuscript.md from a public file (results/scale/runs, results/enrichment.json,
paper/tables/, paper2/v2/results/, the GSM8K-derived outputs shipped there) and checks that the text states it as
rounded there. Rounding is half-up on the unrounded value, and differences are computed before rounding. If a test
fails, the paper is wrong OR the analysis was re-run and the paper has not caught up: reconcile them, never relax the
check.

Scope: the whole manuscript. `head` is the front matter and sections 1-5, `later` is section 6 onward (the later
sections, the changes, code and data, and the appendices). Quoted text (version 1's claims, filed readings) is left out
of the wording checks and checked against its source instead.

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_paper2_numbers.py
"""
import glob
import json
import math
import os
import pathlib
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scale.analyze import welch  # noqa: E402  (numpy only)

MS = ROOT / "paper2" / "manuscript.md"
V2 = ROOT / "paper2" / "v2" / "results"
SPLIT = "## 6. Budget and objective, at inference"


# ----------------------------------------------------------------------------- helpers
def r(x, nd=1):
    """Half-up rounding of the unrounded value (a float, or an exact Fraction), as a string; never '-0.0'."""
    d = Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction) else Decimal(repr(float(x)))
    s = str(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    return ("0." + "0" * nd if nd else "0") if s.strip("-0.") == "" else s


def sg(x, nd=1):
    """Signed: '+4.8', '-0.2'."""
    s = r(x, nd)
    return s if s.startswith("-") else "+" + s


def pct(x, nd=1):
    return r(100 * x, nd)


def j(name):
    """A shipped result; files over 10 MB ship gzipped (`name.gz`, decompressing to the file the run wrote)."""
    if not (V2 / name).exists() and (V2 / (name + ".gz")).exists():
        import gzip
        return json.load(gzip.open(V2 / (name + ".gz"), "rt"))
    return json.load(open(V2 / name))


def flat(text):
    """Whitespace collapsed, so hard-wrapped lines do not matter."""
    return re.sub(r"\s+", " ", text)


def has(text, *frags):
    t = flat(text)
    missing = [f for f in frags if flat(f) not in t]
    assert not missing, f"not in the manuscript: {missing}"


def interval(lo, hi, nd=1, scale=1.0):
    return f"{sg(scale * lo, nd)} to {sg(scale * hi, nd)}"


def section(text, start, end=None):
    """The text from the heading `start` up to the next heading `end` (or to the end): numbers repeated elsewhere are
    checked where they appear, not anywhere in the manuscript."""
    assert text.count(start) == 1, start
    out = text.split(start, 1)[1]
    if end is not None:
        assert out.count(end) == 1, end
        out = out.split(end, 1)[0]
    return out


def para(text, start):
    """The paragraph (up to the next blank line) that begins with `start`."""
    assert text.count(start) == 1, start
    return text.split(start, 1)[1].split("\n\n", 1)[0]


@pytest.fixture(scope="module")
def full():
    return MS.read_text()


@pytest.fixture(scope="module")
def head(full):
    assert full.count(SPLIT) == 1, "section 6's heading is missing or repeated"
    return full.split(SPLIT, 1)[0]


@pytest.fixture(scope="module")
def later(full):
    return SPLIT + full.split(SPLIT, 1)[1]


def unquoted(text):
    """The text without double-quoted spans (version 1's claims and filed readings, quoted on purpose)."""
    return re.sub(r'"[^"]*"', " ", flat(text))


@pytest.fixture(scope="module")
def runs():
    out = {}
    for f in glob.glob(str(ROOT / "results" / "scale" / "runs" / "*" / "result.json")):
        name = os.path.basename(os.path.dirname(f))
        out[name] = json.load(open(f))
    return out


def arm(runs, size, a, clip, drop=()):
    rs = [runs.get(f"{size}_{a}_s{s}" + ("_clip" if clip else "")) for s in range(3) if s not in drop]
    return [x for x in rs if x]


def vals(rs, key):
    return [x[key] * (100 if key.endswith("_acc") else 1) for x in rs]


def mean_sd(v):
    m = sum(v) / len(v)
    return m, math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1))


def wpair(runs, size, a, b, key, clip=(None, None), drop_a=(), drop_b=()):
    """b - a with the seed-level Welch interval of scale.analyze.welch."""
    ca, cb = clip
    x = vals(arm(runs, size, a, ca, drop_a), key)
    y = vals(arm(runs, size, b, cb, drop_b), key)
    d, h = welch(x, y)
    return d, d - h, d + h


TRAIN = {"1m": False, "12m": False, "50m": True}                 # recipe of Table 1: 50M clipped


# ----------------------------------------------------------------------------- front matter and wording
def test_front_matter(head):
    assert head.startswith("---\ntitle: \"When the next byte's type is fixed: why entropy patching skips computed "
                           "values under tight budgets\"")
    has(head, 'author: "Nicolás Vera Zúñiga — Independent researcher, Chile — nicovera@quetru.cl"',
        "bibliography: refs.bib", "link-citations: true", "\\usepackage{needspace}")


def test_abstract_is_plain_ascii_and_short(head):
    a = head.split("## Abstract\n\n", 1)[1].split("\n\n## 1.", 1)[0].strip()
    assert all(ord(c) < 128 for c in a), "the abstract must be plain ASCII"
    assert len(a) <= 1900, len(a)
    for frag in ("teacher-forced", "1M-50M", "128-byte context", "T4", "fp16", "A40", "bf16", "2 of 100",
                 "tight budgets", "cue-formatted", "no venue"):
        assert frag in a, frag


def test_no_placeholders_or_drafting_notes(full):
    for bad in ("[R1", "[R2", "PLACEHOLDER", "FRAMING NOTES", "Anonymous submission", "[To be", "LATER SECTIONS", "[O6c"):
        assert bad not in full, bad
    assert full.count("<!--") == 0, "no HTML comments"


def test_wording(full):
    """Ties are written as ties, with their interval; withdrawn claims of version 1 do not come back (outside quotes)."""
    low = unquoted(full).lower()
    for bad in ("matches", "as good as", "does as well", "as well as", "equivalent", "no worse", "unchanged",
                "same bits", "no cost", "for free", "keeps the gain", "finds them", "text is new", "grows with",
                "stop treating surprisal", "reading the model is not needed", "pending", "training-free fix",
                "replicates", "improves final answers in prose", "without the cue", "at least as good"):
        hits = [m.start() for m in re.finditer(re.escape(bad), low)]
        hits = [i for i in hits if not low[max(0, i - 1):i + len(bad) + 1].startswith("s" + bad)]   # "spending"
        assert not hits, f"{bad!r} at {hits}"


def test_citations_resolve(full):
    bib = (ROOT / "paper2" / "refs.bib").read_text()
    keys = set(re.findall(r"@([A-Za-z][\w]*\d{4}[\w]*)", full))
    missing = [k for k in keys if "{" + k + "," not in bib]
    assert not missing, missing


# ----------------------------------------------------------------------------- setup (section 3)
def test_devices_and_recipes(runs):
    """At 1M and 12M the table arms ran on T4s and the baselines on the A40; at 50M everything on the A40."""
    for size in ("1m", "12m"):
        for a in ("nov10", "novc13", "novc10", "typed14", "final+typed14"):
            for x in arm(runs, size, a, False):
                assert "T4" in x["device"], (size, a)
        for a in ("entropy10", "dep10", "syntax+entropy10"):
            for x in arm(runs, size, a, False):
                assert x["device"] == "NVIDIA A40", (size, a)
    for name, x in runs.items():
        if name.startswith("50m_"):
            assert x["device"] == "NVIDIA A40", name


def test_t4_reruns(head, runs):
    for s, want in ((0, "-3.3"), (1, "+7.1")):
        cross = json.load(open(ROOT / "results" / "scale" / "cross" / f"12m_nov10_s{s}" / "result.json"))
        assert "T4" in cross["device"]
        d = 100 * (cross["final_acc"] - runs[f"12m_nov10_s{s}"]["final_acc"])
        assert sg(d) == want
    has(head, "moved their final answers by -3.3 and +7.1 points")


def test_clipping(head, runs):
    d, lo, hi = wpair(runs, "50m", "entropy10", "entropy10", "final_acc", clip=(False, True))
    assert (sg(d), sg(lo), sg(hi)) == ("+0.9", "-2.4", "+4.2")
    d, lo, hi = wpair(runs, "50m", "entropy10", "entropy10", "computed_acc", clip=(False, True))
    assert (sg(d), sg(lo)) == ("+6.8", "-14.2")
    d, lo, hi = wpair(runs, "50m", "entropy10", "entropy10", "bpb", clip=(False, True))
    has(head, f"{sg(d, 3)}, {interval(lo, hi, 3)}")
    m0, _ = mean_sd(vals(arm(runs, "50m", "entropy10", False), "bpb"))
    m1, _ = mean_sd(vals(arm(runs, "50m", "entropy10", True), "bpb"))
    has(head, f"({r(m0, 3)} to {r(m1, 3)}; {sg(d, 3)}", "+0.9 points, -2.4 to +4.2", "+6.8, -14.2 to +27.8")
    d, lo, hi = wpair(runs, "50m", "dep10", "dep10", "bpb", clip=(False, True))
    has(head, f"bits per byte {sg(d, 3)}, {interval(lo, hi, 3)}")


def test_eval_rates(head, runs):
    for a, want in (("entropy10", "11.2%"), ("dep10", "10.9%"), ("syntax+entropy10", "12.1%"), ("nov10", "11.9%"),
                    ("novc13", "11.4%")):
        rate = runs[f"12m_{a}_s0"]["rate"]
        assert pct(rate) + "%" == want, a
    has(head, "entropy 11.2%, dependence 10.9%, hand-written rule 12.1%, table 11.9%, corpus-fitted table 11.4%",
        "| all sizes | eval rate | 11.2% | 10.9% | 12.1% | 11.9% | 11.4% |")


def test_floors(head, runs):
    table1 = []
    for size, clip in TRAIN.items():
        for a in ("entropy10", "dep10", "syntax+entropy10", "nov10", "novc13"):
            table1 += [(f"{size}_{a}_s{s}", x) for s, x in enumerate(arm(runs, size, a, clip))]
    ok = [(n, x) for n, x in table1 if x["bpb"] < 4]                    # the failed 1M dependence seed: 4.718
    assert len(ok) == 38
    assert max(round(x["e2e_acc"] * x["e2e_n"]) for _, x in ok) == 1
    assert max(round(x["e2e_acc"] * x["e2e_n"]) for x in runs.values()) == 2
    boxed = [100 * x["boxed_acc"] for _, x in ok]
    assert (r(min(boxed)), r(max(boxed))) == ("1.9", "4.0")
    assert runs["1m_dep10_s0"]["boxed_acc"] == 0 and r(runs["1m_dep10_s0"]["bpb"], 3) == "4.718"
    others = [x["bpb"] for n, x in runs.items() if n.startswith("1m_") and n != "1m_dep10_s0"]
    assert (r(min(others), 2), r(max(others), 2)) == ("2.08", "2.26")
    has(head, "more than 1 of 100", "no 1M-50M run in this paper more than 2; Study A's 145M models have no\nreported "
        "end-to-end score", "are 1.9-4.0%", "bits per byte 4.718", "2.08-2.26",
        "the exclusion rule of `scale/analyze.py` (bits per byte above the median of its\nsize plus 1)")
    assert all(n.startswith(("1m_", "12m_", "50m_")) for n in runs)                  # no 145M run is public
    analyze = (ROOT / "scale" / "analyze.py").read_text()
    assert "med = float(np.median(b))" in analyze and '"bpb"] > med + 1' in analyze
    for size in ("1m", "12m", "50m"):                                               # per size, over every run
        b = sorted(x["bpb"] for n, x in runs.items() if n.startswith(size + "_"))
        med = (b[len(b) // 2] + b[(len(b) - 1) // 2]) / 2
        out = [n for n, x in runs.items() if n.startswith(size + "_") and x["bpb"] > med + 1]
        assert out == (["1m_dep10_s0"] if size == "1m" else []), (size, out)


def test_final_answer_copy(head):
    c = j("gsm8k_final_answer_copy.json")["primary"]
    assert c["all"]["n"] == 1319
    has(head, f"in {pct(c['all']['share'])}% of the 1,319 test solutions",
        f"({pct(c['first_660']['share'])}% of the\n660", f"median of {int(c['all']['bytes_to_marker']['median'])} bytes")


def test_exposures(head, runs):
    """GSM8K/MATH training solutions repeated 20 times in a 1.90 GB corpus (notes.md, the corpus line)."""
    line = next(x for x in (ROOT / "notes.md").read_text().splitlines() if x.startswith("Corpus 1.90 GB"))
    m = re.match(r"Corpus (\d+\.\d+) GB: .* GSM8K/MATH train x(\d+) \(10%\)", line)
    gb, reps = float(m.group(1)), int(m.group(2))
    has(head, f"solutions repeated {reps} times to make 10% of bytes", f"the same {r(gb)} GB corpus")
    for size, want in (("1m", "1.4"), ("12m", "5"), ("50m", "16")):
        cfg = runs[f"{size}_entropy10_s0"]["cfg"]
        n = reps * cfg["steps"] * cfg["bs"] * 128 / (gb * 1e9)
        assert r(n, 1 if size == "1m" else 0) == want, (size, n)
    has(head, "about 1.4, 5 and 16 times at 1M, 12M and 50M")


# ----------------------------------------------------------------------------- Table 1 and section 4
def cell(runs, size, a, key, clip, drop=()):
    """Mean ± sd; accuracy means as exact fractions of the target counts (867/2000 is 43.35, printed 43.4)."""
    rs = arm(runs, size, a, clip, drop)
    m, sd = mean_sd(vals(rs, key))
    if key != "bpb":
        n = rs[0][key.replace("_acc", "_n")]
        m = Fraction(100 * sum(round(x[key] * n) for x in rs), n * len(rs))
    nd = 3 if key == "bpb" else 1
    return f"{r(m, nd)} ± {r(sd, nd)}"


def test_table1(head, runs):
    rules = ("entropy10", "dep10", "syntax+entropy10", "nov10", "novc13")
    for size, clip in TRAIN.items():
        for key, label in (("computed_acc", "computed"), ("final_acc", "final"), ("bpb", "bits/byte")):
            cells = []
            for a in rules:
                rs = arm(runs, size, a, clip)
                if not rs:
                    cells.append("not run^b^" if (size, a) == ("50m", "syntax+entropy10") else "–")
                    continue
                drop = (0,) if (size, a) == ("1m", "dep10") else ()
                cells.append(cell(runs, size, a, key, clip, drop) + ("^a^" if drop else ""))
            row = " | ".join(cells)
            assert row in head, (size, label, row)
    has(head, cell(runs, "50m", "syntax+entropy10", "computed_acc", False), cell(runs, "50m", "syntax+entropy10",
        "final_acc", False), cell(runs, "50m", "syntax+entropy10", "bpb", False))
    old = [cell(runs, "1m", "dep10", k, False) for k in ("computed_acc", "final_acc", "bpb")]
    has(head, "giving " + ", ".join(x.split(" ")[0] for x in old[:2]) + " and " + old[2].split(" ")[0])
    nc = arm(runs, "12m", "novc10", False)
    f, c = vals(nc, "final_acc"), vals(nc, "computed_acc")
    has(head, f"only {pct(nc[0]['rate'])}% of evaluation bytes", f"final answers {r(min(f))}-{r(max(f))}%",
        f"computed results\n{r(min(c))}-{r(max(c))}%")


@pytest.mark.parametrize("size,a,b,key,text", [
    ("1m", "entropy10", "nov10", "final_acc", "+5.6 points, interval -20.8 to +32.0"),
    ("1m", "entropy10", "syntax+entropy10", "final_acc", "+38.4, -5.2 to +82.1"),
    ("12m", "entropy10", "nov10", "final_acc", "24.8 points above entropy's (+14.7 to +35.0)"),
    ("12m", "entropy10", "nov10", "computed_acc", "(+4.3, -5.6 to +14.1)"),
    ("50m", "entropy10", "nov10", "computed_acc", "+36.8 points on computed results (+17.0 to +56.5)"),
    ("50m", "entropy10", "nov10", "final_acc", "+29.6 on final answers (+26.6 to\n+32.5)"),
    ("50m", "dep10", "nov10", "computed_acc", "+22.4 (+18.5 to +26.3)"),
    ("50m", "dep10", "nov10", "final_acc", "+11.3 (+8.9 to +13.7)"),
    ("12m", "entropy10", "syntax+entropy10", "final_acc", "+29.6 points on\nfinal answers (+21.2 to +38.1)"),
    ("12m", "entropy10", "syntax+entropy10", "computed_acc", "+7.1 on computed results (+0.8 to +13.3)"),
    ("12m", "syntax+entropy10", "nov10", "final_acc", "-4.8 points on final\nanswers at 12M (-10.7 to +1.1)"),
    ("12m", "syntax+entropy10", "nov10", "computed_acc", "-2.8 on computed results (-11.1 to +5.5)"),
    ("1m", "syntax+entropy10", "nov10", "final_acc", "-32.8 on final answers at 1M (-70.2\nto +4.5)"),
    ("50m", "nov10", "novc13", "computed_acc", "-0.2 points, -6.6 to +6.1"),
    ("50m", "nov10", "novc13", "final_acc", "3.1 points lower on final answers (-5.9 to -0.4)"),
    ("50m", "nov10", "novc13", "bpb", "(-0.011, -0.030 to +0.008)"),
    ("50m", "entropy10", "nov10", "bpb", "+0.031, +0.012\nto +0.049"),
])
def test_training_pairs(head, runs, size, a, b, key, text):
    clip = TRAIN[size]
    d, lo, hi = wpair(runs, size, a, b, key, clip=(clip, clip))
    nd = 3 if key == "bpb" else 1
    nums = [s.lstrip("+-") for s in re.findall(r"[+-]?\d+\.\d+", text)]
    want = [r(abs(d), nd), r(abs(lo), nd), r(abs(hi), nd)]
    assert nums == want, (nums, want)
    has(section(head, "## 4. Results in math", "## 5. Why entropy"), text)        # where section 4 states it


def test_hand_rule_50m_preview(head, runs):
    d, lo, hi = wpair(runs, "50m", "syntax+entropy10", "nov10", "final_acc", clip=(False, True))
    has(head, f"{sg(d)} points on final answers ({interval(lo, hi)})")
    d, lo, hi = wpair(runs, "50m", "syntax+entropy10", "nov10", "computed_acc", clip=(False, True))
    has(head, f"{sg(d)} on computed results ({interval(lo, hi)})")
    d, lo, hi = wpair(runs, "50m", "entropy10", "syntax+entropy10", "final_acc", clip=(False, False))
    has(head, f"the rule gains {sg(d)} points on final answers ({interval(lo, hi)})")
    has(head, "(43.4% computed, 85.5% final)")


def test_seed_spread_50m(head, runs):
    e = vals(arm(runs, "50m", "entropy10", True), "computed_acc")
    t = vals(arm(runs, "50m", "nov10", True), "computed_acc")
    has(head, "({}, {} and {}%)".format(*(r(x) for x in e)), f"table's worst ({r(min(t))}%)")
    assert max(e) < min(t)
    for key in ("computed_acc", "final_acc"):
        assert max(vals(arm(runs, "50m", "entropy10", True), key)) < min(vals(arm(runs, "50m", "nov10", True), key))


def test_final_answer_coverage(head):
    e = json.load(open(ROOT / "results" / "enrichment.json"))
    assert e["entropy10"]["coverage"]["final"] == 1.0 and e["dep10"]["coverage"]["final"] == 0.0
    has(head, f"averages {r(e['surprisal_at_final_starts'], 2)} bits")
    fin = {k: pct(e[k]["acc50m_clip"]["final_acc"]) for k in ("entropy10", "dep10", "novc13", "nov10")}
    cov = {k: pct(e[k]["coverage"]["computed"]) for k in fin}
    assert cov["nov10"] == "100.0"
    has(head, f"entropy covers {cov['entropy10']}% of them and gets {fin['entropy10']}% of final answers, dependence "
              f"{cov['dep10']}% and {fin['dep10']}%, the corpus-fitted table {cov['novc13']}% and {fin['novc13']}%, the "
              f"table 100% and {fin['nov10']}%")


def test_cost_rates(head):
    e = json.load(open(ROOT / "results" / "enrichment.json"))
    has(head, f"({pct(e['nov10']['rate'])}% vs {pct(e['entropy10']['rate'])}%)")


def test_study_a_interim(full):
    """Only the values recorded in the shipped rules file (Study A outputs are not read); reported in Appendix F."""
    rec = (ROOT / "paper2" / "v2" / "rules" / "STUDY_A_INTERIM_RULES.md").read_text()
    for frag in ("91.2%", "68.3%", "1.813", "68.9%", "29.3%", "1.492", "g = +22.3", "0.844", "1.566", "91,500",
                 "amendments A-3 and A-4: entropy10 vs nov10 at\n145M, 3 seeds each, Kaggle T4 fp16",
                 "results/studyA_diag_repetition.json"):
        assert frag in rec, frag
    assert not (ROOT / "paper2" / "v2" / "results" / "studyA_diag_repetition.json").exists()
    f = section(full, "## Appendix F.", "## Appendix G.")
    has(f, "final answers 91.2% against 68.9% (+22.3 points)", "computed results 68.3% against 29.3% (+39.0)",
        "bits per byte 1.813 against 1.492", "0.84 bits per byte", "against 1.57 on new ones", "missed by 0.3 points",
        "Amendment A-4, which is not in the release", "on\nKaggle T4 GPUs in fp16",
        "the diagnostic output is not released")
    assert r(68.3 - 29.3) == "39.0" and r(0.844, 2) == "0.84" and r(1.566, 2) == "1.57"
    s4 = section(full, "## 4. Results in math", "## 5. Why entropy")
    has(s4, "**Study A (145M) was stopped.**", "Appendix F reports the look")
    assert "91.2%" not in s4


# ----------------------------------------------------------------------------- section 5 (BLT-1B)
def test_patcher_budget(head):
    p = j("patcher_budget_by_kind.json")
    m = j("paper2v2_intervals.json")["mechanism"]
    c = p["by_kind"]["computed"]
    has(head, f"mean {r(c['mean_entropy'], 2)} bits, against {r(p['mean_entropy_all'], 2)} over all bytes",
        "mean percentile {}, {} to {}".format(pct(c["mean_percentile"]), *(pct(x) for x in m["computed_entropy_pct"]["ci"])),
        "only {}% of\nthem ({} to {})".format(pct(c["share_above_p90"]), *(pct(x) for x in m["computed_share_above_p90"]["ci"])),
        f"({r(p['threshold_p90'], 2)} bits)", f"They receive {pct(c['share_of_budget'])}% of that",
        f"word starts receive {pct(p['by_kind']['word']['share_of_budget'])}%",
        f"{pct(p['by_kind']['no unit start']['share_of_budget'])}%.")


def test_ceiling(head):
    d = j("blt_cue_budget.json")
    t, v, P = d["thresholds_bits"], d["verdicts"], d["patcher"]
    has(head, f"{r(t['orig']['0.1'], 2)} bits for a 10% budget in the original format and {r(t['prose']['0.1'], 2)} in prose",
        "{}% ({} to {}) have a digit".format(pct(v["C1"]["value"]), *(pct(x) for x in v["C1"]["ci"])))
    assert v["C1"]["n"] == 2672 and v["Q1"]["orig"]["value"] == 0 and v["Q1"]["prose"]["value"] == 0
    dol = P["fresh|orig|res|$"]
    has(head, f"median of {r(dol['H_median'], 2)} bits and a 99th percentile of {r(dol['H_p99'], 2)}",
        f"({dol['n']} and {P['fresh|prose|res|$']['n']} values)")
    eq = P["fresh|orig|res|= "]
    has(head, f"{pct(eq['covered_at_rates']['0.1'])}% of {eq['n']:,}")
    sp = P["fresh|prose|res|space"]
    has(head, f"at {pct(sp['covered_at_rates']['0.1'])}% of {sp['n']:,}",
        "{} points over the `$` values of\nthe same texts, {} to {}".format(sg(100 * v["Q2"]["value"]),
                                                                         *(sg(100 * x) for x in v["Q2"]["ci"])))
    s1 = v["S1"]
    has(head, "from {}% to {}% ({}, {} to {})".format(pct(eq["covered_at_rates"]["0.1"]),
                                                       pct(eq["covered_at_rates"]["0.1"] + s1["value"]),
                                                       sg(100 * s1["value"]), *(sg(100 * x) for x in s1["ci"])),
        f"{pct(s1['covered_after_is_Pd_ge_0.9'])}%\nwhere it is still at least 0.9, {pct(s1['covered_after_is_Pd_lt_0.9'])}%")
    bc = d["by_cue"]
    has(head, f"A start at the value adds {pct(bc['prose|with-E|$']['value'])} points",
        f"against {pct(bc['prose|with-E|space']['value'])} after",
        "(difference {}, {} to\n{})".format(sg(100 * v["Q3"]["value"]), *(sg(100 * x) for x in v["Q3"]["ci"])))
    cr = dol["covered_at_rates"]
    assert cr["0.1"] == 0
    has(head, "entropy covers 0%, {}%, {}% and {}% of them".format(pct(cr["0.15"]), pct(cr["0.2"]), pct(cr["0.25"])))
    D1, m = v["D1"], d["matched"]

    def cov(w, kinds, rate):
        return sum(P[f"fresh|{w}|res|{k}"]["covered_at_rates"][str(rate)] * P[f"fresh|{w}|res|{k}"]["n"]
                   for k in kinds) / sum(P[f"fresh|{w}|res|{k}"]["n"] for k in kinds)

    orig_ro, prose_rp = cov("orig", ("$", "= "), m["R_o"]), cov("prose", ("$", "space"), m["R_p"])
    assert abs((cov("prose", ("$", "space"), 0.1) - orig_ro) - D1["G_A"]) < 1e-9
    assert abs((prose_rp - cov("orig", ("$", "= "), 0.1)) - D1["G_B"]) < 1e-9
    has(head, f"({v['C1']['n']:,} missed values in both writings)",
        "At 10%, entropy covers {}% of the original's values and {}% of the prose values, a {}-point gap".format(
            pct(cov("orig", ("$", "= "), 0.1)), pct(cov("prose", ("$", "space"), 0.1)), pct(D1["G_10"])),
        f"more than prose covers at 10%;")
    assert orig_ro > cov("prose", ("$", "space"), 0.1) and D1["verdict"] == "mixed"
    has(head, f"(an {pct(m['R_o'])}% rate, threshold {r(t['orig'][str(m['R_o'])], 2)} bits)",
        f"covers {pct(orig_ro)}% of its values", f"(a {pct(m['R_p'])}% rate), prose covers {pct(prose_rp)}%",
        f"closes {r(100 * D1['s_B']['value'], 0)}% of the {pct(D1['G_10'])}-point gap")


def test_prose_rewrite(head):
    p = j("blt_prose.json")
    v, L = p["verdicts"], p["by_layout"]
    assert (v["n_problems"], v["n_values"], v["overall"]) == (221, 711, "format-tied")
    has(head, "from\n{}% to {}% ({} points, {} to {})".format(
        pct(L["orig|entropy|res"]["covered"]), pct(L["prose|entropy|res"]["covered"]),
        sg(100 * v["V1_coverage_prose_minus_orig"]["value"]), *(sg(100 * x) for x in v["V1_coverage_prose_minus_orig"]["ci"])))
    has(head, "from {}% to\n{}%; {}, {} to {}".format(pct(L["prose|without|res"]["exact"]), pct(L["prose|with|res"]["exact"]),
                                                     sg(100 * v["V2_gain_prose"]["value"]), *(sg(100 * x) for x in v["V2_gain_prose"]["ci"])))
    has(head, "from {}% to {}%, {}; difference {}, {} to {}".format(
        pct(L["orig|without|res"]["exact"]), pct(L["orig|with|res"]["exact"]), sg(100 * v["V2_gain_orig"]["value"]),
        sg(100 * v["V2_gain_prose_minus_orig"]["value"]), *(sg(100 * x) for x in v["V2_gain_prose_minus_orig"]["ci"])))
    v3 = v["V3_nov_minus_entropy"]
    has(head, "from {} points ({} to {})".format(sg(100 * v3["orig"]["value"]), *(sg(100 * x) for x in v3["orig"]["ci"])),
        "{} ({} to {}) in prose".format(sg(100 * v3["prose"]["value"]), *(sg(100 * x) for x in v3["prose"]["ci"])),
        f"covers\n{pct(L['prose|nov|res']['covered'])}% of the values",
        f"gets {pct(L['orig|default|res']['exact'])}% of them\nin the original and {pct(L['prose|default|res']['exact'])}% in prose")
    s = j("blt_prose_cue_split.json")
    has(head, f"entropy\ncovers {pct(s['orig|= ']['entropy_covered'])}% of the original's values after `= `",
        f"{pct(s['prose|space']['entropy_covered'])}% of the prose values after a plain\nspace")
    assert s["orig|$"]["entropy_covered"] == 0 and s["prose|$"]["entropy_covered"] == 0
    T = j("v2_blt1b_numbers.json")["prose_by_cue"]["split"]
    assert T["orig|nov|$"]["covered"] == T["prose|nov|$"]["covered"] == T["orig|nov|= "]["covered"] == 1.0
    assert T["prose|nov|space"]["covered"] == 0.0
    has(head, "The table starts a patch at every value after `$` in both writings ({} and {}\nvalues) and at every "
              "value after `= `, but at none of the {} prose values after a plain space".format(
                  T["orig|nov|$"]["n"], T["prose|nov|$"]["n"], T["prose|nov|space"]["n"]))
    s5 = section(head, "## 5. Why entropy")
    has(para(s5, "**The same problems written as prose**"), f"On the {v['n_problems']} test problems with computed values "
        f"({v['n_values']} values)")


def test_cuefree(head):
    c = j("blt_cues.json")["cuefree"]
    S = c["summary"]
    assert S["nov_cf@10|res"]["n"] == 1156
    has(head, f"value starts are {pct(c['base_rate_computed_starts'])}% of bytes",
        f"the table covers {pct(S['nov_cf@10|res']['covered'])}%", f"covers\n{pct(S['lz_oracle@10|res']['covered'])}%",
        f"entropy {pct(S['entropy@10|res']['covered'])}%", f"gets {pct(S['entropy@10|res']['exact'])}% of these values",
        f"and\n{pct(S['oracle@10|res']['exact'])}% with a start forced")


def test_table2(head):
    g = j("blt_cues.json")["gsm8k"]
    comp = j("blt_compress.json")["summary"]
    waste = j("blt_waste.json")["curves"]
    p1 = {row.split(",")[0]: row.strip().split(",") for row in open(ROOT / "paper" / "tables" / "table1_blt1b.csv")}

    def row(name, rate, cov, res, fin):
        cov_s = "–" if cov is None else ("100%" if cov == 1 else f"{r(100 * cov, 0)}%")
        line = f"| {name} | {pct(rate)}% | {cov_s} | {pct(res)}% | {pct(fin)}% |"
        assert line in head, line

    S, R = g["summary"], g["rates"]
    for b in (10, 15):
        row("entropy (BLT's rule)" if b == 10 else "entropy", R[f"entropy@{b}"], S[f"entropy@{b}|res"]["covered"],
            S[f"entropy@{b}|res"]["exact"], S[f"entropy@{b}|fin"]["exact"])
        row("`=` rule (hand-written)" if b == 10 else "`=` rule", R[f"eq@{b}"], S[f"eq@{b}|res"]["covered"],
            S[f"eq@{b}|res"]["exact"], S[f"eq@{b}|fin"]["exact"])
        row("SpaceByte-style (entropy after non-alphanumeric)" if b == 10 else "SpaceByte-style", R[f"space@{b}"],
            S[f"space@{b}|res"]["covered"], S[f"space@{b}|res"]["exact"], S[f"space@{b}|fin"]["exact"])
        row("novelty table", R[f"nov@{b}"], S[f"nov@{b}|res"]["covered"], S[f"nov@{b}|res"]["exact"],
            S[f"nov@{b}|fin"]["exact"])
    row("syntax rule: `=` and the answer marker", R["syntax@10"], S["syntax@10|res"]["covered"],
        S["syntax@10|res"]["exact"], S["syntax@10|fin"]["exact"])
    nc = comp["novc@10"]
    row("corpus-fitted table", nc["rate"], nc["cov_res"], nc["res_exact"], nc["fin_exact"])
    tm = waste["table+marker@0.1"]
    row("novelty table + answer marker", tm["rate"], None, tm["acc_res"], tm["acc_fin"])
    for key, name in (("dep@10", "boundary dependence"), ("entdep@10", "entropy + boundary dependence"),
                      ("entdep@15", "entropy + boundary dependence")):
        _, rate, cov, res, fin = p1[key]
        assert f"| {name} | {r(100 * float(rate))}% | {cov} | {res} | {fin} |" in head, key


def test_cue_rule_pairs(head):
    L = j("paper2v2_intervals.json")["layouts"]
    x = L["res eq-nov@10"]
    has(head, "(+4.8 points,\n+0.9 to +8.5)")
    assert (sg(-x["diff"]), sg(-x["ci"][1]), sg(-x["ci"][0])) == ("+4.8", "+0.9", "+8.5")
    x = L["res eq-nov@15"]
    assert (sg(-x["diff"]), sg(-x["ci"][1]), sg(-x["ci"][0])) == ("+0.4", "-3.2", "+4.1")
    has(head, "(+0.4, -3.2 to +4.1)")
    x = L["res nov-space@15"]
    has(head, "ahead by {}, {} to {}".format(sg(x["diff"]), *(sg(t) for t in x["ci"])))
    e = j("paper2v2_intervals.json")["extras"]["blt1b res novc-entropy@10"]
    has(head, "{} points below entropy ({} to {}".format(r(-e["diff"]), *(sg(t) for t in e["ci"])))
    b = j("v2_blt1b_numbers.json")["bpb_vs_entropy"]["table@0.1 - entropy@0.1 | all"]
    has(head, "({} against {}; {}, {} to {})".format(r(b["b"], 3), r(b["a"], 3), sg(b["diff"], 3),
                                                     *(sg(t, 3) for t in b["ci"])))


def test_detector_summary(head):
    a = j("blt_autolabel.json")["verdicts"]["Z1_auto_prose_minus_entropy"]
    has(head, "gains {} points over\nentropy on computed values ({} to {}".format(
        sg(100 * a["value"]), *(sg(100 * x) for x in a["ci"])))
    v = j("blt_cue_budget.json")["verdicts"]
    d = j("blt_cue_budget.json")["diff"]
    has(head, "({}, {} and {} points at 10%, 15% and 20%".format(
        sg(100 * d["prose|D3-E|exact"]["value"]), sg(100 * v["G1@15"]["value"]), sg(100 * v["G1@20"]["value"])))
    f = d["prose|D3-E|fin"]
    has(head, "({} points, {} to {})".format(sg(100 * f["value"]), *(sg(100 * x) for x in f["ci"])))


def test_figure1_exists():
    assert (ROOT / "paper2" / "figures" / "fig1_scale_v2.png").stat().st_size > 10_000
    assert (ROOT / "paper2" / "fig1_v2.py").exists()


def test_submission_metadata_in_sync(head):
    """SUBMISSION.md's version 2 block carries the manuscript's title and abstract (code quotes as plain quotes)."""
    sub = (ROOT / "paper2" / "SUBMISSION.md").read_text()
    v2 = sub.split("## Version 2", 1)[1].split("## Version 1", 1)[0]
    title = re.search(r'^title: "(.*)"$', head, re.M).group(1)
    assert f"```\n{title}\n```" in v2
    a = head.split("## Abstract\n\n", 1)[1].split("\n\n## 1.", 1)[0].strip()
    plain = a.replace("`= `", '"= "').replace("`is `", '"is "')
    assert "`" not in plain and f"```\n{plain}\n```" in v2
    # arXiv renders $-delimited TeX in the abstract with MathJax: no dollar sign, escaped or not, may appear in it
    assert "$" not in a and "$" not in plain and "\\" not in plain
    assert len(plain) <= 1920
    assert f"{len(plain):,} of 1,920 characters" in v2
    assert "10.5281/zenodo.24023934" in v2 and "ZENODO_" + "V210_DOI" not in v2


# ============================================================================= section 6 onward (`later`)
def ivl(v, nd=1, scale=100.0):
    """'+a, +lo to +hi' for a stored [value, [lo, hi]] or {'value', 'ci'} (scaled to points by default)."""
    if isinstance(v, dict):
        v = [v.get("value", v.get("diff")), v["ci"]]
    return f"{sg(scale * v[0], nd)}, {interval(v[1][0], v[1][1], nd, scale)}"


def ci_only(v, nd=1, scale=100.0):
    if isinstance(v, dict):
        v = [v.get("value", v.get("diff")), v["ci"]]
    return interval(v[1][0], v[1][1], nd, scale)


def test_later_sections_and_appendices_present(full):
    for h in ("## 6. Budget and objective, at inference", "## 7. Other results, in brief", "## 8. Limitations",
              "## 9. Conclusion", "## Changes in version 2", "## Code and data", "## References",
              "## Appendix A.", "## Appendix B.", "## Appendix C.", "## Appendix D.", "## Appendix E.",
              "## Appendix F.", "## Appendix G.", "## Appendix H."):
        assert full.count("\n" + h) == 1, h
    assert "::: {#refs}\n:::" in full
    for ref in re.findall(r"Appendi(?:x|ces) ([A-H])\b", full):
        assert f"## Appendix {ref}." in full, ref


# ----------------------------------------------------------------------------- section 6 (BLT-1B, budget and objective)
def test_budget_by_objective(later):
    w = j("blt_waste.json")
    B, C = w["B"], w["curves"]
    assert B["verdict"] == "HOLDS" and B["table@0.1 acc_res"]["ci"][0] >= 1.5
    has(later, "about 1.75 times the table's patch rate (problem-bootstrap interval {} to {};".format(
        r(B["table@0.1 acc_res"]["ci"][0], 2), r(B["table@0.1 acc_res"]["ci"][1], 2)))
    assert r(B["table@0.1 acc_res"]["k"], 2) == "1.75"
    kb = j("v2_blt1b_numbers.json")["k_bracket"]
    assert abs(kb["k_linear"] - B["table@0.1 acc_res"]["k"]) < 1e-12
    has(later, "measured points at {}% and {}% of bytes, which alone bound the ratio between {} and {}".format(
        pct(kb["entropy_below"]["rate"]), pct(kb["entropy_above"]["rate"]),
        *(r(x, 2) for x in kb["k_range_from_bracketing_points"])))
    has(later, f"({pct(C['table@0.1']['acc_res'])}%, at a {pct(C['table@0.1']['rate'])}% patch rate)")
    for key, txt in (("eq@0.1 acc_res", "entropy needs {} times its rate ({} to {})"),
                     ("table+marker@0.1 acc_fin", "answer marker, {} times ({} to {})")):
        has(later, txt.format(r(B[key]["k"], 2), *(r(x, 2) for x in B[key]["ci"])))
    has(later, f"entropy's {r(C['entropy@0.1']['bpb_all'], 3)} at 10% (it gets {r(C['table@0.15']['bpb_all'], 3)} at "
               f"{pct(C['table@0.15']['rate'])}% of bytes), so it needs more than "
               f"{r(C['table@0.15']['rate'] / C['entropy@0.1']['rate'], 2)} times")
    assert C["table@0.15"]["bpb_all"] > C["entropy@0.1"]["bpb_all"]


def test_value_per_start(later):
    A = j("blt_waste.json")["A"]
    e, t = A["entropy@0.1"], A["table@0.1"]
    has(later, "{}% of entropy's starts buy no bits ({} to {}), against {}% of the table's ({} to {})".format(
        pct(e["share_value_le_0"]), *(pct(x) for x in e["share_ci"]), pct(t["share_value_le_0"]),
        *(pct(x) for x in t["share_ci"])),
        f"(+{r(e['mean_value'], 2)} bits against +{r(t['mean_value'], 2)})", "On 150 of the problems")
    assert j("blt_waste.json")["n_problems_A"] == 150
    z = j("v2_blt1b_numbers.json")["zero_value"]["table@0.1"]["by_type"]
    has(later, "question text ({}%) or inside words ({}%)".format(
        r(100 * z["question"]["share_of_zero_value_starts"], 0), r(100 * z["mid-word"]["share_of_zero_value_starts"], 0)),
        "({}% and {}% of them) but are only {}% of its starts".format(
            r(100 * z["space"]["share_value_le_0"], 0), r(100 * z["newline"]["share_value_le_0"], 0),
            r(100 * (z["space"]["share_of_starts"] + z["newline"]["share_of_starts"]), 0)))


def test_both_objectives(later):
    v = j("v2_blt1b_numbers.json")["bpb_vs_entropy"]
    eq = v["eq@0.1 - entropy@0.1 | all"]
    has(later, "bits per byte {} against entropy's {} ({}, {}; {} on the question and {} on the solution)".format(
        r(eq["b"], 3), r(eq["a"], 3), sg(eq["diff"], 3), interval(*eq["ci"], 3),
        sg(v["eq@0.1 - entropy@0.1 | q"]["diff"], 3), sg(v["eq@0.1 - entropy@0.1 | sol"]["diff"], 3)))
    h = j("blt_hybrid.json")
    assert h["verdict"] == "HOLDS"
    P = h["paired"]
    p = P["hybrid(0.03)@0.1 - entropy@0.1"]
    assert p["acc_res"]["ci"][0] >= 0.20 and p["bpb_all"]["ci"][1] <= 0.02
    has(later, "gains {} points on computed results over entropy at 10% ({}) and {} on final answers ({})".format(
        sg(100 * p["acc_res"]["diff"]), ci_only(p["acc_res"]), sg(100 * p["acc_fin"]["diff"]), ci_only(p["acc_fin"])),
        "whole-text bits per byte ({}, {};".format(sg(p["bpb_all"]["diff"], 3), ci_only(p["bpb_all"], 3, 1.0)))
    q, s = v["hybrid(0.03)@0.1 - entropy@0.1 | q"], v["hybrid(0.03)@0.1 - entropy@0.1 | sol"]
    has(later, "{} bits per byte on the question ({}) and {} on the solution ({})".format(
        sg(q["diff"], 3), interval(*q["ci"], 3), sg(s["diff"], 3), interval(*s["ci"], 3)))
    x = P["hybrid(0.03)@0.1 - eq@0.1"]
    has(later, "computed results {} points, {}; final answers {}, {}) and costs {} bits per byte more ({})".format(
        sg(100 * x["acc_res"]["diff"]), ci_only(x["acc_res"]), sg(100 * x["acc_fin"]["diff"]), ci_only(x["acc_fin"]),
        r(x["bpb_all"]["diff"], 3), ci_only(x["bpb_all"], 3, 1.0)))
    for q_, txt in (("0.015", "below entropy's ({}, {}) for a similar gain on computed results ({}, {})"),):
        y = P[f"hybrid({q_})@0.1 - entropy@0.1"]
        has(later, txt.format(sg(y["bpb_all"]["diff"], 3), ci_only(y["bpb_all"], 3, 1.0), sg(100 * y["acc_res"]["diff"]),
                              ci_only(y["acc_res"])))
    y = P["hybrid(0.05)@0.1 - entropy@0.1"]
    has(later, "cost bits ({}, {})".format(sg(y["bpb_all"]["diff"], 3), ci_only(y["bpb_all"], 3, 1.0)))
    y = P["hybrid(0.03)@0.15 - entropy@0.15"]
    has(later, "to {} at 15% ({})".format(sg(100 * y["acc_res"]["diff"]), ci_only(y["acc_res"])))
    c = j("blt_waste.json")["curves"]["entropy@0.25"]
    has(later, f"({pct(c['rate'])}% of bytes), entropy gets {pct(c['acc_res'])}% of computed results")
    t = v["table@0.1 - entropy@0.1 | all"]
    has(later, "costs {} bits per byte ({}; §5), against {} when".format(sg(t["diff"], 3), interval(*t["ci"], 3),
                                                                         "+0.031"))


def test_table3(later):
    w, h = j("blt_waste.json")["curves"], j("blt_hybrid.json")["curves"]
    rows = [("entropy", w, k) for k in ("entropy@0.05", "entropy@0.1", "entropy@0.15", "entropy@0.2", "entropy@0.25",
                                        "entropy@0.3")]
    rows += [("novelty table", w, k) for k in ("table@0.05", "table@0.075", "table@0.1", "table@0.15")]
    rows += [("novelty table + answer marker", w, "table+marker@0.1")]
    rows += [("entropy + a start at each value (`=` rule)", w, k) for k in ("eq@0.075", "eq@0.1", "eq@0.15")]
    rows += [(f"entropy + table's top {q} (label-free hybrid)", h, f"hybrid({f})@{b}")
             for q, f, b in (("1.5%", "0.015", "0.1"), ("3%", "0.03", "0.1"), ("5%", "0.05", "0.1"), ("3%", "0.03", "0.15"))]
    for name, src, k in rows:
        c = src[k]
        line = f"| {name} | {pct(c['rate'])}% | {pct(c['acc_res'])}% | {pct(c['acc_fin'])}% | {r(c['bpb_all'], 3)} |"
        assert line in later, line
    a = (ROOT / "paper2" / "figures" / "fig_tradeoff.png").read_bytes()
    assert a == (ROOT / "paper2" / "v2" / "figures" / "fig_tradeoff.png").read_bytes()
    assert "](figures/fig_tradeoff.png)" in later


def test_table4_quadrant(later):
    q = j("blt_quadrant.json")["by_kind"]
    names = {"computed": "computed result", "final": "final answer", "word_new": "new word", "copied": "copied number",
             "word_rep": "repeated word", "id_new": "random ticket number", "id_copy": "copy of the ticket"}
    for k, name in names.items():
        x = q[k]
        line = f"| {name} | {x['n']:,} | {sg(x['gain_bits'], 2)} | [{sg(x['gain_bits_ci'][0], 2)}, {sg(x['gain_bits_ci'][1], 2)}] |"
        assert line in later, line
    c = q["id_copy"]
    has(later, f"exact copying falls from {pct(c['exact_without'])}% without the start to {pct(c['exact_with'])}% with it")
    v = j("blt_quadrant_verdicts.json")
    assert v["P1"]["verdict"] == "INCONCLUSIVE" and v["P2"]["verdict"] == "HOLDS"
    has(later, "gain {} bits more than the hardest third ({};".format(
        r(v["P2"]["easy_minus_hard"]["diff"], 2), interval(*v["P2"]["easy_minus_hard"]["ci"], 2)),
        f"copied numbers gain {r(100 * v['P1']['copied']['share_of_computed'], 0)}% as much as computed results")


# ----------------------------------------------------------------------------- copies (section 6 and Appendix B)
@pytest.fixture(scope="module")
def o6b():
    return {k: j(f"blt_copy_latch_fresh_{k}.json") for k in ("j1", "j2", "j5", "j7", "verdict")}


def test_copy_o6b(later, o6b):
    V, F1 = o6b["verdict"], o6b["j2"]["F1"]
    assert V["fix_claim"]["holds"] and V["generality_claim"]["holds"] and F1["n"] == 600
    L = F1["by_layout"]
    row = "| O6b, 4-digit ticket, default ({}%) | 600 | {} | {} | {} | {} | – |".format(
        pct(L["D"]["rate"]), *(pct(L[k]["id_copy"]["exact"]) for k in ("D", "D-all", "D.L8", "D.SIVPO")))
    assert row in later, row
    row = "| O6b, 4-digit ticket, 10% ({}%) | 600 | {} | {} | {} | {} | – |".format(
        pct(L["E10"]["rate"]), *(pct(L[k]["id_copy"]["exact"]) for k in ("E10", "E10-all", "E10.L8", "E10.SIVPO")))
    assert row in later, row
    for kind, label in (("digits6", "6-digit ID"), ("mixed", "mixed ID (`QX7-4821`)")):
        b = o6b["j5"]["F3"]["by_kind"][kind]
        row = "| O6b, {}, default ({}%) | {} | {} | {} | {} | {} | – |".format(
            label, pct(b["by_layout"]["D"]["rate"]), b["n"],
            *(pct(b["by_layout"][k]["id_copy"]["exact"]) for k in ("D", "D-all", "D.L8", "D.SIVPO")))
        assert row in later, row
    F5 = o6b["j7"]["F5"]
    assert o6b["j7"]["window"] == 0
    row = "| O6b, 4-digit ticket, as released, default ({}%) | {} | {} | {} | {} | {} | – |".format(
        pct(F5["by_layout"]["D"]["rate"]), F5["n"],
        *(pct(F5["by_layout"][k]["id_copy"]["exact"]) for k in ("D", "D-all", "D.L8", "D.SIVPO")))
    assert row in later, row
    bars = V["bars"]
    has(later, "+74.8 points, +71.3 to +78.2", "recovers +73.7 of those points (+70.2 to +77.2), and +13.8 at 10% (+10.8 to +17.0)",
        "by {} points at the default rate ({})".format(sg(100 * bars["C0"]["value"][0]), ci_only(bars["C0"]["value"])),
        "and by {} at 10% ({})".format(sg(100 * bars["C1e"]["onset_removal_bound"][0]), ci_only(bars["C1e"]["onset_removal_bound"])),
        "L8 gains {} ({}) and {} ({})".format(sg(100 * bars["C1"]["value"][0]), ci_only(bars["C1"]["value"]),
                                              sg(100 * bars["C1e"]["E10.L8 - E10"][0]), ci_only(bars["C1e"]["E10.L8 - E10"])))
    assert ivl(bars["C0"]["value"]) == "+74.8, +71.3 to +78.2"
    P = F1["pairs"]
    d, e = P["D.L8 vs D"], P["E10.L8 vs E10"]
    has(later, "computed values {} ({}) and {} ({}) points".format(
        sg(100 * d["computed"]["exact"][0]), ci_only(d["computed"]["exact"]), sg(100 * e["computed"]["exact"][0]),
        ci_only(e["computed"]["exact"])),
        "final answers {} ({}) and {} ({})".format(sg(100 * d["final"]["exact"][0]), ci_only(d["final"]["exact"]),
                                                  sg(100 * e["final"]["exact"][0]), ci_only(e["final"]["exact"])),
        "copied numbers {} ({}) and {} ({})".format(r(100 * d["copied"]["exact"][0]), ci_only(d["copied"]["exact"]),
                                                   r(100 * e["copied"]["exact"][0]), ci_only(e["copied"]["exact"])),
        "outside the copy {} ({}) and {} ({})".format(sg(d["bpb_without_copy"][0], 4), ci_only(d["bpb_without_copy"], 4, 1.0),
                                                      sg(e["bpb_without_copy"][0], 4), ci_only(e["bpb_without_copy"], 4, 1.0)))
    c = o6b["j1"]["F6_counts"]
    assert c["D-L8"]["removed"] == 0 and c["E10-L8"]["removed"] == 0
    has(later, "L8 removed none of the {:,} and {} starts".format(c["D-L8"]["in_layout"], c["E10-L8"]["in_layout"]),
        "entropy's computed-value starts, which were {:,})".format(c["D-L8"]["in_layout"]))
    v8 = P["D.V8 vs D"]["computed"]["exact"]
    has(later, "(computed values {} points, {})".format(sg(100 * v8[0]), ci_only(v8)))
    G3 = bars["G3"]
    assert not G3["letters4"]["applies"] and not bars["G1"]["testable"]
    w, rel = V["paired"]["windowed"], V["paired"]["released"]
    assert w["class"] == "SIV+PO better" and rel["class"] == "L8 better"
    has(later, '"SIV+PO better" (L8 minus SIV+PO {} points in F1 at D, {};'.format(
        sg(100 * w["class rows"][0]["diff"][0]), ci_only(w["class rows"][0]["diff"])),
        '"L8 better" (L8 minus SIV+PO {}, {}):'.format(sg(100 * rel["class row"]["diff"][0]), ci_only(rel["class row"]["diff"])))
    assert all(x["class"] == "tie" for x in w["class rows"][1:])
    for sentence in V["sentences"][:2] + [w["reading"], rel["reading"], V["sentences"][4]]:
        has(later, '"' + sentence + '"')
    bl = V["reported"]["F5"]["by_length"]
    lo, hi = bl["tokens 0-600"], bl["tokens 700-2000"]
    has(later, "in the {} texts of 700 tokens or more, removing entropy's starts on the copy drops exact copies from "
               "{}% to {}%; L8, which fires at the copy's first byte in {}% of them, keeps {}%, and SIV+PO {}%".format(
                   hi["n"], pct(hi["D"]), pct(hi["D-all"]), pct(hi["L8_fires_c1"]), pct(hi["D.L8"]), pct(hi["D.SIVPO"])),
        "Under 600 tokens ({} texts) L8 and SIV+PO both keep {}%".format(lo["n"], pct(lo["D.L8"])))
    assert lo["D.L8"] == lo["D.SIVPO"]


def test_copy_o6c(later):
    V, S = j("blt_copy_latch_cutoff_verdict.json"), j("blt_copy_latch_cutoff_score.json")
    assert V["class"] == "the cutoff better" and V["n"] == 600 and S["window"] == 0
    has(later, '"' + V["reading"] + '"', "-3.17 points of exact copies, -5.00 to -1.33")
    p = V["primary"]
    assert (sg(100 * p[0], 2), sg(100 * p[1][0], 2), sg(100 * p[1][1], 2)) == ("-3.17", "-5.00", "-1.33")
    dt = V["beside"]["discordant texts (primary)"]
    has(later, "(-5.00 to -1.33; {} copies lost and {} kept; L8 did not fire at the copy's first byte on {} of the {} and "
               "fired on {})".format(dt["lost"]["copies"], dt["kept"]["copies"], dt["lost"]["L8 did not fire at c1"],
                                     dt["lost"]["copies"], dt["lost"]["L8 fired at c1"]))
    b = V["beside"]
    x = b["D.L8 - D.SIVPO (O6b's released row, replicated)"]
    assert x["class"] == "L8 better"
    has(later, "L8 minus SIV+PO is {} ({})".format(sg(100 * x["diff"][0], 2), ci_only(x["diff"], 2)),
        "copy starts gains {} ({})".format(sg(100 * b["D-all - D"][0], 2), ci_only(b["D-all - D"], 2)),
        "SIV+PO<620 gains {} over entropy ({})".format(sg(100 * b["D.SIVPO620 - D"][0], 2), ci_only(b["D.SIVPO620 - D"], 2)))
    L = S["by_layout"]
    row = "| O6c, 4-digit ticket, as released, new texts, default ({}%) | {} | {} | {} | {} | {} | {} |".format(
        pct(L["D"]["rate"]), S["n"], *(pct(L[k]["id_copy"]["exact"]) for k in ("D", "D-all", "D.L8", "D.SIVPO", "D.SIVPO620")))
    assert row in later, row
    bp = S["pairs"]["D.L8 vs D.SIVPO620"]["id_copy"]["bits_per_unit"]
    has(later, "({} bits, {}; read after the data)".format(sg(bp[0], 3), ci_only(bp, 3, 1.0)),
        f"(entropy's test rate {pct(L['D']['rate'])}%)")
    block = j("blt_copy_latch_cutoff_blocks.json")["block"]
    fit = sorted(x["idx"] for x in block["fit"])
    test = sorted(x["idx"] for x in block["test"])
    assert (len(fit), len(test)) == (300, 600)
    has(later, f"(problems {test[0]:,} to {test[-1]:,}; fit texts {fit[0]:,} to {fit[-1]:,})")
    check = (ROOT / "paper2" / "v2" / "checks" / "O6c_INDEPENDENT_CHECK.md").read_text()
    for frag in ("About 9% of bootstrap replications", "617-621", "idx 6691", "Exact copies only", "No cost bars"):
        assert frag in check, frag
    has(later, "About 9% of bootstrap replications of the whole test would tie", "cutoffs from 617 to 621 do best",
        "The rule had predicted a tie.")


def test_copy_table_onsets(later):
    t = j("blt_table_copy_onsets.json")["F1"]
    assert t["table@D"]["copy_c1"]["num"] == 0 and t["table@10"]["copy_c1"]["num"] == 0
    has(later, "inside {}% and {}% of copies".format(pct(t["table@D"]["copy_inside"]["value"]),
                                                    pct(t["table@10"]["copy_inside"]["value"])))
    assert j("blt_table_copy_onsets.json")["table_value_copy_equals_source"] == 293 == t["table@D"]["copy_c1"]["den"]


# ----------------------------------------------------------------------------- section 7 and Appendix A (detector)
def test_detector(later):
    cb = j("blt_cue_budget.json")
    D, V, C = cb["diff"], cb["verdicts"], cb["by_cue"]
    ms = j("blt_matched_share.json")["diff"]
    pr = j("blt_probe_reads.json")
    has(later, "(AUC 0.77, 0.74 to 0.80)")
    w1 = pr["verdicts"]["W1_within_number_auc"]
    assert (r(w1["value"], 2), r(w1["ci"][0], 2), r(w1["ci"][1], 2)) == ("0.77", "0.74", "0.80")
    m = ms["prose|M-E|fin"]
    has(section(later, "## Appendix A.", "## Appendix B."), "still loses 1.9 on final answers ({})".format(ci_only(m)),
        "keeps +12.3 on computed values ({})".format(ci_only(ms["prose|M-E|exact"])))
    assert sg(100 * m["value"]) == "-1.9" and sg(100 * ms["prose|M-E|exact"]["value"]) == "+12.3"
    f2 = D["prose|F2-E|exact"]
    has(later, "+11.1 points, {}) with no detectable final-answer difference ({}, {})".format(
        ci_only(f2), sg(100 * D["prose|F2-E|fin"]["value"]), ci_only(D["prose|F2-E|fin"])))
    assert sg(100 * f2["value"]) == "+11.1"
    has(later, "keeps +11.1 ({}) with no detectable final-answer difference ({}, {}) and slightly lower whole-text bits "
               "per byte ({}, {})".format(ci_only(f2), sg(100 * D["prose|F2-E|fin"]["value"]), ci_only(D["prose|F2-E|fin"]),
                                          sg(D["prose|F2-E|bpb_whole"]["value"], 3), ci_only(D["prose|F2-E|bpb_whole"], 3, 1.0)))
    s7 = section(later, "## 7. Other results", "## 8. Limitations")
    A = section(later, "## Appendix A.", "## Appendix B.")
    bw, bq = D["prose|dep-E|bpb_whole"]["value"], D["prose|dep-E|bpb_question"]["value"]
    assert "dependence" in s7 and "+3.8" not in s7          # section 7 points to Appendix A, which gives gain and cost
    has(A, "({} points behind in prose, {}; {} in equation text, {})".format(
        r(100 * D["prose|D3-dep|exact"]["value"]), ci_only(D["prose|D3-dep|exact"]), r(100 * D["orig|D3-dep|exact"]["value"]),
        ci_only(D["orig|D3-dep|exact"])),
        "raises final answers\nover entropy ({}), at {} bits per byte on the whole text ({} on the question)".format(
            ivl(D["prose|dep-E|fin"]), sg(bw, 3), sg(bq, 3)),
        "drops them to {}%".format(pct(cb["layouts"]["orig|dep"]["exact_fin"])))
    has(later, "The 3% layout gains {} points in prose ({})".format(sg(100 * D["prose|D3-E|exact"]["value"]),
                                                                  ci_only(D["prose|D3-E|exact"])),
        "{} points there ({}; {} values) against {} after a plain space ({}; {:,} values)".format(
            sg(100 * C["prose|D3-E|$"]["value"]), ci_only(C["prose|D3-E|$"]), C["prose|D3-E|$"]["n"],
            sg(100 * C["prose|D3-E|space"]["value"]), ci_only(C["prose|D3-E|space"]), C["prose|D3-E|space"]["n"]),
        "({} at 15%, {}; {} at 20%, {})".format(sg(100 * V["G1@15"]["value"]), ci_only(V["G1@15"]),
                                                 sg(100 * V["G1@20"]["value"], 2), ci_only(V["G1@20"], 2)),
        "random 3% of candidates by {} ({})".format(sg(100 * V["K1"]["value"]), ci_only(V["K1"])),
        "loses final answers ({} points, {})".format(sg(100 * D["prose|D3-E|fin"]["value"]), ci_only(D["prose|D3-E|fin"])),
        "whole-text bits per byte ({}, {}; {} on the question, {} on the solution)".format(
            sg(V["W1"]["value"], 4), ci_only(V["W1"], 4, 1.0), sg(D["prose|D3-E|bpb_question"]["value"], 3),
            sg(D["prose|D3-E|bpb_solution"]["value"], 3)))
    F1 = V["F1"]
    assert F1["lost"]["problems"] == 67
    has(later, "the 67 problems where its forced starts displace entropy's start at the final answer ({}, {})".format(
        sg(100 * F1["lost"]["value"]), ci_only(F1["lost"])),
        "recovers {} ({}) but still loses {} ({})".format(sg(100 * D["prose|D3F-D3|fin"]["value"]), ci_only(D["prose|D3F-D3|fin"]),
                                                         r(-100 * D["prose|D3F-E|fin"]["value"]), ci_only(D["prose|D3F-E|fin"])),
        "gains {} on prose values after `$` ({}) but loses {} points of final answers ({})".format(
            sg(100 * V["V1"]["VU-E_dollar"]["value"]), ci_only(V["V1"]["VU-E_dollar"]),
            r(-100 * V["V1"]["VU-E_fin"]["value"]), ci_only(V["V1"]["VU-E_fin"])),
        "gains {} over entropy and {} over the `=` rule ({}), raises final answers ({}) and lowers whole-text bits per byte "
        "({}, {})".format(sg(100 * D["orig|D3-E|exact"]["value"]), sg(100 * D["orig|D3-eq|exact"]["value"]),
                          ci_only(D["orig|D3-eq|exact"]), ivl(D["orig|D3-E|fin"]), sg(D["orig|D3-E|bpb_whole"]["value"], 3),
                          ci_only(D["orig|D3-E|bpb_whole"], 3, 1.0)))
    assert j("blt_matched_share.json")["q"]["prose"] == 0.0153


def test_detector_probe_and_reads(later):
    L = j("blt_learned.json")
    a, v = L["auc"], L["verdicts"]
    has(later, "AUC {} ({} to {}) when trained on prose and {} ({} to {}) when trained on equation text, against {} for "
               "entropy".format(r(a["prose|rp_prose"]["value"], 2), *(r(x, 2) for x in a["prose|rp_prose"]["ci"]),
                                r(a["prose|rp_orig"]["value"], 2), *(r(x, 2) for x in a["prose|rp_orig"]["ci"]),
                                r(a["prose|entropy"]["value"], 2)),
        "gains {} and {} points on computed values over entropy ({}; {})".format(
            sg(100 * v["L1_router_patcher_prose"]["value"]), sg(100 * v["L2_router_patcher_orig_on_prose"]["value"]),
            ci_only(v["L1_router_patcher_prose"]), ci_only(v["L2_router_patcher_orig_on_prose"])))
    A = j("blt_autolabel.json")
    ag, Lz, Vz = A["auto_label_agreement"]["prose"], A["layouts"], A["verdicts"]
    has(later, "precision {} and recall {})".format(r(ag["precision"], 2), r(ag["recall"], 2)),
        "it gains {} points in prose ({})".format(sg(100 * Vz["Z1_auto_prose_minus_entropy"]["value"]),
                                                  ci_only(Vz["Z1_auto_prose_minus_entropy"])),
        "slightly above the supervised probe refitted in the\nsame run ({}; difference {})".format(
            sg(100 * Lz["prose|marker_prose"]["minus_entropy_res"]["value"]), ivl(Vz["Z2_auto_minus_marker"])),
        "applied to prose, {} ({})".format(sg(100 * Lz["prose|auto_orig"]["minus_entropy_res"]["value"]),
                                          ci_only(Lz["prose|auto_orig"]["minus_entropy_res"])),
        "in equation text, {} ({})".format(sg(100 * Lz["orig|auto_orig"]["minus_entropy_res"]["value"]),
                                          ci_only(Lz["orig|auto_orig"]["minus_entropy_res"])),
        "bits per byte of the solution ({}, {})".format(sg(Lz["prose|auto_prose"]["minus_entropy_bpb"]["value"], 3),
                                                       ci_only(Lz["prose|auto_prose"]["minus_entropy_bpb"], 3, 1.0)),
        "loses final answers ({}% against {}%)".format(pct(Lz["prose|auto_prose"]["exact_fin"]), pct(Lz["prose|entropy"]["exact_fin"])))
    R = j("blt_probe_reads.json")
    c, f = R["counts"]["prose"], R["forced"]["prose|auto_prose"]
    has(later, "{} computed values among {:,} numbers".format(c["computed"], c["numbers"]),
        "gets {}".format(r(R["auc"]["prose|anynum_prose|computed_within_numbers"]["value"], 2)),
        "({:,} against {:,} numbers in prose, {}% of them on non-numbers)".format(f["n_forced"], c["numbers"],
                                                                                r(100 * f["share_non_number"], 0)),
        "ahead of it by {} points in prose ({})".format(sg(100 * R["verdicts"]["W2_auto_minus_anynum_res"]["value"]),
                                                        ci_only(R["verdicts"]["W2_auto_minus_anynum_res"])),
        "behind it by {} in equation text ({})".format(r(-100 * R["layouts"]["orig|auto_minus_anynum_res"]["value"]),
                                                       ci_only(R["layouts"]["orig|auto_minus_anynum_res"])))
    P = j("blt_profiles.json")["auc_computed_vs_copy"]["prose"]
    has(later, "at the first digit (AUC {})".format(r(P["entropy@+0"]["auc"], 2)),
        "({} at the second digit)".format(r(P["entropy@+1"]["auc"], 2)),
        "(AUC {} there, {} one byte before)".format(r(P["probe@+0"]["auc"], 2), r(P["probe@-1"]["auc"], 2)))
    S = j("blt_second_digit.json")["diff"]
    has(later, "({}, {}, against {} for a start at the first digit, {})".format(
        sg(100 * S["prose|O2-E|exact"]["value"]) + " points", ci_only(S["prose|O2-E|exact"]),
        sg(100 * S["prose|O1-E|exact"]["value"]), ci_only(S["prose|O1-E|exact"])))
    pv = j("patcher_variants.json")
    assert pv["verdicts"]["U1"]["verdict"] == "FAILS" and pv["verdicts"]["U2"]["verdict"] == "FAILS"
    d = pv["diff"]
    for k in ("V1", "V2", "V4"):
        lo, hi = d[f"prose|all|{k}-V0"]["ci"]
        assert lo < 0 < hi, k
    has(later, "({}, {} and {}; each interval includes 0)".format(*(sg(d[f"prose|all|{k}-V0"]["value"], 3) for k in ("V1", "V2", "V4"))),
        "did ({})".format(ivl(d["prose|all|V3-V0"], 3, 1.0)))
    assert 4000 * 32 * 128 == 16_384_000


def test_table_a1(later):
    cb, ms = j("blt_cue_budget.json"), j("blt_matched_share.json")
    L, D = cb["layouts"], cb["diff"]
    assert ms["layouts"]["prose|E"]["exact_res"] == pytest.approx(L["prose|E"]["exact_res"])
    rows = [("entropy", "E", None), ("entropy + a start at every value (labels)", "with", "with"),
            ("boundary dependence", "dep", "dep"), ("probe, top 3%", "D3", "D3"),
            ("probe, top 3%, entropy's final-line starts kept", "D3F", "D3F"),
            ("fused ranking (entropy + 2 × probe)", "F2", "F2"),
            ("fused, value uncertainty instead of the probe", "VU", "VU"), ("random 3% of candidates", "R3", "R3")]
    for name, k, b in rows:
        bits = "–" if b is None else sg(D[f"prose|{b}-E|bpb_whole"]["value"], 4)
        line = "| {} | {} | {} | {} | {} | {} |".format(name, pct(L[f"prose|{k}"]["exact_res"]), pct(L[f"prose|{k}"]["exact_fin"]),
                                                        bits, pct(L[f"orig|{k}"]["exact_res"]), pct(L[f"orig|{k}"]["exact_fin"]))
        assert line in later, line
    M = ms["layouts"]
    line = "| probe, as many starts as numbers | {} | {} | – | {} | {} |".format(
        pct(M["prose|M"]["exact_res"]), pct(M["prose|M"]["exact_fin"]), pct(M["orig|M"]["exact_res"]), pct(M["orig|M"]["exact_fin"]))
    assert line in later, line
    assert cb["n"]["prose"]["fresh"] == 915 and cb["n"]["prose"]["values"] == 2448


# ----------------------------------------------------------------------------- Appendix C (outside math)
GEN = ROOT / "results" / "generic"
PCT_KEYS = ("acc_computed", "acc_copied", "acc_id_new", "acc_rhs_copy")


def gen(folder, rule):
    rs = [json.load(open(f)) for f in sorted(glob.glob(str(GEN / folder / f"{rule}_s*.json")))]
    assert len(rs) == 2 and all(x["device"] == "Tesla T4" for x in rs), (folder, rule)
    return rs


def gvals(rs, m):
    return [x[m] * (100 if m in PCT_KEYS else 1) for x in rs]


def test_outside_math(later):
    out = j("v2_outside_math.json")
    assert out["devices"] == ["Tesla T4"]
    for folder, a, b, m, nd, txt in (
            ("prose", "entropy10", "novelty", "bpb", 3, "(1.770 against 1.832; {})"),
            ("code", "entropy10", "novelty", "bpb", 3, "(1.649 against 1.684; {})"),
            ("code_acc", "entropy10", "novelty", "acc_id_new", 1, "128 bytes {}"),
            ("code_acc", "entropy10", "novelty", "acc_rhs_copy", 1, "`(` {})"),
            ("trace", "entropy10", "novelty", "acc_computed", 1, "on computed values ({})"),
            ("trace", "entropy10", "novelty", "acc_copied", 1, "copied values ({})")):
        d, h = welch(gvals(gen(folder, a), m), gvals(gen(folder, b), m))
        stored = out["pairs"][f"{folder}: {b} - {a} | {m}"]
        assert abs(stored["diff"] - d) < 1e-9 and abs(stored["ci"][0] - (d - h)) < 1e-9
        s = f"{sg(d, nd)}, {interval(d - h, d + h, nd)}"
        if folder == "code_acc":
            s = (s.split(",")[0] + " points," + s.split(",", 1)[1]) if m == "acc_id_new" else s
        has(later, txt.format(s))
    for folder, rule, name in (("prose", "entropy10", "prose | entropy"), ("prose", "novelty", "prose | novelty table"),
                               ("code", "entropy10", "code | entropy"), ("code", "novelty", "code | novelty table"),
                               ("code", "entropy_t17", "code | entropy (higher budget)"),
                               ("code", "novelty_t10", "code | novelty table (higher budget)"),
                               ("trace", "entropy10", "traces | entropy"), ("trace", "novelty", "traces | novelty table"),
                               ("trace_novnet", "novelty_net", "traces | novelty network (128 bytes)")):
        rs = gen(folder, rule)
        m, s = mean_sd(gvals(rs, "bpb"))
        cells = [f"{pct(sum(x['val_rate'] for x in rs) / 2)}%", f"{r(m, 3)} ± {r(s, 3)}"]
        for k in ("acc_computed", "acc_copied"):
            if k in rs[0]:
                mm, ss = mean_sd(gvals(rs, k))
                cells.append(f"{r(mm)} ± {r(ss)}")
            else:
                cells.append("–")
        line = f"| {name} | " + " | ".join(cells) + " |"
        assert line in later, line
    nn = json.load(open(GEN / "trace" / "novnet.json"))["network (128 bytes)"]
    assert nn["copied value"]["covered"] == 0
    has(later, f"places patch starts at {r(100 * nn['computed value']['covered'], 0)}% of computed trace values and none of "
               "the copied ones")
    corp = {d: json.load(open(GEN / d / "corpus.json"))["train_bytes"] for d in ("code", "trace")}
    has(later, f"Python source code ({r(corp['code'] / 1e9, 2)} GB", f"execution traces ({r(corp['trace'] / 1e9, 2)} GB)")


def test_no_general_rule(later):
    X = j("xdomain.json")
    v, dom = X["verdicts"], X["domains"]
    assert v["X1"]["verdict"] == "fails" and v["X1"]["domains_H_above_E_unit"] == ["code"]
    assert v["X2"]["verdict"] == "does not hold" and v["X3"]["bits"]["verdict"] == "fails"
    assert v["X3"]["unit"]["verdict"] == "fails" and v["X3"]["bits"]["lodo_above_E"] == [] == v["X3"]["unit"]["lodo_above_E"]
    assert v["X4"]["verdict"] == "does not hold"
    f = lambda x, nd=2: f"{sg(x['value'], nd)}, {interval(*x['ci'], nd)}"
    has(later, "on units only in code ({})".format(f(dom["code"]["unit"]["minus_E"]["H"])),
        "below it in prose ({})".format(f(dom["prose"]["unit"]["minus_E"]["H"])),
        "(algebra {}, code {}, prose {}, clinical {};\n  each interval below 0)".format(
            *(sg(dom[d]["bits"]["minus_E"]["N"]["value"], 2) for d in ("algebra", "code", "prose", "clinical"))),
        "math ({}), algebra ({}) and traces ({}; each interval above 0)".format(
            *(sg(dom[d]["bits"]["minus_E"]["G_dom"]["value"], 2) for d in ("math", "algebra", "traces"))),
        "algebra on units ({}), nor does the table ({})".format(f(v["X4"]["G_math_minus_E"]), f(v["X4"]["N_minus_E"])))
    assert sorted(v["X2"]["N_minus_E_bits_below0"]) == sorted(["algebra", "code", "prose", "clinical"])
    vc = dom["math"]["by_class"]["value"]
    has(later, f"({r(vc['gain_unit'], 2)} per start), and only {r(100 * vc['in_base'], 0)}% of them")
    assert vc["gain_unit"] == max(c["gain_unit"] for c in dom["math"]["by_class"].values() if c["gain_unit"] is not None)
    u = {d: dom[d]["bits"]["minus_E"]["U"] for d in dom}
    above = [x for x in u.values() if x["ci"][0] > 0]
    assert len(above) == 5 and u["traces"]["ci"][0] <= 0
    assert all(dom[d]["unit"]["minus_E"]["U"]["value"] == 0 == dom[d]["unit"]["minus_E"]["U"]["ci"][1] for d in dom)
    has(later, "in five of six domains ({} to {}, each\ninterval above 0; traces {}, {})".format(
        sg(min(x["value"] for x in above), 2), sg(max(x["value"] for x in above), 2), sg(u["traces"]["value"], 3),
        interval(*u["traces"]["ci"], 3)))
    has(later, "In math it is not distinguishable ({})".format(f(dom["math"]["unit"]["minus_E"]["H"])))
    assert dom["math"]["unit"]["minus_E"]["H"]["ci"][0] < 0
    for d in ("algebra", "code", "prose", "clinical"):
        assert dom[d]["bits"]["minus_E"]["N"]["ci"][1] < 0
    for d in ("math", "algebra", "traces"):
        assert dom[d]["bits"]["minus_E"]["G_dom"]["ci"][0] > 0
    P = j("xdomain_probe.json")
    pv = P["verdicts"]
    assert pv["Y1_bits"]["verdict"] == "holds" and len(pv["Y1_bits"]["domains"]) == 5
    assert pv["Y1_unit"]["domains"] == ["traces"] and pv["Y2_bits"]["domains"] == ["traces"] and pv["Y2_unit"]["domains"] == []
    pd = [P["domains"][d]["bits"]["minus_E"]["P_dom"]["value"] for d in pv["Y1_bits"]["domains"]]
    assert all(P["domains"][d]["bits"]["minus_E"]["P_dom"]["ci"][0] > 0 for d in pv["Y1_bits"]["domains"])
    has(later, "within five domains ({} to\n{}, each interval above 0)".format(sg(min(pd), 2), sg(max(pd), 2)))
    N = j("xdomain_newtok.json")
    nv = N["verdicts"]
    assert nv["V1_bits"]["domains"] == ["code"] and nv["V2_bits"]["domains"] == ["code", "traces"]
    assert nv["V1_unit"]["domains"] == [] == nv["V2_unit"]["domains"]
    au = [x for k, x in N["auc"].items() if k.split("|")[1] in ("N_dom", "N_lodo")]
    has(later, "(AUC {} to {})".format(r(min(au), 2), r(max(au), 2)),
        "(-0.11 on bits, {})".format(ci_only(N["domains"]["math"]["bits"]["N_minus_NUM"], 2, 1.0)))
    assert sg(N["domains"]["math"]["bits"]["N_minus_NUM"]["value"], 2) == "-0.11"


# ----------------------------------------------------------------------------- Appendix D (other negative results)
def test_negative_results(later, runs):
    comp = j("blt_compress.json")["summary"]
    ex = j("paper2v2_intervals.json")["extras"]
    has(later, "gets {}% of computed results and PPM-C's escape probability {}%".format(
        pct(comp["lz@10"]["res_exact"]), pct(comp["ppm@10"]["res_exact"])),
        "({}, {}; {}, {})".format(sg(ex["blt1b res lz-entropy@10"]["diff"]) + " points", ci_only(ex["blt1b res lz-entropy@10"], 1, 1.0),
                                  sg(ex["blt1b res ppm-entropy@10"]["diff"]), ci_only(ex["blt1b res ppm-entropy@10"], 1, 1.0)),
        "at only {}% and {}% of computed results".format(r(100 * comp["lz@10"]["cov_res"], 0), r(100 * comp["ppm@10"]["cov_res"], 0)))
    cp = j("v2_blt1b_numbers.json")["compress_pairs"]["lz+nov@10 - nov@10"]
    has(later, "({}% against {}%; {}, {})".format(r(cp["acc_b"]), r(cp["acc_a"]), sg(cp["diff"]), interval(*cp["ci"])))
    t = comp["typed@10"]
    has(later, "({}% at 10%, {} points over entropy, {})".format(pct(t["res_exact"]), sg(ex["blt1b res typed-entropy@10"]["diff"]),
                                                                ci_only(ex["blt1b res typed-entropy@10"], 1, 1.0)),
        "({}% against {}%)".format(pct(t["fin_exact"]), pct(comp["entropy@10"]["fin_exact"])))
    m_ = lambda size, a, key, clip: cell(runs, size, a, key, clip).split(" ")[0]
    dd = section(later, "## Appendix D.", "## Appendix E.")
    d, lo, hi = wpair(runs, "12m", "entropy10", "typed14", "final_acc", clip=(False, False))
    has(dd, "(final answers {}% against {}% for entropy; {}, {};".format(
        m_("12m", "typed14", "final_acc", False), m_("12m", "entropy10", "final_acc", False), sg(d), interval(lo, hi)))
    d, lo, hi = wpair(runs, "50m", "entropy10", "typed14", "final_acc", clip=(True, True))
    has(dd, "({}% against {}%; {}, {})".format(m_("50m", "typed14", "final_acc", True), m_("50m", "entropy10", "final_acc", True),
                                                sg(d), interval(lo, hi)),
        "with one seed at {}%".format(r(min(vals(arm(runs, "50m", "typed14", True), "final_acc")))))
    d, lo, hi = wpair(runs, "1m", "entropy10", "typed14", "final_acc", clip=(False, False))
    has(dd, "(final\nanswers {}% against {}%; {}, {}; across devices)".format(
        m_("1m", "typed14", "final_acc", False), m_("1m", "entropy10", "final_acc", False), sg(d), interval(lo, hi)))
    ft = arm(runs, "50m", "final+typed14", True)
    has(dd, "rise to {}% while computed results stay near 9% ({}%, against {}% without the forced start)".format(
        r(mean_sd(vals(ft, "final_acc"))[0]), m_("50m", "final+typed14", "computed_acc", True),
        m_("50m", "typed14", "computed_acc", True)))
    tr = j("v2_training_numbers.json")["t4_reruns"]
    has(later, "(final answers {}% and {}% under their own layout)".format(r(tr["seed 0"]["final_rerun"]), r(tr["seed 1"]["final_rerun"])),
        "to {}% under entropy and {}% under value uncertainty".format(r(tr["mean over the two re-runs"]["entropy10"]),
                                                                     r(tr["mean over the two re-runs"]["typed14"])))
    sw = [tr[s]["switched_layout"][k]["bpb"] for s in ("seed 0", "seed 1") for k in ("entropy10", "typed14", "final+typed14")]
    own = [tr[s]["bpb_rerun"] for s in ("seed 0", "seed 1")]
    has(later, "from {} to {}-{}".format(r(sum(own) / 2, 2), r(min(sw), 1), r(max(sw), 1)))


# ----------------------------------------------------------------------------- Appendix E (R13 pilot)
def test_r13_pilot(later, runs):
    p = json.load(open(V2 / "r13_pilot" / "result.json"))
    assert p["device"] == "mps" and p["size"] == "12m" and p["seed"] == 0
    has(later, "It got {}% of final answers, {}% of computed results and {} bits per byte".format(
        pct(p["final_acc"]), pct(p["computed_acc"]), r(p["bpb"], 3)),
        "an evaluation patch rate of {}% against entropy's {}%".format(pct(p["rate"]), pct(runs["12m_entropy10_s0"]["rate"])))
    e, n, s = (vals(arm(runs, "12m", a, False), "final_acc") for a in ("entropy10", "nov10", "syntax+entropy10"))
    has(later, f"entropy seed ({r(min(e))}-{r(max(e))}%) and table seed ({r(min(n))}-{r(max(n))}%)",
        f"the lowest seed of the hand-written rule ({r(min(s))}%)")
    assert max(e + n) < 100 * p["final_acc"] < min(s)
    b = [x["bpb"] for a in ("entropy10", "nov10", "syntax+entropy10") for x in arm(runs, "12m", a, False)]
    assert p["bpb"] < min(b)
    has(later, f"below every run of those arms ({r(min(b), 3)}-{r(max(b), 3)})")
    rule = (ROOT / "paper2" / "v2" / "rules" / "R13_PROBE_TRAIN_RULES.md").read_text()
    assert "**promising** if final answers >= 60% with bits per byte <= 1.70" in rule


# ----------------------------------------------------------------------------- Appendix G (H-Net)
def test_hnet(later):
    two = json.load(open(ROOT / "results" / "hnet" / "hnet_check_2stage_gaps.json"))
    both = json.load(open(ROOT / "results" / "hnet" / "hnet_check_both_models.json"))
    m2, m1 = two["models"]["hnet_2stage_XL"], both["models"]["hnet_1stage_XL"]
    assert two["n_problems"] == 1292 and m2["s1_res"]["n"] == 3547
    has(later, f"(a chunk start at {pct(m2['s1_all']['mean'])}% of bytes) starts a chunk at the `=` two bytes before "
               f"{pct(m2['s1_res_gap2']['mean'])}% of the computed results, and at the result's own first byte for "
               f"{r(100 * m2['s1_res']['mean'], 2)}% of them; it starts a chunk at the space before "
               f"{pct(m2['s1_fin_gap1']['mean'])}% of final answers",
        f"In the 1-stage model ({pct(m1['s0_all']['mean'])}% of bytes), a chunk starts at a result's first byte for "
        f"{pct(m1['s0_res']['mean'])}% of computed results",
        f"(teacher-forced exact match {pct(m1['exact_res_covered']['mean'])}% of {m1['exact_res_covered']['n']} against "
        f"{pct(m1['exact_res_uncovered']['mean'])}% of {m1['exact_res_uncovered']['n']:,}; correlational)")
    assert m1["s0_res_sp"]["mean"] == 1.0
    paper1 = (ROOT / "paper" / "manuscript.md").read_text()
    assert "hnet_check" not in paper1 and "outer chunk" not in paper1      # paper 1 has no H-Net measurement


# ----------------------------------------------------------------------------- Appendix H
def test_appendix_h(later):
    e = json.load(open(ROOT / "results" / "enrichment.json"))
    names = {"entropy10": "entropy", "dep10": "dependence", "syntax+entropy10": "hand-written rule",
             "nov10": "novelty table", "novc13": "corpus-fitted table"}
    runs50 = [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "results" / "scale" / "runs" / "50m_syntax+entropy10_s*" / "result.json")))]
    assert len(runs50) == 2 and not any((x.get("cfg") or {}).get("clip") for x in runs50)
    for k, name in names.items():
        x = e[k]
        cov_c = "100%" if x["coverage"]["computed"] == 1 else f"{pct(x['coverage']['computed'])}%"
        cov_f = f"{r(100 * x['coverage']['final'], 0)}%"
        if x.get("acc50m_clip"):
            acc = (pct(x["acc50m_clip"]["computed_acc"]), pct(x["acc50m_clip"]["final_acc"]))
        else:
            acc = (r(Fraction(sum(round(1000 * o["computed_acc"]) for o in runs50), 20)),
                   r(Fraction(sum(round(660 * o["final_acc"]) for o in runs50), 2 * 660) * 100))
        line = f"| {name} | {pct(x['rate'])}% | {r(x['B'], 2)} | {cov_c} | {cov_f} | {acc[0]} | {acc[1]} |"
        assert line in later, line
    has(later, f"{r(e['surprisal_at_computed_starts'], 2)} bits at computed-result starts and "
               f"{r(e['surprisal_at_final_starts'], 2)} at final-answer starts")
    m = j("paper2v2_intervals.json")["mechanism"]
    has(later, "at BLT-1B's {}th surprisal percentile and the table's at the {}th (difference {} points, {} to {}".format(
        pct(m["entropy_sel_s_large_pct"]["value"]), pct(m["nov_sel_s_large_pct"]["value"]),
        pct(m["diff_s_large_pct_entropy_minus_nov"]["value"]), *(pct(x) for x in m["diff_s_large_pct_entropy_minus_nov"]["ci"])),
        "{}% and {}% of their selections".format(pct(m["entropy_sel_new_share"]["value"]), pct(m["nov_sel_new_share"]["value"])),
        "(the table slightly more, {} points, {}; {} among numbers, {})".format(
            sg(100 * m["diff_new_share_nov_minus_entropy"]["value"]), ci_only(m["diff_new_share_nov_minus_entropy"]),
            sg(100 * m["diff_new_share_numbers_nov_minus_entropy"]["value"]), ci_only(m["diff_new_share_numbers_nov_minus_entropy"])))
    rs = j("reducible_stripped.json")
    has(later, f"against {pct(rs['tok_new']['all bytes']['share_new'])}% of all positions",
        "computed-result starts sit at the {}rd and word starts at the {}th".format(
            r(100 * rs["plane"]["computed"]["s_large_pct"], 0), r(100 * rs["plane"]["word"]["s_large_pct"], 0)),
        "(Spearman {} for the table, {} for entropy)".format(r(rs["spearman"]["nov"]["with_lz_true"], 2),
                                                            r(rs["spearman"]["ent_small"]["with_lz_true"], 2)),
        "covers {}% of computed-result starts".format(pct(rs["coverage_top10"]["lz_true (oracle)"]["computed"])))
    lb = {d: json.load(open(ROOT / "results" / "learned_boundary" / f"{d}.json"))["novelty_table"] for d in ("math", "code")}
    ls = lb["math"]["line/statement start"]
    has(later, "(at a {}% rate) starts a patch at {}% of line starts, which are {}% of its boundaries against {}% of bytes".format(
        pct(lb["math"]["rate"]), r(100 * ls["covered"], 0), r(100 * ls["share"], 0), r(100 * ls["chance"], 0)),
        "{}% of its boundaries fall inside identifiers ({}% of bytes)".format(
            r(100 * lb["code"]["mid-identifier"]["share"], 0), r(100 * lb["code"]["mid-identifier"]["chance"], 0)))
    assert lb["code"]["line/statement start"]["covered"] == 0


# ----------------------------------------------------------------------------- conclusion, changes, code and data
def test_conclusion_agrees(later):
    w, h = j("blt_waste.json")["curves"], j("blt_hybrid.json")["curves"]
    has(later, "raises exact match on computed values from {}% to {}% at a 10% budget, with no detectable difference in "
               "whole-text bits per byte".format(pct(w["entropy@0.1"]["acc_res"]), pct(h["hybrid(0.03)@0.1"]["acc_res"])))
    concl = flat(later.split("## 9. Conclusion", 1)[1].split("## Changes", 1)[0])
    for frag in ("teacher-forced", "no detectable difference", "without the `= ` cue", "closed", "not run", "tight"):
        assert frag in concl, frag


def test_changes_quote_version_1(later):
    """Every quotation in 'Changes in version 2' is version 1's own wording (from git history, when available)."""
    import subprocess
    try:
        v1 = subprocess.run(["git", "-C", str(ROOT), "show", "1985984:paper2/manuscript.md"], capture_output=True,
                            text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("version 1's text is not available from git here")
    v1 = flat(v1).lower()
    ch = later.split("## Changes in version 2", 1)[1].split("## Code and data", 1)[0]
    quotes = re.findall(r'"([^"]+)"', flat(ch))
    assert len(quotes) >= 20
    missing = [q for q in quotes if q.lower() not in v1]
    assert not missing, missing


def test_changes_numbers(later, runs):
    ch = later.split("## Changes in version 2", 1)[1].split("## Code and data", 1)[0]
    d, lo, hi = wpair(runs, "50m", "nov10", "novc13", "final_acc", clip=(True, True))
    has(ch, f"(final answers {sg(d)} points, {interval(lo, hi)})", "the table's mean is 71.5%",
        "a start at a computed value buys {} bits".format(r(j("blt_quadrant.json")["by_kind"]["computed"]["gain_bits"], 2)),
        "falls from 73.6%")
    assert cell(runs, "12m", "nov10", "final_acc", False).startswith("71.5 ")
    assert r(j("v2_training_numbers.json")["t4_reruns"]["mean over the two re-runs"]["nov10"]) == "73.6"


def test_code_and_data(later):
    cd = flat(later.split("## Code and data", 1)[1].split("## References", 1)[0])
    for frag in ("release v2.1.0 (doi:10.5281/zenodo.24023934)", "`paper2/v2/README.md`", "`tests/test_paper2_numbers.py`",
                 "**Funding:** none", "**Competing interests:** none", "**Use of AI.**", "`claude-opus-5-5`"):
        assert frag in cd, frag
    assert (ROOT / "paper2" / "v2" / "README.md").exists()
    n = {f: sum(1 for _ in open(ROOT / "paper2" / "v2" / "data" / f)) for f in
         ("prose_rewrites/train.jsonl", "prose_rewrites/test.jsonl", "prose_rewrites_fresh/test.jsonl")}
    assert (n["prose_rewrites/train.jsonl"], n["prose_rewrites/test.jsonl"], n["prose_rewrites_fresh/test.jsonl"]) == (1382, 275, 915)
    c = j("prose_rewrite_cost.json")
    total = sum(c[k]["usd"] for k in ("pilot", "batch", "retry", "fresh_batch", "fresh_retry") if isinstance(c.get(k), dict) and "usd" in c[k])
    assert round(total) == 11
    has(cd, "1,382 training and 275 test problems", "(221 test problems with 711 computed values)",
        "915 of 1,019 new test problems", "about $11")
    p = j("blt_prose.json")["verdicts"]
    assert (p["n_problems"], p["n_values"]) == (221, 711)


def test_referenced_files_exist(full):
    """Every repository path the text names exists in this release."""
    for path in set(re.findall(r"`((?:paper2|results|scale|prereg|tests)/[^`\s]*?)`", full)):
        if "*" in path:
            assert glob.glob(str(ROOT / path)), path
        else:
            assert (ROOT / path).exists(), path
    rules = ROOT / "paper2" / "v2" / "rules"
    for name in set(re.findall(r"`(?:rules/)?([A-Za-z0-9_]+_(?:RULES|NOTE|RULE)\.md)`", full)):
        assert (rules / name).exists(), name
    for name in set(re.findall(r"`([a-z0-9_]+(?:\*)?\.json)`", full)):
        assert glob.glob(str(V2 / name)) or glob.glob(str(ROOT / "results" / "**" / name), recursive=True), name


# ============================================================================= restatements
# The abstract, the introduction and the summaries repeat numbers that the sections above test where they are first
# reported. These tests recompute each repeated number at the place where it is repeated, so a repeated copy cannot
# drift from its source.
def p1_blt1b():
    """Paper 1's BLT-1B layouts (its Table 1): layout -> (rate, covered, computed, final)."""
    rows = [x.strip().split(",") for x in open(ROOT / "paper" / "tables" / "table1_blt1b.csv")][1:]
    return {x[0]: x[1:] for x in rows}


def default_rate(p1):
    """BLT-1B's own layout's patch rate on GSM8K (paper 1's 'default' row)."""
    return float(next(v for k, v in p1.items() if k.startswith("default"))[0])


def test_abstract_numbers(head, runs):
    a = flat(head.split("## Abstract\n\n", 1)[1].split("\n\n## 1.", 1)[0])
    cb = j("blt_cue_budget.json")
    P, t = cb["patcher"], cb["thresholds_bits"]
    dol, eq = (P[f"fresh|orig|res|{k}"]["covered_at_rates"]["0.1"] for k in ("$", "= "))
    assert all(abs(t[w]["0.1"] - math.log2(10)) < 0.1 for w in ("orig", "prose"))   # "capped near log2 10 bits"
    v3 = j("blt_prose.json")["verdicts"]["V3_nov_minus_entropy"]
    c = wpair(runs, "50m", "entropy10", "nov10", "computed_acc", clip=(True, True))
    f = wpair(runs, "50m", "entropy10", "nov10", "final_acc", clip=(True, True))
    e2e = max(round(x["e2e_acc"] * x["e2e_n"]) for x in runs.values())
    assert {x["e2e_n"] for x in runs.values()} == {100}
    for size in TRAIN:                                                     # 3 seeds per rule in Table 1
        for rule in ("entropy10", "dep10", "nov10", "novc13", "syntax+entropy10"):
            n = len(arm(runs, size, rule, TRAIN[size]))
            assert n in (3, 0) and (n == 3 or (size, rule) in (("1m", "novc13"), ("50m", "syntax+entropy10"))), (size, rule)
    s1 = cb["verdicts"]["S1"]
    T = j("v2_blt1b_numbers.json")["prose_by_cue"]["split"]
    assert T["prose|nov|$"]["covered"] == 1.0 and T["prose|nov|space"]["covered"] == 0.0
    assert c[1] > 0 and f[1] > 0
    d1 = cb["verdicts"]["D1"]                                              # "partly because ... more starts per value"
    assert d1["verdict"] == "mixed" and 0 < d1["s_B"]["ci"][0] and d1["s_B"]["ci"][1] < 1
    for frag in (f"Entropy then starts a patch at {r(100 * dol, 0)}% of values after a dollar sign and {r(100 * eq, 0)}% "
                 "after `= `",
                 f"raises the {r(100 * eq, 0)}% to {r(100 * (eq + s1['value']), 0)}%",
                 "it covers every value after a dollar sign and none after a plain space",
                 "its margin over entropy falls from {} to {} points, partly because prose gives entropy more starts "
                 "per value".format(sg(100 * v3["orig"]["value"]), sg(100 * v3["prose"]["value"])),
                 "acts as a learned detector of cues that tend to precede new values in its fitting text",
                 "the table beats entropy at 50M by {} points on computed results and {} on final answers".format(
                     r(c[0]), r(f[0])),
                 f"no 1M-50M model solves more than {e2e} of 100 GSM8K problems",
                 "(1M-50M parameters, 128-byte context, 3 seeds per rule)",
                 "At 1M and 12M the table ran on a T4 GPU in fp16 and the baselines on an A40 in bf16; at 50M all ran "
                 "on the A40"):
        assert frag in a, frag


def test_introduction_numbers(head, runs):
    intro = head.split("## 1. Introduction", 1)[1].split("## 2. Related work", 1)[0]
    p1 = p1_blt1b()
    cb = j("blt_cue_budget.json")
    t, C1 = cb["thresholds_bits"], cb["verdicts"]["C1"]
    has(intro, f"about {r(100 / 8)}-{r(100 / 6)}% of bytes start a patch",
        f"BLT-1B\nstarts a patch at {pct(default_rate(p1))}% of the bytes",
        f"starts a patch at {p1['entropy@10'][1]} of\ncomputed results and gets {p1['entropy@10'][2]} of them exactly right",
        f"$\\log_2 10 = {r(math.log2(10), 2)}$ bits",
        f"({r(min(t['orig']['0.1'], t['prose']['0.1']), 2)}-{r(max(t['orig']['0.1'], t['prose']['0.1']), 2)} bits)",
        f"misses, {r(100 * C1['value'], 0)}% have a digit\nprobability of at least 0.9",
        f"Without a fixed cue it covers {pct(j('blt_cues.json')['cuefree']['summary']['nov_cf@10|res']['covered'])}% of values")
    c = wpair(runs, "50m", "entropy10", "nov10", "computed_acc", clip=(True, True))
    f = wpair(runs, "50m", "entropy10", "nov10", "final_acc", clip=(True, True))
    f12 = wpair(runs, "12m", "entropy10", "nov10", "final_acc", clip=(False, False))
    v3 = j("blt_prose.json")["verdicts"]["V3_nov_minus_entropy"]
    B, curves = j("blt_waste.json")["B"], j("blt_waste.json")["curves"]
    ratio = curves["table@0.15"]["rate"] / curves["entropy@0.1"]["rate"]
    assert curves["table@0.15"]["bpb_all"] > curves["entropy@0.1"]["bpb_all"]        # the table never gets there
    hy = j("blt_hybrid.json")["paired"]["hybrid(0.03)@0.1 - entropy@0.1"]
    has(intro, "(computed results\n   {} points, {}; final answers {}, {})".format(sg(c[0]), interval(c[1], c[2]), sg(f[0]),
                                                                               interval(f[1], f[2])),
        f"final answers only\n   ({sg(f12[0])}, {interval(f12[1], f12[2])})",
        "falls from {} to {} points".format(sg(100 * v3["orig"]["value"]), sg(100 * v3["prose"]["value"])),
        f"about {r(B['table@0.1 acc_res']['k'], 2)} times the table's patch starts",
        f"more than {r(ratio, 2)}\n   times entropy's", f"gains {sg(100 * hy['acc_res']['diff'])} points on\n   computed results")


def test_prior_work_and_setup_numbers(head, runs):
    p1 = p1_blt1b()
    hand = arm(runs, "50m", "syntax+entropy10", False)
    assert len(hand) == 2 and r(hand[0]["params"] / 1e6) == "53.7"
    has(head, "at 53.7M parameters ({}% of computed\nresults, {}% of final answers; two unclipped seeds)".format(
        *(cell(runs, "50m", "syntax+entropy10", k, False).split(" ")[0] for k in ("computed_acc", "final_acc"))),
        f"({p1['results@10 (hand-written)'][2]} and {p1['entdep@10'][2]} of computed results at a 10% budget)")
    for size, want in (("1m", "1M (width 128, 4 global layers, 131M training bytes)"),
                       ("12m", "12M (320,\n8 layers, 0.5B bytes)"), ("50m", "50M (576, 12 layers, 1.5B bytes)")):
        for x in (x for n, x in runs.items() if n.startswith(size + "_")):
            cfg = x["cfg"]
            nbytes = cfg["steps"] * cfg["bs"] * 128
            got = (cfg["d"], cfg["glayers"], f"{r(nbytes / 1e6, 0)}M" if size == "1m" else f"{r(nbytes / 1e9)}B")
            assert got == {"1m": (128, 4, "131M"), "12m": (320, 8, "0.5B"), "50m": (576, 12, "1.5B")}[size], (size, got)
        has(head, want)
    assert "CTX = 128" in (ROOT / "scale" / "train.py").read_text()
    assert "window=32" in (ROOT / "scale" / "model.py").read_text()
    assert all(x["cfg"].get("clip") == 1.0 for n, x in runs.items() if n.startswith("50m_") and n.endswith("_clip"))
    assert {(x["computed_n"], x["final_n"]) for x in runs.values()} == {(1000, 660)}
    has(head, "gradient clipping at norm 1.0", "for 1,000 computed results and 660 final answers",
        "except the diverged 1M seed (0.0%)")
    S = j("blt_cues.json")["gsm8k"]["summary"]
    assert (S["entropy@10|fin"]["n"], S["entropy@10|res"]["n"]) == (294, 796)
    has(head, "Unless stated otherwise: 294 GSM8K test problems", "(796 computed results)",
        f"Its default\nlayout starts a patch at {pct(default_rate(p1))}% of GSM8K bytes")
    readme = (ROOT / "paper2" / "v2" / "README.md").read_text()
    assert "all 7,473 GSM8K train solutions" in readme
    has(head, "on all 7,473 GSM8K training solutions")
    train = pathlib.Path.home() / ".cache" / "segresearch-gsm8k" / "train.jsonl"     # the public GSM8K train split
    if train.exists():
        assert sum(1 for _ in open(train)) == 7473


def test_section4_restatements(head, full, runs):
    s4 = section(head, "## 4. Results in math", "## 5. Why entropy")
    f12 = wpair(runs, "12m", "entropy10", "nov10", "final_acc", clip=(False, False))
    f50 = wpair(runs, "50m", "entropy10", "nov10", "final_acc", clip=(True, True))
    has(s4, f"moves from {sg(f12[0])} at 12M (unclipped, across devices) to {sg(f50[0])} at 50M (clipped, one device)",
        "({}% vs {}%; -0.2 points".format(*(cell(runs, "50m", a, "computed_acc", True).split(" ")[0] for a in ("novc13", "nov10"))),
        "({} vs {}; +0.031".format(*(cell(runs, "50m", a, "bpb", True).split(" ")[0] for a in ("nov10", "entropy10"))))
    pre = flat((ROOT / "prereg" / "2026-10-13_scale150_bltft.md").read_text())
    for frag in ("145M parameters, 91,500 updates", "A1: nov10 - entropy10, final answers, margin +10 points",
                 "final answers, margin -5", "A4: nov10 - entropy10, bits per byte, upper margin +0.03",
                 "missed by 0.3 points on one criterion", "+9.7 vs the threshold 10",
                 "(entropy10, dep10, nov10, novc13; 12 runs)", "at least 10 points on final answers AND by at least 10 "
                 "points on computed results", "Pre-registration: novelty patching at 150M"):
        assert frag in pre, frag
    assert "A-4" not in pre                                     # A-4 is summarized in the rules file only
    f = section(full, "## Appendix F.", "## Appendix G.")        # Study A's design is reported in Appendix F
    has(f, "at 145M parameters, 3 seeds each, 91,500 updates", "final answers at least 10\npoints higher",
        "no more than 5 points below the 50M gap", "bits per byte within +0.03", "where the\nsize is called \"150M\"",
        "with entropy, boundary dependence and the\ncorpus-fitted table", "each table seed at least 10 points above "
        "dependence on final answers and\non computed results")
    has(s4, "called\n\"150M\" in its preregistration", "before two of its three seeds per rule had\nfinished")
    s7 = section(full, "## 7. Other results", "## 8. Limitations")
    m = re.search(r"Two earlier unclipped runs are public in @vera2026easy \((\d+\.\d)% computed,\s+(\d+\.\d)% final\)", s4)
    assert m and (m.group(1), m.group(2)) == tuple(cell(runs, "50m", "syntax+entropy10", k, False).split(" ")[0]
                                                   for k in ("computed_acc", "final_acc"))
    assert s7


def test_section5_restatements(head):
    s5 = head.split("## 5. Why entropy", 1)[1]
    rs, cb = j("reducible_stripped.json"), j("blt_cue_budget.json")
    assert rs["n_problems"] == 1292 and rs["n_bytes"] == j("patcher_budget_by_kind.json")["n_bytes"]
    assert (cb["n"]["orig"]["fresh"], cb["n"]["orig"]["values"]) == (915, 2448)
    S, R = j("blt_cues.json")["gsm8k"]["summary"], j("blt_cues.json")["gsm8k"]["rates"]
    p1 = p1_blt1b()
    tm = j("blt_waste.json")["curves"]["table+marker@0.1"]
    has(s5, "On 1,292 GSM8K test solutions", "on the 915 later GSM8K test\nproblems of Appendix A (2,448 computed values).",
        "Their coverage by cue and the patcher's digit probabilities had not been computed in any writing, and the `is ` "
        "writing was new; their prose rewrites had been scored for exact match in Appendix A's tests, and in the original "
        "writing they are among the 1,292 solutions above (pooled), so their exact-match results by cue had been seen",
        "gets {}% of them right, against\n{}% for the `=` rule".format(pct(S["nov@10|res"]["exact"]), pct(S["eq@10|res"]["exact"])),
        f"and {pct(S['entropy@10|res']['exact'])}% for entropy",
        "helps little\nat 10% ({}%) and much at 15% ({}%".format(pct(S["space@10|res"]["exact"]), pct(S["space@15|res"]["exact"])),
        "({} at 10% and {} at 15%, against {}% and\n{}%".format(p1["entdep@10"][2], p1["entdep@15"][2],
                                                             pct(S["nov@10|res"]["exact"]), pct(S["nov@15|res"]["exact"])),
        "({}% against {}% for entropy at 10%)".format(pct(S["nov@10|fin"]["exact"]), pct(S["entropy@10|fin"]["exact"])),
        "({}% of bytes) it gets {}% of computed results and {}% of\nfinal answers".format(
            pct(tm["rate"]), pct(tm["acc_res"]), pct(tm["acc_fin"])))
    assert S["nov@10|res"]["covered"] == 1.0 and S["nov@10|fin"]["covered"] == 0.0
    assert abs(R["nov@10"] - 0.1032) < 0.001


def test_section6_to_8_restatements(full, runs, o6b):
    s6 = full.split("## 6. Budget and objective", 1)[1].split("## 7. Other results", 1)[0]
    S = j("blt_cues.json")["gsm8k"]["summary"]
    has(s6, "and {}% of computed results against {}%".format(pct(S["eq@10|res"]["exact"]), pct(S["entropy@10|res"]["exact"])))
    q = j("blt_quadrant.json")["by_kind"]
    has(s6, "A start\nbuys {} bits at computed results and {} at final answers; it costs at the random ticket ({}) and at its pure\n"
            "copy ({};".format(r(q["computed"]["gain_bits"], 2), r(q["final"]["gain_bits"], 2), sg(q["id_new"]["gain_bits"], 2),
                               sg(q["id_copy"]["gain_bits"], 2)),
        "Copied numbers in the solution gain\n{} bits".format(r(q["copied"]["gain_bits"], 2)))
    hy = j("blt_hybrid.json")["paired"]
    has(para(s6, "**Tight budgets only.**"), "the\nhybrid's gain over entropy falls from {} points at 10% to {} at 15%".format(
        sg(100 * hy["hybrid(0.03)@0.1 - entropy@0.1"]["acc_res"]["diff"]),
        sg(100 * hy["hybrid(0.03)@0.15 - entropy@0.15"]["acc_res"]["diff"])))
    has(s6, "(2,000 for the bits-per-byte comparisons computed for this version:\nthe `=` rule and the table against "
            "entropy, and the splits into question and solution)")
    assert "2000 resamples" in j("v2_blt1b_numbers.json")["convention"]
    assert "BOOT = 1000" in (ROOT / "paper2" / "v2" / "scripts" / "realblt_hybrid.py").read_text()
    assert "BOOT = 1000" in (ROOT / "paper2" / "v2" / "scripts" / "realblt_waste.py").read_text()
    L = o6b["j2"]["F1"]["by_layout"]
    V = j("blt_copy_latch_cutoff_verdict.json")
    S_ = j("blt_copy_latch_cutoff_score.json")
    bp = S_["pairs"]["D.L8 vs D.SIVPO620"]["id_copy"]["bits_per_unit"]
    bars = o6b["verdict"]["bars"]
    s7 = section(full, "## 7. Other results", "## 8. Limitations")
    cp = para(s7, "**Starts inside a verbatim copy (Appendix B).**")
    has(cp, "raises exact copies from {}% to {}% (+74.8 points, +71.3 to +78.2; 600\ntexts".format(
        pct(L["D"]["id_copy"]["exact"]), pct(L["D-all"]["id_copy"]["exact"])),
        "recovers {} of those\npoints ({}), and {} at 10% ({})".format(
            sg(100 * bars["C1"]["value"][0]), ci_only(bars["C1"]["value"]), sg(100 * bars["C1e"]["E10.L8 - E10"][0]),
            ci_only(bars["C1e"]["E10.L8 - E10"])),
        "the byte veto then beat L8 under the released conversion at the\ndefault rate (L8 minus the byte veto: {} points of exact copies, {})".format(
            sg(100 * V["primary"][0], 2), ci_only(V["primary"], 2)),
        "on exact copies only (bits per copy show no detectable difference, {}, {})".format(sg(bp[0], 3), ci_only(bp, 3, 1.0)),
        "about 9% of replications tying", "sits on these\ntexts' own optimum", "no claim about costs",
        "new to this work by index", "in texts of 700 tokens or more, removing entropy's starts\non the copy loses every copy")
    assert o6b["verdict"]["reported"]["F5"]["by_length"]["tokens 700-2000"]["D-all"] == 0
    m = j("blt_matched_share.json")["n"]["prose"]
    assert (m["test"], m["values"]) == (915, 2448)
    has(s7, "On the 915 later problems")
    ls = section(full, "## 8. Limitations", "## 9. Conclusion")
    e2e = max(round(x["e2e_acc"] * x["e2e_n"]) for x in runs.values())
    has(ls, f"the 1M-50M models trained from\nscratch solve at most {e2e} of 100 GSM8K problems", "about 1.4 to 16 times",
        "every\nrun in Table 1 was clipped, and every 50M run used the A40", "holds on\nexact copies only")
    assert all(x["device"] == "NVIDIA A40" for n, x in runs.items() if n.startswith("50m_"))
    assert any(not (x.get("cfg") or {}).get("clip") for n, x in runs.items() if n.startswith("50m_"))   # unclipped 50M runs exist


def test_appendix_restatements(later, runs):
    b = flat(later.split("## Appendix B.", 1)[1].split("## Appendix C.", 1)[0])
    V = j("blt_copy_latch_cutoff_verdict.json")
    has(b, "L8 minus SIV+PO<620 is {} points\nof exact copies ({};".format(sg(100 * V["primary"][0], 2), ci_only(V["primary"], 2)),
        "dropping it leaves {}".format(sg(100 * V["primary"][0], 2)))
    c = later.split("## Appendix C.", 1)[1].split("## Appendix D.", 1)[0]
    rate = {(f, x): pct(sum(o["val_rate"] for o in gen(f, x)) / 2) for f, x in
            (("code", "entropy10"), ("code", "novelty"), ("code", "novelty_t10"), ("code", "entropy_t17"))}
    has(c, "at matched evaluation rates ({}% and {}%)".format(rate[("code", "entropy10")], rate[("code", "novelty")]),
        "evaluates at\n{}% of bytes against entropy's\n{}%".format(rate[("code", "novelty_t10")], rate[("code", "entropy_t17")]))
    nn, tb = gen("trace_novnet", "novelty_net"), gen("trace", "novelty")
    has(c, "scores {}%\non computed values and {}% on copied ones, against {}% and {}% with the table".format(
        *(r(mean_sd(gvals(rs, k))[0]) for rs in (nn, tb) for k in ("acc_computed", "acc_copied"))))
    e = json.load(open(ROOT / "results" / "enrichment.json"))
    h = later.split("## Appendix H.", 1)[1]
    has(h, f"($B = {r(e['entropy10']['B'], 2)}$) and starts a patch at {pct(e['entropy10']['coverage']['computed'])}% of\n"
           f"computed-result starts", f"($B = {r(e['nov10']['B'], 2)}$)", f"($B = {r(e['syntax+entropy10']['B'], 2)}$)",
        f"On BLT-1B ({j('reducible_stripped.json')['n_problems']:,} GSM8K test problems)")
    two = json.load(open(ROOT / "results" / "hnet" / "hnet_check_2stage_gaps.json"))
    has(later, "on {:,} GSM8K test solutions without\nannotations ({:,} in-line computed results)".format(
        two["n_problems"], two["models"]["hnet_2stage_XL"]["s1_res"]["n"]))
    has(later, "(at least 60% of final answers with at most 1.70 bits\nper byte)")


def test_appendix_constants(full):
    p1 = p1_blt1b()
    s5 = full.split("## 5. Why entropy", 1)[1].split("## 6. Budget", 1)[0]
    has(s5, f"BLT-1B's own layout starts a patch at {pct(default_rate(p1))}% of bytes")
    a = full.split("## Appendix A.", 1)[1].split("## Appendix B.", 1)[0]
    q = j("blt_matched_share.json")["q"]
    has(a, f"as many starts as there are numbers ({pct(q['prose'], 2)}% of prose positions)")
    has(full, "(AUC 0.77, 0.74 to 0.80; 711 computed values among 3,085 numbers")
    c = full.split("## Appendix C.", 1)[1].split("## Appendix D.", 1)[0]
    rule = flat((ROOT / "paper2" / "v2" / "rules" / "XDOMAIN_RULES.md").read_text())
    assert "spacing 48 0.88 and 0.93" in rule
    assert "one math and one prose TEST document, 12 random positions each" in rule
    has(c, "Gains were measured with 1/48 of positions toggled at once; against single starts (12 positions in one "
           "document each) the grouped gain correlates 0.88 in math and 0.93 in prose")
    assert "50 FIT + 60 TEST documents each" in rule and len(j("xdomain.json")["domains"]) == 6
    has(c, "in six domains with 60 test documents each")


def test_submission_comments_and_counts(full):
    """The Comments line of version 2 carries the paper's counts, the code link and the DOI of release v2.1.0."""
    import shutil
    import subprocess
    sub = (ROOT / "paper2" / "SUBMISSION.md").read_text()
    v2 = sub.split("## Version 2", 1)[1].split("## Version 1", 1)[0]
    comments = v2.split("### Comments", 1)[1].split("```\n", 2)[1].strip()
    assert all(ord(ch) < 128 for ch in comments)
    tables = len(re.findall(r"^\*\*Table [0-9A-H]+\.\*\*", full, re.M))
    figures = len(re.findall(r"^!\[", full, re.M))
    assert (tables, figures) == (8, 2)
    m = re.search(r"(\d+) pages, (\d+) figures, (\d+) tables", comments)
    assert m and (int(m.group(2)), int(m.group(3))) == (figures, tables), comments
    for frag in ("v2 corrects v1's main claims", "final record", "no venue is intended",
                 "Code, logs and results: https://github.com/nicoveraz/segresearch", "release v2.1.0",
                 "doi:10.5281/zenodo.24023934"):
        assert frag in comments, frag
    assert "10.5281/zenodo.24023934" in v2   # the v2.1.0 DOI, filled in after the release (2026-11-06)
    pdf = ROOT / "paper2" / "arxiv.pdf"
    if shutil.which("pdfinfo") and pdf.exists():
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        assert f"Pages:           {m.group(1)}\n" in info or re.search(rf"Pages:\s+{m.group(1)}\n", info), info
    arxiv = (ROOT / "paper2" / "ARXIV_V2.md").read_text()
    assert comments in arxiv, "ARXIV_V2.md repeats the Comments line"


def test_table_fit_follows_the_code(full):
    """The novelty target looks back 127 bytes and the solution table is fitted on 2 MB (reuse_screen.py)."""
    src = (ROOT / "reuse_screen.py").read_text()
    assert "CTX = 128" in src and "for d in range(1, W):" in src and "N_FIT = 2_000_000" in src and "CAP = 16" in src
    has(full, "within the previous 127 bytes (capped at 16)", "the table is fitted on 2 MB of GSM8K and MATH",
        "contexts seen fewer than 20 times")
    assert "np.where(c2 >= 20" in src


# ============================================================================= version 2, second pass
# Numbers that the first pass checked anywhere in the text are checked here inside the section that states them, and
# constants that were hard-coded are read from their files.
def test_setup_constants_in_place(head, runs):
    s3 = section(head, "## 3. Method and setup", "## 4. Results in math")
    rates = [runs[f"12m_{a}_s0"]["rate"] for a in ("entropy10", "dep10", "syntax+entropy10", "nov10", "novc13")]
    has(s3, f"which gives\n{pct(min(rates))}-{pct(max(rates))}% of evaluation bytes")
    assert {(x["computed_n"], x["final_n"]) for x in runs.values()} == {(1000, 660)}
    has(s3, "for 1,000 computed results and 660 final answers (the first 660\nGSM8K test problems)")
    prep = (ROOT / "scale" / "prep.py").read_text()
    assert "b[s:s + 200_000]" in prep and "len(b) - 200_000, 20)" in prep
    has(s3, f"on {r(20 * 200_000 / 1e6, 0)} MB sampled from the pretraining corpus")
    S = j("blt_cues.json")["gsm8k"]["summary"]
    rs, cb = j("reducible_stripped.json"), j("blt_cue_budget.json")
    pb = j("patcher_budget_by_kind.json")["by_kind"]["computed"]["share_above_p90"]
    pr = j("blt_prose.json")["by_layout"]["orig|entropy|res"]["covered"]
    eq = cb["patcher"]["fresh|orig|res|= "]["covered_at_rates"]["0.1"]
    has(s3, "1,292 test solutions for the\npatcher's entropy", "221 test problems with 711\ncomputed values",
        "915 later test problems with 2,448 computed values", "60 documents\nper domain",
        "appears as {}% (the 294\nproblems), {}% (the 1,292), {}% (the 221) and {}% of the values after `= ` (the 915)".format(
            r(100 * S["entropy@10|res"]["covered"], 0), pct(pb), pct(pr), pct(eq)))
    assert rs["n_problems"] == 1292 and cb["n"]["orig"]["values"] == 2448


def test_section5_constants_in_place(head):
    s5 = section(head, "## 5. Why entropy")
    src = (ROOT / "paper2" / "v2" / "scripts" / "realblt_cues.py").read_text()
    m = re.search(r"cuefree_problems\((\d+), seed=1\)", src)
    assert m and "def main(N=300)" in src and "cuefree_problems(N, seed=2)" in src
    n = j("blt_cues.json")["cuefree"]["summary"]["nov_cf@10|res"]["n"]
    has(para(s5, "**Templated text without a fixed cue.**"), f"table on {int(m.group(1)):,} of them. On 300 test problems "
        f"({n:,} values")
    cap = para(s5, "**Table 2.**")
    S = j("blt_cues.json")["gsm8k"]["summary"]
    has(cap, f"{S['entropy@10|fin']['n']} GSM8K test problems without annotations, {S['entropy@10|res']['n']} computed results")
    has(para(s5, "**What the table is.**"), "starts a patch at neither (Appendix C)", "(Table H1)")


def test_introduction_cue_contrasts(head):
    intro = section(head, "## 1. Introduction", "## 2. Related work")
    cb = j("blt_cue_budget.json")
    P, S1 = cb["patcher"], cb["verdicts"]["S1"]
    sp = P["fresh|prose|res|space"]["covered_at_rates"]["0.1"]
    eq = P["fresh|orig|res|= "]["covered_at_rates"]["0.1"]
    has(intro, f"entropy covers {pct(sp)}% of the values that\nfollow a plain space",
        f"raises entropy's coverage of those values from {pct(eq)}% to {pct(eq + S1['value'])}%")
    kb = j("v2_blt1b_numbers.json")["k_bracket"]["k_range_from_bracketing_points"]
    v = j("v2_blt1b_numbers.json")["bpb_vs_entropy"]
    has(intro, "the interpolation alone\n   allows {} to {}".format(r(kb[0], 2), r(kb[1], 2)),
        "move from the\n   solution ({}) to the question ({})".format(sg(v["hybrid(0.03)@0.1 - entropy@0.1 | sol"]["diff"], 3),
                                                                   sg(v["hybrid(0.03)@0.1 - entropy@0.1 | q"]["diff"], 3)))
    T = j("v2_blt1b_numbers.json")["prose_by_cue"]["split"]
    assert T["prose|nov|$"]["covered"] == 1.0 and T["prose|nov|space"]["covered"] == 0.0
    has(intro, "it starts a patch at\nevery value after `$` and at none after a plain space")


def unit_wins():
    """For every rule of the six-domain tests: the domains where it beats entropy on whole units (interval above 0)."""
    out = {}
    X = j("xdomain.json")["domains"]
    for d, v in X.items():
        for k, x in v["unit"]["minus_E"].items():
            out.setdefault(f"X:{k}", []).extend([d] if x["ci"][0] > 0 else [])
    P = j("xdomain_probe.json")["domains"]
    for d, v in P.items():
        for k, x in v["unit"]["minus_E"].items():
            out.setdefault(f"P:{k}", []).extend([d] if x["ci"][0] > 0 else [])
    N = j("xdomain_newtok.json")["verdicts"]
    for k in ("V1_unit", "V2_unit"):
        out[f"N:{k}"] = list(N[k]["domains"])
    return out


def test_no_rule_beat_entropy_on_units_in_more_than_one_domain(full):
    w = unit_wins()
    assert max(len(v) for v in w.values()) == 1, w
    assert w["X:H"] == ["code"] and w["P:P_dom"] == ["traces"]
    phrase = "no rule we tested beat entropy on exact match of whole words and numbers in more than one domain"
    for start, end in (("## Abstract", "## 1. Introduction"), ("## 8. Limitations", "## 9. Conclusion"),
                       ("## 9. Conclusion", "## Changes in version 2")):
        has(section(full, start, end), phrase)
    assert "general rule across six domains" not in flat(full) and "best general rule" not in flat(full)


def test_copy_pools(later):
    b = section(later, "## Appendix B.", "## Appendix C.")
    o6b = flat((ROOT / "paper2" / "v2" / "rules" / "O6b_COPY_LATCH_RULES.md").read_text())
    o6c = flat((ROOT / "paper2" / "v2" / "rules" / "O6c_CUTOFF_RULES.md").read_text())
    assert "about 2,650 distinct fresh problems, 47% of the pool" in o6b
    assert "554 of the 1,220 when the draws are replayed" in o6c and "idx 6253 to the end" in o6c
    has(b, "(47% of O6b's pool and 554 of the 1,220 problems of O6c's", "(O6b from index 1,800, O6c from index 6,253)",
        "*SIV+PO* (a segment-initial veto plus a pointer-onset veto)", "O6b and O6c (the test names used in the rules files)")
    rules = (ROOT / "paper2" / "v2" / "analysis" / "2026-10-30_copyfix" / "bytes" / "final_rules.py").read_text()
    assert "segment-initial veto" in rules and "pointer onset" in rules


def test_appendix_constants_in_place(later):
    a = section(later, "## Appendix A.", "## Appendix B.")
    learned = (ROOT / "paper2" / "v2" / "scripts" / "realblt_learned.py").read_text()
    pca = int(re.search(r"^R, Q, NEG, PCA, LAM, NBOOT = [^,]+, [^,]+, [^,]+, (\d+),", learned, re.M).group(1))
    has(a, f"reduced to {pca}\nprincipal components")
    pv = (ROOT / "paper2" / "v2" / "scripts" / "patcher_variants.py").read_text()
    assert '"V3": dict(d=64, layers=2' in pv and '"V0": dict(d=32, layers=1' in pv and "DEV = " in pv
    has(a, "larger patcher (width 64, 2 layers)", "one-layer, width-32 patcher", "one seed per variant, on the laptop GPU",
        "the 275 test\nproblems of §5 (221 of them with computed values; 711 values)")
    c = section(later, "## Appendix C.", "## Appendix D.")
    nn = json.load(open(ROOT / "results" / "generic" / "trace" / "novnet.json"))["2-byte table"]
    assert nn["computed value"]["covered"] == 0 == nn["copied value"]["covered"] and nn["line start"]["covered"] == 1.0
    has(c, "the two-byte table starts a patch at none of either and at every line start\n"
           "(`results/generic/trace/novnet.json`, after the data)")
    f = section(later, "## Appendix F.", "## Appendix G.")
    has(f, "Planned, with no decision rules written: the detector on prose from a second writer")
    designed = f.split("Planned, with no decision rules written", 1)[0]
    assert "BLT-7B" not in designed and "second writer" not in designed
    e = section(later, "## Appendix E.", "## Appendix F.")
    has(e, "where every cue rule beats entropy on final answers\nat 12M (the table only in a comparison across devices)")


def test_appendix_c_and_section7_dependence_and_scope(later, runs):
    s7 = section(later, "## 7. Other results", "## 8. Limitations")
    cb = j("blt_cue_budget.json")["diff"]
    has(section(later, "## Appendix A.", "## Appendix B."), "(+3.8, +1.1 to +6.4), at +0.096 bits per byte")
    assert ivl(cb["prose|dep-E|fin"]) == "+3.8, +1.1 to +6.4" and sg(cb["prose|dep-E|bpb_whole"]["value"], 3) == "+0.096"
    w = unit_wins()
    has(s7, "no rule beat it in more than one", "beat entropy on bits in five of six domains (exploratory, no verdict set")
    assert max(len(v) for v in w.values()) == 1


def test_changes_second_pass(later, runs):
    ch = section(later, "## Changes in version 2", "## Code and data")
    e = json.load(open(ROOT / "results" / "enrichment.json"))
    note = next(x for x in (ROOT / "notes.md").read_text().splitlines() if "Boundary enrichment (SOMBRERO" in x)
    m = re.search(r"\((\d\.\d\d) vs (\d\.\d\d) bits all bytes\)", note)
    assert m and m.group(1) == r(e["surprisal_at_computed_starts"], 2)
    pb = j("patcher_budget_by_kind.json")
    has(ch, f"({m.group(1)} bits at computed-result starts against {m.group(2)} over all bytes;",
        "(2.90 against 1.11 bits)")
    assert r(pb["by_kind"]["computed"]["mean_entropy"], 2) == "2.90" and r(pb["mean_entropy_all"], 2) == "1.11"
    mech = j("paper2v2_intervals.json")["mechanism"]
    has(ch, "({}% and {}% of their selections; the table slightly more".format(
        pct(mech["nov_sel_new_share"]["value"]), pct(mech["entropy_sel_new_share"]["value"])))
    assert mech["diff_new_share_nov_minus_entropy"]["ci"][0] > 0
    t12 = cell(runs, "12m", "typed14", "bpb", False).split(" ")[0]
    e12 = cell(runs, "12m", "entropy10", "bpb", False).split(" ")[0]
    d, lo, hi = wpair(runs, "50m", "entropy10", "typed14", "bpb", clip=(True, True))
    has(ch, f"at 12M its bits per byte ({t12}) are above entropy's ({e12})", f"({sg(d, 3)}, {interval(lo, hi, 3)})")
    assert float(t12) > float(e12) and lo < 0 < hi
    lb = json.load(open(ROOT / "results" / "learned_boundary" / "math.json"))["novelty_table"]["line/statement start"]
    has(ch, f"line starts in math is {r(100 * lb['share'], 0)}% (version 1: 10%")
    src = (ROOT / "reuse_screen.py").read_text()
    w = int(re.search(r"^CTX = (\d+)", src, re.M).group(1))
    has(ch, f"a window of the previous {w - 1} bytes; version 1 said 4 MB and {w}")
    assert (ROOT / "results" / "generic" / "code_acc").is_dir()
    for verdict in ("*Withdrawn.*", "*Corrected.*", "*Qualified.*"):
        assert verdict in ch, verdict
    assert "No experiments were run after that." in flat(ch)


def test_code_and_data_scope(later):
    cd = flat(section(later, "## Code and data", "## References"))
    c = j("prose_rewrite_cost.json")
    assert (c["fresh_batch"]["n"], c["fresh_batch"]["valid"]) == (1019, 915)
    has(cd, f"{c['fresh_batch']['valid']} of {c['fresh_batch']['n']:,} new test problems",
        "checks the paper's result numbers against those files", "the scripts and results behind every table, figure and "
        "number", "the preregistration with amendments A-1 to A-3", "Study A's 145M runs on Kaggle T4 GPUs (fp16)",
        "the small-patcher variants of Appendix A")
    assert "checks every number" not in cd and "rebuild every table" not in cd
    pre = (ROOT / "prereg" / "2026-10-13_scale150_bltft.md").read_text()
    assert "A-3 (supersedes" in pre and "A-4" not in pre


def test_gzipped_result_is_the_run_output():
    """blt_cue_budget.json ships gzipped (13 MB as written); it decompresses to the run's file, byte for byte."""
    import gzip
    import hashlib
    raw = gzip.open(V2 / "blt_cue_budget.json.gz", "rb").read()
    assert hashlib.sha256(raw).hexdigest() == "1726a63f57eabf2c7c6bf40adbd3455ba2e6e7002996e90377a7848d206d7512"
    readme = (ROOT / "paper2" / "v2" / "README.md").read_text()
    assert "1726a63f57eabf2c7c6bf40adbd3455ba2e6e7002996e90377a7848d206d7512" in readme
    big = [f for f in V2.rglob("*") if f.is_file() and f.stat().st_size > 10_000_000]
    assert not big, big



# ============================================================================= third pass (verifier issues of 2026-11-02)
def test_framing_is_scoped_to_cues_that_precede_new_values(full, head, later):
    """The table detects cues that tend to precede new values in its fitting text, not every fixed cue: in traces the
    same `-> ` precedes computed and copied values and the table covers neither."""
    T = json.load(open(ROOT / "results" / "generic" / "trace" / "novnet.json"))["2-byte table"]
    assert T["computed value"]["covered"] == 0.0 and T["copied value"]["covered"] == 0.0
    assert T["line start"]["covered"] == 1.0
    phrase = "cues that tend to precede new values in its fitting text"
    abstract = section(head, "## Abstract", "## 1. Introduction")
    intro = section(head, "## 1. Introduction", "## 2. Related work")
    s5 = para(head, "**What the table is.**")
    concl = section(later, "## 9. Conclusion", "## Changes in version 2")
    for part in (abstract, intro, s5, concl):
        has(part, phrase)
    has(s5, "It does not detect every\nfixed cue", "the same `-> ` precedes computed and copied values alike",
        "the table starts a patch at neither (Appendix C)")
    has(intro, "In execution traces, where the same\ncue precedes computed and copied values, it starts a patch at neither")
    has(concl, "in execution traces, where the same cue precedes copies, it starts a patch at no value")
    for bad in ("detector of the cues that precede values", "detector of such cues", "learns that cue."):
        assert bad not in flat(full), bad


def test_cue_contrasts_are_not_called_isolation(head):
    """The within-text contrast compares different values and the `is ` swap adds a byte: they point to the cue."""
    intro = section(head, "## 1. Introduction", "## 2. Related work")
    assert "isolate the cue" not in flat(head)
    assert len("is ".encode()) - len("= ".encode()) == 1
    has(intro, "Two\ncontrasts point to the cue", "Neither isolates the cue cleanly",
        "the swap adds a byte before each value",
        "falls below entropy, partly\nbecause prose has more bytes per value and so gives entropy more starts per value")
    d1 = j("blt_cue_budget.json")["verdicts"]["D1"]
    assert d1["verdict"] == "mixed"


def test_ceiling_test_problems_were_in_the_pooled_analysis(head):
    """The 915 problems' original writings are among the 1,292 solutions of the pooled entropy analysis."""
    src = (ROOT / "paper2" / "v2" / "scripts" / "reducible2.py").read_text()
    maxt = int(re.search(r"^MAXT = (\d+)$", src, re.M).group(1))
    n_problems = int(re.search(r"def main\(n_problems=(\d+)", src).group(1))
    rows = [json.loads(x) for x in open(ROOT / "paper2" / "v2" / "data" / "prose_rewrites_fresh" / "test.jsonl")]
    assert len(rows) == 915
    assert all(x["idx"] < n_problems and len(x["orig"].encode()) + 1 <= maxt for x in rows)
    assert j("reducible_stripped.json")["n_problems"] == 1292
    s5 = section(head, "## 5. Why entropy", None)
    assert "not used to design this test" not in flat(s5)


def test_section7_byte_veto_is_one_cell(later, o6b):
    """Inside the windows SIV+PO was better than L8 in one paired comparison only; the other four tie."""
    w, rel = o6b["verdict"]["paired"]["windowed"], o6b["verdict"]["paired"]["released"]
    first, rest = w["class rows"][0], w["class rows"][1:]
    assert (first["cell"], first["rate"], first["class"]) == ("F1", "D", "SIV+PO better")
    assert len(rest) == 4 and all(x["class"] == "tie" for x in rest)
    cp = para(section(later, "## 7. Other results", "## 8. Limitations"), "**Starts inside a verbatim copy (Appendix B).**")
    has(cp, "was better than L8 inside the windows in one cell, 4-digit\ntickets at the default rate (L8 minus SIV+PO {} "
            "points, {}; tied in the other four comparisons), and worse\nunder the conversion as released ({}, {})".format(
                sg(100 * first["diff"][0]), ci_only(first["diff"]), sg(100 * rel["class row"]["diff"][0]),
                ci_only(rel["class row"]["diff"])))


def test_appendix_c_math_reason_is_marked_untested(later):
    c = section(later, "## Appendix C.", "## Appendix D.")
    has(c, "In math it is not distinguishable (+0.02, -0.04 to +0.09). A possible reason, not tested, is that")
    assert "so the large gain on computed values of §6 does not show" not in flat(c)
