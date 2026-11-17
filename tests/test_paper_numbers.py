"""The paper must not contradict the results it is derived from (adapted from textca/tests/test_paper_numbers.py).

Tables are generated (make_tables.py -> paper/tables/), so they cannot drift. Numbers in the prose are typed by
hand, and this is where drift happens: the first draft had five wordings that went beyond the data, caught by a
manual reread. Each test recomputes a load-bearing number from results/registry/ or results/items/ and checks
that the manuscript states it. If one fails, the paper is wrong OR the analysis was re-run and the paper has not
caught up: reconcile them, never relax the check.

Version 2 adds tests for every number new in it; their inputs are in this repository (results/, and paper/v2/ for the
results, scripts and data new in version 2; see paper/v2/README.md). Tests that rebuild something from the GSM8K test
set check it against the committed result when ~/.cache/segresearch-gsm8k/test.jsonl is present.

    uv run pytest tests/test_paper_numbers.py
"""
import glob
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
V2 = ROOT / "paper" / "v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(V2 / "scripts"))
import flops  # noqa: E402
import registry  # noqa: E402
import stats  # noqa: E402
import blt1b_numbers  # noqa: E402  (paper/v2/scripts)
import cue_budget_numbers  # noqa: E402
import gsm8k_final_copy  # noqa: E402
import r9_cue_split  # noqa: E402
import r9_numbers  # noqa: E402
import trained_compare  # noqa: E402

MS = ROOT / "paper" / "manuscript.md"
GSM = pathlib.Path(os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl"))


@pytest.fixture(scope="module")
def text():
    s = MS.read_text()
    tables = lambda m: (ROOT / "paper" / "tables" / f"{m.group(1)}.md").read_text()
    return re.sub(r"\{\{table:([^}]+)\}\}", tables, s)


@pytest.fixture(scope="module")
def reg():
    return [r for r in registry.load() if not r.get("smoke") and not r.get("superseded")]


def blt(reg):
    return {r["layout"]: r for r in reg if r.get("experiment") == "blt_budget" and r.get("thresh") == "train" and r.get("window") == 512}


def trained(reg, exp, rule, key):
    v = [r[key] for r in reg if r.get("experiment") == exp and r.get("format") == "mathexp" and r["rule"] == rule]
    assert len(v) >= 3, f"{exp}/{rule}: {len(v)} seeds; the paper claims 3 or more"
    return 100 * float(np.mean(v))


def items(pattern):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "items" / pattern))):
        out[pathlib.Path(f).stem] = json.load(open(f))
    assert out, f"no per-target files match {pattern}"
    return out


def has(text, *frags):
    missing = [f for f in frags if f not in text]
    assert not missing, f"manuscript does not state: {missing}"


def p1(x):
    return f"{x:.1f}"


def sg(x, nd=1):
    """A signed number as the manuscript prints it: '+1.4', '−7.8' (Unicode minus)."""
    return f"{x:+.{nd}f}".replace("-", "−")


def iv(lo, hi, nd=1):
    """An interval as the manuscript prints it: '−7.8 to +4.5'."""
    return f"{sg(lo, nd)} to {sg(hi, nd)}"


def sci(p):
    """A p-value to one significant figure, as the manuscript's LaTeX: '2 \\times 10^{-31}'."""
    e = math.floor(math.log10(p)); m = round(p / 10 ** e)
    if m == 10:
        m, e = 1, e + 1
    return f"{m} \\times 10^{{{e}}}"


# ------------------------------------------------------------------ abstract and §4: BLT-1B at a tight budget
def test_blt1b_headline(text, reg):
    b = blt(reg)
    has(text, f"puts a patch start at {100 * b['entropy@10']['results_covered']:.0f}% of the computed results",
        f"gets {p1(100 * b['entropy@10']['results_exact'])}% of them exactly right",
        f"gets {p1(100 * b['results@10']['results_exact'])}%", f"signal gets {p1(100 * b['entdep@10']['results_exact'])}%",
        f"default layout at {100 * b['default']['rate']:.0f}% of bytes: {p1(100 * b['default']['results_exact'])}%")
    has(text, f"at 15% it covers {100 * b['entropy@15']['results_covered']:.0f}% of them, at 10% only {100 * b['entropy@10']['results_covered']:.0f}%")
    # §4: the default layout's coverage and final answers as the prose restates them (version 1 numbers)
    has(text, f"its patcher covers computed results well ({100 * b['default']['results_covered']:.0f}% of them start a patch)",
        f"which gives {p1(100 * b['default']['final_exact'])}% for the default layout")


def test_blt1b_patch_rates(text, reg):
    """§4: every layout runs at or slightly above its budget; entropy and the hand-written layout have the same printed
    rate at each budget (Table 1), the others span 0.100 to 0.157."""
    b = blt(reg)
    for R in (10, 15):
        for lay in ("entropy", "results", "dep", "entdep"):
            assert b[f"{lay}@{R}"]["rate"] >= R / 100, f"{lay}@{R} runs below budget"
        assert f"{b[f'entropy@{R}']['rate']:.3f}" == f"{b[f'results@{R}']['rate']:.3f}"
    others = [b[f"{lay}@{R}"]["rate"] for lay in ("dep", "entdep") for R in (10, 15)]
    has(text, "All layouts run at or slightly above budget on the test problems",
        f"(patch rates {b['entropy@15']['rate']:.3f} and {b['entropy@10']['rate']:.3f} for entropy and the hand-written layout "
        f"at 15% and 10%, {min(others):.3f} to {max(others):.3f} for the others)")


def test_blt1b_paired_differences(text):
    # from the per-target outcomes (exact); the registry holds accuracies rounded to 0.1 points
    it = items("blt_budget_train_w512.json")["blt_budget_train_w512"]["data"]
    L = it["layouts"]
    acc = lambda lay, key: 100 * np.mean(L[lay][key])
    d = lambda a, b, key="res_exact": p1(acc(a, key) - acc(b, key))
    has(text, f"+{d('results@15', 'entropy@15')} points at 15%", f"+{d('results@10', 'entropy@10')} at 10%",
        f"+{d('entdep@15', 'entropy@15')} points at 15%", f"+{d('entdep@15', 'results@15')}")
    x, y = np.array(L["results@15"]["res_exact"]), np.array(L["entropy@15"]["res_exact"])
    has(text, f"{int((x & ~y).sum())} results right only under *results* vs {int((~x & y).sum())} only under *entropy*")
    # exact McNemar p-values, to one significant figure (version 1 printed 3e-31 for 2.46e-31)
    P = lambda a, b: stats.mcnemar(np.array(L[a]["res_exact"], bool), np.array(L[b]["res_exact"], bool))[0]
    has(text, f"McNemar $p = {sci(P('results@15', 'entropy@15'))}$", f"($p = {sci(P('results@10', 'entropy@10'))}$)")
    # §7, final answers, with intervals that resample problems: tied with the hand-written rule (version 2)
    fpid = list(range(len(L["default"]["fin_exact"])))

    def fin(a, b):
        x, y = np.array(L[a]["fin_exact"], bool), np.array(L[b]["fin_exact"], bool)
        lo, hi = stats.boot(x, y, fpid)
        return 100 * (x.mean() - y.mean()), 100 * lo, 100 * hi, stats.mcnemar(x, y)[0]
    d10, lo10, hi10, p10 = fin("entdep@10", "entropy@10")
    d15, lo15, hi15, _ = fin("entdep@15", "entropy@15")
    h10, hlo, hhi, _ = fin("entdep@10", "results@10")
    has(text, f"{sg(d10)} points at 10% ({iv(lo10, hi10)}; $p = {sci(p10)}$) but not at 15% ({sg(d15)}, {iv(lo15, hi15)})",
        f"it is tied with the hand-written rule ({sg(h10)}, {iv(hlo, hhi)} at 10%)")
    assert lo10 > 0 and lo15 < 0 < hi15 and hlo < 0 < hhi, "the text calls these a gain, no detectable gain and a tie"


def test_blt1b_final_answer_coverage(text, reg):
    """§7: entropy starts a patch at nearly every final answer at 10% and 15% (realblt_typed.py's coverage lines)."""
    cov = {}
    for r in reg:
        if r.get("experiment") == "blt_typed" and r.get("window") == 512:
            m = re.search(r"covered: results ([\d.]+)%, final\s+([\d.]+)%", r["raw"])
            cov[r["layout"]] = (float(m.group(1)), float(m.group(2)))
    has(text, f"at both budgets ({cov['entropy@10'][1]:.1f}% and {cov['entropy@15'][1]:.1f}%); what differs is how many "
              f"computed results it covers ({cov['entropy@10'][0]:.0f}% and {cov['entropy@15'][0]:.0f}%)")


# ------------------------------------------------------------------ §3 and §5: final answers and the cue (version 2)
def test_final_answer_copy(text):
    """§3: a GSM8K final answer mostly repeats the solution's last number (paper/v2/scripts/gsm8k_final_copy.py)."""
    res = json.load(open(V2 / "results" / "gsm8k_final_copy.json"))
    if GSM.exists():                                       # rebuild from the GSM8K test set
        raw = GSM.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        assert hashlib.sha256(raw).hexdigest() == res["sha256"]
        blt294 = [p for p in rows[:300] if gsm8k_final_copy.blt_fits(p)]
        for key, rs, view in (("all_raw", rows, "raw"), ("first660_raw", rows[:660], "raw"), ("blt294_blt", blt294, "blt")):
            s = gsm8k_final_copy.summarize(rs, view)
            assert (s["n"], s["same_digits"], s["tail"]) == (res[key]["n"], res[key]["same_digits"], res[key]["tail"]), key
    a, b = res["all_raw"], res["blt294_blt"]
    assert (a["n"], b["n"]) == (1319, 294)
    has(text, f"in {100 * a['same_digits_share']:.1f}% of the 1,319 test problems ({100 * b['same_digits_share']:.1f}% of our 294; "
              "thousands separators ignored)", f"that number ends a median of {a['tail']['median']:.0f} bytes before the final-answer line")
    # §6: the 660 evaluation problems of the models trained from scratch are the first 660 of the test set (mathexp.py;
    # scale/data.py keeps the same split), read in the released format
    src = (ROOT / "mathexp.py").read_text()
    assert re.search(r"N_VAL_GSM, N_VAL_MATH = \(60, 60\) if SMOKE else \(660, 700\)", src) and "realtext._encode(gsm_te[:N_VAL_GSM])" in src
    f6 = res["first660_raw"]
    assert f6["n"] == 660
    has(text, f"Final answers mostly repeat the solution's last number (§3; {100 * f6['same_digits_share']:.1f}% of the 660 evaluation problems)")


def test_blt1b_cue_split(text):
    """§3 and §5: paper 1's 796 results by the cue before them, an analysis after the fact (paper/v2/scripts/blt1b_numbers.py)."""
    res = json.load(open(V2 / "results" / "blt1b_numbers.json"))
    D = items("blt_budget_train_w512.json")["blt_budget_train_w512"]["data"]
    if GSM.exists():                                       # rebuild the spans exactly as realblt_budget.problem() does
        kinds, pids, used = blt1b_numbers.result_kinds()
        assert used == 294 and pids == D["res_pid"], "spans do not match the per-target file"
        assert {k: int((kinds == k).sum()) for k in ("$", "digit")} == res["n"]
        for lay in D["layouts"]:
            cov = np.array(D["covered"][lay], bool)
            for k in ("$", "digit"):
                assert abs(cov[kinds == k].mean() - res["by_cue"][lay][k]["covered"]) < 1e-12
    e, d = res["by_cue"]["entropy@10"], res["by_cue"]["default"]
    assert e["$"]["covered"] == 0.0 and res["n"]["$"] + res["n"]["digit"] == 796
    has(text, f"({res['n']['$']} are written `= $N`)",
        f"entropy at 10% starts a patch at none of the {res['n']['$']} written `= $N` and at {100 * e['digit']['covered']:.0f}% "
        f"of the {res['n']['digit']} written `= N`")
    # §5: Table 1 counts coverage at the '$'; the default layout rarely starts a patch there, yet gets most such results right
    has(text, f"there the default layout starts a patch at only {100 * d['$']['covered']:.0f}% of the {res['n']['$']} results "
              f"written `= $N`, yet gets {100 * d['$']['exact']:.0f}% of them right, against {100 * d['digit']['exact']:.0f}% of the others")


def test_cue_budget(text):
    """§5: BLT-1B's patcher at the computed values of 915 further GSM8K test problems (paper/v2/results/
    blt_cue_budget_entropy.json, a released subset; paper/v2/scripts/cue_budget_numbers.py, which also checks the
    summaries stored with the rows)."""
    c = cue_budget_numbers.compute()
    th, cov = c["threshold_bits"], c["coverage"]
    C = json.load(open(V2 / "results" / "blt_cue_budget_entropy.json"))
    n_train = C["n"]["train_problems"]
    assert "(300-1,318)" in C["about"] and n_train == sum(1 for _ in open(V2 / "data" / "prose_rewrites" / "train.jsonl"))
    has(text, f"$\\log_2 10 = {c['log2_10']:.2f}$ bits",
        f"{c['problems']} further GSM8K test problems (those among problems 300–1,318 that fit the context and had a valid "
        f"prose rewrite, made for a measurement not reported here; {c['values']:,} computed values)",
        f"thresholds are fitted on the same {n_train:,} GSM8K training problems in the writing tested, and a value counts as "
        "covered when a patch starts at its first digit, after any `$`",
        f"BLT-1B's threshold for a 10% budget lies at about this level ({th['0.1']:.2f} bits)",
        f"median probability of {c['after $']['Pd_median']:.3f} at the first digit after `= $` (median entropy "
        f"{c['after $']['H_median']:.2f} bits), where entropy at 10% starts no patch",
        f"after `= ` the median is {c['after =']['Pd_median']:.3f} ({c['after =']['H_median']:.2f} bits) and entropy covers "
        f"{100 * cov['= ']['0.1']:.1f}% of the values")
    # §1: the patcher is unsure which digit comes next (median entropy at the first digit)
    has(text, f"BLT-1B's patcher is unsure there, at a median of {c['after $']['H_median']:.1f}–{c['after =']['H_median']:.1f} bits")
    assert cov["$"]["0.1"] == 0.0 and abs(th["0.1"] - c["log2_10"]) < 0.05, "no value after '= $' covered; threshold near log2 10"
    assert c["after $"]["H_p99"] < th["0.1"], "after '= $' even the 99th percentile of entropy is below the 10% threshold"
    mm = c["missed_at_10_Pd_ge_0.9"]
    has(text, f"{100 * mm['value']:.1f}% ({100 * mm['ci'][0]:.1f} to {100 * mm['ci'][1]:.1f}) have a digit probability of at least 0.9")
    s = c["is_instead_of_eq"]; d = s["difference"]
    has(text, f"before the {s['n']:,} values written `= N` lowers that median from {s['Pd_median_eq']:.3f} to {s['Pd_median_is']:.3f}",
        f"from {100 * s['covered_eq']:.1f}% to {100 * s['covered_is']:.1f}% at the same budget "
        f"({sg(100 * d['value'])} points, {iv(100 * d['ci'][0], 100 * d['ci'][1])})")
    R = ("0.1", "0.15", "0.2", "0.25")
    has(text, "entropy starts a patch at {}% of computed values at 10% of bytes, {}% at 15%, {}% at 20% and {}% at 25%".format(
            *(f"{100 * cov['all'][r]:.0f}" for r in R)),
        "(thresholds {}, {}, {} and {} bits)".format(*(f"{th[r]:.2f}" for r in R)),
        "after `= $` at {}%, {}%, {}% and {}%".format(*(f"{100 * cov['$'][r]:.0f}" for r in R)),
        f"It starts one at {100 * cov['final']['0.1']:.0f}% of final answers already at 10%",
        f"average patches of 6 and 8 bytes [@pagnoni2025blt], about {100 / 6:.0f}% and {100 / 8:.1f}% of bytes; at 12.5% entropy "
        f"covers {100 * cov['all']['0.125']:.0f}% of computed values")
    # the abstract's figure is for values right after '= ', where a value's first byte is its first digit (both counts agree)
    has(text, f"(entropy covers about {round(100 * cov['= ']['0.2'], -1):.0f}% of the values right after `= ` at 20% of bytes)")


def test_prose_rewrite(text):
    """§5: the same solutions rewritten as prose (paper/v2/results/blt_prose.json from realblt_prose.py; numbers by
    paper/v2/scripts/r9_numbers.py, which reproduces the verdicts stored with the rows)."""
    r = r9_numbers.compute()
    c, v1 = r["covered_entropy"], r["V1_coverage_prose_minus_orig"]
    pt = lambda b: f"{sg(100 * b['value'])}"
    ci = lambda b: iv(100 * b["ci"][0], 100 * b["ci"][1])
    n_train = sum(1 for _ in open(V2 / "data" / "prose_rewrites" / "train.jsonl"))
    has(text, f"a prose rewrite of {r['n_problems']} of the first 300 ({r['n_values']} values)",
        f"thresholds are fitted on the same {n_train:,} GSM8K training problems in the writing tested",
        f"entropy starts a patch at {100 * c['prose']:.1f}% of the values in prose against {100 * c['orig']:.1f}% in the original "
        f"({pt(v1)} points, {ci(v1)})")
    wo, wp = r["with_minus_entropy"]["orig"], r["with_minus_entropy"]["prose"]
    for f in ("orig", "prose"):                            # the layout with a start at every value runs slightly above budget
        assert r["rates"][f"{f}|entropy"] < r["rates"][f"{f}|with"] < 0.11
    has(text, "Adding a start at every value to entropy's layout, slightly beyond the budget, raises exact match by "
              f"{100 * wo['value']:.1f} points in the original ({ci(wo)}) but only by {100 * wp['value']:.1f} in prose ({ci(wp)})")
    g = r["V2_gain_prose"]
    has(text, f"against the same layout without those starts it adds {100 * g['value']:.1f} points ({ci(g)}; "
              f"{100 * r['V2_gain_orig']['value']:.1f} in the original)")
    assert g["ci"][0] > 0, "the text says a start still pays in prose"
    ex, b = r["exact"], r["bytes_per_value"]
    has(text, f"(default layout {100 * ex['prose']['default']:.1f}% against {100 * ex['orig']['default']:.1f}%",
        f"prose has {b['prose_over_orig']:.2f} times as many bytes per value")
    # values after '$' stay uncovered in both writings (r9_cue_split.py, after the fact)
    split = json.load(open(V2 / "results" / "blt_prose_cue_split.json"))
    rows = json.load(open(V2 / "results" / "blt_prose.json"))["rows"]
    recs = {x["idx"]: x for x in map(json.loads, open(V2 / "data" / "prose_rewrites" / "test.jsonl"))}
    n = defaultdict(int)
    for x in rows:
        if x["kind"] == "res" and x["layout"] == "entropy":
            q = recs[x["problem"]]
            if r9_cue_split.cue(q[x["format"]], q[x["format"] + "_spans"][x["j"]]) == "$":
                n[x["format"]] += 1
                assert not x["covered"]
    assert dict(n) == {"orig": split["orig|$"]["n"], "prose": split["prose|$"]["n"]}
    # most of what the added starts gain in prose is at the values after '$' (a start's gain there and after a plain space)
    gain = {k: 100 * (split[f"prose|{k}"]["exact_with"] - split[f"prose|{k}"]["exact_entropy"]) for k in ("$", "space")}
    assert set(split) == {"orig|$", "orig|= ", "prose|$", "prose|space"}
    assert split["prose|$"]["n"] * gain["$"] > split["prose|space"]["n"] * gain["space"], "most of the prose gap is after '$'"
    has(text, f"Most of those {100 * wp['value']:.1f} points come from the {n['prose']} values after `$`",
        f"entropy starts a patch at none of them (nor at any of the original's {n['orig']}), and a start adds "
        f"{gain['$']:.0f} points there, against {gain['space']:.0f} after a plain space, where entropy already covers "
        f"{100 * split['prose|space']['entropy_covered']:.0f}%")


# ------------------------------------------------------------------ §6: trained at tight budgets
def welch_seeds(va, vb):
    """Difference of seed means with stats.compare's Welch 95% interval (degrees of freedom rounded down), for metrics such
    as bits per byte that stats.compare's pooled interval does not cover. Returns (difference, low, high)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    d = va.mean() - vb.mean()
    sa, sb = va.var(ddof=1) / len(va), vb.var(ddof=1) / len(vb)
    se = math.sqrt(sa + sb)
    df = (sa + sb) ** 2 / (sa ** 2 / (len(va) - 1) + sb ** 2 / (len(vb) - 1))
    t = stats.T975.get(max(1, min(8, int(df))), 2.0)
    return d, d - t * se, d + t * se


def test_trained_budget_claims(text, reg):
    T = lambda rule, k: trained(reg, "math_tight_budget", rule, k)
    rows = stats.mlx_rows()
    C = lambda a, b, k, exp="math_tight_budget": stats.compare(rows, a, b, k, experiment=(exp, None))
    w = lambda c: (100 * c["d"], 100 * c["welch"][0], 100 * c["welch"][1])
    c = C("dep10", "entropy20", "final_acc")
    assert abs(100 * c["d"] - (T("dep10", "final_acc") - T("entropy20", "final_acc"))) < 1e-9
    has(text, f"by {p1(100 * c['d'])} points (95% interval over seeds {iv(100 * c['welch'][0], 100 * c['welch'][1])})")
    # the seed means are ordered hand-written > dependence > entropy at every budget, on both accuracies (Table 2's caption
    # says where the seeds overlap; several of these gaps have Welch intervals that include 0, see results/stats.md)
    for R in (10, 15, 20):
        for k in ("final_acc", "computed_acc"):
            assert T(f"syntax+entropy{R}", k) > T(f"dep{R}", k) > T(f"entropy{R}", k), (R, k)
    has(text, "The ordering hand-written > dependence > entropy holds at every budget in the seed means.")
    # from 10% to 20%: entropy's final answers rise; the result-aware rules' final answers and every rule's computed results
    # show no detectable change (version 1: "flat from 10% up", "does not improve with budget")
    d, lo, hi = w(C("entropy20", "entropy10", "final_acc"))
    assert lo > 0
    has(text, f"From 10% to 20%, entropy's final-answer accuracy rises ({T('entropy10', 'final_acc'):.0f}% $\\rightarrow$ "
              f"{T('entropy15', 'final_acc'):.0f}% $\\rightarrow$ {T('entropy20', 'final_acc'):.0f}%; {sg(d)} points, {iv(lo, hi)})")
    tie = {}
    for rule, k in (("dep", "final_acc"), ("syntax+entropy", "final_acc"), ("entropy", "computed_acc"), ("dep", "computed_acc"),
                    ("syntax+entropy", "computed_acc")):
        d, lo, hi = w(C(f"{rule}20", f"{rule}10", k))
        assert lo < 0 < hi, f"{rule} {k}: the paper says no detectable change from 10% to 20%"
        tie[(rule, k)] = f"{sg(d)}, {iv(lo, hi)}"
    has(text, f"while the result-aware rules show no detectable change (dependence {tie[('dep', 'final_acc')]}; "
              f"hand-written {tie[('syntax+entropy', 'final_acc')]}), nor does any rule's computed-result accuracy (entropy "
              f"{tie[('entropy', 'computed_acc')]}; dependence {tie[('dep', 'computed_acc')]}; hand-written {tie[('syntax+entropy', 'computed_acc')]})",
        "we read this as arithmetic capacity capping each rule's level at this size")
    L = lambda rule, k: trained(reg, "math_larger", rule, k)
    d, lo, hi = w(C("words+syntax", "entropy", "computed_acc", "math_larger"))
    assert lo > 0
    has(text, f"gets {p1(L('words+syntax', 'computed_acc'))}% of computed results against {p1(L('entropy', 'computed_acc'))}% for entropy "
              f"({sg(d)} points, {iv(lo, hi)}, 3 seeds)",
        f"word starts alone get {p1(L('words', 'computed_acc'))}%", f"math syntax {p1(L('stride6+syntax', 'computed_acc'))}%")
    S = lambda rule: trained(reg, "math_small", rule, "final_acc")
    has(text, f"gets {p1(S('entropy'))}% of final answers, its jump variant {p1(S('jump'))}%", f"math syntax {p1(S('words+syntax'))}%")


def test_jump_rule(text, reg):
    T = lambda rule, k: p1(trained(reg, "math_tight_budget", rule, k))
    C = lambda k: stats.compare(stats.mlx_rows(), "jump10", "entropy10", k, experiment=("math_tight_budget", None))
    f, c = C("final_acc"), C("computed_acc")
    assert f["welch"][0] > 0 and c["welch"][0] < 0 < c["welch"][1], "the paper says jump recovers final answers, not computed results"
    # (the abstract stated this too until the scaling study; it now says jump helps neither target at 50M, test_scaling)
    has(text, f"recovers final answers ({T('jump10', 'final_acc')}% against {T('entropy10', 'final_acc')}% for entropy) but not computed results "
              f"({T('jump10', 'computed_acc')}% against {T('entropy10', 'computed_acc')}%; {sg(100 * c['d'])} points, "
              f"{iv(100 * c['welch'][0], 100 * c['welch'][1])})")


def test_small_corpus_coverage(text, reg):
    """§6: on held-out text of Table 2's corpus at 10% (results/reuse_screen.json), entropy starts a patch at no final
    answer and no computed result; dependence at no final answer, at 37% of computed results and at every copy after '>>'."""
    c = json.load(open(ROOT / "results" / "reuse_screen.json"))["math"]
    e, d = c["entropy10"], c["dep[math]10"]
    assert e["final"] == e["computed"] == d["final"] == 0.0 and d["copy"] == 1.0
    has(text, "neither entropy nor dependence starts a patch at a final answer (they follow `#### `)",
        f"yet dependence, which starts one at {100 * d['computed']:.0f}% of computed results (entropy: none) and at every copy "
        f"of a result after `>>`, gets {p1(trained(reg, 'math_tight_budget', 'dep10', 'final_acc'))}% of final answers")


def test_parameter_counts(text):
    """The parameter counts stated in the paper, recomputed from the model definition."""
    import subprocess
    out = {}
    for d, gl in ((128, 4), (64, 2)):
        code = ("import mlx.core as mx, mlx.utils, harness; p = harness._init(mx.random.key(0)); "
                "print(sum(v.size for _, v in mlx.utils.tree_flatten(p)), sum(v.size for _, v in mlx.utils.tree_flatten(p['g'])))")
        env = {**__import__("os").environ, "SEGR_D": str(d), "SEGR_GLAYERS": str(gl), "SEGR_POOL": "xattn", "SEGR_LOCAL": "window"}
        n, g = map(int, subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT).stdout.split())
        out[d] = (n, g)
    has(text, f"{out[128][0] / 1e6:.1f}M parameters in all, {out[128][1] / 1e6:.1f}M of them global", f"D = 64 is {out[64][0] / 1e6:.1f}M",
        f"D = 64 ({out[64][0] / 1e6:.1f}M parameters)")


def test_hand_written_rule_bpb(text, reg):
    """§6: the hand-written rule's bits per byte are lower than dependence's at every budget and than entropy's at 15% and
    20%; at 10% they show no detectable difference from entropy's or jump's (Welch intervals over seeds). Version 1 said
    "the lowest bits per byte at every budget", which jump's seed mean contradicts at 10%."""
    bpb = lambda rule: [r["bpb"] for r in reg if r.get("experiment") == "math_tight_budget" and r.get("format") == "mathexp" and r["rule"] == rule]
    for R in (10, 15, 20):
        assert welch_seeds(bpb(f"syntax+entropy{R}"), bpb(f"dep{R}"))[2] < 0, R
        if R > 10:
            assert welch_seeds(bpb(f"syntax+entropy{R}"), bpb(f"entropy{R}"))[2] < 0, R
    e, j = welch_seeds(bpb("syntax+entropy10"), bpb("entropy10")), welch_seeds(bpb("syntax+entropy10"), bpb("jump10"))
    assert e[1] < 0 < e[2] and j[1] < 0 < j[2]
    f = lambda g: f"{sg(g[0], 3)}, {iv(g[1], g[2], 3)}"
    has(text, "The hand-written rule's bits per byte are lower than dependence's at every budget and than entropy's at 15% and 20%, and "
              f"tied with entropy's and jump's at 10% ({f(e)}; {f(j)}), so its gain shows no detectable cost in bits per byte")
    # dependence costs 1.5-7% in bits per byte against entropy, with intervals above 0 at every budget
    rel = [100 * (np.mean(bpb(f"dep{R}")) / np.mean(bpb(f"entropy{R}")) - 1) for R in (10, 15, 20)]
    assert all(welch_seeds(bpb(f"dep{R}"), bpb(f"entropy{R}"))[1] > 0 for R in (10, 15, 20))
    assert round(min(rel), 1) == 1.5 and round(max(rel)) == 7, rel
    has(text, "dependence costs 1.5–7% in bits per byte against entropy.")


# ------------------------------------------------------------------ §7 and §13: the two dependence variants in trained models
def test_dependence_variants_trained(text):
    """§7: every entropy-plus-dependence seed (2) is below every dependence seed (3) on both accuracies, but the Welch
    intervals over seeds include 0 (version 1: "dependence alone is the better variant")."""
    rows = stats.mlx_rows()
    seeds = lambda rule, k: [100 * r[k] for r in rows if r.get("experiment") == "math_tight_budget" and r["rule"] == rule]
    out = {}
    for k in ("computed_acc", "final_acc"):
        assert (len(seeds("entdep10", k)), len(seeds("dep10", k))) == (2, 3)
        assert max(seeds("entdep10", k)) < min(seeds("dep10", k)), k
        c = stats.compare(rows, "dep10", "entdep10", k, experiment=("math_tight_budget", None))
        assert c["welch"][0] < 0 < c["welch"][1], k
        out[k] = f"{sg(100 * c['d'])}" + (" points, " if k == "computed_acc" else ", ") + iv(100 * c["welch"][0], 100 * c["welch"][1])
    m = lambda k: p1(np.mean(seeds("entdep10", k)))
    has(text, f"in the trained models dependence alone scores higher on every seed (Table 2; entropy plus dependence: {m('computed_acc')}% / "
              f"{m('final_acc')}%, 2 seeds), though with two seeds the intervals include 0 ({out['computed_acc']}; {out['final_acc']})",
        "differs from the one that scores higher on every seed in trained models (dependence alone; §7)")


# ------------------------------------------------------------------ §8: BLT-1B adapted to the budget
def test_finetune(text):
    ft = defaultdict(dict)
    for it in items("blt_finetune_*_w512.json").values():
        ft[it["trained_rule"]][it["seed"]] = it
    lay = {"entropy": "entropy@10", "results": "results@10", "entdep": "entdep@10"}
    acc = {r: {s: np.array(ft[r][s]["data"]["layouts"][lay[r]]["res_exact"]) for s in ft[r]} for r in ft}
    assert all(len(v) == 3 for v in acc.values()), "the paper claims three runs per rule"
    m = {r: 100 * np.mean([a.mean() for a in acc[r].values()]) for r in acc}
    has(text, f"({p1(m['entropy'])}% vs {p1(m['entdep'])}%, three runs per rule")
    pooled = lambda a, b: 100 * (np.concatenate([acc[a][s] for s in (0, 1, 2)]).mean() - np.concatenate([acc[b][s] for s in (0, 1, 2)]).mean())
    has(text, f"entdep − entropy is +{p1(pooled('entdep', 'entropy'))} points", f"hand-written − entropy +{p1(pooled('results', 'entropy'))}",
        f"entdep − hand-written +{p1(pooled('entdep', 'results'))}")
    cross = 100 * np.mean([np.mean(ft["entropy"][s]["data"]["layouts"]["entdep@10"]["res_exact"]) for s in (0, 1, 2)])
    has(text, f"under entdep ({p1(cross)}% on average)")


# ------------------------------------------------------------------ §9: which positions; §10: Scratchpad Patching; §11: compute
def test_traces(text):
    """§9: the trace models: ties, with their intervals (window 32, 2 seeds; window 8, 3 seeds); entropy at 10% skips 83%
    of trace values."""
    rows = stats.mlx_rows()
    mean = lambda rule, w: 100 * np.mean([r["computed_acc"] for r in rows if r.get("experiment") == "reason_trained"
                                          and r.get("window", 32) == w and r["rule"] == rule])
    c = stats.compare(rows, "syntax+entropy10", "entropy10", "computed_acc", experiment=("reason_trained", None), window=(32, 32))
    assert (c["na"], c["nb"]) == (2, 2) and c["welch"][0] < 0 < c["welch"][1]
    has(text, f"yet the hand-written rule is tied with it ({p1(mean('syntax+entropy10', 32))}% vs {p1(mean('entropy10', 32))}%, "
              f"2 seeds; {sg(100 * c['d'])} points, 95% interval over seeds {iv(100 * c['welch'][0], 100 * c['welch'][1])})")
    c8 = stats.compare(rows, "syntax+entropy10", "entropy10", "computed_acc", experiment=("reason_trained", None), window=(8, 8))
    assert (c8["na"], c8["nb"]) == (3, 3) and c8["welch"][0] < 0 < c8["welch"][1]
    has(text, f"the hand-written rule is {100 * c8['d']:.1f} points higher ({p1(mean('syntax+entropy10', 8))}% vs "
              f"{p1(mean('entropy10', 8))}%, 3 seeds), with no detectable difference ({iv(100 * c8['welch'][0], 100 * c8['welch'][1])})")
    t = json.load(open(ROOT / "results" / "typed_screen.json"))["reason"]["entropy10"]["computed"]
    has(text, f"entropy skips most values ({100 * (1 - t):.0f}% at 10%)")


def test_scope_numbers(text):
    """§1, contribution 4: values BLT-1B cannot compute (program-trace arithmetic, Table 5) show no detectable gain from a
    patch start at 10% (problem bootstrap, as in results/stats.md); Scratchpad Patching's entropy trigger against random."""
    it = json.load(open(ROOT / "results" / "items" / "blt_trace_train_w512.json"))["data"]
    L, pid = it["layouts"], it["pid"]["computed"]
    x, y = (np.array(L[lay]["computed"]["exact"], bool) for lay in ("oracle@10", "entropy@10"))
    lo, hi = stats.boot(x, y, pid)
    assert lo < 0 < hi
    c = stats.compare(stats.mlx_rows(), "sp16:entropy", "sp16:random", "final_acc", experiment=("scratchpad16", None))
    has(text, f"values the model cannot compute show no detectable gain ({sg(100 * (x.mean() - y.mean()))} points, {iv(100 * lo, 100 * hi)}; §9)",
        f"no detectable difference from random scratchpads on final answers ({sg(100 * c['d'])} points, "
        f"{iv(100 * c['welch'][0], 100 * c['welch'][1])}; §10)")


def test_scratchpad(text, reg):
    sp = lambda rule, k="final_acc": [r[k] for r in reg if r.get("experiment") == "scratchpad16" and r["rule"] == rule]
    for rule in ("sp16:syntax", "sp16:entropy", "sp16:random"):
        assert len(sp(rule)) == 5, f"{rule}: the paper claims 5 seeds"
    m = lambda rule, k="final_acc": 100 * np.mean(sp(rule, k))
    has(text, f"({p1(m('sp16:entropy'))}% vs {p1(m('sp16:random'))}%, 5 seeds;",
        f"answer-start scratchpads give {p1(m('sp16:syntax'))}%, more than six times as much")
    assert m("sp16:syntax") > 6 * max(m("sp16:entropy"), m("sp16:random"))
    c = stats.compare(stats.mlx_rows(), "sp16:entropy", "sp16:random", "final_acc", experiment=("scratchpad16", None))
    has(text, f"{sg(100 * c['d'])} points, Welch 95% interval {iv(100 * c['welch'][0], 100 * c['welch'][1])}")
    # computed results near the floor under every trigger of Table 6
    comp = [m(r, "computed_acc") for r in ("sp16:syntax", "sp16:entropy", "sp16:random", "sp16:none", "sp16:dense8")]
    has(text, f"Computed results stay near the floor under every trigger ({min(comp):.1f}–{max(comp):.1f}%)")
    # the 8.3% answer-start seed has the highest bits per byte of the five, in the range of the entropy and random triggers
    syn = [r for r in reg if r.get("experiment") == "scratchpad16" and r["rule"] == "sp16:syntax"]
    low = [r for r in syn if round(100 * r["final_acc"], 1) == 8.3]
    rest = [r["bpb"] for r in syn if r is not low[0]]
    flat = sp("sp16:entropy", "bpb") + sp("sp16:random", "bpb")
    assert len(low) == 1 and low[0]["bpb"] > max(rest) and min(flat) < low[0]["bpb"] < max(flat)
    has(text, f"one of their seeds stays at 8.3%, with the highest bits per byte of the five ({low[0]['bpb']:.3f}, against "
              f"{min(rest):.3f}–{max(rest):.3f}), in the range of the entropy and random triggers ({min(flat):.3f}–{max(flat):.3f})")


def test_compute(text, reg):
    pb, pp, pt = flops.blt1b()
    has(text, f"the local layers cost {pb / 1e6:.0f} MFLOPs per byte, the global model {pp / 1e6:,.0f} MFLOPs per patch, and the "
              f"entropy model {pt / 1e6:.0f} MFLOPs per byte")
    ref = blt(reg)["default"]["rate"] * pp + pb + pt
    with_ent = lambda r: 100 * (r * pp + pb + pt) / ref
    lookup = lambda r: 100 * (r * pp + pb) / ref
    has(text, f"A 10% budget then costs {with_ent(0.10):.0f}% of the default's forward compute and 15% costs {with_ent(0.15):.0f}%",
        f"{lookup(0.10):.0f}% at 10% and {lookup(0.15):.0f}% at 15%")
    hb, hp, ht = flops.harness()
    has(text, f"a 10% budget saves only {100 - 100 * (0.10 * hp + hb + ht) / (0.25 * hp + hb + ht):.0f}% of compute")
    # dependence alone at 15% against the default layout (version 2: a tie, with its interval; Figure 3 plots the measured rate)
    L = items("blt_budget_train_w512.json")["blt_budget_train_w512"]["data"]
    fpid = list(range(len(L["layouts"]["default"]["fin_exact"])))
    pair = lambda key, pid: (np.array(L["layouts"]["dep@15"][key], bool), np.array(L["layouts"]["default"][key], bool), pid)
    x, y, pid = pair("res_exact", L["res_pid"]); lo, hi = stats.boot(x, y, pid)
    has(text, f"({100 * x.mean():.1f}% against {100 * y.mean():.1f}%; {sg(100 * (x.mean() - y.mean()))} points, {iv(100 * lo, 100 * hi)})")
    assert lo < 0 < hi
    x, y, pid = pair("fin_exact", fpid); lo, hi = stats.boot(x, y, pid)
    has(text, f"it loses {100 * (y.mean() - x.mean()):.1f} points on final answers ({iv(100 * lo, 100 * hi)}; §7)")
    has(text, f"at about {round(lookup(blt(reg)['dep@15']['rate']), -1):.0f}% of its compute (Figure 3)")
    # paper/v2/results/blt1b_numbers.json holds the same comparisons
    v = json.load(open(V2 / "results" / "blt1b_numbers.json"))["pairs"]["dep@15 - default|res_exact"]
    assert abs(v["lo"] - stats.boot(*pair("res_exact", L["res_pid"]))[0]) < 1e-12


# ------------------------------------------------------------------ §6: scaling to 50M parameters (results/scale/)
def scale_runs(recipe):
    """Table 3's runs ('unclipped': no gradient clipping, the rules of Table 3) or the 50M re-runs of entropy10 and dep10
    with gradient clipping ('clipped'). results/scale/runs also holds runs of other rules, some on other hardware, and
    scale.analyze.load() keys runs by size, rule and seed only, so it would mix the two recipes: keep them apart here."""
    from scale import analyze
    runs = defaultdict(dict)
    for f in sorted(glob.glob(str(ROOT / "results" / "scale" / "runs" / "*" / "result.json"))):
        o = json.load(open(f))
        clip = bool(o["cfg"].get("clip"))
        if recipe == "unclipped" and (clip or o["arm"] not in analyze.ARMS + analyze.EXTRA):
            continue
        if recipe == "clipped" and not (clip and o["arm"] in ("entropy10", "dep10")):
            continue
        assert o["seed"] not in runs[(o["size"], o["arm"])], f"two {recipe} runs for {o['size']} {o['arm']} seed {o['seed']}"
        runs[(o["size"], o["arm"])][o["seed"]] = o
    return runs


@pytest.fixture(scope="module")
def scale():
    from scale import analyze
    un, cl = scale_runs("unclipped"), scale_runs("clipped")
    bad = analyze.failed(un)
    ok = {k: {sd: o for sd, o in v.items() if (k[0], k[1], sd) not in bad} for k, v in un.items()}
    return un, ok, bad, cl


def test_scaling(text, scale):
    from scale import analyze
    un, ok, bad, cl = scale
    assert sum(len(v) for v in un.values()) == 28, "Table 3 has 28 runs"
    assert {(s, a, sd) for s, a, sd in bad} == {("1m", "dep10", 0)}, f"the paper names one failed run: {bad}"
    assert all(o["device"] == "NVIDIA A40" for v in un.values() for o in v.values()), "Table 3's runs ran on the A40"
    assert set(cl) == {("50m", "entropy10"), ("50m", "dep10")} and all(len(v) == 3 for v in cl.values())
    assert all(o["device"] == "NVIDIA A40" and o["cfg"]["clip"] == 1.0 for v in cl.values() for o in v.values())
    v = lambda runs, size, arm, k: [100 * r[k] for _, r in sorted(runs[(size, arm)].items())]
    m = lambda runs, size, arm, k: float(np.mean(v(runs, size, arm, k)))
    W = lambda runs, size, k: analyze.welch(v(runs, size, "entropy10", k), v(runs, size, "dep10", k))   # dep - entropy
    ci = lambda g: iv(g[0] - g[1], g[0] + g[1])
    for size in ("12m", "50m"):
        for arm in ("entropy10", "dep10"):
            assert len(ok[(size, arm)]) == 3, f"{size} {arm}: the paper claims 3 seeds"
    # final answers: not detectable at 1.1M, present at 11.7M and 53.7M; computed results: present at 53.7M only
    g = {s: W(ok, s, "final_acc") for s in ("1m", "12m", "50m")}
    assert g["1m"][0] - g["1m"][1] < 0 < g["1m"][0] + g["1m"][1] and all(g[s][0] - g[s][1] > 0 for s in ("12m", "50m"))
    g1c, g12 = W(ok, "1m", "computed_acc"), W(ok, "12m", "computed_acc")
    assert g1c[0] - g1c[1] < 0 < g1c[0] + g1c[1] and g12[0] - g12[1] < 0 < g12[0] + g12[1], \
        "the paper says dependence and entropy separate on computed results only at 53.7M"
    # version 1 read the 53.7M gap as appearing "once the models can compute", but at 11.7M the hand-written rule is already
    # ahead of entropy on computed results (Table 3; paper/v2/results/scale_compare.json): version 2 drops that reading
    h12 = analyze.welch(v(ok, "12m", "entropy10", "computed_acc"), v(ok, "12m", "syntax+entropy10", "computed_acc"))
    sc = json.load(open(V2 / "results" / "scale_compare.json"))["unclipped"]["compare"]["12m|syntax+entropy10-entropy10"]["computed_acc"]
    assert h12[0] - h12[1] > 0 and abs(sc["diff"] - h12[0]) < 1e-9 and abs(sc["lo"] - (h12[0] - h12[1])) < 1e-9
    assert "once the models can compute" not in text
    has(text, "on a 1.9 GB math corpus, it is not detectable at 1M parameters, appears on final answers at 12M and on both targets at 50M",
        "Dependence is ahead of entropy on final answers at 11.7M and 53.7M and on computed results at 53.7M (Figure 2)",
        f"On computed results, dependence and entropy separate only at 53.7M: at 1.1M and 11.7M they show no detectable difference "
        f"({sg(g1c[0])} points, {ci(g1c)}; {sg(g12[0])}, {ci(g12)}), and at 53.7M",
        f"{sg(g['1m'][0])} points at 1.1M (95% interval over seeds {ci(g['1m'])})", f"{sg(g['12m'][0])} at 11.7M ({ci(g['12m'])})",
        f"{sg(g['50m'][0])} at 53.7M ({ci(g['50m'])}; {p1(m(ok, '50m', 'dep10', 'final_acc'))}% against {p1(m(ok, '50m', 'entropy10', 'final_acc'))}%)")
    gc = W(ok, "50m", "computed_acc")
    assert gc[0] - gc[1] > 0
    has(text, f"({sg(g['12m'][0], 0)} points on final answers) and 50M ({sg(g['50m'][0], 0)} on final answers",
        f"and {sg(gc[0], 0)} on computed results; §6)")
    has(text, f"gets {p1(m(ok, '50m', 'dep10', 'computed_acc'))}% of computed results against {p1(m(ok, '50m', 'entropy10', 'computed_acc'))}% "
              f"for entropy and {p1(m(ok, '50m', 'dep10', 'final_acc'))}% of final answers against {p1(m(ok, '50m', 'entropy10', 'final_acc'))}% "
              "(3 seeds each)",
        f"and at 53.7M dependence gets {p1(m(ok, '50m', 'dep10', 'computed_acc'))}% against "
        f"{p1(m(ok, '50m', 'entropy10', 'computed_acc'))}% for entropy ({sg(gc[0])} points, {ci(gc)})")
    # "every dependence seed above every entropy seed": final answers at 12M and 50M, computed results at 50M
    for size, k in (("12m", "final_acc"), ("50m", "final_acc"), ("50m", "computed_acc")):
        assert min(v(ok, size, "dep10", k)) > max(v(ok, size, "entropy10", k)), f"{size} {k}: seeds overlap"
    assert max(v(ok, "1m", "dep10", "final_acc")) > min(v(ok, "1m", "entropy10", "final_acc")), "the paper says the 1.1M seeds overlap"
    bpb = lambda runs, size, arm: [r["bpb"] for _, r in sorted(runs[(size, arm)].items())]
    gb = analyze.welch(bpb(ok, "50m", "entropy10"), bpb(ok, "50m", "dep10"))
    assert gb[0] + gb[1] < 0, "the paper says dependence has lower bits per byte at 53.7M"
    has(text, f"({np.mean(bpb(ok, '50m', 'dep10')):.3f} against {np.mean(bpb(ok, '50m', 'entropy10')):.3f}; {sg(gb[0], 3)}, {iv(gb[0] - gb[1], gb[0] + gb[1], 3)})")
    # the second recipe: gradient clipping at norm 1.0, same GPU type, 3 seeds per rule
    gcf, gcc = W(cl, "50m", "final_acc"), W(cl, "50m", "computed_acc")
    gcb = analyze.welch(bpb(cl, "50m", "entropy10"), bpb(cl, "50m", "dep10"))
    ent_c = v(cl, "50m", "entropy10", "computed_acc")
    for k in ("final_acc", "computed_acc"):
        assert min(v(cl, "50m", "dep10", k)) > max(v(cl, "50m", "entropy10", k)), "the paper says every dependence seed is above"
    assert gcf[0] - gcf[1] > 0 and gcc[0] - gcc[1] < 0 < gcc[0] + gcc[1] and gcb[0] - gcb[1] < 0 < gcb[0] + gcb[1]
    assert "gradient clipping" not in re.search(r"## Abstract\n\n(.*?)\n\n## 1\.", text, flags=re.S).group(1), \
        "the abstract does not report the second recipe (on computed results it gives no detectable difference)"
    has(text, f"dependence again leads entropy at 53.7M on final answers, {p1(m(cl, '50m', 'dep10', 'final_acc'))}% against "
              f"{p1(m(cl, '50m', 'entropy10', 'final_acc'))}% ({sg(gcf[0])} points, {ci(gcf)})",
        "On computed results every dependence seed is again above every entropy seed, but the interval includes 0: "
        f"{p1(m(cl, '50m', 'dep10', 'computed_acc'))}% against {p1(m(cl, '50m', 'entropy10', 'computed_acc'))}% "
        f"({sg(gcc[0])}, {ci(gcc)}; entropy's seeds spread from {min(ent_c):.1f}% to {max(ent_c):.1f}%)",
        f"Bits per byte show no detectable difference ({sg(gcb[0], 3)}, {iv(gcb[0] - gcb[1], gcb[0] + gcb[1], 3)})",
        f"is smaller with gradient clipping, and its interval then includes 0 ({sg(gcc[0])}, {ci(gcc)}, against {sg(gc[0])}, {ci(gc)} without)")
    # jump, the hand-written rule, entropy at 20%
    j = next(iter(ok[("50m", "jump10")].values()))
    has(text, f"{100 * j['final_acc']:.1f}% of final answers and {100 * j['computed_acc']:.1f}% of computed results", f"({j['bpb']:.3f})",
        "the lowest bits per byte of the rules in Table 3 at a 10% budget")
    assert 100 * j["final_acc"] < min(v(ok, "50m", "entropy10", "final_acc")) and 100 * j["computed_acc"] < min(v(ok, "50m", "entropy10", "computed_acc"))
    assert j["bpb"] == min(r["bpb"] for (s, a), rs in ok.items() if s == "50m" and a.endswith("10") for r in rs.values()), \
        "jump: lowest bpb of Table 3's 10% rules"
    hw = list(ok[("50m", "syntax+entropy10")].values())
    assert len(hw) == 2, "the paper says two hand-written seeds at 50M"
    has(text, f"and the hand-written rule {p1(100 * np.mean([r['computed_acc'] for r in hw]))}% (two seeds)")
    e20 = next(iter(ok[("50m", "entropy20")].values()))
    dep_final = v(ok, "50m", "dep10", "final_acc")
    assert min(dep_final) <= 100 * e20["final_acc"] <= max(dep_final) + 1e-9, "the paper says within the range of the dependence seeds"
    has(text, f"(20%, which is {100 * e20['rate']:.1f}% of the evaluation text)", f"entropy gets {100 * e20['final_acc']:.1f}% of final answers (one seed)",
        f"within the range of the dependence seeds at 10% ({min(dep_final):.1f}–{max(dep_final):.1f}%)",
        f"and {100 * e20['computed_acc']:.1f}% of computed results against {p1(m(ok, '50m', 'dep10', 'computed_acc'))}%",
        f"({p1(m(ok, '50m', 'entropy10', 'computed_acc'))}% to {100 * e20['computed_acc']:.1f}%)")
    assert max(r["e2e_acc"] for runs in (un, cl) for rs in runs.values() for r in rs.values()) <= 0.01, "the paper says at most 1 of 100 end to end"
    has(text, "at most 1 of 100 for any run")
    fail = un[("1m", "dep10")][0]["bpb"]
    others = [r["bpb"] for (s, a), rs in ok.items() if s == "1m" for r in rs.values()]
    has(text, f"(bits per byte {fail:.2f} against {min(others):.2f}–{max(others):.2f}", "One of the 28 runs of Table 3",
        "the 28 runs of Table 3 ran on one rented A40 GPU for about $19")
    assert "TBD_" not in text, "placeholders left in the manuscript"


def test_table3_and_figure2_rebuild(text, scale, monkeypatch):
    """Table 3 (paper/tables/table6_scaling.md) is what make_tables.scale_table() writes through the loader of
    paper/v2/scripts/scaling_unclipped.py, which the Declarations name: Table 3's runs alone (make_figures.scaling()
    plots the same runs for Figure 2). scale.analyze.load() itself mixes in the runs with gradient clipping at commit
    1985984; once it keeps the recipes apart, make_tables.py must rebuild Table 3 directly as well."""
    import make_tables
    import scaling_unclipped
    from scale import analyze
    un, ok, bad, cl = scale
    published = (ROOT / "paper" / "tables" / "table6_scaling.md").read_text()

    def table():
        out = {}
        monkeypatch.setattr(make_tables, "write_table", lambda name, header, body: out.update(name=name, header=header, body=body))
        make_tables.scale_table()
        md = "| " + " | ".join(out["header"]) + " |\n|" + "---|" * len(out["header"]) + "\n"
        return out["name"], md + "".join("| " + " | ".join(str(c) for c in row) + " |\n" for row in out["body"])

    real_ok, _ = analyze.runs_ok()
    mixed = any(o["cfg"].get("clip") for v in real_ok.values() for o in v.values())
    if not mixed:                                          # the loader keeps the recipes apart: make_tables.py alone suffices
        assert table() == ("table6_scaling", published)
    monkeypatch.setattr(analyze, "load", scaling_unclipped.load_unclipped)
    got_ok, got_bad = analyze.runs_ok()
    assert got_bad == bad and {k: sorted(v) for k, v in got_ok.items()} == {k: sorted(v) for k, v in ok.items()}
    assert table() == ("table6_scaling", published)
    has(text, "`paper/v2/scripts/scaling_unclipped.py` rebuilds Table 3 and Figure 2 from the runs in `results/scale/` without "
              "gradient clipping")


def test_scaling_coverage(text):
    """§6: on the scaling corpus's evaluation text, entropy at 10% starts a patch at every final answer and dependence at
    none (results/enrichment.json, scale/enrichment.py; the masks of Table 3's rules, same evaluation patch rates)."""
    e = json.load(open(ROOT / "results" / "enrichment.json"))
    assert e["entropy10"]["coverage"]["final"] == 1.0 and e["dep10"]["coverage"]["final"] == 0.0
    for rule in ("entropy10", "dep10", "syntax+entropy10"):
        r = json.load(open(ROOT / "results" / "scale" / "runs" / f"50m_{rule}_s0" / "result.json"))["rate"]
        assert abs(e[rule]["rate"] - r) < 1e-3, f"{rule}: not the mask of Table 3's runs"
    c = lambda rule: 100 * e[rule]["coverage"]["computed"]
    has(text, "entropy at 10% starts a patch at every final answer and dependence at none",
        f"(entropy {c('entropy10'):.1f}%, dependence {c('dep10'):.1f}%, hand-written {c('syntax+entropy10'):.0f}%)")
    # "at 11.7M and 53.7M their accuracy is ordered like" that share; at 1.1M it is not (dependence is below entropy there)
    rules = ("entropy10", "dep10", "syntax+entropy10")
    assert c(rules[0]) < c(rules[1]) < c(rules[2])
    from scale import analyze
    ok = {k: v for k, v in scale_runs("unclipped").items()}
    bad = analyze.failed(ok)
    fin = lambda size, rule: np.mean([o["final_acc"] for sd, o in ok[(size, rule)].items() if (size, rule, sd) not in bad])
    for size in ("12m", "50m"):
        assert fin(size, rules[0]) < fin(size, rules[1]) < fin(size, rules[2]), size
    assert fin("1m", "dep10") < fin("1m", "entropy10")
    has(text, "at 11.7M and 53.7M their accuracy is ordered like the share of computed results that get a patch start")


def test_seed_counts(text, scale):
    """Appendix A (corrected in version 2): the seed counts of Table 3 and of the text."""
    un, ok, bad, cl = scale
    fewer = {("1m", "dep10"): 2, ("50m", "syntax+entropy10"): 2, ("50m", "jump10"): 1, ("50m", "entropy20"): 1}
    assert all(len(rs) == fewer.get(k, 3) for k, rs in ok.items())
    has(text, "in Table 3, 1.1M dependence and 53.7M hand-written have two seeds, and jump and entropy at 20% one")
    # the text: entropy plus dependence (§7) has 2 seeds; the trace models (§9) have 2 with the 32-byte window, 3 with 8
    rows = stats.mlx_rows()
    n = lambda exp, rule, **w: sum(1 for r in rows if r.get("experiment") == exp and r["rule"] == rule
                                   and all(r.get(k, 32) == x for k, x in w.items()))
    assert n("math_tight_budget", "entdep10") == 2
    assert all(n("reason_trained", rule, window=32) == 2 and n("reason_trained", rule, window=8) == 3 for rule in ("entropy10", "syntax+entropy10"))
    has(text, "in the text, entropy plus dependence (§7) and the window-32 trace models (§9) have two.",
        "effects at two seeds that were not detectable at five (§12)")


def test_passes_over_solutions(text, scale):
    """§6 and §13: bytes seen (steps x batch x 128-byte windows) and passes over the GSM8K and MATH solutions (9.7 MB, 10%
    of the scaling corpus): 1.35 at 1.1M, 15.5 at 53.7M and 13.5 for the models of Table 2 (version 1 rounded these to
    'about once' and 'about 13 times')."""
    un, ok, bad, cl = scale
    seen = {s: (lambda c: c["steps"] * c["bs"] * 128)(next(iter(ok[(s, "entropy10")].values()))["cfg"]) for s in ("1m", "12m", "50m")}
    has(text, f"The models see {seen['1m'] / 1e9:.2f}, {seen['12m'] / 1e9:.1f} and {seen['50m'] / 1e9:.1f} billion bytes")
    passes = {s: 0.10 * b / 9.7e6 for s, b in seen.items()}
    table2 = 32_000 * 32 * 128 / 9.7e6                    # Table 2: 32,000 steps of 32 windows of 128 bytes, solutions only
    has(text, f"(about {passes['1m']:.1f} at 1.1M, about {passes['50m']:.0f} at 53.7M)",
        f"about {passes['1m']:.1f} times, against about {table2:.0f} times")


# ------------------------------------------------------------------ §3: BLT-1B with its 512-byte window restored
def test_window_fix(text, reg):
    """The paper uses BLT-1B with the window restored (window=512 rows); it compares with the released conversion."""
    w = json.loads((ROOT / "results" / "blt_window512.json").read_text())
    has(text, f"{w['problems_longer']} of our {w['problems']} test problems are longer than 512 bytes")
    fixed = blt(reg)
    released = {r["layout"]: r for r in reg if r.get("experiment") == "blt_budget" and r.get("thresh") == "train" and r.get("rerun") == 26}
    has(text, f"the default layout starts a patch on {100 * fixed['default']['rate']:.1f}% of bytes on GSM8K, against {100 * released['default']['rate']:.1f}% without it")
    old = items("blt_budget_train.json")["blt_budget_train"]["data"]["layouts"]           # released conversion, per target
    d = lambda R: p1(100 * (np.mean(old[f"results@{R}"]["res_exact"]) - np.mean(old[f"entropy@{R}"]["res_exact"])))
    has(text, f"(results − entropy at 15% and 10%: +{d(15)} and +{d(10)} points)")
    fig = json.loads((ROOT / "results" / "fig1_layouts_w512.json").read_text())
    n = len(fig["text"].encode())
    assert len(fig["entropy@10"]) == len(fig["entdep@10"]), "Figure 1: equal patch counts"
    starts = [s for s, _ in fig["results"]]
    assert not any(t in set(fig["entropy@10"]) for t in starts) and all(t in set(fig["entdep@10"]) for t in starts)
    has(text, f"Figure 1 shows a single solution ({n} bytes)")


def test_threshold_fit_problems(text):
    """§3: thresholds fitted on the first 300 GSM8K training problems, of which 297 fit the context (the run's log)."""
    log = (ROOT / "results" / "logs" / "blt1b_w512" / "budget.log").read_text(errors="replace")
    n = int(re.search(r"thresholds fitted on (\d+) GSM8K train problems", log).group(1))
    has(text, f"the first 300 GSM8K training problems ({n} fit the context)")


def test_threshold_modes(text, reg):
    """§13: train-fitted thresholds against the original top-R selection per problem (released conversion); version 1
    said 'as large or larger', but at 10% the train-fitted gap is 0.5 points smaller."""
    old = items("blt_budget_train.json")["blt_budget_train"]["data"]["layouts"]
    train = lambda R: 100 * (np.mean(old[f"results@{R}"]["res_exact"]) - np.mean(old[f"entropy@{R}"]["res_exact"]))
    prob = {r["layout"]: r["results_exact"] for r in reg
            if r.get("experiment") == "blt_budget" and r.get("thresh") == "problem" and r.get("rerun") is None}
    per = lambda R: 100 * (prob[f"results@{R}"] - prob[f"entropy@{R}"])
    has(text, f"(released conversion, results − entropy: {sg(train(15))} against {sg(per(15))} points at 15%, "
              f"{sg(train(10))} against {sg(per(10))} at 10%)", "with similar or larger gaps")


def test_answer_boundary(text, reg):
    """§4: removing the patch start at the final answer (BLT-1B, window restored)."""
    r = {x["layout"]: x["answer_exact"] for x in reg if str(x.get("experiment", "")).startswith("blt_answer") and x.get("window") == 512}
    drop = 100 * (r["default"] - r["-answer"])
    has(text, f"costs {drop:.0f} points of exact match ({p1(100 * r['default'])}% $\\rightarrow$ {p1(100 * r['-answer'])}%)")


# ------------------------------------------------------------------ §12: negative results
def test_learned_patcher_correlations(text):
    """§12: held-out correlations of the learned patcher and the tables (recorded in notes.md, issue #19), stated with
    their values instead of version 1's 'matches'."""
    notes = (ROOT / "notes.md").read_text()
    assert "math table 0.140 vs network 8 bytes 0.140; logic/traces table 0.158 vs network 0.147" in notes
    has(text, "(math 0.140 against 0.140; logic and traces 0.147 against 0.158)")


def test_negative_results_intervals(text, reg):
    """§12, with Welch intervals over seeds: per-digit patches (on computed results) and operand-aligned patches (on final
    answers) against their matched random controls, 5 seeds each (notes.md: rand07 is the twin of +digits, rand22 of
    +ops; version 1: "vanished at five seeds"); the learned patcher (ndep10) against the table (dep10)."""
    rows = stats.mlx_rows()
    C = lambda a, b, k, exp: stats.compare(rows, a, b, k, experiment=(exp, None))
    f = lambda c, unit="": f"{sg(100 * c['d'])}{unit}, {iv(100 * c['welch'][0], 100 * c['welch'][1])}"
    notes = (ROOT / "notes.md").read_text()
    assert "rand07 (twin of +digits)" in notes and "rand22 (twin of +ops)" in notes
    dg = C("words+syntax+digits", "words+syntax+rand07", "computed_acc", "math_new_strategies")
    op = C("words+syntax+ops", "words+syntax+rand22", "final_acc", "math_new_strategies")
    for c in (dg, op):
        assert (c["na"], c["nb"]) == (5, 5) and c["welch"][0] < 0 < c["welch"][1]
    has(text, "Per-digit patches after `=` and operand-aligned patches looked promising at two seeds (on computed results and final "
              f"answers); at five seeds neither differs detectably from a matched random control ({f(dg, ' points')}; {f(op)}).")
    lc, lf = C("ndep10", "dep10", "computed_acc", "math_tight_budget"), C("ndep10", "dep10", "final_acc", "math_tight_budget")
    assert lc["welch"][0] < 0 < lc["welch"][1] and lf["welch"][1] < 0
    T = lambda rule, k: p1(trained(reg, "math_tight_budget", rule, k))
    has(text, f"it gets {T('ndep10', 'computed_acc')}% / {T('ndep10', 'final_acc')}% against the table's {T('dep10', 'computed_acc')}% / "
              f"{T('dep10', 'final_acc')}% ({f(lc, ' points')}; {f(lf)}).")


def test_trained_compare_file():
    """paper/v2/results/trained_compare.json, which lists the comparisons of §6, §7 and §12 that version 2 gives with
    intervals, is what paper/v2/scripts/trained_compare.py computes from the registry now (the tests above recompute
    them independently and check the manuscript)."""
    saved = json.load(open(V2 / "results" / "trained_compare.json"))["comparisons"]
    fresh = trained_compare.compute()
    keys = ("section", "experiment", "a", "b", "metric", "seeds", "includes_zero", "separated")
    assert [{k: s[k] for k in keys} for s in saved] == [{k: f[k] for k in keys} for f in fresh]
    assert all(abs(s[k] - f[k]) < 1e-9 for s, f in zip(saved, fresh) for k in ("diff", "lo", "hi"))
    bpb = {(o["a"], o["b"]): o for o in fresh if o["metric"] == "bpb"}
    e, j = bpb[("syntax+entropy10", "entropy10")], bpb[("syntax+entropy10", "jump10")]
    for g, o in ((welch_seeds(o["a_values"], o["b_values"]), o) for o in (e, j)):
        assert all(abs(x - y) < 1e-12 for x, y in zip(g, (o["diff"], o["lo"], o["hi"])))


# ------------------------------------------------------------------ wording and release hygiene (version 2)
FORBIDDEN = ("does as well", "as good as", "matches", "equivalent", "no worse", "at no cost", "indistinguishable")


def test_wording(text):
    """Ties are 'tied' or 'no detectable difference', with the interval, never one of the phrases above; the follow-up
    is cited with its arXiv number."""
    low = text.lower()
    assert not [w for w in FORBIDDEN if w in low], [w for w in FORBIDDEN if w in low]
    has(text, "arXiv:2610.22100 [@verazuniga2026patch]", "[@verazuniga2026patch] used another two-byte table")
    bib = (ROOT / "paper" / "refs.bib").read_text()
    assert "@article{verazuniga2026patch," in bib and "arXiv:2610.22100" in bib


def test_v2_support_files():
    """paper/v2/: no absolute local paths, every file described in the README, and GSM8K's licence next to the GSM8K text."""
    readme = (V2 / "README.md").read_text()
    for f in sorted(p for p in V2.rglob("*") if p.is_file()):
        rel = str(f.relative_to(V2))
        if f.suffix in (".py", ".md", ".json", "") and f.name != ".DS_Store":
            s = f.read_text()
            assert "/Users/" not in s and "/private/" not in s, f"absolute local path in {rel}"
        assert rel == "README.md" or rel in readme, f"{rel} is not described in paper/v2/README.md"
    lic = (V2 / "data" / "prose_rewrites" / "LICENSE-GSM8K").read_text()
    assert "Copyright (c) 2021 OpenAI" in lic and "The above copyright notice and this permission notice" in lic
    assert "LICENSE-GSM8K" in (V2 / "data" / "prose_rewrites" / "README.md").read_text()


def test_arxiv_abstract():
    """paper/SUBMISSION.md, version 2: the arXiv abstract is plain ASCII, at most 1,920 characters, and says what the
    manuscript's abstract says."""
    sub = (ROOT / "paper" / "SUBMISSION.md").read_text()
    v2 = sub[sub.index("## Version 2"):]
    block = re.search(r"### Abstract.*?```\n(.*?)\n```", v2, flags=re.S).group(1)
    assert block.isascii() and len(block) <= 1920, (block.isascii(), len(block))
    ms = MS.read_text()
    a = re.search(r"## Abstract\n\n(.*?)\n\n## 1\.", ms, flags=re.S).group(1)
    plain = (a.replace("$\\log_2 10 = 3.32$", "log2(10) = 3.32").replace("`= `", '"= "').replace("`=`", '"="').replace("*", ""))
    assert plain == block, "the arXiv abstract differs from the manuscript's"
    # the Comments line is the same in SUBMISSION.md and ARXIV_V2.md, plain ASCII, within arXiv's 400-character limit, and
    # names the release and its version DOI
    c1 = re.search(r"### Comments\n.*?```\n(.*?)\n```", v2, flags=re.S).group(1)
    c2 = re.search(r"## Comments line for version 2\n.*?```\n(.*?)\n```", (ROOT / "paper" / "ARXIV_V2.md").read_text(), flags=re.S).group(1)
    assert c1 == c2 and c1.isascii() and len(c1) <= 400, (c1 == c2, c1.isascii(), len(c1))
    assert "(release v2.1.0, doi:10.5281/zenodo.24023934)" in c1
