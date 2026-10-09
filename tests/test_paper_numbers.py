"""The paper must not contradict the results it is derived from (adapted from textca/tests/test_paper_numbers.py).

Tables are generated (make_tables.py -> paper/tables/), so they cannot drift. Numbers in the prose are typed by
hand, and this is where drift happens: the first draft had five wordings that went beyond the data, caught by a
manual reread. Each test recomputes a load-bearing number from results/registry/ or results/items/ and checks
that the manuscript states it. If one fails, the paper is wrong OR the analysis was re-run and the paper has not
caught up: reconcile them, never relax the check.

    uv run pytest tests/test_paper_numbers.py
"""
import glob
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flops  # noqa: E402
import registry  # noqa: E402

MS = ROOT / "paper" / "manuscript.md"


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


# ------------------------------------------------------------------ abstract and §4: BLT-1B at a tight budget
def test_blt1b_headline(text, reg):
    b = blt(reg)
    has(text, f"puts a patch start at {100 * b['entropy@10']['results_covered']:.0f}% of the computed results",
        f"gets {p1(100 * b['entropy@10']['results_exact'])}% of them exactly right",
        f"gets {p1(100 * b['results@10']['results_exact'])}%", f"signal gets {p1(100 * b['entdep@10']['results_exact'])}%",
        f"default layout at {100 * b['default']['rate']:.0f}% of bytes: {p1(100 * b['default']['results_exact'])}%")
    has(text, f"at 15% it covers {100 * b['entropy@15']['results_covered']:.0f}% of them, at 10% only {100 * b['entropy@10']['results_covered']:.0f}%")


def test_blt1b_paired_differences(text):
    # from the per-target outcomes (exact); the registry holds accuracies rounded to 0.1 points
    it = items("blt_budget_train_w512.json")["blt_budget_train_w512"]["data"]["layouts"]
    acc = lambda lay, key: 100 * np.mean(it[lay][key])
    d = lambda a, b, key="res_exact": p1(acc(a, key) - acc(b, key))
    has(text, f"+{d('results@15', 'entropy@15')} points at 15%", f"+{d('results@10', 'entropy@10')} at 10%",
        f"+{d('entdep@15', 'entropy@15')} points at 15%", f"+{d('entdep@15', 'results@15')}",
        f"+{d('entdep@10', 'entropy@10', 'fin_exact')} points at 10%", f"not at 15% (+{d('entdep@15', 'entropy@15', 'fin_exact')}, p = 0.26)")
    x, y = np.array(it["results@15"]["res_exact"]), np.array(it["entropy@15"]["res_exact"])
    has(text, f"{int((x & ~y).sum())} results right only under *results* vs {int((~x & y).sum())} only under *entropy*")


# ------------------------------------------------------------------ §5: trained at tight budgets
def test_trained_budget_claims(text, reg):
    T = lambda rule, k: trained(reg, "math_tight_budget", rule, k)
    has(text, f"by {p1(T('dep10', 'final_acc') - T('entropy20', 'final_acc'))} points (pooled 95% interval")
    has(text, f"final-answer accuracy {T('entropy10', 'final_acc'):.0f}% $\\rightarrow$ {T('entropy15', 'final_acc'):.0f}% $\\rightarrow$ {T('entropy20', 'final_acc'):.0f}%")
    L = lambda rule, k: trained(reg, "math_larger", rule, k)
    has(text, f"gets {p1(L('words+syntax', 'computed_acc'))}% of computed results against {p1(L('entropy', 'computed_acc'))}% for entropy",
        f"word starts alone get {p1(L('words', 'computed_acc'))}%", f"math syntax {p1(L('stride6+syntax', 'computed_acc'))}%")
    S = lambda rule: trained(reg, "math_small", rule, "final_acc")
    has(text, f"gets {p1(S('entropy'))}% of final answers, its jump variant {p1(S('jump'))}%", f"math syntax {p1(S('words+syntax'))}%")


def test_jump_rule(text, reg):
    T = lambda rule, k: p1(trained(reg, "math_tight_budget", rule, k))
    # (the abstract stated this too until the scaling study; it now says jump helps neither target at 50M, test_scaling)
    has(text, f"recovers final answers ({T('jump10', 'final_acc')}% against {T('entropy10', 'final_acc')}% for entropy) but not computed results ({T('jump10', 'computed_acc')}% against {T('entropy10', 'computed_acc')}%)")


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


def test_hand_written_rule_has_lowest_bpb(text, reg):
    for R in (10, 15, 20):
        hw = trained(reg, "math_tight_budget", f"syntax+entropy{R}", "bpb")
        assert hw < trained(reg, "math_tight_budget", f"entropy{R}", "bpb"), f"the paper says the hand-written rule has the lowest bpb at {R}%"


# ------------------------------------------------------------------ §7: BLT-1B adapted to the budget
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


# ------------------------------------------------------------------ §9: Scratchpad Patching; §10: compute
def test_scratchpad(text, reg):
    sp = lambda rule: [r["final_acc"] for r in reg if r.get("experiment") == "scratchpad16" and r["rule"] == rule]
    for rule in ("sp16:syntax", "sp16:entropy", "sp16:random"):
        assert len(sp(rule)) == 5, f"{rule}: the paper claims 5 seeds"
    has(text, f"({p1(100 * np.mean(sp('sp16:entropy')))}% vs {p1(100 * np.mean(sp('sp16:random')))}%, 5 seeds)",
        f"answer-start scratchpads give {p1(100 * np.mean(sp('sp16:syntax')))}%")


def test_compute(text, reg):
    pb, pp, pt = flops.blt1b()
    ref = blt(reg)["default"]["rate"] * pp + pb + pt
    with_ent = lambda r: 100 * (r * pp + pb + pt) / ref
    lookup = lambda r: 100 * (r * pp + pb) / ref
    has(text, f"A 10% budget then costs {with_ent(0.10):.0f}% of the default's forward compute and 15% costs {with_ent(0.15):.0f}%",
        f"{lookup(0.10):.0f}% at 10% and {lookup(0.15):.0f}% at 15%")
    hb, hp, ht = flops.harness()
    has(text, f"a 10% budget saves only {100 - 100 * (0.10 * hp + hb + ht) / (0.25 * hp + hb + ht):.0f}% of compute")


# ------------------------------------------------------------------ §5: scaling to 50M parameters (results/scale/)
def test_scaling(text):
    from scale import analyze
    ok, bad = analyze.runs_ok()
    assert {(s, a, sd) for s, a, sd in bad} == {("1m", "dep10", 0)}, f"the paper names one failed run: {bad}"
    v = lambda size, arm, k: [100 * r[k] for r in ok[(size, arm)].values()]
    m = lambda size, arm, k: float(np.mean(v(size, arm, k)))
    f1 = lambda x: f"{x:+.1f}".replace("-", "−")
    for size in ("12m", "50m"):
        for arm in ("entropy10", "dep10"):
            assert len(ok[(size, arm)]) == 3, f"{size} {arm}: the paper claims 3 seeds"
    gaps = [m(s, "dep10", "final_acc") - m(s, "entropy10", "final_acc") for s in ("1m", "12m", "50m")]
    has(text, f"by {f1(gaps[0])}, {f1(gaps[1])} and {f1(gaps[2])} points",
        f"it is {f1(gaps[0])} points at 1.1M", f"{f1(gaps[1])} at 11.7M and {f1(gaps[2])} at 53.7M")
    has(text, f"({p1(m('50m', 'dep10', 'final_acc'))}% against {p1(m('50m', 'entropy10', 'final_acc'))}%)",
        f"gets {p1(m('50m', 'dep10', 'computed_acc'))}% of computed results against {p1(m('50m', 'entropy10', 'computed_acc'))}%",
        f"dependence gets {p1(m('50m', 'dep10', 'computed_acc'))}% of them against {p1(m('50m', 'entropy10', 'computed_acc'))}% for entropy")
    # "every dependence seed above every entropy seed": final answers at 12M and 50M, computed results at 50M
    for size, k in (("12m", "final_acc"), ("50m", "final_acc"), ("50m", "computed_acc")):
        assert min(v(size, "dep10", k)) > max(v(size, "entropy10", k)), f"{size} {k}: seeds overlap"
    assert max(v("1m", "dep10", "final_acc")) > min(v("1m", "entropy10", "final_acc")), "the paper says the 1.1M seeds overlap"
    bpb = lambda size, arm: float(np.mean([r["bpb"] for r in ok[(size, arm)].values()]))
    has(text, f"({bpb('50m', 'dep10'):.3f} against {bpb('50m', 'entropy10'):.3f})")
    j = next(iter(ok[("50m", "jump10")].values()))
    has(text, f"{100 * j['final_acc']:.1f}% of final answers and {100 * j['computed_acc']:.1f}% of computed results", f"({j['bpb']:.3f})")
    assert 100 * j["final_acc"] < min(v("50m", "entropy10", "final_acc")) and 100 * j["computed_acc"] < min(v("50m", "entropy10", "computed_acc"))
    assert j["bpb"] == min(r["bpb"] for (s, a), rs in ok.items() if s == "50m" and a.endswith("10") for r in rs.values()), \
        "jump: lowest bpb of the 10% rules"
    hw = list(ok[("50m", "syntax+entropy10")].values())
    assert len(hw) == 2, "the paper says two hand-written seeds at 50M"
    has(text, f"hand-written rule {p1(100 * np.mean([r['computed_acc'] for r in hw]))}% ({p1(100 * np.mean([r['final_acc'] for r in hw]))}% of final answers; two seeds)")
    e20 = next(iter(ok[("50m", "entropy20")].values()))
    has(text, f"(20%, which is {100 * e20['rate']:.1f}% of the evaluation text)", f"on final answers ({100 * e20['final_acc']:.1f}%, one seed)",
        f"computed results ({100 * e20['computed_acc']:.1f}% against {p1(m('50m', 'dep10', 'computed_acc'))}%)")
    assert abs(100 * e20["final_acc"] - m("50m", "dep10", "final_acc")) < 2, "the paper says entropy at 20% matches dependence on final answers"
    assert max(r["e2e_acc"] for rs in ok.values() for r in rs.values()) <= 0.01, "the paper says at most 1 of 100 end to end"
    fail = analyze.load(analyze.RESULTS)[("1m", "dep10")][0]["bpb"]
    others = [r["bpb"] for (s, a), rs in ok.items() if s == "1m" for r in rs.values()]
    has(text, f"(bits per byte {fail:.2f} against {min(others):.2f}–{max(others):.2f}")
    assert "TBD_" not in text, "placeholders left in the manuscript"


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


def test_answer_boundary(text, reg):
    """§4: removing the patch start at the final answer (BLT-1B, window restored)."""
    r = {x["layout"]: x["answer_exact"] for x in reg if str(x.get("experiment", "")).startswith("blt_answer") and x.get("window") == 512}
    drop = 100 * (r["default"] - r["-answer"])
    has(text, f"costs {drop:.0f} points of exact match ({p1(100 * r['default'])}% $\\rightarrow$ {p1(100 * r['-answer'])}%)")
