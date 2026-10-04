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
    return {r["layout"]: r for r in reg if r.get("experiment") == "blt_budget" and r.get("thresh") == "train" and r.get("rerun") == 26}


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
    it = items("blt_budget_train.json")["blt_budget_train"]["data"]["layouts"]
    acc = lambda lay, key: 100 * np.mean(it[lay][key])
    d = lambda a, b, key="res_exact": p1(acc(a, key) - acc(b, key))
    has(text, f"+{d('results@15', 'entropy@15')} points at 15%", f"+{d('results@10', 'entropy@10')} at 10%",
        f"+{d('entdep@15', 'entropy@15')} points at 15%", f"+{d('entdep@15', 'results@15')}",
        f"+{d('entdep@15', 'entropy@15', 'fin_exact')} points at 15%", f"+{d('entdep@10', 'entropy@10', 'fin_exact')} at 10%")
    x, y = np.array(it["results@15"]["res_exact"]), np.array(it["entropy@15"]["res_exact"])
    has(text, f"{int((x & ~y).sum())} results right only under *results* vs {int((~x & y).sum())} only under *entropy*")


# ------------------------------------------------------------------ §5: trained at tight budgets
def test_trained_budget_claims(text, reg):
    T = lambda rule, k: trained(reg, "math_tight_budget", rule, k)
    has(text, f"by {p1(T('dep10', 'final_acc') - T('entropy20', 'final_acc'))} points (pooled 95% interval")
    has(text, f"final-answer accuracy {T('entropy10', 'final_acc'):.0f}% → {T('entropy15', 'final_acc'):.0f}% → {T('entropy20', 'final_acc'):.0f}%")
    L = lambda rule, k: trained(reg, "math_larger", rule, k)
    has(text, f"gets {p1(L('words+syntax', 'computed_acc'))}% of computed results against {p1(L('entropy', 'computed_acc'))}% for entropy",
        f"word starts alone get {p1(L('words', 'computed_acc'))}%", f"math syntax {p1(L('stride6+syntax', 'computed_acc'))}%")
    S = lambda rule: trained(reg, "math_small", rule, "final_acc")
    has(text, f"gets {p1(S('entropy'))}% of final answers, its jump variant {p1(S('jump'))}%", f"math syntax {p1(S('words+syntax'))}%")


def test_jump_rule(text, reg):
    T = lambda rule, k: p1(trained(reg, "math_tight_budget", rule, k))
    has(text, f"recovers final answers ({T('jump10', 'final_acc')}% vs {T('entropy10', 'final_acc')}% at 10%) but not computed results ({T('jump10', 'computed_acc')}% vs {T('entropy10', 'computed_acc')}%)",
        f"recovers final answers ({T('jump10', 'final_acc')}% against {T('entropy10', 'final_acc')}% for entropy) but not computed results ({T('jump10', 'computed_acc')}% against {T('entropy10', 'computed_acc')}%)")


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
    for it in items("blt_finetune_*.json").values():
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
