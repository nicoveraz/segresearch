"""Odds for the O6b windowed paired verdict as amended after the fourth check (2026-10-30): its class is set by the
id_copy exact rows (L8 - SIV+PO) of F1, F2b and F3's 6-digit, letter and mixed IDs, each counted only where its cell
shows the hazard at that rate. Exact: each row's class is read from a table of the class rule over every (lost, kept)
count, and the counts' distribution is summed exactly, plug-in or Dirichlet-multinomial (no simulation of fresh rows).
Dev only: the 293 F1 rows of results/blt_copy_latch_step2.json, where SIV stands in for SIV+PO (both mask c1..c4 in
293 of 293 F1 dev rewrites). The gate chances are the rule's predicted testabilities (see GATES). No model; writes
nothing.

    ../segresearch/.venv-blt/bin/python -B copy_latch_paired_odds.py | tee results/blt_copy_latch_fresh_paired_odds.log

A row's class follows realblt_copy_latch_fresh.classify on a 95% percentile interval of 2,000 bootstrap resamples over
problems (seed 6). With one copy per text the paired difference takes the values -1, 0 and +1, so a resample is a
multinomial draw of the lost and kept counts. Rows are independent, each with its own truth when uncertain.
"""
import itertools
import json
import math
import os

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
MARGIN, B = 0.02, 2000
TABLES = {}


def class_table(n, top):
    """class[l, k] of a row with l lost and k kept copies of n: 1 = L8 better, -1 = SIV+PO better, 0 = tie"""
    if (n, top) not in TABLES:
        tab = np.zeros((top + 1, top + 1), int)
        for l, k in itertools.product(range(top + 1), repeat=2):
            v = (k - l) / n
            if l + k > n or abs(v) < MARGIN:
                continue
            c = np.random.default_rng(6).multinomial(n, [l / n, k / n, 1 - (l + k) / n], size=B)
            lo, hi = np.percentile((c[:, 1] - c[:, 0]) / n, [2.5, 97.5])
            tab[l, k] = 1 if v >= MARGIN and lo > 0 else -1 if v <= -MARGIN and hi < 0 else 0
        TABLES[(n, top)] = tab
    return TABLES[(n, top)]


def pmf(n, top, p=None, alpha=None):
    """P(l lost, k kept) for l, k <= top: multinomial with rates p = (lost, kept), or Dirichlet-multinomial alpha"""
    lf = [math.lgamma(i + 1) for i in range(n + 1)]
    out = np.zeros((top + 1, top + 1))
    for l, k in itertools.product(range(top + 1), repeat=2):
        m = n - l - k
        if m < 0:
            continue
        c = lf[n] - lf[l] - lf[k] - lf[m]
        if alpha is None:
            if (l and p[0] == 0) or (k and p[1] == 0):
                continue
            out[l, k] = math.exp(c + (l * math.log(p[0]) if l else 0) + (k * math.log(p[1]) if k else 0)
                                 + m * math.log(1 - p[0] - p[1]))
        else:
            a = alpha
            out[l, k] = math.exp(c + math.lgamma(sum(a)) - math.lgamma(n + sum(a)) + math.lgamma(l + a[0])
                                 - math.lgamma(a[0]) + math.lgamma(k + a[1]) - math.lgamma(a[1]) + math.lgamma(m + a[2])
                                 - math.lgamma(a[2]))
    assert out.sum() > 1 - 1e-6, (n, top, p, alpha, out.sum())
    return out


def row_odds(n, p=None, alpha=None, top=None):
    """{L8 better, SIV+PO better, tie, tie at 2 points or more} for one row"""
    top = top or min(n, 200)
    w, tab = pmf(n, top, p, alpha), class_table(n, top)
    far = np.abs(np.subtract.outer(np.arange(top + 1), np.arange(top + 1))) * 50 >= n      # |k - l| >= 2 points
    return {"L8 better": float(w[tab == 1].sum()), "SIV+PO better": float(w[tab == -1].sum()),
            "tie": float(w[tab == 0].sum()), "tie, 2 points or more": float(w[(tab == 0) & far].sum())}


rows = json.load(open("results/blt_copy_latch_step2.json"))["rows"]
dev = {}
for b in ("D", "E10"):
    d = [r["layouts"][f"{b}.L8"]["kinds"]["id_copy"][1] - r["layouts"][f"{b}.SIV"]["kinds"]["id_copy"][1] for r in rows]
    dev[b] = (d.count(-1), d.count(1), d.count(0))
print("dev F1 discordance, L8 against SIV (lost, kept, same):", dev)

# the class rows: size, the rate whose dev discordance they borrow, and their gate chance (low, high)
GATES = {
    "F1 D": (600, "D", (1, 1)),                 # F1's D gate is C0's point condition: it passes whenever no K5 stops
    "F1 E10": (600, "E10", (1, 1)),             # dev: E10-all - E10 = +15.7 [+11.6, +19.8] at n = 293
    "6-digit D": (100, "D", (0.95, 0.97)),      # the rule's testabilities (the smoke's D hazards; third check)
    "mixed D": (100, "D", (0.83, 0.89)),
    "letters D": (100, "D", (0.2, 0.4)),
    "F2b D": (300, "D", (0.03, 0.09)),
    "6-digit E10": (100, "E10", (0.5, 0.9)),    # F1's dev E10 hazard (46 of 293 lost to E10's starts), with D breaking
    "mixed E10": (100, "E10", (0.4, 0.7)),      # 2 of 3 6-digit and 1 of 2 mixed smoke copies (fourth check, N1)
}                                               # letters and F2b have no E10 row: D broke none of their smoke copies
res = {}
for name, (n, b, _) in GATES.items():
    lost, kept, same = dev[b]
    nd = lost + kept + same
    res[(name, "dev")] = row_odds(n, p=(lost / nd, kept / nd))
    res[(name, "Jeffreys")] = row_odds(n, alpha=(lost + 0.5, kept + 0.5, same + 0.5))
    for rec in (0.97, 0.95, 0.90):              # L8 missing c1, and so losing the copy, on 1 - recall of copies (D rows)
        res[(name, f"recall {rec}")] = (row_odds(n, p=(1 - rec, kept / nd)) if b == "D" else res[(name, "dev")])
for (name, basis), o in res.items():
    print(f"  {name:12s} {basis:12s} " + "  ".join(f"P({k}) {v:.3f}" for k, v in o.items()))


def combine(per_row, g):
    """P(class) over the gated rows, independent: tie, SIV+PO better, L8 better, mixed, and tie with a row 2 points off"""
    q = lambda f: float(np.prod([1 - g[r] + g[r] * f(per_row[r]) for r in g]))
    q_none = q(lambda o: o["tie"])
    q_sb, q_lb = q(lambda o: 1 - o["SIV+PO better"]), q(lambda o: 1 - o["L8 better"])
    return {"tie": q_none, "SIV+PO better": q_lb - q_none, "L8 better": q_sb - q_none,
            "mixed": 1 - q_sb - q_lb + q_none, "tie with a class row 2 points or more off":
                q_none - q(lambda o: o["tie"] - o["tie, 2 points or more"])}


print("\nthe windowed class")
for lohi in (0, 1):
    g = {r: v[2][lohi] for r, v in GATES.items()}
    print(f"  gates {'low' if lohi == 0 else 'high'}: expected class rows {sum(g.values()):.2f}")
    for basis in ("dev", "Jeffreys", "recall 0.97", "recall 0.95"):
        c = combine({r: res[(r, basis)] for r in g}, g)
        print(f"    {basis:12s}: " + "  ".join(f"P({k}) {v:.3f}" for k, v in c.items()))

print("\nnull: a true difference of zero, with refit side effects flipping f of copies each way")
for f in (0.005, 0.01, 0.03, 0.10):
    one = {n: 1 - row_odds(n, p=(f, f))["tie"] for n in (600, 300, 100)}
    fw = [1 - float(np.prod([1 - v[2][lohi] * one[v[0]] for v in GATES.values()])) for lohi in (0, 1)]
    print(f"  f {f:.3f}: P(a row not tied) at n 600 {one[600]:.3f}, 300 {one[300]:.3f}, 100 {one[100]:.3f}; "
          f"P(the windowed class leaves 'tie') {fw[0]:.3f}-{fw[1]:.3f}")
