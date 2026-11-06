"""Predictions for O6c (followup/O6c_CUTOFF_RULES.md, proposed), from stored outputs only: O6b's J7 rows (F5, fresh
F1 texts under the released conversion: results/blt_copy_latch_fresh_j7.json), each copy's first token from J1's
patcher outputs (results/blt_copy_latch_fresh_patcher.npz, git-ignored; its sha256 is checked against J1's record),
and [JW0]'s dev rows (results/blt_copy_gate_w0.json: the released conversion on F1 texts from GSM8K test problems
0-299). No model, and no new problem. The J7 numbers were read before O6c was written, and J7-based predictions for
cutoffs 600-720 were seen before the cutoff was set; the cutoff, 620, comes from the dev rows, by the copy's position.

    ../segresearch/.venv-blt/bin/python -B copy_latch_cutoff_predict.py | tee results/blt_copy_latch_cutoff_predict.log

The cutoff rule, SIV+PO<620, is SIV+PO at token positions t < 620 and nothing at t >= 620. On J7's texts its outcome
at the copy is approximated per text: where the copy's first token c1 < 620 it masks the copy as SIV+PO does, so it
takes D.SIVPO's outcome; where c1 >= 620 it masks nothing there, so it takes D.L8's outcome when L8 did not fire at c1
(neither layout masks the copy) and D's when L8 fired. The refitted threshold's small shift is ignored.

On [JW0]'s dev rows each copy's first token is rebuilt with F1's builder (realblt_quadrant.build) from GSM8K test
problems 0-299 alone. Tickets are always 4 digits, so positions do not depend on the ticket drawn; every rebuilt text's
token count and distance must equal its [JW0] row's. No train row is read. There the cutoff is scored as D-all where
c1 < 620 and D elsewhere, and [JW0]'s argmax canary (D-all where the patcher's argmax at c1 is right, else D) stands in
for L8.
"""
import itertools
import json
import math
import os
import sys

sys.dont_write_bytecode = True
os.environ["O6_SMOKE"] = "1"                    # import only: no pool read and no output; load() is a pure reader
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

import realblt_copy_latch_fresh as F

CUT, MARGIN, B, N_NEW = 620, 0.02, 2000, 600
OTHER = (665, 700)      # for comparison only: [JW0]'s 700-token text length as a position, and the first draft's cutoff

j1 = json.load(open("results/blt_copy_latch_fresh_j1.json"))
npz = "results/blt_copy_latch_fresh_patcher.npz"
assert F.sha256(npz) == j1["patcher_sha256"], "the patcher outputs differ from those J1 recorded"
copy_of = {r["idx"]: F.copy_tokens(r) for r in F.load(npz, ("F1.test",))["F1.test"]}
rows = json.load(open("results/blt_copy_latch_fresh_j7.json"))["rows"]["F5"]
ex = lambda r, name: r["layouts"][name]["kinds"]["id_copy"][1]


def j7(cut):
    """J7's texts, with the outcome of a cutoff at token `cut` approximated per text"""
    per = []
    for r in rows:
        c = copy_of[r["idx"]]
        fired = r["fires_c1"]["L8"]
        out = ex(r, "D.SIVPO") if c[0] < cut else (ex(r, "D") if fired else ex(r, "D.L8"))
        per.append({"n": r["n_tokens"], "c1": int(c[0]), "cL": int(c[-1]), "fired": fired, "D": ex(r, "D"),
                    "D-all": ex(r, "D-all"), "D.L8": ex(r, "D.L8"), "D.SIVPO": ex(r, "D.SIVPO"), "cut": out})
    d = [x["D.L8"] - x["cut"] for x in per]
    return per, d, d.count(-1), d.count(1)


per, d, lost, kept = j7(CUT)
n = len(per)
mean = lambda k, xs=per: sum(x[k] for x in xs) / len(xs) if xs else float("nan")
print(f"J7 (O6b, F5): {n} fresh F1 test texts under the released conversion; the cutoff at token {CUT}")
print(f"  the copy's first token sits {min(x['n'] - x['c1'] for x in per)}-{max(x['n'] - x['c1'] for x in per)} tokens "
      f"before the end; c1 >= {CUT}: {sum(x['c1'] >= CUT for x in per)} texts; copies straddling token {CUT} (c1 < "
      f"{CUT} <= their last token): {sum(x['c1'] < CUT <= x['cL'] for x in per)}")
print("  exact copies:", {k: round(mean(k), 4) for k in ("D", "D-all", "D.L8", "D.SIVPO", "cut")})
print("  by length (n, then D, D-all, D.L8, D.SIVPO and the cutoff's approximated outcome):")
for lo, hi in ((0, 600), (600, 700), (700, 100000)):
    xs = [x for x in per if lo <= x["n"] < hi]
    print(f"    tokens {lo}-{hi if hi < 100000 else ''}: n {len(xs)}, "
          + ", ".join(f"{k} {mean(k, xs):.3f}" for k in ("D", "D-all", "D.L8", "D.SIVPO", "cut"))
          + f"; L8 fires at c1 {mean('fired', xs):.3f}")
print("  by the copy's first token (n; L8 fires at c1; D.L8, D.SIVPO and the cutoff exact; copies L8 lost to the "
      "cutoff and kept):")
for lo, hi in ((0, 575), (575, CUT), (CUT, 700), (700, 100000)):
    sel = [(x, y) for x, y in zip(per, d) if lo <= x["c1"] < hi]
    xs = [x for x, _ in sel]
    print(f"    c1 {lo}-{hi - 1 if hi < 100000 else ''}: n {len(xs)}; fires {sum(x['fired'] for x in xs)}; "
          + ", ".join(f"{k} {mean(k, xs):.3f}" for k in ("D.L8", "D.SIVPO", "cut"))
          + f"; lost {sum(y == -1 for _, y in sel)}, kept {sum(y == 1 for _, y in sel)}")
print(f"  D.L8 - cutoff, approximated: {100 * (kept - lost) / n:+.2f} points ({lost} copies lost, {kept} kept); lost "
      f"where L8 did not fire at c1: {sum(1 for x, y in zip(per, d) if y == -1 and not x['fired'])}")
print(f"  D.L8 - D.SIVPO on J7 itself: {100 * (mean('D.L8') - mean('D.SIVPO')):+.2f} points; D-all - D: "
      f"{100 * (mean('D-all') - mean('D')):+.2f}")

# the class of a fresh 600-text row, exactly over the (lost, kept) counts, one seeded bootstrap draw per count
TAB = {}


def class_table(n, top):
    if (n, top) not in TAB:
        tab = np.zeros((top + 1, top + 1), int)
        for l, k in itertools.product(range(top + 1), repeat=2):
            v = (k - l) / n
            if l + k > n or abs(v) < MARGIN:
                continue
            c = np.random.default_rng(6).multinomial(n, [l / n, k / n, 1 - (l + k) / n], size=B)
            lo, hi = np.percentile((c[:, 1] - c[:, 0]) / n, [2.5, 97.5])
            tab[l, k] = 1 if v >= MARGIN and lo > 0 else -1 if v <= -MARGIN and hi < 0 else 0
        TAB[(n, top)] = tab
    return TAB[(n, top)]


def pmf(n, top, p=None, alpha=None):
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
    assert out.sum() > 1 - 1e-6, out.sum()
    return out


def odds(w, tab):
    return {"L8 better": float(w[tab == 1].sum()), "cutoff better": float(w[tab == -1].sum()),
            "tie": float(w[tab == 0].sum())}


top = 120
tab = class_table(N_NEW, top)
r3 = lambda o: {k: round(v, 3) for k, v in o.items()}
truth = lambda l, k, m: r3(odds(pmf(N_NEW, top, p=(l / m, k / m)), tab))
jeffreys = lambda l, k, m: r3(odds(pmf(N_NEW, top, alpha=(l + 0.5, k + 0.5, m - l - k + 0.5)), tab))
print(f"\nthe primary class on {N_NEW} fresh texts (exact over the counts; each count's class is one seeded draw):")
print("  J7 as truth:", truth(lost, kept, n))
print("  J7's uncertainty (Jeffreys):", jeffreys(lost, kept, n))
for extra in (2, 5, 10):          # if the cutoff loses `extra` more copies per 600 than the approximation says
    print(f"  J7 as truth, with {extra} more copies per 600 lost by the cutoff:", truth(lost, kept + extra, n))
print("  for comparison, the same calculation at other cutoffs:")
for c in OTHER:
    _, _, l2, k2 = j7(c)
    print(f"    token {c}: D.L8 - cutoff {100 * (k2 - l2) / n:+.2f} points ({l2} lost, {k2} kept); J7 as truth "
          f"{truth(l2, k2, n)}; Jeffreys {jeffreys(l2, k2, n)}")

# dev: [JW0]'s rows by the copy's position, rebuilt from GSM8K test problems 0-299 (no train row)
jw0 = json.load(open("results/blt_copy_gate_w0.json"))["rows"]
rng, built = np.random.default_rng(0), []
with open(F.GSM) as f:
    for line in itertools.islice(f, 300):
        b, units, info = F.make(json.loads(line), rng, "F1")
        if len(b) + 1 <= F.MAX_TOKENS:
            built.append({"n": len(b) + 1, "distance": info["distance"], "c1": int(F.copy_tokens({"units": units})[0])})
assert len(built) == len(jw0) and all(x["n"] == r["n_tokens"] and x["distance"] == r["distance"]
                                      for x, r in zip(built, jw0)), "the rebuilt dev texts differ from [JW0]'s rows"
nd = len(jw0)
c1 = np.array([x["c1"] for x in built])
ntok = np.array([x["n"] for x in built])
Dd = np.array([r["D"]["exact"] for r in jw0], int)
Da = np.array([r["D-all"]["exact"] for r in jw0], int)
canary = np.where([r["patcher_ok"][0] for r in jw0], Da, Dd)
near = np.array([r["distance"] < 512 for r in jw0])


def dev(c):
    cut = np.where(c1 < c, Da, Dd)
    return cut, int(((canary == 0) & (cut == 1)).sum()), int(((canary == 1) & (cut == 0)).sum())


print(f"\ndev ([JW0]: {nd} F1 texts from GSM8K test problems 0-299 under the released conversion), by the copy's "
      f"position:")
print(f"  rebuilt with F1's builder: all {nd} texts' token counts and distances equal [JW0]'s; the copy's first token "
      f"sits {(ntok - c1).min()}-{(ntok - c1).max()} tokens before the end")
print(f"  the argmax canary (L8's stand-in) keeps {canary.sum()} of {nd} copies ({canary.mean():.3f})")
print("  masking every copy start (D-all) against D, by the copy's first token: " + "; ".join(
    f"c1 {lo}-{hi - 1 if hi < 100000 else ''}: {Da[(c1 >= lo) & (c1 < hi)].sum()} of {((c1 >= lo) & (c1 < hi)).sum()}"
    f" against {Dd[(c1 >= lo) & (c1 < hi)].sum()}" for lo, hi in ((0, 575), (575, 627), (627, 100000))))
kept_at = {c: int(dev(c)[0].sum()) for c in range(450, 851)}
best = [c for c, v in kept_at.items() if v == max(kept_at.values())]
print(f"  a cutoff c (D-all where c1 < c, D elsewhere) keeps the most, {max(kept_at.values())} of {nd}, for c in "
      f"{min(best)}-{max(best)}" + ("" if best == list(range(min(best), max(best) + 1)) else " (not contiguous)"))
for c in (CUT,) + OTHER:
    cut, l2, k2 = dev(c)
    print(f"  cutoff {c}: keeps {cut.sum()} ({cut.mean():.3f}); canary - cutoff {100 * (k2 - l2) / nd:+.2f} points "
          f"(the canary lost {l2} copies the cutoff kept and kept {k2} it lost); class on {N_NEW} fresh texts, dev as "
          f"truth {truth(l2, k2, nd)}; Jeffreys {jeffreys(l2, k2, nd)}")
bylen = np.where(ntok < 700, Da, Dd)
print(f"  by text length instead (D-all under 700 tokens, D from 700 on), as the first draft computed it: "
      f"{bylen.mean():.3f} on all {nd} (the canary {canary.mean():.3f}); {bylen[near].mean():.3f} on the {near.sum()} "
      f"with distance < 512 (the canary {canary[near].mean():.3f})")
