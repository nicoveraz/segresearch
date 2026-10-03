"""Statistics for the headline claims (issue #18); writes results/stats.md.

1. Seed audit: every MLX training configuration with its seed count, mean, seed range and standard deviation.
   Configurations with fewer than 3 seeds are flagged (experiment standard 3).
2. Headline comparisons between training rules: difference of seed means, whether the seed ranges separate, a
   Welch t interval over seeds (wide with 2-3 seeds), and a pooled-sample interval (treats all targets from all
   seeds as independent; it ignores seed-to-seed variance, so it is the optimistic one).
3. Paired tests on BLT-1B (results/items/*.json, per-target outcomes saved by the BLT-1B scripts): exact McNemar
   test per pair of layouts, and a 95% bootstrap interval for the accuracy difference that resamples whole
   problems (targets from the same problem are correlated).

    uv run stats.py
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

import registry

OUT = os.path.join(registry.ROOT, "results", "stats.md")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}   # t quantiles, df -> value


def mlx_rows():
    return [r for r in registry.load() if r.get("format") == "mathexp" and not r.get("smoke") and not r.get("superseded")]


def config(r):
    return (r.get("experiment"), r.get("corpus", "math"), r.get("D"), r.get("glayers"), r.get("window", 32), r["rule"])


def audit(rows):
    g = defaultdict(list)
    for r in rows:
        g[config(r)].append(r)
    lines = ["| experiment | corpus | D | window | rule | seeds | final acc mean [range] | computed acc mean [range] | bpb mean (sd) |",
             "|---|---|---|---|---|---|---|---|---|"]
    few = []
    for k in sorted(g, key=lambda k: tuple(str(x) for x in k)):
        rs = g[k]; n = len(rs)
        def rng(f):
            v = [100 * r[f] for r in rs if not math.isnan(r.get(f, float("nan")))]
            return f"{np.mean(v):.1f} [{min(v):.1f}, {max(v):.1f}]" if v else ""
        b = [r["bpb"] for r in rs]
        flag = " **<3**" if n < 3 else ""
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[4]} | {k[5]} | {n}{flag} | {rng('final_acc')} | {rng('computed_acc')} | "
                     f"{np.mean(b):.3f} ({np.std(b, ddof=1) if n > 1 else float('nan'):.3f}) |")
        if n < 3:
            few.append(k)
    return lines, few


def compare(rows, a, b, metric, **where):
    pick = lambda rule: [r for r in rows if r["rule"] == rule and all(r.get(k, d) == v for k, (v, d) in where.items())]
    A, B = pick(a), pick(b)
    if not A or not B:
        return None
    va, vb = np.array([r[metric] for r in A]), np.array([r[metric] for r in B])
    d = va.mean() - vb.mean()
    sep = va.min() > vb.max() or vb.min() > va.max()
    # Welch interval over seeds
    if len(va) > 1 and len(vb) > 1:
        sa, sb = va.var(ddof=1) / len(va), vb.var(ddof=1) / len(vb)
        se = math.sqrt(sa + sb)
        df = (sa + sb) ** 2 / (sa ** 2 / (len(va) - 1) + sb ** 2 / (len(vb) - 1)) if se > 0 else 1
        t = T975.get(max(1, min(8, int(df))), 2.0)
        welch = (d - t * se, d + t * se)
    else:
        welch = (float("nan"), float("nan"))
    # pooled-sample interval (targets independent across seeds)
    n_key = metric.replace("_acc", "_n")
    na, nb = sum(r[n_key] for r in A), sum(r[n_key] for r in B)
    pa, pb = sum(r[metric] * r[n_key] for r in A) / na, sum(r[metric] * r[n_key] for r in B) / nb
    se_p = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return dict(a=a, b=b, metric=metric, na=len(va), nb=len(vb), d=d, sep=sep, welch=welch, pooled=(pa - pb - 1.96 * se_p, pa - pb + 1.96 * se_p))


COMPARISONS = [   # (label, rule A, rule B, metric, filter: key -> (value, default))
    ("10%: dependence vs entropy", "dep10", "entropy10", "final_acc", dict(experiment=("math_tight_budget", None))),
    ("10%: dependence vs entropy", "dep10", "entropy10", "computed_acc", dict(experiment=("math_tight_budget", None))),
    ("10%: hand-written vs entropy", "syntax+entropy10", "entropy10", "final_acc", dict(experiment=("math_tight_budget", None))),
    ("10%: hand-written vs entropy", "syntax+entropy10", "entropy10", "computed_acc", dict(experiment=("math_tight_budget", None))),
    ("10%: hand-written vs dependence", "syntax+entropy10", "dep10", "computed_acc", dict(experiment=("math_tight_budget", None))),
    ("15%: dependence vs entropy", "dep15", "entropy15", "final_acc", dict(experiment=("math_tight_budget", None))),
    ("15%: dependence vs entropy", "dep15", "entropy15", "computed_acc", dict(experiment=("math_tight_budget", None))),
    ("20%: dependence vs entropy", "dep20", "entropy20", "final_acc", dict(experiment=("math_tight_budget", None))),
    ("20%: dependence vs entropy", "dep20", "entropy20", "computed_acc", dict(experiment=("math_tight_budget", None))),
    ("dependence 10% vs entropy 20%", "dep10", "entropy20", "final_acc", dict(experiment=("math_tight_budget", None))),
    ("D=128: words+syntax vs entropy", "words+syntax", "entropy", "computed_acc", dict(experiment=("math_larger", None))),
    ("D=128: words+syntax vs words", "words+syntax", "words", "computed_acc", dict(experiment=("math_larger", None))),
    ("D=64: words+syntax vs entropy", "words+syntax", "entropy", "final_acc", dict(experiment=("math_small", None))),
    ("Scratchpad 16: syntax vs entropy trigger", "sp16:syntax", "sp16:entropy", "final_acc", dict(experiment=("scratchpad16", None))),
    ("Scratchpad 16: entropy trigger vs random", "sp16:entropy", "sp16:random", "final_acc", dict(experiment=("scratchpad16", None))),
    ("Traces, window 8: hand-written vs entropy", "syntax+entropy10", "entropy10", "computed_acc", dict(experiment=("reason_trained", None), window=(8, 32))),
    ("Traces, window 32: hand-written vs entropy", "syntax+entropy10", "entropy10", "computed_acc", dict(experiment=("reason_trained", None), window=(32, 32))),
]


def mcnemar(x, y):
    """Exact two-sided McNemar p-value for paired booleans x, y."""
    b = int(np.sum(x & ~y)); c = int(np.sum(~x & y)); n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), b, c


def boot(x, y, pid, reps=4000, seed=0):
    """95% interval for mean(x) - mean(y), resampling whole problems."""
    pid = np.asarray(pid); ids = np.unique(pid)
    by = {i: np.flatnonzero(pid == i) for i in ids}
    d = (x.astype(float) - y.astype(float))
    sums = np.array([d[by[i]].sum() for i in ids]); cnts = np.array([len(by[i]) for i in ids])
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        s = rng.integers(0, len(ids), len(ids))
        out.append(sums[s].sum() / cnts[s].sum())
    return np.quantile(out, [0.025, 0.975])


BLT_PAIRS = [("results@15", "entropy@15"), ("entdep@15", "entropy@15"), ("entdep@15", "results@15"),
             ("results@10", "entropy@10"), ("entdep@10", "entropy@10"), ("entdep@10", "results@10"),
             ("oracle@15", "entropy@15"), ("entdep@15", "entropy@15"), ("oracle@10", "entropy@10"), ("entdep@10", "entropy@10"),
             ("default", "entropy@15")]


def blt_items():
    lines = []
    for f in sorted(glob.glob(os.path.join(registry.ROOT, "results", "items", "*.json"))):
        it = json.load(open(f)); name = os.path.basename(f)[:-5]; L = it["data"]["layouts"]
        targets = []                                   # (label, key path getter, pid)
        if name.startswith("blt_finetune"):
            continue                                   # compared across models below (matched layouts)
        if name.startswith("blt_budget"):
            fin_pid = list(range(len(next(iter(L.values()))["fin_exact"])))
            targets = [("in-line results", lambda l: L[l]["res_exact"], it["data"]["res_pid"]), ("final answers", lambda l: L[l]["fin_exact"], fin_pid)]
        elif name.startswith("blt_code"):
            targets = [("repeated identifiers", lambda l: L[l]["exact"], it["data"]["pid"])]
        elif name.startswith(("blt_trace", "blt_logic", "blt_direct")):
            for kind in it["data"]["pid"]:
                key = "choice" if kind == "answer" else "exact"
                targets.append((f"{kind} ({key})", (lambda k, kk: lambda l: L[l][k][kk])(kind, key), it["data"]["pid"][kind]))
        elif name.startswith("blt_e2e"):
            targets = [("end-to-end final answer", lambda l: L[l]["correct"], list(range(len(it["data"]["gold"]))))]
        if not targets:
            continue
        lines += [f"\n### {name} ({it.get('commit')})\n", "| target | A | B | acc A | acc B | A - B [95% problem bootstrap] | A only / B only | McNemar p |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, get, pid in targets:
            pairs = [(a, b) for a, b in BLT_PAIRS if a in L and b in L]
            if name.startswith("blt_e2e"):
                ls = list(L); pairs = [(a, b) for a in ls for b in ls if a < b]
            for a, b in dict.fromkeys(pairs):
                x, y = np.array(get(a), bool), np.array(get(b), bool)
                if len(x) == 0:
                    continue
                p, nb, nc = mcnemar(x, y); lo, hi = boot(x, y, pid)
                lines.append(f"| {label} | {a} | {b} | {100 * x.mean():.1f}% | {100 * y.mean():.1f}% | {100 * (x.mean() - y.mean()):+.1f} "
                             f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {nb} / {nc} | {p:.2g} |")
    ft = defaultdict(dict)                             # rule -> seed -> items
    for f in sorted(glob.glob(os.path.join(registry.ROOT, "results", "items", "blt_finetune_*.json"))):
        it = json.load(open(f)); ft[it["trained_rule"]][it.get("seed", 0)] = it
    if ft:
        lay = {"entropy": "entropy@10", "entdep": "entdep@10", "results": "results@10"}
        acc = lambda r, sd: np.array(ft[r][sd]["data"]["layouts"][lay[r]]["res_exact"], bool)
        lines += ["\n### BLT-1B fine-tuned at 10% (LoRA, 1500 steps), each model under its own training layout\n",
                  "Per run: in-line computed results exact. Pooled test: all runs of A against all runs of B on the same targets "
                  "(seed s of A paired with seed s of B), bootstrap over problems.\n",
                  "| trained = tested | runs | exact by run | mean |", "|---|---|---|---|"]
        for r in ("entropy", "results", "entdep"):
            if r in ft:
                v = [100 * acc(r, sd).mean() for sd in sorted(ft[r])]
                lines.append(f"| {r} | {len(v)} | {', '.join(f'{x:.1f}' for x in v)} | {np.mean(v):.1f}% |")
        lines += ["", "| A | B | paired runs | A - B [95% problem bootstrap] | A only / B only | McNemar p |", "|---|---|---|---|---|---|"]
        for a, b in (("entdep", "entropy"), ("results", "entropy"), ("entdep", "results")):
            seeds = sorted(set(ft.get(a, {})) & set(ft.get(b, {})))
            if not seeds:
                continue
            x = np.concatenate([acc(a, sd) for sd in seeds]); y = np.concatenate([acc(b, sd) for sd in seeds])
            pid = np.concatenate([ft[a][sd]["data"]["res_pid"] for sd in seeds])
            p, nb, nc = mcnemar(x, y); lo, hi = boot(x, y, pid)
            lines.append(f"| {a} | {b} | {len(seeds)} | {100 * (x.mean() - y.mean()):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {nb} / {nc} | {p:.2g} |")
    return lines or ["\nNo per-item files yet (results/items/); rerun the BLT-1B scripts."]


def main():
    rows = mlx_rows()
    aud, few = audit(rows)
    out = ["# Statistics for the headline claims\n", "Generated by `stats.py` from the results registry and `results/items/`.\n",
           "## 1. Seed audit (MLX training runs)\n", f"{len(few)} configurations have fewer than 3 seeds (flagged **<3**).\n", *aud,
           "\n## 2. Headline comparisons between training rules\n",
           "Differences in percentage points. *Separated*: every seed of one rule beats every seed of the other. "
           "The Welch interval uses the seed means (few seeds, so wide); the pooled interval treats every target of every seed as "
           "independent and ignores seed variance (optimistic).\n",
           "| comparison | metric | seeds A/B | A - B | separated | Welch 95% (seeds) | pooled 95% (targets) |", "|---|---|---|---|---|---|---|"]
    for label, a, b, metric, where in COMPARISONS:
        c = compare(rows, a, b, metric, **where)
        if c is None:
            continue
        w = "" if math.isnan(c["welch"][0]) else f"[{100 * c['welch'][0]:+.1f}, {100 * c['welch'][1]:+.1f}]"
        out.append(f"| {label} | {metric.replace('_acc', '')} | {c['na']}/{c['nb']} | {100 * c['d']:+.1f} | {'yes' if c['sep'] else 'no'} | {w} | "
                   f"[{100 * c['pooled'][0]:+.1f}, {100 * c['pooled'][1]:+.1f}] |")
    out += ["\n## 3. Paired tests on BLT-1B (per-target outcomes)\n",
            "Each layout is scored on the same targets, so differences are paired. McNemar: exact test on the targets where "
            "exactly one layout is right. The bootstrap resamples whole problems.", *blt_items()]
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", os.path.relpath(OUT, registry.ROOT), f"; {len(few)} configurations below 3 seeds")


if __name__ == "__main__":
    main()
