"""Reuse vs novelty: can a copy-aware signal find computed values without spending the budget on copies? (#31)

Computed values have a predictable type but NEW content; copies and lookups repeat content already in the context
and gain little from a patch start. Entropy is low for both. This screen measures reuse directly, as LZ77 does,
within the model's context (CTX = 128 bytes):

    novelty   NON-causal diagnostic: length of the longest prefix of b[t:] that also starts somewhere in the previous
              CTX bytes (forward match, capped at 16). Short = new content. Not a legal rule: it reads byte t onward.
    suffix    causal: length of the longest suffix of b[:t] that also occurs earlier in the window (inside a repeat?)
    ambig     causal: how many different next bytes the earlier occurrences of the current >=2-byte suffix predict
              (an induction-style copy predictor that disagrees with itself: a slot whose filler varies)
    novtab    causal, self-supervised: expected novelty per preceding 2-byte context, fitted on train text (the
              dependence table's form, but the target comes from the text itself, not from model losses)
    novnet    causal, self-supervised: a small network (neural_patcher.Patcher, last K bytes) predicting novelty

Each score alone and combined with entropy (standardized sum) at a 10% budget (threshold = train quantile), against
entropy10, dep10 (the boundary-dependence table) and syntax+entropy10. Reported: coverage of computed / copy / final
/ step starts and of other word starts, and mean novelty and entropy at each. Corpora: the math corpus (GSM8K +
MATH) and the logic + program-trace corpus. The tables and networks are fitted on one corpus and applied to both,
so the transfer column shows whether the signal is domain-free (#5: the GSM8K dependence table covers 0% of traces).

    uv run reuse_screen.py [K] [EPOCHS]
"""
import os
import sys

import numpy as np

import mathexp
import realtext
from prepare import CTX, RI

CORPORA = {"math": os.path.expanduser("~/.cache/segresearch-math/math.npz"),
           "reason": os.path.expanduser("~/.cache/segresearch-reason/math.npz")}
SYNTAX = {"math": [b"=", b"\\boxed{", b"#### ", b">>"], "reason": None}
CAP = 16
N_FIT = 2_000_000                       # train bytes used to fit thresholds, tables and networks
R = 0.10


def _runs_forward(eq, cap):
    """run[t] = number of consecutive True in eq starting at t (capped)."""
    run = np.zeros(len(eq) + 1, np.int16)
    for k in range(cap):                                           # cap passes: run[t] = eq[t] * (1 + run[t+1])
        run[:-1] = eq * (1 + run[1:])
    return np.minimum(run[:-1], cap)


def match_signals(b, W=CTX, cap=CAP):
    """novelty (forward match, non-causal), suffix (backward match, causal) and ambig (causal), per position."""
    n = len(b); b = b.astype(np.int16)
    nov = np.zeros(n, np.int16); suf = np.zeros(n, np.int16)
    preds = []                                                     # (offset run, predicted byte) per offset
    for d in range(1, W):
        eq = np.zeros(n, bool); eq[d:] = b[d:] == b[:-d]           # eq[t]: b[t] == b[t-d]
        f = _runs_forward(eq, cap)
        nov = np.maximum(nov, f)
        # backward run ending at t-1: number of consecutive True in eq at t-1, t-2, ...
        bk = _runs_forward(eq[::-1], cap)[::-1]                     # run ending at t (inclusive)
        bt = np.r_[0, bk[:-1]].astype(np.int16)                     # ending at t-1 -> suffix of b[:t]
        bt[:d] = 0
        suf = np.maximum(suf, bt)
        pb = np.full(n, -1, np.int16); pb[d:] = b[:-d]              # the byte the copy at offset d predicts for t
        preds.append(np.where(bt >= 2, pb, -1))
    P = np.stack(preds)                                            # (W-1, n); -1 = no >=2-byte match at this offset
    P.sort(axis=0)
    distinct = ((P[1:] != P[:-1]) & (P[1:] >= 0)).sum(0) + (P[0] >= 0)
    return nov.astype(np.float32), suf.astype(np.float32), distinct.astype(np.float32)


def starts(roles, role):
    m = roles == RI[role]
    return m & ~np.r_[False, m[:-1]]


def tab_fit(b, target):
    b = b.astype(np.int64); p1 = np.r_[0, b[:-1]]; p2 = np.r_[0, 0, b[:-2]]
    key = p2 * 256 + p1
    s2, c2 = np.bincount(key, target, 65536), np.bincount(key, None, 65536)
    s1, c1 = np.bincount(p1, target, 256), np.bincount(p1, None, 256)
    return (np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan), np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan),
            float(target.mean()))


def net_fit(b, target, K, epochs):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import neural_patcher
    rng = np.random.default_rng(0)
    pos = rng.choice(np.arange(K, len(b)), 600_000, replace=False)
    X, y = neural_patcher.contexts(b, pos, K), target[pos].astype(np.float32)
    model = neural_patcher.Patcher(K); mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    step = nn.value_and_grad(model, lambda m, x, t: ((m(x) - t) ** 2).mean())
    idx = np.arange(len(y))
    for ep in range(epochs):
        rng.shuffle(idx); tot = 0.0
        for s in range(0, len(idx), 1024):
            j = idx[s:s + 1024]
            l, g = step(model, mx.array(X[j]), mx.array(y[j])); opt.update(model, g)
            mx.eval(model.parameters(), opt.state); tot += l.item() * len(j)
        print(f"    epoch {ep + 1}: mse {tot / len(idx):.3f} (target var {y.var():.3f})", flush=True)
    return lambda bb: neural_patcher.predict(model, bb)


def main(K, epochs):
    import deptrigger
    data = {}
    for c, path in CORPORA.items():
        z = np.load(path)
        tr = slice(len(z["train_bytes"]) - N_FIT, None)               # the end of train (problems are shuffled)
        d = {"btr": z["train_bytes"][tr], "Htr": z["train_H"][tr], "bva": z["val_bytes"], "Hva": z["val_H"],
             "roles": z["val_roles"]}
        print(f"{c}: match signals on {len(d['btr']):,} train + {len(d['bva']):,} val bytes", flush=True)
        d["tr"] = match_signals(d["btr"]); d["va"] = match_signals(d["bva"])
        t = np.load(os.path.join(os.path.dirname(path), "deptrigger.npz"))
        d["dep"] = lambda bb, t=t: deptrigger.table_lookup(t["T2"], t["T1"], float(t["glob"]), bb)
        data[c] = d

    # self-supervised novelty predictors, fitted on each corpus (target: -novelty, so higher = newer content)
    fitted = {}
    for c, d in data.items():
        fitted[f"dep[{c}]"] = d["dep"]                                 # the boundary-dependence table (model losses)
    for c, d in data.items():
        target = -d["tr"][0]
        fitted[f"novtab[{c}]"] = (lambda tb: (lambda bb: deptrigger.table_lookup(*tb, bb)))(tab_fit(d["btr"], target))
        print(f"  novnet fitted on {c} (K={K})", flush=True)
        fitted[f"novnet[{c}]"] = net_fit(d["btr"], target, K, epochs)

    roles_of = {"computed": "ANS_LOCAL", "copy": "VALUE", "final": "ANS_LONG", "step": "VAR"}
    out = {}
    for c, d in data.items():
        roles = d["roles"]
        tgt = {k: starts(roles, v) for k, v in roles_of.items()}
        ans = np.zeros(len(roles), bool)
        for v in tgt.values():
            ans |= v
        tgt["word"] = realtext._word_starts(d["bva"]) & ~ans
        print(f"\n== {c}: diagnostic (mean at each kind of start)")
        nov, suf, amb = d["va"]
        for k, m in tgt.items():
            print(f"  {k:9s} n={m.sum():6d} | novelty {nov[m].mean():5.2f} (share with no 3-byte match {(nov[m] < 3).mean():5.1%}) "
                  f"| entropy {d['Hva'][m].mean():4.2f} | suffix {suf[m].mean():4.1f} | ambig {amb[m].mean():4.1f}")
        print(f"  {'all':9s} n={len(nov):6d} | novelty {nov.mean():5.2f} (share with no 3-byte match {(nov < 3).mean():5.1%}) "
              f"| entropy {d['Hva'].mean():4.2f} | suffix {suf.mean():4.1f} | ambig {amb.mean():4.1f}")

        # scores (higher = start a patch); each as (train values, val values)
        S = {"entropy": (d["Htr"], d["Hva"]),
             "novelty(oracle)": (-d["tr"][0], -nov),
             "suffix": (-d["tr"][1], -suf),
             "ambig": (d["tr"][2], amb)}
        for name, f in fitted.items():
            S[name] = (f(d["btr"]), f(d["bva"]))
        z = lambda a, ref: (a - ref.mean()) / (ref.std() + 1e-9)
        rows = {}
        for name, (a, v) in S.items():
            rows[f"{name}10"] = v > np.quantile(a, 1 - R)
            if name != "entropy":
                ca, cv = z(d["Htr"], d["Htr"]) + z(a, a), z(d["Hva"], d["Htr"]) + z(v, a)
                rows[f"entropy+{name}10"] = cv > np.quantile(ca, 1 - R)
        if SYNTAX[c]:
            syn_tr, syn_va = mathexp._syntax_starts(d["btr"]), mathexp._syntax_starts(d["bva"])
            rest = ~syn_tr
            thr = np.quantile(d["Htr"][rest], 1 - (R - syn_tr.mean()) / rest.mean())
            rows["syntax+entropy10"] = syn_va | (d["Hva"] > thr)
        print(f"\n== {c}: coverage at a {R:.0%} budget (share of each kind of start that starts a patch)")
        print(f"  {'rule':34s} rate  " + " ".join(f"{k:>8s}" for k in tgt))
        out[c] = {}
        for name, m in rows.items():
            cov = {k: float(m[v].mean()) for k, v in tgt.items()}
            out[c][name] = {"rate": float(m.mean()), **cov}
            print(f"  {name:34s} {m.mean():.3f} " + " ".join(f"{cov[k]:8.1%}" for k in tgt), flush=True)
    import json
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/reuse_screen.json", "w"), indent=1)
    print("\n-> results/reuse_screen.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 16, int(a[1]) if len(a) > 1 else 3)
