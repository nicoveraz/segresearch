"""R15 (paper3/R15_NEWTOKEN_XDOMAIN_RULES.md): the label-free probe widened from numbers to any content. Logistic
probe on BLT-1B's patcher state (PCA-32, as R12) for "a word or number not seen earlier in the document starts
here" (new_any), fitted per domain (N_dom) or on the other five (N_lodo); new_num (digits only, NUM_dom) for
reference. Same capture analysis as the cross-domain test and R11 (whose gain-trained P_dom and two-byte G_dom are
recomputed alongside).

    .venv-blt/bin/python xdomain_newtok.py      -> results/xdomain_newtok.json   (XN_SMOKE=1: 10 + 10 documents)
"""
import json
import os

import numpy as np

import _seg  # noqa: F401
import realblt_learned as rl
import xdomain_analysis as xa
import xdomain_probe as xp

SMOKE = os.environ.get("XN_SMOKE") == "1"
NEG, MAXTR, NBOOT = 0.25, 60000, 1000
RULES = ("E", "G_dom", "P_dom", "NUM_dom", "N_dom", "N_lodo")


def labels(b):
    """new_any, new_num per token (token t = byte t-1; index 0 is the start token)."""
    L = len(b); na = np.zeros(L + 1); nn = np.zeros(L + 1); seen = set(); i = 0
    while i < L:
        if xa.isal(b[i]) and (i == 0 or not xa.isal(b[i - 1])):
            j = i
            while j < L and xa.isal(b[j]):
                j += 1
            w = bytes(b[i:j])
            if w not in seen:
                na[i + 1] = 1.0; nn[i + 1] = float(48 <= b[i] <= 57)
            seen.add(w); i = j
        else:
            i += 1
    return na, nn


def train_rows(D, H, doms, lab, rng):
    X, y = [], []
    for dom in doms:
        for i, d in enumerate(D[dom]["fit"]):
            yy = d[lab]; t = np.arange(2, len(yy))
            keep = (yy[t] > 0) | (np.random.default_rng(1000 * xa.DOMAINS.index(dom) + i).random(len(t)) < NEG)
            t = t[keep]
            X.append(H[f"{dom}_fit{i}"][t - 1]); y.append(yy[t])
    X = np.concatenate(X).astype(np.float32); y = np.concatenate(y)
    if len(y) > MAXTR:
        s = rng.choice(len(y), MAXTR, replace=False); X, y = X[s], y[s]
    return X, y


def score(m, h):
    s = np.full(len(h), -np.inf)                               # s[t] from h[t-1]
    s[1:] = rl.apply(m, h[:-1].astype(np.float32))
    s[:2] = -np.inf
    return s


def main():
    D = {d: xa.load(d) for d in xa.DOMAINS}
    H = xp.patcher_states(D)                                   # all 660 documents (cached by R11)
    if SMOKE:
        D = {d: {"fit": v["fit"][:10], "test": v["test"][:10]} for d, v in D.items()}
    for dom in xa.DOMAINS:
        for sp in ("fit", "test"):
            for d in D[dom][sp]:
                d["new_any"], d["new_num"] = labels(bytes(d["bytes"]))
    rng = np.random.default_rng(0)
    probes = {}
    for dom in xa.DOMAINS:
        probes[("N_dom", dom)] = rl.fit_logit(*train_rows(D, H, [dom], "new_any", rng))
        probes[("N_lodo", dom)] = rl.fit_logit(*train_rows(D, H, [o for o in xa.DOMAINS if o != dom], "new_any", rng))
        probes[("NUM_dom", dom)] = rl.fit_logit(*train_rows(D, H, [dom], "new_num", rng))
        print(f"probes fitted {dom}", flush=True)
    ref = {}
    for obj in ("bits", "unit"):
        for dom in xa.DOMAINS:
            ref[(obj, "P_dom", dom)] = xp.fit_ridge(*xp.train_rows(D, H, [dom], obj, rng))
            ref[(obj, "G_dom", dom)] = xa.fit_table(D[dom]["fit"], xp.FIELD[obj])
    brng = np.random.default_rng(1)
    res = {"domains": {}, "label_rates": {}, "auc": {}, "verdicts": {}}
    for dom in xa.DOMAINS:
        T = D[dom]["test"]
        na = np.concatenate([d["new_any"][2:] for d in T]); nn = np.concatenate([d["new_num"][2:] for d in T])
        res["label_rates"][dom] = {"new_any": float(na.mean()), "new_num": float(nn.mean())}
        sc_all = {r: [score(probes[(r, dom)], H[f"{dom}_test{i}"]) for i in range(len(T))] for r in ("N_dom", "N_lodo", "NUM_dom")}
        for r, lab in (("N_dom", na), ("N_lodo", na), ("NUM_dom", nn)):
            res["auc"][f"{dom}|{r}"] = float(rl.auc(lab, np.concatenate([s[2:] for s in sc_all[r]])))
        R = {}
        for obj in ("bits", "unit"):
            picked = {r: [] for r in RULES}; orac = []
            for i, d in enumerate(T):
                n = len(d["base"]); g = d[xp.FIELD[obj]]
                ok = np.isfinite(g); ok[:2] = False
                cand = ok & ~d["base"]
                if obj == "unit":
                    cand &= d["unit_len"] > 0
                k = min(int(round(xa.EXTRA * n)), int(cand.sum()))
                sc = {"E": d["entropy"], "G_dom": xa.lookup(ref[(obj, "G_dom", dom)], d),
                      "P_dom": xp.score(ref[(obj, "P_dom", dom)], H[f"{dom}_test{i}"])}
                for r in ("N_dom", "N_lodo", "NUM_dom"):
                    sc[r] = sc_all[r][i]
                for r in RULES:
                    picked[r].append(float(g[xa.pick(sc[r], cand, k, n)].sum()))
                orac.append(float(g[xa.pick(np.where(ok, g, -np.inf), cand, k, n)].sum()))
            P = {r: np.array(v) for r, v in picked.items()}; O = np.array(orac)
            B = [brng.integers(0, len(O), len(O)) for _ in range(NBOOT)]

            def diff(a, b):
                bd = [(P[a][x].sum() - P[b][x].sum()) / O[x].sum() for x in B]
                return {"value": float((P[a].sum() - P[b].sum()) / O.sum()),
                        "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
            R[obj] = {"capture": {r: float(P[r].sum() / O.sum()) for r in RULES},
                      "minus_E": {r: diff(r, "E") for r in RULES if r != "E"},
                      "N_minus_NUM": diff("N_dom", "NUM_dom"), "N_minus_P": diff("N_dom", "P_dom")}
            o = R[obj]
            print(f"{dom:9s} {obj:4s} | " + " | ".join(f"{r} {o['capture'][r]:.2f}" + ("" if r == "E" else
                  f" ({o['minus_E'][r]['value']:+.2f} [{o['minus_E'][r]['ci'][0]:+.2f},{o['minus_E'][r]['ci'][1]:+.2f}])") for r in RULES), flush=True)
        res["domains"][dom] = R
    dm = res["domains"]
    verdict = lambda c: "holds" if c >= 5 else "fails" if c <= 2 else "mixed"
    for obj in ("bits", "unit"):
        for v, r in (("V1", "N_dom"), ("V2", "N_lodo")):
            doms = [d for d in xa.DOMAINS if dm[d][obj]["minus_E"][r]["ci"][0] > 0]
            res["verdicts"][f"{v}_{obj}"] = {"verdict": verdict(len(doms)), "domains": doms}
    json.dump(res, open(os.path.join(_seg.HERE, "results", "xdomain_newtok_smoke.json" if SMOKE else "xdomain_newtok.json"), "w"), indent=1)
    print("AUC", json.dumps({k: round(v, 3) for k, v in res["auc"].items()}))
    print("LABEL RATES", json.dumps({d: {k: round(x, 3) for k, x in v.items()} for d, v in res["label_rates"].items()}))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
