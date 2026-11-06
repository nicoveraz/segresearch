"""R11 (paper3/R11_PROBE_XDOMAIN_RULES.md): the cross-domain test (X3) repeated with a probe on BLT-1B's patcher state
in place of the two-byte gain map. Ridge on PCA-32 of the state at token t-1, predicting the gain of a start at t;
fitted on the domain's own FIT documents (P_dom), the other five domains' (P_lodo) or math's (P_math). Same capture
analysis as xdomain_analysis.py; patcher states cached in ~/.cache/segresearch-xdomain/patcher_states.npz.

    .venv-blt/bin/python xdomain_probe.py        -> results/xdomain_probe.json
"""
import json
import os

import numpy as np

import _seg  # noqa: F401
import xdomain_analysis as xa

CACHE = os.path.expanduser("~/.cache/segresearch-xdomain/patcher_states.npz")
PCA, LAM, MAXTR, NBOOT = 32, 10.0, 60000, 1000
RULES = ("E", "G_dom", "G_lodo", "P_dom", "P_lodo", "P_math")
FIELD = {"bits": "gain_bits", "unit": "gain_unit"}


def patcher_states(D):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    import torch
    from transformers import AutoTokenizer
    import blt_load
    from realblt_budget import MODEL
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    cap = {}
    hook = model.model.patcher.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    out = {}
    for dom in xa.DOMAINS:
        for sp in ("fit", "test"):
            for i, d in enumerate(D[dom][sp]):
                b = bytes(d["bytes"])
                ids = tok(b.decode(), return_tensors="pt").input_ids.to(dev)
                assert ids.shape[1] == len(b) + 1
                with torch.no_grad():
                    model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                        max_patch_length=cfg.max_patch_length)
                out[f"{dom}_{sp}{i}"] = cap["h"][0].float().cpu().numpy().astype(np.float16)
                if dev == "mps" and i % 20 == 0:
                    torch.mps.empty_cache()
        print(f"patcher states {dom}", flush=True)
    hook.remove()
    np.savez(CACHE, **out)
    return out


def train_rows(D, H, doms, obj, rng):
    X, y = [], []
    for dom in doms:
        for i, d in enumerate(D[dom]["fit"]):
            g = d[FIELD[obj]]; m = np.isfinite(g) & ~d["base"]; m[:2] = False
            if obj == "unit":
                m &= d["unit_len"] > 0
            t = np.flatnonzero(m)
            X.append(H[f"{dom}_fit{i}"][t - 1]); y.append(g[t])
    X = np.concatenate(X).astype(np.float32); y = np.concatenate(y)
    if len(y) > MAXTR:
        s = rng.choice(len(y), MAXTR, replace=False); X, y = X[s], y[s]
    return X, y


def fit_ridge(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A = (X - mu) / sd
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    V = Vt[:PCA].T
    A1 = np.c_[A @ V, np.ones(len(A))]
    w = np.linalg.solve(A1.T @ A1 + LAM * np.eye(A1.shape[1]), A1.T @ y)
    return mu, sd, V, w


def score(m, h):
    mu, sd, V, w = m
    s = np.full(len(h), -np.inf)                               # one state per token; s[t] from h[t-1]
    s[1:] = np.c_[((h[:-1].astype(np.float32) - mu) / sd) @ V, np.ones(len(h) - 1)] @ w
    s[:2] = -np.inf
    return s


def main():
    D = {d: xa.load(d) for d in xa.DOMAINS}
    H = patcher_states(D)
    rng = np.random.default_rng(0)
    probes = {}
    for obj in ("bits", "unit"):
        for dom in xa.DOMAINS:
            probes[(obj, "P_dom", dom)] = fit_ridge(*train_rows(D, H, [dom], obj, rng))
            probes[(obj, "P_lodo", dom)] = fit_ridge(*train_rows(D, H, [o for o in xa.DOMAINS if o != dom], obj, rng))
        pm = fit_ridge(*train_rows(D, H, ["math"], obj, rng))
        for dom in xa.DOMAINS:
            probes[(obj, "P_math", dom)] = pm
        print(f"probes fitted ({obj})", flush=True)
    tabs = {}
    for obj in ("bits", "unit"):
        for dom in xa.DOMAINS:
            tabs[(obj, "G_dom", dom)] = xa.fit_table(D[dom]["fit"], FIELD[obj])
            tabs[(obj, "G_lodo", dom)] = xa.fit_table([x for o in xa.DOMAINS if o != dom for x in D[o]["fit"]], FIELD[obj])
    brng = np.random.default_rng(1)
    res = {"domains": {}, "verdicts": {}}
    for dom in xa.DOMAINS:
        R = {}
        for obj in ("bits", "unit"):
            picked = {r: [] for r in RULES}; orac = []; rho = {r: ([], []) for r in ("E", "P_dom", "P_lodo")}
            for i, d in enumerate(D[dom]["test"]):
                n = len(d["base"]); g = d[FIELD[obj]]
                ok = np.isfinite(g); ok[:2] = False
                cand = ok & ~d["base"]
                if obj == "unit":
                    cand &= d["unit_len"] > 0
                k = min(int(round(xa.EXTRA * n)), int(cand.sum()))
                h = H[f"{dom}_test{i}"]
                sc = {"E": d["entropy"], "G_dom": xa.lookup(tabs[(obj, "G_dom", dom)], d),
                      "G_lodo": xa.lookup(tabs[(obj, "G_lodo", dom)], d)}
                for r in ("P_dom", "P_lodo", "P_math"):
                    sc[r] = score(probes[(obj, r, dom)], h)
                for r in RULES:
                    picked[r].append(float(g[xa.pick(sc[r], cand, k, n)].sum()))
                orac.append(float(g[xa.pick(np.where(ok, g, -np.inf), cand, k, n)].sum()))
                for r in rho:
                    rho[r][0].append(sc[r][cand]); rho[r][1].append(g[cand])
            P = {r: np.array(v) for r, v in picked.items()}; O = np.array(orac)
            B = [brng.integers(0, len(O), len(O)) for _ in range(NBOOT)]
            out = {"capture": {}, "minus_E": {}, "P_minus_G": {}, "spearman": {}}
            for r in RULES:
                out["capture"][r] = float(P[r].sum() / O.sum())
                if r != "E":
                    bd = [(P[r][b].sum() - P["E"][b].sum()) / O[b].sum() for b in B]
                    out["minus_E"][r] = {"value": float((P[r].sum() - P["E"].sum()) / O.sum()),
                                         "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
            for p_, g_ in (("P_dom", "G_dom"), ("P_lodo", "G_lodo")):
                bd = [(P[p_][b].sum() - P[g_][b].sum()) / O[b].sum() for b in B]
                out["P_minus_G"][p_] = {"value": float((P[p_].sum() - P[g_].sum()) / O.sum()),
                                        "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
            for r, (a, b) in rho.items():
                out["spearman"][r] = xa.spearman(np.concatenate(a), np.concatenate(b))
            R[obj] = out
        res["domains"][dom] = R
        for obj in ("bits", "unit"):
            o = R[obj]
            print(f"{dom:9s} {obj:4s} | " + " | ".join(f"{r} {o['capture'][r]:.2f}" + ("" if r == "E" else
                  f" ({o['minus_E'][r]['value']:+.2f} [{o['minus_E'][r]['ci'][0]:+.2f},{o['minus_E'][r]['ci'][1]:+.2f}])") for r in RULES), flush=True)
    dm = res["domains"]
    above = lambda dom, obj, r: dm[dom][obj]["minus_E"][r]["ci"][0] > 0
    verdict = lambda c: "holds" if c >= 5 else "fails" if c <= 2 else "mixed"
    for obj in ("bits", "unit"):
        y1 = [d for d in xa.DOMAINS if above(d, obj, "P_dom")]
        y2 = [d for d in xa.DOMAINS if above(d, obj, "P_lodo")]
        res["verdicts"][f"Y1_{obj}"] = {"verdict": verdict(len(y1)), "domains": y1}
        res["verdicts"][f"Y2_{obj}"] = {"verdict": verdict(len(y2)), "domains": y2}
    y3 = dm["algebra"]["unit"]["minus_E"]["P_math"]
    res["verdicts"]["Y3"] = {"verdict": "holds" if y3["ci"][0] > 0 else "does not hold", **y3}
    json.dump(res, open(os.path.join(_seg.HERE, "results", "xdomain_probe.json"), "w"), indent=1)
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
