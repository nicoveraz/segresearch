"""Analysis of the cross-domain test (paper3/XDOMAIN_RULES.md): which rule for spending an extra 3% of patch starts
captures the most gain, in each of six domains, on the bits and unit objectives; verdicts X1-X4.

Reads results/xdomain_gains_{domain}.npz (realblt_xdomain.py), writes results/xdomain.json.

    ~/anaconda3/bin/python xdomain_analysis.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS = ("math", "algebra", "code", "prose", "clinical", "traces")
EXTRA, MIN_COUNT, NBOOT = 0.03, 20, 1000
RULES = ("E", "N", "H", "U", "G_dom", "G_lodo", "G_math")


def load(dom):
    z = np.load(os.path.join(HERE, "results", f"xdomain_gains_{dom}.npz"))
    out = {}
    for sp, n in (("fit", int(z["n_fit"])), ("test", int(z["n_test"]))):
        out[sp] = [{k: z[f"{sp}{i}_{k}"] for k in ("bytes", "entropy", "novelty", "base", "gain_bits", "gain_unit",
                                                   "unit_len")} for i in range(n)]
    out["fit_rate"] = float(z["fit_rate"])
    return out


def keys(d):
    """Two-byte and one-byte context of every token t: bytes t-3 and t-2 (token t is byte t-1), 0-padded."""
    b = np.r_[0, 0, 0, d["bytes"].astype(np.int64)]          # b[j + 3] is byte j
    n = len(d["bytes"]) + 1
    t = np.arange(n)
    p2, p1 = b[t], b[t + 1]                                  # bytes t-3, t-2
    return p2 * 256 + p1, p1


def fit_table(docs, field):
    s2 = np.zeros(65536); c2 = np.zeros(65536); s1 = np.zeros(256); c1 = np.zeros(256); tot = []
    for d in docs:
        k2, k1 = keys(d)
        g = d[field]; m = np.isfinite(g); m[:2] = False
        np.add.at(s2, k2[m], g[m]); np.add.at(c2, k2[m], 1); np.add.at(s1, k1[m], g[m]); np.add.at(c1, k1[m], 1)
        tot.append(g[m])
    glob = float(np.concatenate(tot).mean())
    t1 = np.where(c1 >= MIN_COUNT, s1 / np.maximum(c1, 1), glob)
    t2 = np.where(c2 >= MIN_COUNT, s2 / np.maximum(c2, 1), np.tile(t1, 256))
    return {"t2": t2, "c2": c2}


def lookup(tab, d):
    k2, _ = keys(d)
    v = tab["t2"][k2].astype(float); v[:2] = -np.inf
    return v


def isal(x):
    return (48 <= x <= 57) or (65 <= x <= 90) or (97 <= x <= 122)


def byte_class(d):
    """Class of token t by its byte b[t-1]: value (digit run after '=' or '->'), digit, word, inside a run, space, other."""
    b = d["bytes"]; n = len(b) + 1
    cls = np.array(["other"] * n, dtype=object)
    for t in range(2, n):
        c = b[t - 1]
        if c in (32, 10, 9):
            cls[t] = "space"
        elif isal(c) and isal(b[t - 2]):
            cls[t] = "inside"
        elif 48 <= c <= 57:
            j = t - 2
            while j >= 0 and b[j] in (32, 36):                       # skip spaces and '$'
                j -= 1
            cue = j >= 0 and (b[j] == 61 or (b[j] == 62 and j >= 1 and b[j - 1] == 45))   # '=' or '->'
            cls[t] = "value" if cue else "digit"
        elif isal(c):
            cls[t] = "word"
    cls[:2] = "none"
    return cls


def tiebreak(n):
    return (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def pick(score, cand, k, n):
    idx = np.where(cand)[0]
    if not len(idx) or k <= 0:
        return idx[:0]
    order = np.lexsort((-tiebreak(n)[idx], -score[idx]))
    return idx[order[:k]]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    D = {d: load(d) for d in DOMAINS}
    tabs = {}
    for obj, field in (("bits", "gain_bits"), ("unit", "gain_unit")):
        for d in DOMAINS:
            tabs[(obj, "G_dom", d)] = fit_table(D[d]["fit"], field)
            tabs[(obj, "G_lodo", d)] = fit_table([x for o in DOMAINS if o != d for x in D[o]["fit"]], field)
            tabs[(obj, "G_math", d)] = fit_table(D["math"]["fit"], field)
        tabs[(obj, "pooled")] = fit_table([x for o in DOMAINS for x in D[o]["fit"]], field)
    rng = np.random.default_rng(0)
    res = {"rules": RULES, "extra": EXTRA, "domains": {}}
    for dom in DOMAINS:
        fitd = D[dom]["fit"]
        zs = {}
        for f in ("entropy", "novelty"):
            x = np.concatenate([d[f][2:][np.isfinite(d[f][2:])] for d in fitd]); zs[f] = (x.mean(), x.std())
        R = {"fit_rate": D[dom]["fit_rate"], "test_rate": float(np.mean([d["base"].mean() for d in D[dom]["test"]]))}
        for obj, field in (("bits", "gain_bits"), ("unit", "gain_unit")):
            picked = {r: [] for r in RULES}; orac = []; rho = {r: [] for r in ("E", "N", "H", "G_dom", "G_lodo")}
            for d in D[dom]["test"]:
                n = len(d["base"]); g = d[field]
                ok = np.isfinite(g); ok[:2] = False
                cand = ok & ~d["base"]
                if obj == "unit":
                    cand &= d["unit_len"] > 0
                k = min(int(round(EXTRA * n)), int(cand.sum()))
                zE = (d["entropy"] - zs["entropy"][0]) / zs["entropy"][1]
                zN = (d["novelty"] - zs["novelty"][0]) / zs["novelty"][1]
                sc = {"E": d["entropy"], "N": d["novelty"], "H": zE + zN,
                      "U": np.where(d["unit_len"] > 0, 1e6, 0.0) + zE}
                for r in ("G_dom", "G_lodo", "G_math"):
                    sc[r] = lookup(tabs[(obj, r, dom)], d)
                for r in RULES:
                    picked[r].append(float(g[pick(sc[r], cand, k, n)].sum()))
                orac.append(float(g[pick(np.where(ok, g, -np.inf), cand, k, n)].sum()))
                if obj == "bits":
                    for r in rho:
                        rho[r].append((sc[r][ok], g[ok]))
            P = {r: np.array(v) for r, v in picked.items()}; O = np.array(orac)
            B = [rng.integers(0, len(O), len(O)) for _ in range(NBOOT)]
            out = {"oracle_gain_per_doc": float(O.mean()), "capture": {}, "minus_E": {}}
            for r in RULES:
                cap = P[r].sum() / O.sum()
                bs = np.array([P[r][i].sum() / O[i].sum() for i in B])
                out["capture"][r] = {"value": float(cap), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     "gain_per_doc": float(P[r].mean())}
                if r != "E":
                    bd = np.array([(P[r][i].sum() - P["E"][i].sum()) / O[i].sum() for i in B])
                    out["minus_E"][r] = {"value": float((P[r].sum() - P["E"].sum()) / O.sum()),
                                         "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
            if obj == "bits":
                out["spearman"] = {r: spearman(np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v]))
                                   for r, v in rho.items()}
            R[obj] = out
        cls = {}
        for d in D[dom]["test"]:
            c = byte_class(d)
            for name in ("value", "digit", "word", "inside", "space", "other"):
                m = (c == name) & np.isfinite(d["gain_bits"])
                cls.setdefault(name, {"bits": [], "unit": [], "base": []})
                cls[name]["bits"] += list(d["gain_bits"][m]); cls[name]["base"] += list(d["base"][m])
                mu = m & np.isfinite(d["gain_unit"])
                cls[name]["unit"] += list(d["gain_unit"][mu])
        R["by_class"] = {k: {"n": len(v["bits"]), "share": None, "gain_bits": float(np.mean(v["bits"])) if v["bits"] else None,
                             "gain_unit": float(np.mean(v["unit"])) if v["unit"] else None,
                             "in_base": float(np.mean(v["base"])) if v["base"] else None} for k, v in cls.items()}
        tot = sum(v["n"] for v in R["by_class"].values())
        for v in R["by_class"].values():
            v["share"] = v["n"] / tot
        allg = np.concatenate([d["gain_bits"][2:] for d in D[dom]["test"]]); allb = np.concatenate([d["base"][2:] for d in D[dom]["test"]])
        R["removal_side"] = {"gain_bits_at_base_starts": float(np.nanmean(allg[allb])), "gain_bits_elsewhere": float(np.nanmean(allg[~allb]))}
        res["domains"][dom] = R
    # top contexts of the pooled tables
    for obj in ("bits", "unit"):
        t = tabs[(obj, "pooled")]; c = t["c2"]; m = c >= 50
        top = np.argsort(-np.where(m, t["t2"], -np.inf))[:12]
        res[f"top_contexts_{obj}"] = [{"context": repr(bytes([k // 256, k % 256]).decode("latin1")), "gain": float(t["t2"][k]),
                                       "count": int(c[k])} for k in top]
    # verdicts
    dm = res["domains"]
    above = lambda dom, obj, r: dm[dom][obj]["minus_E"][r]["ci"][0] > 0
    below = lambda dom, obj, r: dm[dom][obj]["minus_E"][r]["ci"][1] < 0
    x1 = [d for d in DOMAINS if above(d, "unit", "H")]
    v1 = "holds" if len(x1) >= 5 else "fails" if len(x1) <= 2 else "mixed"
    x2 = [d for d in DOMAINS if not above(d, "bits", "N") and above(d, "unit", "H")]
    v2 = "holds" if len(x2) >= 5 else "does not hold"
    v3 = {}
    for obj in ("bits", "unit"):
        a = [d for d in DOMAINS if above(d, obj, "G_lodo")]
        b = [d for d in DOMAINS if dm[d][obj]["capture"]["G_lodo"]["value"] >= 0.8 * dm[d][obj]["capture"]["G_dom"]["value"]]
        v3[obj] = {"verdict": "holds" if len(a) >= 5 and len(b) >= 5 else "fails" if len(a) <= 2 else "mixed",
                   "lodo_above_E": a, "lodo_within_80pct_of_dom": b}
    al = dm["algebra"]["unit"]["minus_E"]
    v4 = "holds" if al["G_math"]["ci"][0] > 0 and al["N"]["ci"][0] > 0 else "does not hold"
    res["verdicts"] = {"X1": {"verdict": v1, "domains_H_above_E_unit": x1},
                       "X2": {"verdict": v2, "domains": x2,
                              "N_minus_E_bits_below0": [d for d in DOMAINS if below(d, "bits", "N")]},
                       "X3": v3, "X4": {"verdict": v4, "G_math_minus_E": al["G_math"], "N_minus_E": al["N"]}}
    json.dump(res, open(os.path.join(HERE, "results", "xdomain.json"), "w"), indent=1)
    for dom in DOMAINS:
        R = dm[dom]
        print(f"== {dom}: base rate test {R['test_rate']:.3f} (fit {R['fit_rate']:.3f}); removal side: gain at base starts "
              f"{R['removal_side']['gain_bits_at_base_starts']:+.2f} bits, elsewhere {R['removal_side']['gain_bits_elsewhere']:+.2f}")
        for obj in ("bits", "unit"):
            o = R[obj]
            print(f"   {obj:4s} oracle/doc {o['oracle_gain_per_doc']:.2f} | " + " | ".join(
                f"{r} {o['capture'][r]['value']:.2f}" + ("" if r == "E" else f" ({o['minus_E'][r]['value']:+.2f} [{o['minus_E'][r]['ci'][0]:+.2f},{o['minus_E'][r]['ci'][1]:+.2f}])")
                for r in RULES))
        print("   by class (share, bits gain, unit gain, in base): " + "; ".join(
            f"{k} {v['share']:.2f} {v['gain_bits'] if v['gain_bits'] is None else round(v['gain_bits'], 2)} "
            f"{v['gain_unit'] if v['gain_unit'] is None else round(v['gain_unit'], 3)} {v['in_base'] if v['in_base'] is None else round(v['in_base'], 2)}"
            for k, v in R["by_class"].items()))
        print("   spearman with bits gain: " + ", ".join(f"{k} {v:.2f}" for k, v in R["bits"]["spearman"].items()))
    for obj in ("bits", "unit"):
        print(f"top pooled contexts ({obj}):", ", ".join(f"{c['context']} {c['gain']:+.2f} (n {c['count']})" for c in res[f"top_contexts_{obj}"]))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
