"""R14 (paper3/R14_PROBE_READS_RULES.md): does the patcher-state probe read "a computed number comes next" or only
"some number comes next"? R12's setup plus a probe trained on "a number starts here" (any number), within-number
AUCs, the composition of each probe's forced set, BLT-1B layouts with each probe's top 3% forced, and the words
before high-scoring candidates.

    .venv-blt/bin/python realblt_probe_reads.py      -> results/blt_probe_reads.json   (R14_SMOKE=1: 40 / 20 problems)
"""
import collections
import json
import os
import re

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import realblt_learned as rl
from realblt_autolabel import autolabel
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts

SMOKE = os.environ.get("R14_SMOKE") == "1"
PROBES = {"marker_prose": ("prose", "y"), "auto_prose": ("prose", "ya"), "anynum_prose": ("prose", "num"),
          "auto_orig": ("orig", "ya"), "anynum_orig": ("orig", "num")}
LAYOUTS = ("marker_prose", "auto_prose", "anynum_prose")
R, Q, NEG, NBOOT = rl.R, rl.Q, rl.NEG, rl.NBOOT
WORD = re.compile(rb"[A-Za-z$%]+")


def context(b, i, n=2):
    """The last n words before byte i, lowercased (the phrase a candidate follows)."""
    return " ".join(w.decode().lower() for w in WORD.findall(b[max(0, i - 40):i])[-n:])


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    recs = {sp: [json.loads(l) for l in open(os.path.join(rl.DATA, f"{sp}.jsonl"))] for sp in ("train", "test")}
    if SMOKE:
        recs = {"train": recs["train"][:40], "test": recs["test"][:20]}
    btok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    cap = {}
    hook = model.model.patcher.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    P = {}
    for f in rl.FORMATS:
        for sp, rs in recs.items():
            xs = []
            for r in rs:
                x = rl.build(r, f)
                ids = btok(x["b"].decode(), return_tensors="pt").input_ids
                if ids.shape[1] > MAX_TOKENS:
                    continue
                b = x["b"]
                x["idx"] = r["idx"]; x["ya"] = autolabel(x)
                x["num"] = np.array([48 <= b[i] <= 57 for i in x["cand"]], float)
                ids = ids.to(dev); n = ids.shape[1]
                with torch.no_grad():
                    ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                    max_patch_length=cfg.max_patch_length)
                e = ent[0].float().cpu().numpy(); h = cap["h"][0].float().cpu().numpy()
                s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
                x.update({"ids": ids.cpu(), "sig": {"entropy": s}, "Hp": h[x["cand"]].astype(np.float16), "score": {}})
                xs.append(x)
            P[(f, sp)] = xs
            print(f"patcher {f}/{sp}: {len(xs)} problems", flush=True)
    hook.remove()

    rng = np.random.default_rng(0)
    stack = lambda xs, fld, sel=None: np.concatenate([(x[fld] if sel is None else x[fld][x[sel]]) for x in xs]).astype(np.float32)
    for pname, (ftr, lab) in PROBES.items():
        tr = P[(ftr, "train")]
        for x in tr:
            x["keep"] = (x[lab] > 0) | (np.random.default_rng(x["idx"]).random(len(x[lab])) < NEG)
        m = rl.fit_logit(stack(tr, "Hp", "keep"), stack(tr, lab, "keep"))
        for (f, sp), xs in P.items():
            if f == ftr and sp == "train":                    # out-of-fold scores on the probe's own train problems
                fold = {i: j % 5 for j, i in enumerate(rng.permutation(len(xs)))}
                for fo in range(5):
                    trx = [x for i, x in enumerate(xs) if fold[i] != fo]
                    mf = rl.fit_logit(stack(trx, "Hp", "keep"), stack(trx, lab, "keep"))
                    for i, x in enumerate(xs):
                        if fold[i] == fo:
                            x["score"][pname] = rl.apply(mf, x["Hp"].astype(np.float32))
            else:
                for x in xs:
                    x["score"][pname] = rl.apply(m, x["Hp"].astype(np.float32))
        print(f"probe {pname} fitted", flush=True)

    res = {"counts": {}, "auc": {}, "forced": {}, "layouts": {}, "contexts": {}, "verdicts": {}}
    brng = np.random.default_rng(1)

    def boot_auc(xs, lab, pname, mask=None):
        def one(pk):
            y = np.concatenate([x[lab] for x in pk]); s = np.concatenate([x["score"][pname] for x in pk])
            if mask:
                k = np.concatenate([x[mask] for x in pk]) > 0; y, s = y[k], s[k]
            return rl.auc(y, s)
        bs = [one([xs[i] for i in brng.integers(0, len(xs), len(xs))]) for _ in range(NBOOT)]
        return {"value": float(one(xs)), "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}

    for f in rl.FORMATS:
        xs = P[(f, "test")]
        y, ya, nm = (np.concatenate([x[k] for x in xs]) for k in ("y", "ya", "num"))
        res["counts"][f] = {"candidates": int(len(y)), "numbers": int(nm.sum()), "computed": int(y.sum()),
                            "other_new": int(((ya > 0) & (y == 0)).sum()), "copies": int(((nm > 0) & (ya == 0) & (y == 0)).sum()),
                            "computed_not_new": int(((y > 0) & (ya == 0)).sum()),
                            "bytes": int(sum(x["ids"].shape[1] for x in xs))}
        print(f"counts {f}: {res['counts'][f]}", flush=True)
        for pname in PROBES:
            for meas, lab, mask in (("computed_all", "y", None), ("computed_within_numbers", "y", "num"),
                                    ("number_all", "num", None)):
                res["auc"][f"{f}|{pname}|{meas}"] = r_ = boot_auc(xs, lab, pname, mask)
                print(f"AUC {f:5s} {pname:12s} {meas:24s} {r_['value']:.3f} [{r_['ci'][0]:.3f}, {r_['ci'][1]:.3f}]", flush=True)

    rows = []
    for f in rl.FORMATS:
        tr = P[(f, "train")]; ntok = sum(x["ids"].shape[1] for x in tr)
        for pname in PROBES:
            allsc = np.sort(np.concatenate([x["score"][pname] for x in tr]))[::-1]
            thr = float(allsc[min(len(allsc) - 1, int(Q * ntok))])
            for x in P[(f, "train")] + P[(f, "test")]:
                x.setdefault("forced", {})[pname] = (x["cand"][x["score"][pname] > thr] + 1).tolist()
            xs = P[(f, "test")]
            fm = np.concatenate([x["score"][pname] > thr for x in xs])
            y, ya, nm = (np.concatenate([x[k] for x in xs]) for k in ("y", "ya", "num"))
            k = max(1, int(fm.sum()))
            res["forced"][f"{f}|{pname}"] = {"n_forced": int(fm.sum()), "share_computed": float((fm & (y > 0)).sum() / k),
                                             "share_other_new": float((fm & (ya > 0) & (y == 0)).sum() / k),
                                             "share_copies": float((fm & (nm > 0) & (ya == 0) & (y == 0)).sum() / k),
                                             "share_non_number": float((fm & (nm == 0)).sum() / k),
                                             "recall_computed": float((fm & (y > 0)).sum() / max(1, (y > 0).sum()))}
            print(f"FORCED {f:5s} {pname:12s} {json.dumps({a: round(b, 3) for a, b in res['forced'][f'{f}|{pname}'].items()})}", flush=True)
        fit = blt_layouts.Fit((R,))
        for x in tr:
            fit.add(x["sig"], x["forced"])
        fit.done()
        for kk, x in enumerate(P[(f, "test")]):
            ids = x["ids"].to(dev); n = ids.shape[1]
            lay = {"entropy": blt_layouts.layout("entropy", R, x["sig"], fit).tolist()}
            for pname in LAYOUTS:
                lay[pname] = blt_layouts.layout(f"forced[{pname}]:entropy", R, x["sig"], fit, x["forced"]).tolist()
            for lname, st in lay.items():
                st = sorted(set(st))
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                bt = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()
                h0 = x["head"] + 1
                for j, (kind, s, e) in enumerate(x["spans"]):
                    rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": kind, "j": j,
                                 "ok": bool((lp[s - 1:e - 1].argmax(-1) == ids[0, s:e]).all())})
                rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": "bits", "j": -1,
                             "bits": float(bt[h0 - 1:].sum()), "nbytes": int(n - h0), "rate": len(st) / n})
            if dev == "mps" and kk % 20 == 0:
                torch.mps.empty_cache()

    def sel(f, lay, kind="res"):
        return {(r["problem"], r["j"]): r for r in rows if r["format"] == f and r["layout"] == lay and r["kind"] == kind}

    def boot(keys, d):
        Pk = np.array([k[0] for k in keys]); d = np.asarray(d, float)
        probs = np.unique(Pk); by = {p: d[Pk == p] for p in probs}
        bs = [np.concatenate([by[p] for p in brng.choice(probs, len(probs))]).mean() for _ in range(NBOOT)]
        return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    for f in rl.FORMATS:
        E = sel(f, "entropy"); keys = sorted(E); BE = sel(f, "entropy", "bits")
        for lname in ("entropy",) + LAYOUTS:
            L = sel(f, lname); B = sel(f, lname, "bits"); Fn = sel(f, lname, "fin")
            out = {"exact_res": float(np.mean([L[k]["ok"] for k in keys])), "exact_fin": float(np.mean([r["ok"] for r in Fn.values()])),
                   "bpb_solution": float(sum(r["bits"] for r in B.values()) / sum(r["nbytes"] for r in B.values())),
                   "rate": float(np.mean([r["rate"] for r in B.values()]))}
            if lname != "entropy":
                out["minus_entropy_res"] = boot(keys, [float(L[k]["ok"]) - float(E[k]["ok"]) for k in keys])
                bk = sorted(B)
                out["minus_entropy_bpb"] = boot([(k[0], 0) for k in bk], [(B[k]["bits"] - BE[k]["bits"]) / B[k]["nbytes"] for k in bk])
            res["layouts"][f"{f}|{lname}"] = out
            print(f"LAYOUT {f:5s} {lname:12s} exact res {out['exact_res']:.3f} fin {out['exact_fin']:.3f} bpb {out['bpb_solution']:.4f} rate {out['rate']:.3f}"
                  + (f" | vs entropy {100 * out['minus_entropy_res']['value']:+.1f} [{100 * out['minus_entropy_res']['ci'][0]:+.1f}, "
                     f"{100 * out['minus_entropy_res']['ci'][1]:+.1f}]" if lname != "entropy" else ""), flush=True)
        A, N = sel(f, "auto_prose"), sel(f, "anynum_prose")
        res["layouts"][f"{f}|auto_minus_anynum_res"] = boot(keys, [float(A[k]["ok"]) - float(N[k]["ok"]) for k in keys])

    xs = P[("prose", "test")]                                    # descriptive: the phrases the probe fires after
    ctx = collections.defaultdict(lambda: collections.defaultdict(list))
    allc = []
    for x in xs:
        for j, i in enumerate(x["cand"]):
            kind = ("computed" if x["y"][j] else "other_new" if x["ya"][j] else "copy") if x["num"][j] else "non_number"
            c = context(x["b"], int(i)); sc = float(x["score"]["auto_prose"][j])
            ctx[kind][c].append(sc); allc.append((sc, c, kind))
    for kind, d in ctx.items():
        top = sorted(d.items(), key=lambda kv: -len(kv[1]))[:15]
        res["contexts"][kind] = [{"words": c, "n": len(v), "mean_score": float(np.mean(v))} for c, v in top]
    top = sorted(allc, reverse=True)[:100]
    res["contexts"]["top100_by_auto_prose"] = {"kinds": dict(collections.Counter(k for _, _, k in top)),
                                               "words": collections.Counter(c for _, c, _ in top).most_common(15)}

    w1 = res["auc"]["prose|auto_prose|computed_within_numbers"]
    w2 = res["layouts"]["prose|auto_minus_anynum_res"]
    res["verdicts"] = {"W1_within_number_auc": {**w1, "verdict": "HOLDS" if w1["ci"][0] > 0.70 else "FAILS" if w1["ci"][1] < 0.60 else "MIXED"},
                       "W2_auto_minus_anynum_res": {**w2, "verdict": "HOLDS" if w2["ci"][0] > 0 else "REVERSED" if w2["ci"][1] < 0 else "NO DIFFERENCE"}}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_probe_reads_smoke.json" if SMOKE else "blt_probe_reads.json"), "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))
    for kind in ("computed", "copy", "other_new"):
        print(f"CONTEXT {kind}: " + "; ".join(f"'{c['words']}' x{c['n']} ({c['mean_score']:+.1f})" for c in res["contexts"].get(kind, [])[:8]))
    print("TOP100", json.dumps(res["contexts"]["top100_by_auto_prose"]))


if __name__ == "__main__":
    main()
