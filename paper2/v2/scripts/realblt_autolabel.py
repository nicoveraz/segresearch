"""R12 (paper3/R12_AUTOLABEL_RULES.md): R10's patcher-state probe trained without labels. Auto-label: a candidate
is positive if a number starts there whose value does not occur earlier in the text (a training label only; the
probe reads the state before the number). Probes on BLT-1B's patcher state (as R10): marker_prose (R10's labels, the
reference), auto_prose, auto_orig; BLT-1B layouts at 10% with each probe's top 3% forced, rest entropy.

    .venv-blt/bin/python realblt_autolabel.py        -> results/blt_autolabel.json   (R12_SMOKE=1: 40 / 20 problems)
"""
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
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts

SMOKE = os.environ.get("R12_SMOKE") == "1"
NUM = re.compile(rb"\d[\d,]*(?:\.\d+)?")
PROBES = ("marker_prose", "auto_prose", "auto_orig")
R, Q, NEG, NBOOT = rl.R, rl.Q, rl.NEG, rl.NBOOT


def autolabel(x):
    b = x["b"]; out = np.zeros(len(x["cand"]))
    for j, i in enumerate(x["cand"]):
        if 48 <= b[i] <= 57:
            v = NUM.match(b, i).group(0).replace(b",", b"")
            out[j] = float(v not in {m.replace(b",", b"") for m in NUM.findall(b[:i])})
    return out


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
                x["idx"] = r["idx"]; x["ya"] = autolabel(x); ids = ids.to(dev); n = ids.shape[1]
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

    res = {"auto_label_agreement": {}, "auc": {}, "layouts": {}, "verdicts": {}}
    for f in rl.FORMATS:
        y = np.concatenate([x["y"] for x in P[(f, "train")]]); ya = np.concatenate([x["ya"] for x in P[(f, "train")]])
        res["auto_label_agreement"][f] = {"precision": float((y * ya).sum() / max(1, ya.sum())), "recall": float((y * ya).sum() / max(1, y.sum())),
                                          "n_true": int(y.sum()), "n_auto": int(ya.sum())}
        print(f"auto-labels {f}: {res['auto_label_agreement'][f]}", flush=True)
    rng = np.random.default_rng(0)
    stack = lambda xs, fld, sel=None: np.concatenate([(x[fld] if sel is None else x[fld][x[sel]]) for x in xs]).astype(np.float32)
    for pname, (ftr, lab) in {"marker_prose": ("prose", "y"), "auto_prose": ("prose", "ya"), "auto_orig": ("orig", "ya")}.items():
        tr = P[(ftr, "train")]
        for x in tr:
            x["keep"] = (x[lab] > 0) | (np.random.default_rng(x["idx"]).random(len(x[lab])) < NEG)
        m = rl.fit_logit(stack(tr, "Hp", "keep"), stack(tr, lab, "keep"))
        for (f, sp), xs in P.items():
            if f == ftr and sp == "train":
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
    brng = np.random.default_rng(1)
    for f in rl.FORMATS:
        xs = P[(f, "test")]
        for pname in PROBES:
            val = rl.auc(np.concatenate([x["y"] for x in xs]), np.concatenate([x["score"][pname] for x in xs]))
            bs = []
            for _ in range(NBOOT):
                pk = [xs[i] for i in brng.integers(0, len(xs), len(xs))]
                bs.append(rl.auc(np.concatenate([x["y"] for x in pk]), np.concatenate([x["score"][pname] for x in pk])))
            res["auc"][f"{f}|{pname}"] = {"value": val, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
            print(f"AUC {f:5s} {pname:12s} {val:.3f}", flush=True)

    rows = []
    for f in rl.FORMATS:
        tr = P[(f, "train")]; ntok = sum(x["ids"].shape[1] for x in tr)
        for pname in PROBES:
            allsc = np.sort(np.concatenate([x["score"][pname] for x in tr]))[::-1]
            thr = float(allsc[min(len(allsc) - 1, int(Q * ntok))])
            for x in P[(f, "train")] + P[(f, "test")]:
                x.setdefault("forced", {})[pname] = (x["cand"][x["score"][pname] > thr] + 1).tolist()
        fit = blt_layouts.Fit((R,))
        for x in tr:
            fit.add(x["sig"], x["forced"])
        fit.done()
        for k, x in enumerate(P[(f, "test")]):
            ids = x["ids"].to(dev); n = ids.shape[1]
            lay = {"entropy": blt_layouts.layout("entropy", R, x["sig"], fit).tolist()}
            for pname in PROBES:
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
            if dev == "mps" and k % 20 == 0:
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
        for lname in ("entropy",) + PROBES:
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
    A, M = sel("prose", "auto_prose"), sel("prose", "marker_prose"); keys = sorted(A)
    z2 = boot(keys, [float(A[k]["ok"]) - float(M[k]["ok"]) for k in keys])
    z1 = res["layouts"]["prose|auto_prose"]["minus_entropy_res"]
    res["verdicts"] = {"Z1_auto_prose_minus_entropy": {**z1, "holds": z1["ci"][0] > 0},
                       "Z2_auto_minus_marker": {**z2, "holds": z2["ci"][0] > -0.05}}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_autolabel_smoke.json" if SMOKE else "blt_autolabel.json"), "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
