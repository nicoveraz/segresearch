"""R19 (paper3/R19_FORCED_SHARE_RULES.md): how many starts should R12's label-free detector force? BLT-1B at 10%:
entropy (E); the detector's top q% forced, rest entropy (D_q, q = 0.5-3%); forced where its calibrated probability
passes 0.5 (Dp); one fused ranking, entropy + w * probability (F_w). L* chosen on 300 train problems (out-of-fold
scores), then scored on the test problems.

    .venv-blt/bin/python realblt_forced_share.py      -> results/blt_forced_share.json   (R19_SMOKE=1: 40 / 20 problems)
"""
import json
import os

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import realblt_learned as rl
from realblt_autolabel import autolabel
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts

SMOKE = os.environ.get("R19_SMOKE") == "1"
R, NBOOT = rl.R, rl.NBOOT
QS = {"D0_5": 0.005, "D1": 0.01, "D1_5": 0.015, "D2": 0.02, "D3": 0.03}
WS = {"F0_5": 0.5, "F1": 1.0, "F2": 2.0, "F4": 4.0}
LAYOUTS = ("E",) + tuple(QS) + ("Dp",) + tuple(WS)
NDEV, TOL = (20 if SMOKE else 300), 0.01


def kind_of(name):
    return "entropy" if name == "E" else name if name in WS else f"forced[{name}]:entropy"


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
                x["idx"] = r["idx"]; x["ya"] = autolabel(x)
                ids = ids.to(dev); n = ids.shape[1]
                with torch.no_grad():
                    ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                    max_patch_length=cfg.max_patch_length)
                e = ent[0].float().cpu().numpy(); h = cap["h"][0].float().cpu().numpy()
                s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
                x.update({"ids": ids.cpu(), "sig": {"entropy": s}, "Hp": h[x["cand"]].astype(np.float16)})
                xs.append(x)
            P[(f, sp)] = xs
            print(f"patcher {f}/{sp}: {len(xs)} problems", flush=True)
    hook.remove()

    res = {"thresholds": {}, "dev": {}, "selected": {}, "layouts": {}, "diff": {}, "verdicts": {}}
    rng = np.random.default_rng(0)
    rows = []

    def score(xs, split, f, fit):
        for kk, x in enumerate(xs):
            ids = x["ids"].to(dev); n = ids.shape[1]
            for lname in LAYOUTS:
                st = sorted(set(blt_layouts.layout(kind_of(lname), R, x["sig"], fit, x["forced"]).tolist()))
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                am = lp.argmax(-1)
                bt = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()
                h0 = x["head"] + 1
                for j, (kind, s, e) in enumerate(x["spans"]):
                    rows.append({"split": split, "format": f, "problem": x["idx"], "layout": lname, "kind": kind, "j": j,
                                 "ok": bool((am[s - 1:e - 1] == ids[0, s:e]).all())})
                nf = len(set(x["forced"].get(lname, ())) & set(st))
                rows.append({"split": split, "format": f, "problem": x["idx"], "layout": lname, "kind": "bits", "j": -1,
                             "bits": float(bt[h0 - 1:].sum()), "nbytes": int(n - h0), "rate": len(st) / n, "forced": nf / n})
            if dev == "mps" and kk % 20 == 0:
                torch.mps.empty_cache()

    def summary(split, f, lname):
        R_ = [r_ for r_ in rows if r_["split"] == split and r_["format"] == f and r_["layout"] == lname]
        B = [r_ for r_ in R_ if r_["kind"] == "bits"]
        return {"exact_res": float(np.mean([r_["ok"] for r_ in R_ if r_["kind"] == "res"])),
                "exact_fin": float(np.mean([r_["ok"] for r_ in R_ if r_["kind"] == "fin"])),
                "bpb_solution": float(sum(r_["bits"] for r_ in B) / sum(r_["nbytes"] for r_ in B)),
                "rate": float(np.mean([r_["rate"] for r_ in B])), "forced_share": float(np.mean([r_["forced"] for r_ in B]))}

    for f in rl.FORMATS:
        tr, te = P[(f, "train")], P[(f, "test")]
        # the detector: R12's label-free probe; out-of-fold scores on train (thresholds, dev), full fit on test
        for x in tr:
            x["keep"] = (x["ya"] > 0) | (np.random.default_rng(x["idx"]).random(len(x["ya"])) < rl.NEG)
        stack = lambda xs, fld, sel=None: np.concatenate([(x[fld] if sel is None else x[fld][x[sel]]) for x in xs]).astype(np.float32)
        m = rl.fit_logit(stack(tr, "Hp", "keep"), stack(tr, "ya", "keep"))
        fold = {i: j % 5 for j, i in enumerate(rng.permutation(len(tr)))}
        for fo in range(5):
            trx = [x for i, x in enumerate(tr) if fold[i] != fo]
            mf = rl.fit_logit(stack(trx, "Hp", "keep"), stack(trx, "ya", "keep"))
            for i, x in enumerate(tr):
                if fold[i] == fo:
                    x["dscore"] = rl.apply(mf, x["Hp"].astype(np.float32))
        for x in te:
            x["dscore"] = rl.apply(m, x["Hp"].astype(np.float32))
        ntok = sum(x["ids"].shape[1] for x in tr)
        allsc = np.sort(np.concatenate([x["dscore"] for x in tr]))[::-1]
        thr = {k: float(allsc[min(len(allsc) - 1, int(q * ntok))]) for k, q in QS.items()}
        res["thresholds"][f] = thr
        for x in tr + te:
            n = x["ids"].shape[1]
            p = 1 / (1 + np.exp(-(x["dscore"] + np.log(rl.NEG))))          # calibrated for the 25% negative sampling
            x["forced"] = {k: (x["cand"][x["dscore"] > t] + 1).tolist() for k, t in thr.items()}
            x["forced"]["Dp"] = (x["cand"][p > 0.5] + 1).tolist()
            pt = np.zeros(n); pt[x["cand"] + 1] = p
            for k, w in WS.items():
                x["sig"][k] = x["sig"]["entropy"] + w * pt
        fit = blt_layouts.Fit((R,))
        for x in tr:
            fit.add(x["sig"], x["forced"])
        fit.done()

        # selection on dev (train problems, out-of-fold scores), fixed before the test problems are scored
        dv = [tr[i] for i in sorted(np.random.default_rng(19).choice(len(tr), min(NDEV, len(tr)), replace=False))]
        score(dv, "dev", f, fit)
        res["dev"][f] = {l: summary("dev", f, l) for l in LAYOUTS}
        for l in LAYOUTS:
            o = res["dev"][f][l]
            print(f"DEV    {f:5s} {l:5s} exact {o['exact_res']:.3f} fin {o['exact_fin']:.3f} bpb {o['bpb_solution']:.4f} rate {o['rate']:.3f} forced {o['forced_share']:.4f}", flush=True)
        ok = [l for l in LAYOUTS[1:] if res["dev"][f][l]["exact_fin"] >= res["dev"][f]["E"]["exact_fin"] - TOL - 1e-9]
        star = max(ok, key=lambda l: res["dev"][f][l]["exact_res"]) if ok else max(LAYOUTS[1:], key=lambda l: res["dev"][f][l]["exact_fin"])
        res["selected"][f] = {"L*": star, "qualified": ok}
        print(f"SELECTED {f}: {star} (qualified {ok})", flush=True)

        score(te, "test", f, fit)
        for l in LAYOUTS:
            res["layouts"][f"{f}|{l}"] = o = summary("test", f, l)
            print(f"LAYOUT {f:5s} {l:5s} exact {o['exact_res']:.3f} fin {o['exact_fin']:.3f} bpb {o['bpb_solution']:.4f} rate {o['rate']:.3f} forced {o['forced_share']:.4f}", flush=True)

    brng = np.random.default_rng(1)

    def sel(f, lay, kind):
        return {(r_["problem"], r_["j"]): r_ for r_ in rows
                if r_["split"] == "test" and r_["format"] == f and r_["layout"] == lay and r_["kind"] == kind}

    def boot(keys, d):
        Pk = np.array([k[0] for k in keys]); d = np.asarray(d, float)
        probs = np.unique(Pk); by = {p: d[Pk == p] for p in probs}
        bs = [np.concatenate([by[p] for p in brng.choice(probs, len(probs))]).mean() for _ in range(NBOOT)]
        return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    for f in rl.FORMATS:
        star = res["selected"][f]["L*"]
        pairs = [(star, "E"), (star, "D3")] + [(l, "E") for l in LAYOUTS[1:]] + [(l, "D3") for l in LAYOUTS[1:] if l != "D3"]
        for a, b_ in dict.fromkeys(pairs):
            for meas, kind in (("exact", "res"), ("fin", "fin")):
                A, Bl = sel(f, a, kind), sel(f, b_, kind); keys = sorted(Bl)
                tag = f"{f}|{'L*' if a == star else a}-{b_}|{meas}"
                d = boot(keys, [float(A[k]["ok"]) - float(Bl[k]["ok"]) for k in keys])
                res["diff"][f"{f}|{a}-{b_}|{meas}"] = d
                if a == star:
                    res["diff"][tag] = d
                print(f"DIFF {f:5s} {a:5s}-{b_:5s} {meas:5s} {100 * d['value']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]", flush=True)
    D_ = res["diff"]
    for t, key, lo in (("B1", "prose|L*-E|exact", 0.0), ("B2", "prose|L*-D3|fin", 0.0), ("B3", "prose|L*-E|fin", -0.03)):
        res["verdicts"][t] = {**D_[key], "L*": res["selected"]["prose"]["L*"],
                              "verdict": "HOLDS" if D_[key]["ci"][0] > lo else "DOES NOT HOLD"}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_forced_share_smoke.json" if SMOKE else "blt_forced_share.json"), "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
