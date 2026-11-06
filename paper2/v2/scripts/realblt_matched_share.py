"""R20 (paper3/R20_MATCHED_SHARE_RULES.md): R12's label-free detector with a forced share equal to the share of number
starts in the format's train text (M: prose 1.53%, equation text 2.68%), against entropy (E), R12's 3% (D3) and R19's
fused ranking (F2), on fresh GSM8K test problems (data/prose_rewrites_fresh). BLT-1B at 10%; everything fitted on the
1,382 train problems of R12-R19.

    .venv-blt/bin/python realblt_matched_share.py   -> results/blt_matched_share.json   (R20_SMOKE=1: 40 train / 20 of R9's test)
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

SMOKE = os.environ.get("R20_SMOKE") == "1"
FRESH = os.path.join(_seg.HERE, "data", "prose_rewrites_fresh", "test.jsonl")
R, NBOOT = rl.R, rl.NBOOT
QM = {"prose": 0.0153, "orig": 0.0268}                 # number starts / positions in each format's train text (fixed in the rules)
W = 2.0                                                # F2: entropy + 2 x the detector's calibrated probability (R19's L*)
LAYOUTS = ("E", "M", "D3", "F2")
KIND = {"E": "entropy", "M": "forced[M]:entropy", "D3": "forced[D3]:entropy", "F2": "F2"}


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    recs = {"train": [json.loads(l) for l in open(os.path.join(rl.DATA, "train.jsonl"))],
            "test": [json.loads(l) for l in open(os.path.join(rl.DATA, "test.jsonl") if SMOKE else FRESH)]}
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

    res = {"q": QM, "thresholds": {}, "n": {}, "layouts": {}, "diff": {}, "verdicts": {}}
    rng = np.random.default_rng(0)
    rows = []
    for f in rl.FORMATS:
        tr, te = P[(f, "train")], P[(f, "test")]
        res["n"][f] = {"train": len(tr), "test": len(te), "values": sum(len(x["vtok"]) for x in te)}
        # the detector, exactly as R19: out-of-fold scores on train (thresholds, entropy fill), full fit on test
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
        thr = {k: float(allsc[min(len(allsc) - 1, int(q * ntok))]) for k, q in (("M", QM[f]), ("D3", 0.03))}
        res["thresholds"][f] = thr
        for x in tr + te:
            n = x["ids"].shape[1]
            p = 1 / (1 + np.exp(-(x["dscore"] + np.log(rl.NEG))))          # calibrated for the 25% negative sampling
            x["forced"] = {k: (x["cand"][x["dscore"] > t] + 1).tolist() for k, t in thr.items()}
            pt = np.zeros(n); pt[x["cand"] + 1] = p
            x["sig"]["F2"] = x["sig"]["entropy"] + W * pt
        fit = blt_layouts.Fit((R,))
        for x in tr:
            fit.add(x["sig"], x["forced"])
        fit.done()
        for kk, x in enumerate(te):
            ids = x["ids"].to(dev); n = ids.shape[1]
            for lname in LAYOUTS:
                st = sorted(set(blt_layouts.layout(KIND[lname], R, x["sig"], fit, x["forced"]).tolist()))
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                am = lp.argmax(-1)
                bt = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()
                h0 = x["head"] + 1
                for j, (kind, s, e) in enumerate(x["spans"]):
                    rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": kind, "j": j,
                                 "ok": bool((am[s - 1:e - 1] == ids[0, s:e]).all())})
                nf = len(set(x["forced"].get(lname, ())) & set(st))
                rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": "bits", "j": -1,
                             "bits": float(bt[h0 - 1:].sum()), "nbytes": int(n - h0), "rate": len(st) / n, "forced": nf / n})
            if dev == "mps" and kk % 20 == 0:
                torch.mps.empty_cache()
        for l in LAYOUTS:
            R_ = [r_ for r_ in rows if r_["format"] == f and r_["layout"] == l]
            B = [r_ for r_ in R_ if r_["kind"] == "bits"]
            res["layouts"][f"{f}|{l}"] = o = {
                "exact_res": float(np.mean([r_["ok"] for r_ in R_ if r_["kind"] == "res"])),
                "exact_fin": float(np.mean([r_["ok"] for r_ in R_ if r_["kind"] == "fin"])),
                "bpb_solution": float(sum(r_["bits"] for r_ in B) / sum(r_["nbytes"] for r_ in B)),
                "rate": float(np.mean([r_["rate"] for r_ in B])), "forced_share": float(np.mean([r_["forced"] for r_ in B]))}
            print(f"LAYOUT {f:5s} {l:3s} exact {o['exact_res']:.3f} fin {o['exact_fin']:.3f} bpb {o['bpb_solution']:.4f} rate {o['rate']:.3f} forced {o['forced_share']:.4f}", flush=True)

    brng = np.random.default_rng(1)

    def sel(f, lay, kind):
        return {(r_["problem"], r_["j"]): r_ for r_ in rows if r_["format"] == f and r_["layout"] == lay and r_["kind"] == kind}

    def boot(keys, d):
        Pk = np.array([k[0] for k in keys]); d = np.asarray(d, float)
        probs = np.unique(Pk); by = {p: d[Pk == p] for p in probs}
        bs = [np.concatenate([by[p] for p in brng.choice(probs, len(probs))]).mean() for _ in range(NBOOT)]
        return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    for f in rl.FORMATS:
        for a, b_ in (("M", "E"), ("M", "D3"), ("M", "F2"), ("D3", "E"), ("F2", "E")):
            for meas, kind in (("exact", "res"), ("fin", "fin")):
                A, Bl = sel(f, a, kind), sel(f, b_, kind); keys = sorted(Bl)
                d = res["diff"][f"{f}|{a}-{b_}|{meas}"] = boot(keys, [float(A[k]["ok"]) - float(Bl[k]["ok"]) for k in keys])
                print(f"DIFF {f:5s} {a:3s}-{b_:3s} {meas:5s} {100 * d['value']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]", flush=True)
    D_ = res["diff"]
    for t, key, lo in (("N1", "prose|M-E|fin", -0.02), ("N2", "prose|M-D3|exact", -0.03), ("N3", "prose|M-F2|exact", 0.0)):
        res["verdicts"][t] = {**D_[key], "verdict": "HOLDS" if D_[key]["ci"][0] > lo else "DOES NOT HOLD"}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_matched_share_smoke.json" if SMOKE else "blt_matched_share.json"), "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
