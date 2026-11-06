"""R18 (paper3/R18_SECOND_DIGIT_RULES.md): a patch start at a number's second digit where BLT-1B's patcher entropy
stays high (P), against entropy (E), every second digit (P_all), R12's label-free detector at the decision point
(D), both (DP), and label oracles at the first (O1) or second (O2) byte of every computed value. BLT-1B at 10%.

    .venv-blt/bin/python realblt_second_digit.py      -> results/blt_second_digit.json   (R18_SMOKE=1: 40 / 20 problems)
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
from realblt_autolabel import autolabel
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts

SMOKE = os.environ.get("R18_SMOKE") == "1"
R, Q, NBOOT = rl.R, rl.Q, rl.NBOOT
NUM = re.compile(rb"\d[\d,]*(?:\.\d+)?")
LAYOUTS = ("P", "P_all", "D", "DP", "O1", "O2")


def isdig(c):
    return 48 <= c <= 57


def second_digits(x, r, f):
    """Byte positions i in the solution whose previous byte is a digit that starts a number."""
    b = x["b"]; line = len(b) - len(("\nThe final answer is " + r["answer"]).encode())
    return np.array([i for i in range(max(x["head"], 2), line)
                     if isdig(b[i - 1]) and not rl.isal(b[i - 2])], int)


def kind_of(x, j, vstart):
    """computed / other_new / copy for the number starting at byte j."""
    if j in vstart:
        return "computed"
    v = NUM.match(x["b"], j).group(0).replace(b",", b"")
    return "other_new" if v not in {m.replace(b",", b"") for m in NUM.findall(x["b"][:j])} else "copy"


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
                x["idx"] = r["idx"]; x["ya"] = autolabel(x); x["sd"] = second_digits(x, r, f)
                ids = ids.to(dev); n = ids.shape[1]
                with torch.no_grad():
                    ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                    max_patch_length=cfg.max_patch_length)
                e = ent[0].float().cpu().numpy(); h = cap["h"][0].float().cpu().numpy()
                s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
                x.update({"ids": ids.cpu(), "sig": {"entropy": s}, "e": e, "Hp": h[x["cand"]].astype(np.float16)})
                xs.append(x)
            P[(f, sp)] = xs
            print(f"patcher {f}/{sp}: {len(xs)} problems", flush=True)
    hook.remove()

    res = {"tau": {}, "forced": {}, "layouts": {}, "diff": {}, "verdicts": {}}
    rng = np.random.default_rng(0)
    rows = []
    for f in rl.FORMATS:
        tr, te = P[(f, "train")], P[(f, "test")]
        # D: R12's label-free probe trained on this format, out-of-fold scores on train for the threshold
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
        thr = float(allsc[min(len(allsc) - 1, int(Q * ntok))])
        tau = float(np.median(np.concatenate([x["e"][x["sd"]] for x in tr if len(x["sd"])])))
        res["tau"][f] = tau
        for x in tr + te:
            vstart = {t - 1 for t in x["vtok"]}
            sd = x["sd"]; keep = x["e"][sd] > tau
            D = (x["cand"][x["dscore"] > thr] + 1).tolist(); Pp = (sd[keep] + 1).tolist()
            o2 = [s + 1 for kind, s, e in x["spans"] if kind == "res" and e - s >= 2]
            x["forced"] = {"P": Pp, "P_all": (sd + 1).tolist(), "D": D, "DP": sorted(set(D) | set(Pp)),
                           "O1": list(x["vtok"]), "O2": o2}
            x["pkinds"] = [kind_of(x, int(i) - 1, vstart) for i in sd[keep]]
        kinds = [k for x in te for k in x["pkinds"]]
        res["forced"][f] = {"P_per_token": float(sum(len(x["forced"]["P"]) for x in te) / sum(x["ids"].shape[1] for x in te)),
                            "P_all_per_token": float(sum(len(x["forced"]["P_all"]) for x in te) / sum(x["ids"].shape[1] for x in te)),
                            "P_shares": {k: kinds.count(k) / max(1, len(kinds)) for k in ("computed", "other_new", "copy")}}
        print(f"{f}: tau {tau:.3f} forced {json.dumps(res['forced'][f])}", flush=True)
        fit = blt_layouts.Fit((R,))
        for x in tr:
            fit.add(x["sig"], x["forced"])
        fit.done()
        for kk, x in enumerate(te):
            ids = x["ids"].to(dev); n = ids.shape[1]
            lay = {"E": blt_layouts.layout("entropy", R, x["sig"], fit).tolist()}
            for name in LAYOUTS:
                lay[name] = blt_layouts.layout(f"forced[{name}]:entropy", R, x["sig"], fit, x["forced"]).tolist()
            for lname, st in lay.items():
                st = sorted(set(st))
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                am = lp.argmax(-1)
                bt = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()
                h0 = x["head"] + 1
                for j, (kind, s, e) in enumerate(x["spans"]):
                    ok = bool((am[s - 1:e - 1] == ids[0, s:e]).all())
                    tail = bool((am[s:e - 1] == ids[0, s + 1:e]).all()) if e - s >= 2 else None
                    rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": kind, "j": j, "ok": ok, "tail": tail})
                rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": "bits", "j": -1,
                             "bits": float(bt[h0 - 1:].sum()), "nbytes": int(n - h0), "rate": len(st) / n})
            if dev == "mps" and kk % 20 == 0:
                torch.mps.empty_cache()

    brng = np.random.default_rng(1)

    def sel(f, lay, kind="res"):
        return {(r_["problem"], r_["j"]): r_ for r_ in rows if r_["format"] == f and r_["layout"] == lay and r_["kind"] == kind}

    def boot(keys, d):
        Pk = np.array([k[0] for k in keys]); d = np.asarray(d, float)
        probs = np.unique(Pk); by = {p: d[Pk == p] for p in probs}
        bs = [np.concatenate([by[p] for p in brng.choice(probs, len(probs))]).mean() for _ in range(NBOOT)]
        return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    for f in rl.FORMATS:
        E = sel(f, "E"); keys = sorted(E); tkeys = [k for k in keys if E[k]["tail"] is not None]
        for lname in ("E",) + LAYOUTS:
            L = sel(f, lname); B = sel(f, lname, "bits"); Fn = sel(f, lname, "fin")
            res["layouts"][f"{f}|{lname}"] = {"exact_res": float(np.mean([L[k]["ok"] for k in keys])),
                                              "tail_res": float(np.mean([L[k]["tail"] for k in tkeys])),
                                              "exact_fin": float(np.mean([r_["ok"] for r_ in Fn.values()])),
                                              "bpb_solution": float(sum(r_["bits"] for r_ in B.values()) / sum(r_["nbytes"] for r_ in B.values())),
                                              "rate": float(np.mean([r_["rate"] for r_ in B.values()]))}
            o = res["layouts"][f"{f}|{lname}"]
            print(f"LAYOUT {f:5s} {lname:5s} exact {o['exact_res']:.3f} tail {o['tail_res']:.3f} fin {o['exact_fin']:.3f} bpb {o['bpb_solution']:.4f} rate {o['rate']:.3f}", flush=True)
        for a, b_ in (("O2", "E"), ("P", "E"), ("DP", "D"), ("P", "P_all"), ("O1", "E"), ("D", "E"), ("P_all", "E")):
            A, Bl = sel(f, a), sel(f, b_)
            res["diff"][f"{f}|{a}-{b_}|exact"] = boot(keys, [float(A[k]["ok"]) - float(Bl[k]["ok"]) for k in keys])
            res["diff"][f"{f}|{a}-{b_}|tail"] = boot(tkeys, [float(A[k]["tail"]) - float(Bl[k]["tail"]) for k in tkeys])
            for meas in ("exact", "tail"):
                d = res["diff"][f"{f}|{a}-{b_}|{meas}"]
                print(f"DIFF {f:5s} {a:5s}-{b_:5s} {meas:5s} {100 * d['value']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]", flush=True)
    D_ = res["diff"]
    for t, key in (("T1", "prose|O2-E|exact"), ("T2", "prose|P-E|exact"), ("T3", "prose|DP-D|exact")):
        res["verdicts"][t] = {**D_[key], "verdict": "HOLDS" if D_[key]["ci"][0] > 0 else "DOES NOT HOLD"}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_second_digit_smoke.json" if SMOKE else "blt_second_digit.json"), "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
