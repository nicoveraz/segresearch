"""R17 (paper3/R17_SIGNAL_SHAPE_NOTE.md, descriptive): BLT-1B's patcher entropy and R12's label-free probe score byte by
byte around numbers (8 bytes before the first digit to 4 after), by kind (computed value, other new number, copy;
word starts as a reference), on the GSM8K prose rewrites and the equation-style originals.

    .venv-blt/bin/python realblt_profiles.py      -> results/blt_profiles.json
"""
import json
import os
import re

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoTokenizer

import blt_load
import realblt_learned as rl
from realblt_autolabel import autolabel
from realblt_budget import MAX_TOKENS, MODEL

OFF = range(-8, 5)
NUM = re.compile(rb"\d[\d,]*(?:\.\d+)?")
KINDS = ("computed", "other_new", "copy", "word")
NBOOT = 1000


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    recs = {sp: [json.loads(l) for l in open(os.path.join(rl.DATA, f"{sp}.jsonl"))] for sp in ("train", "test")}
    btok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    cap = {}
    hook = model.model.patcher.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))

    def patch(x):
        ids = btok(x["b"].decode(), return_tensors="pt").input_ids
        if ids.shape[1] > MAX_TOKENS:
            return None
        with torch.no_grad():
            ent, _, _ = model.model.patcher(ids.to(dev), patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                            max_patch_length=cfg.max_patch_length)
        return ent[0].float().cpu().numpy(), cap["h"][0].float().cpu().numpy()     # index i: predicting byte i

    res = {"profiles": {}, "auc_computed_vs_copy": {}, "counts": {}}
    brng = np.random.default_rng(1)
    for f in rl.FORMATS:
        Xs, Ys = [], []
        for r in recs["train"]:                                     # R12's probe for this format, refitted
            x = rl.build(r, f); x["idx"] = r["idx"]; out = patch(x)
            if out is None:
                continue
            ya = autolabel(x); keep = (ya > 0) | (np.random.default_rng(x["idx"]).random(len(ya)) < rl.NEG)
            Xs.append(out[1][x["cand"]][keep]); Ys.append(ya[keep])
        pm = rl.fit_logit(np.concatenate(Xs).astype(np.float32), np.concatenate(Ys))
        print(f"probe fitted ({f})", flush=True)
        vals = {k: {o: {"entropy": [], "probe": []} for o in OFF} for k in KINDS}   # lists of (problem, value)
        sep = {"probe": {o: ([], []) for o in range(-4, 1)}, "entropy": {o: ([], []) for o in range(0, 3)}}
        for pi, r in enumerate(recs["test"]):
            x = rl.build(r, f); out = patch(x)
            if out is None:
                continue
            e, h = out; b = x["b"]; L = len(b)
            score = rl.apply(pm, h[:L].astype(np.float32)); ya = autolabel(x)
            for j, i in enumerate(x["cand"]):
                i = int(i)
                if 48 <= b[i] <= 57:
                    kind = "computed" if x["y"][j] else "other_new" if ya[j] else "copy"
                    nlen = NUM.match(b, i).end() - i
                elif 65 <= b[i] <= 90 or 97 <= b[i] <= 122:
                    kind, nlen = "word", 0
                else:
                    continue
                for o in OFF:
                    p = i + o
                    if p < 0 or p >= L or (o > 0 and kind != "word" and o >= nlen):
                        continue
                    vals[kind][o]["entropy"].append((pi, float(e[p]))); vals[kind][o]["probe"].append((pi, float(score[p])))
                if kind in ("computed", "copy"):
                    for o in range(-4, 1):
                        if i + o >= 0:
                            sep["probe"][o][0].append(kind == "computed"); sep["probe"][o][1].append(float(score[i + o]))
                    for o in range(0, 3):
                        if o < nlen:
                            sep["entropy"][o][0].append(kind == "computed"); sep["entropy"][o][1].append(float(e[i + o]))
            if dev == "mps" and pi % 25 == 0:
                torch.mps.empty_cache()
        prof = {}
        for k in KINDS:
            for sig in ("entropy", "probe"):
                rows = []
                for o in OFF:
                    v = vals[k][o][sig]
                    if len(v) < 20:
                        rows.append({"offset": o, "n": len(v)}); continue
                    P = np.array([a for a, _ in v]); V = np.array([c for _, c in v]); probs = np.unique(P)
                    by = {p: V[P == p] for p in probs}
                    bs = [np.concatenate([by[p] for p in brng.choice(probs, len(probs))]).mean() for _ in range(NBOOT)]
                    rows.append({"offset": o, "n": int(len(V)), "mean": float(V.mean()),
                                 "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]})
                prof[f"{k}|{sig}"] = rows
        res["profiles"][f] = prof
        res["auc_computed_vs_copy"][f] = {f"{sig}@{o:+d}": {"auc": float(rl.auc(np.array(a), np.array(s))), "n": len(a)}
                                          for sig, d in sep.items() for o, (a, s) in d.items()}
        res["counts"][f] = {k: len(vals[k][0]["entropy"]) for k in KINDS}
        print(f"== {f}: counts {res['counts'][f]}", flush=True)
        for sig in ("entropy", "probe"):
            print(f"  {sig:8s} offset " + " ".join(f"{o:+5d}" for o in OFF), flush=True)
            for k in KINDS:
                print(f"  {sig:8s} {k:9s} " + " ".join(f"{r_['mean']:5.2f}" if "mean" in r_ else "    ." for r_ in prof[f"{k}|{sig}"]), flush=True)
        print("  AUC computed vs copy: " + json.dumps({k: round(v["auc"], 3) for k, v in res["auc_computed_vs_copy"][f].items()}), flush=True)
    hook.remove()
    json.dump(res, open(os.path.join(_seg.HERE, "results", "blt_profiles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
