"""The prose-rewrite measurement of paper 1 v2 (section 5): the same GSM8K problems and computed values in two
writings, the stripped original (values after '= ') and an LLM rewrite as prose without equations (paper/v2/data/
prose_rewrites/, described in its data card). On BLT-1B at a 10% budget: does entropy start patches at the values,
does a start there pay, and does the novelty table of arXiv:2610.22100 still find them without the cue? (Docstring
rewritten for the public copy; the code is unchanged.)

Layouts per format (thresholds and tables fitted on that format's train problems):
    default   BLT-1B's own patching          entropy   BLT-1B's entropy at 10%
    nov       novelty table fitted on the format's train text (paper 2's procedure)
    oracle    a start at every computed value, filled to the budget by entropy
    with / without   the entropy layout plus / minus a start at every computed value
Per computed value: covered (entropy starts a patch at its first byte, the first digit: a '$' before a value is not
part of it), exact (teacher-forced greedy).

    R9_DATA=paper/v2/data/prose_rewrites python paper/v2/scripts/realblt_prose.py     -> paper/v2/results/blt_prose.json
"""
import json
import os

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import reuse_screen
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts
from realblt_cues import signals

DATA = os.path.expanduser(os.environ.get("R9_DATA", "~/.cache/segresearch-prose"))
R = 0.10
FORMATS = ("orig", "prose")
LAYOUTS = ("default", "entropy", "nov", "oracle", "with", "without")


def units(r, f):
    """[(kind, j, start_tok, end_tok)]: computed values (j = their index, shared by both formats) and the final answer."""
    t = r[f]
    bpos = lambda c: len(t[:c].encode())                       # char offset -> byte offset
    out = [("res", j, bpos(s) + 1, bpos(s) + len(v.encode()) + 1) for j, (s, v) in enumerate(zip(r[f + "_spans"], r["values"]))]
    fs = r["final_start_" + f]
    out.append(("fin", -1, bpos(fs) + 1, len(t.encode()) + 1))
    return out


def boot(keys, d, n=2000, seed=0):
    """Mean of d with a 95% bootstrap interval over problems (keys[i][0] is the problem)."""
    rng = np.random.default_rng(seed)
    P = np.array([k[0] for k in keys]); d = np.asarray(d, float)
    probs = np.unique(P); by = {p: d[P == p] for p in probs}
    bs = [np.concatenate([by[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(n)]
    return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d))}


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    recs = {sp: [json.loads(l) for l in open(os.path.join(DATA, f"{sp}.jsonl"))] for sp in ("train", "test")}
    rows, rates = [], {}
    for f in FORMATS:
        fb = np.frombuffer("\n\n".join(r[f] for r in recs["train"]).encode(), np.uint8)
        T2, T1, glob = reuse_screen.tab_fit(fb, -reuse_screen.match_signals(fb)[0])
        tabs = {"nov": {"T2": T2, "T1": T1, "glob": glob}}

        def prep(r):
            b = r[f].encode()
            ids = tok(r[f], return_tensors="pt").input_ids.to(dev)
            if ids.shape[1] > MAX_TOKENS or ids.shape[1] != len(b) + 1:
                return None
            us = units(r, f)
            return r, b, ids, us, {"comp": [s for k, _, s, _ in us if k == "res"]}, signals(model, cfg, ids, b, tabs, dev)

        train = [x for x in map(prep, recs["train"]) if x]
        test = [x for x in map(prep, recs["test"]) if x]
        fit = blt_layouts.Fit((R,))
        for *_, forced, sig in train:
            fit.add(sig, forced)
        fit.done()
        for i, (r, b, ids, us, forced, sig) in enumerate(test):
            n = ids.shape[1]
            ent = set(blt_layouts.layout("entropy", R, sig, fit).tolist())
            F = set(forced["comp"])
            lay = {"default": None, "entropy": sorted(ent), "nov": blt_layouts.layout("nov", R, sig, fit).tolist(),
                   "oracle": blt_layouts.layout("forced[comp]:entropy", R, sig, fit, forced).tolist(),
                   "with": sorted(ent | F), "without": sorted(ent - F)}
            for name in LAYOUTS:
                st = lay[name]
                with torch.no_grad():
                    out = model(input_ids=ids, use_cache=False) if st is None else \
                        model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev), use_cache=False)
                lp = torch.log_softmax(out.logits[0].float(), -1)
                if st is not None:
                    rates.setdefault(f"{f}|{name}", []).append(len(st) / n)
                S = set(st) if st is not None else set()
                for kind, j, s, e in us:
                    tgt = ids[0, s:e]; pred = lp[s - 1:e - 1]
                    rows.append({"format": f, "problem": r["idx"], "layout": name, "kind": kind, "j": j,
                                 "covered": s in S if st is not None else None,
                                 "ok": bool((pred.argmax(-1) == tgt).all())})
            if dev == "mps" and (i + 1) % 20 == 0:
                torch.mps.empty_cache()
            if (i + 1) % 25 == 0:
                print(f"  {f}: {i + 1}/{len(test)}", flush=True)
        print(f"{f}: {len(train)} train / {len(test)} test; entropy train rate {fit.rate('entropy', R):.3f}", flush=True)

    get = lambda f, lay, kind="res": {(x["problem"], x["j"]): x for x in rows
                                      if x["format"] == f and x["layout"] == lay and x["kind"] == kind}
    res = {"rates": {k: float(np.mean(v)) for k, v in rates.items()}, "by_layout": {}, "verdicts": {}}
    for f in FORMATS:
        for lay in LAYOUTS:
            for kind in ("res", "fin"):
                g = get(f, lay, kind)
                if g:
                    res["by_layout"][f"{f}|{lay}|{kind}"] = {"n": len(g), "exact": float(np.mean([x["ok"] for x in g.values()])),
                                                            "covered": None if lay == "default" else float(np.mean([x["covered"] for x in g.values()]))}
    eo, ep = get("orig", "entropy"), get("prose", "entropy")
    keys = sorted(set(eo) & set(ep))
    v1 = boot(keys, [float(ep[k]["covered"]) - float(eo[k]["covered"]) for k in keys])
    gain = {}
    for f in FORMATS:
        w, wo = get(f, "with"), get(f, "without")
        gain[f] = {k: float(w[k]["ok"]) - float(wo[k]["ok"]) for k in keys}
    v2 = boot(keys, [gain["prose"][k] for k in keys])
    v2_diff = boot(keys, [gain["prose"][k] - gain["orig"][k] for k in keys])
    v2_orig = boot(keys, [gain["orig"][k] for k in keys])
    v3 = {}
    for f in FORMATS:
        nv, en = get(f, "nov"), get(f, "entropy")
        v3[f] = boot(keys, [float(nv[k]["ok"]) - float(en[k]["ok"]) for k in keys])
    fmt_tied = v1["ci"][0] > 0 and v2_diff["ci"][1] < 0
    follows = v1["ci"][0] <= 0 and v2["ci"][0] > 0
    res["verdicts"] = {
        "V1_coverage_prose_minus_orig": {**v1, "above_0": v1["ci"][0] > 0},
        "V2_gain_prose": {**v2, "above_0": v2["ci"][0] > 0}, "V2_gain_orig": v2_orig, "V2_gain_prose_minus_orig": v2_diff,
        "V3_nov_minus_entropy": {"prose": v3["prose"], "orig": v3["orig"], "holds": v3["prose"]["ci"][0] <= 0},
        "overall": "format-tied" if fmt_tied else "follows computed values" if follows else "mixed",
        "n_problems": len({k[0] for k in keys}), "n_values": len(keys)}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_prose.json"), "w"))
    for k, v in res["by_layout"].items():
        print(f"RESULT {k:24s} n {v['n']:4d} exact {v['exact']:.3f}" + ("" if v["covered"] is None else f" covered {v['covered']:.3f}"))
    print("rates", json.dumps({k: round(v, 3) for k, v in res["rates"].items()}))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
