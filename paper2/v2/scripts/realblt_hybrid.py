"""R8b H (issue #29): a label-free hybrid at inference on BLT-1B. Rules: paper2v2/R8b_HYBRID_D_RULES.md (committed
before any run). hybrid(q)@R = patch starts at every position whose novelty-table value is in the top q of training
bytes, the rest of the budget R filled by entropy. Paired by problem with R8's layouts (results/blt_waste.json: same
294 problems, same training problems and thresholds).

    .venv-blt/bin/python realblt_hybrid.py [N=300]      -> results/blt_hybrid.json
"""
import json
import os
import sys

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import realblt_waste as w
from realblt_budget import GSM, GSM_TRAIN, MODEL, lengths_from_starts

HYB = [(0.015, 0.10), (0.03, 0.10), (0.05, 0.10), (0.03, 0.15)]
PRIMARY = "hybrid(0.03)@0.1"
BOOT = 1000


def main(N=300):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    tab = np.load(w.NOVTAB_GSM)

    def entropy_sig(x):
        with torch.no_grad():
            ent, _, _ = model.model.patcher(x["ids"].to(dev), patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                            max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy(); n = x["ids"].shape[1]
        s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
        return {"entropy": s, "nov": x["nov"]}

    train = [x for x in (w.prep(json.loads(l), tok, tab) for l in list(open(GSM_TRAIN))[:N]) if x]
    test = [x for x in (w.prep(json.loads(l), tok, tab) for l in list(open(GSM))[:N]) if x]
    allv = np.concatenate([x["nov"][np.isfinite(x["nov"])] for x in train])
    thr = {q: float(np.quantile(allv, 1 - q)) for q, _ in HYB}
    for x in train + test:
        for q in thr:
            x["forced"][f"q{int(round(q * 1000))}"] = np.flatnonzero(x["nov"] >= thr[q]).tolist()   # keys must be \w+
    fit = blt_layouts.Fit(tuple(sorted({R for _, R in HYB})))
    for x in train:
        fit.add(entropy_sig(x), x["forced"])
    fit.done()
    rows = []
    for i, x in enumerate(test):
        sig = entropy_sig(x); ids = x["ids"].to(dev); n = ids.shape[1]; qe = x["q_end"] + 1
        for q, R in HYB:
            st = sorted(set(blt_layouts.layout(f"forced[q{int(round(q * 1000))}]:entropy", R, sig, fit, x["forced"]).tolist()))
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                             use_cache=False).logits[0].float(), -1)
            bt = np.r_[0.0, (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()]
            ok = {(k, s): bool((lp[s - 1:e - 1].argmax(-1) == ids[0, s:e]).all()) for k, s, e in x["spans"]}
            forced_share = len([t for t in x["forced"][f"q{int(round(q * 1000))}"] if t >= 2]) / n
            rows.append({"problem": i, "layout": f"hybrid({q})@{R}", "rate": len(st) / n, "forced_share": forced_share,
                         "n_res": sum(k == "res" for k, _ in ok), "ok_res": sum(v for (k, _), v in ok.items() if k == "res"),
                         "n_fin": sum(k == "fin" for k, _ in ok), "ok_fin": sum(v for (k, _), v in ok.items() if k == "fin"),
                         "bits_all": float(bt[1:].sum()), "n_all": n - 1, "bits_q": float(bt[1:qe].sum()), "n_q": qe - 1,
                         "bits_sol": float(bt[qe:].sum()), "n_sol": n - qe,
                         "covered_res": sum(s in set(st) for k, s, _ in x["spans"] if k == "res")})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} problems", flush=True)
    res = analyse(rows)
    json.dump({**res, "thresholds": thr, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_hybrid.json"), "w"))


def analyse(rows):
    waste = json.load(open(os.path.join(_seg.HERE, "results", "blt_waste.json")))["rowsB"]
    allr = rows + waste
    M = {(r["layout"], r["problem"]): r for r in allr}
    probs = sorted({r["problem"] for r in rows})
    for p in probs:                                             # same problems as R8
        assert M[(PRIMARY, p)]["n_res"] == M[("entropy@0.1", p)]["n_res"] and M[(PRIMARY, p)]["n_all"] == M[("entropy@0.1", p)]["n_all"], p
    rng = np.random.default_rng(2)

    def stat(nm, pick):
        rr = [M[(nm, p)] for p in pick]
        s = lambda k: sum(r[k] for r in rr)
        return {"rate": float(np.mean([r["rate"] for r in rr])), "acc_res": s("ok_res") / s("n_res"), "acc_fin": s("ok_fin") / s("n_fin"),
                "bpb_all": s("bits_all") / s("n_all"), "bpb_q": s("bits_q") / s("n_q"), "bpb_sol": s("bits_sol") / s("n_sol")}
    names = sorted({r["layout"] for r in rows}) + ["entropy@0.1", "entropy@0.15", "eq@0.1", "eq@0.15", "table@0.1", "table@0.15"]
    res = {"curves": {nm: stat(nm, probs) for nm in names}, "paired": {}}
    res["curves"].update({nm + " forced_share": {"value": float(np.mean([M[(nm, p)]["forced_share"] for p in probs]))}
                          for nm in sorted({r["layout"] for r in rows})})
    pairs = [(f"hybrid({q})@{R}", f"entropy@{R}") for q, R in HYB] + [(f"hybrid({q})@{R}", f"eq@{R}") for q, R in HYB] \
        + [("hybrid(0.03)@0.1", "table@0.1")]
    for a, b in pairs:
        d0 = {m: stat(a, probs)[m] - stat(b, probs)[m] for m in ("acc_res", "acc_fin", "bpb_all")}
        bs = {m: [] for m in d0}
        for _ in range(BOOT):
            pick = rng.choice(probs, len(probs))
            sa, sb = stat(a, pick), stat(b, pick)
            for m in d0:
                bs[m].append(sa[m] - sb[m])
        res["paired"][f"{a} - {b}"] = {m: {"diff": float(d0[m]), "ci": [float(np.percentile(bs[m], 2.5)), float(np.percentile(bs[m], 97.5))]} for m in d0}
    p = res["paired"][f"{PRIMARY} - entropy@0.1"]
    gain_lo, gain_hi = 100 * p["acc_res"]["ci"][0], 100 * p["acc_res"]["ci"][1]
    bpb_lo, bpb_hi = p["bpb_all"]["ci"]
    res["verdict"] = "HOLDS" if (gain_lo >= 20 and bpb_hi <= 0.02) else ("FAILS" if (gain_hi < 10 or bpb_lo > 0.05) else "INCONCLUSIVE")
    for nm, c in res["curves"].items():
        print(f"  {nm:22s} {c}")
    for k, v in res["paired"].items():
        print(f"PAIRED {k:34s} computed {100 * v['acc_res']['diff']:+6.1f} [{100 * v['acc_res']['ci'][0]:+6.1f}, {100 * v['acc_res']['ci'][1]:+6.1f}]"
              f" final {100 * v['acc_fin']['diff']:+6.1f} [{100 * v['acc_fin']['ci'][0]:+6.1f}, {100 * v['acc_fin']['ci'][1]:+6.1f}]"
              f" bpb {v['bpb_all']['diff']:+.3f} [{v['bpb_all']['ci'][0]:+.3f}, {v['bpb_all']['ci'][1]:+.3f}]")
    print("H verdict:", res["verdict"])
    return res


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))
