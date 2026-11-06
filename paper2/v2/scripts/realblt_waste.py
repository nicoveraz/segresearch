"""R8 (issue #29): does entropy patching spend its budget where it cannot help? Decision rules: paper2v2/R8_WASTE_RULES.md
(committed before any run). BLT-1B, stripped GSM8K (problem() of realblt_budget), thresholds fitted on train problems.

A  value of every patch start (entropy at 10%, the novelty table at 10%; first N_A test problems): starts split at
   random into 5 groups; each group removed in turn; value of a start = increase in bits over the bytes of the patch it
   started in the full layout. Starts labelled by position type.
B  patch starts needed for equal answers: accuracy against patch rate for entropy (5-30%), the table (5-15%), the '='
   rule (7.5-15%) and the table plus the answer marker (10%); k = entropy's rate at the table's 10% computed-result
   accuracy / the table's rate.
C  bits per byte (whole text, question, solution) against patch rate, same layouts.

    .venv-blt/bin/python realblt_waste.py [N=300] [N_A=150]      -> results/blt_waste.json
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
from realblt_budget import GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem
from scale.rules import table_lookup

NOVTAB_GSM = os.path.expanduser("~/.cache/segresearch-blt/novtab_gsm.npz")
ENT_R = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TAB_R = (0.05, 0.075, 0.10, 0.15)
EQ_R = (0.075, 0.10, 0.15)
LAYOUTS = ([(f"entropy@{r}", "entropy", r) for r in ENT_R] + [(f"table@{r}", "nov", r) for r in TAB_R]
           + [(f"eq@{r}", "forced[eq]:entropy", r) for r in EQ_R] + [("table+marker@0.1", "forced[final]:nov", 0.10)])
GROUPS = 5
BOOT = 1000


def prep(p, tok, tab):
    pr = problem(p, tok, "cpu")
    if pr is None:
        return None
    b, ids, spans, res = pr
    n = ids.shape[1]
    q_end = len(p["question"].encode()) + 1                     # bytes of the question and its newline
    fin = [s for k, s, _ in spans if k == "fin"]
    v = table_lookup(tab["T2"], tab["T1"], float(tab["glob"]), np.frombuffer(b, np.uint8))
    nov = np.full(n, -np.inf); nov[2:] = v[1:n - 1]              # token t = byte t-1
    return {"b": b, "ids": ids, "spans": spans, "res": set(res.tolist()), "fin": set(fin), "q_end": q_end,
            "nov": nov, "forced": {"eq": res.tolist(), "final": fin}}


def pos_type(x, s):
    j = s - 1; b = x["b"]
    if j < x["q_end"]:
        return "question"
    if s in x["res"]:
        return "computed"
    if s in x["fin"]:
        return "final"
    c = chr(b[j]) if b[j] < 128 else "?"
    prev = chr(b[j - 1]) if j > 0 and b[j - 1] < 128 else " "
    if c.isdigit():
        return "number"
    if c.isalpha():
        return "mid-word" if prev.isalpha() else "word start"
    return {" ": "space", "\n": "newline"}.get(c, "punct")


def interp_rate(rates, accs, target):
    """Smallest rate at which the (rate, acc) curve reaches target, linear between points; None if never."""
    pts = sorted(zip(rates, accs))
    for (r0, a0), (r1, a1) in zip(pts, pts[1:]):
        if a0 >= target:
            return r0
        if a1 >= target:
            return r0 + (target - a0) * (r1 - r0) / (a1 - a0)
    return pts[-1][0] if pts and pts[-1][1] >= target else None


def main(N=300, N_A=150):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    tab = np.load(NOVTAB_GSM)

    def entropy_sig(x):
        with torch.no_grad():
            ent, _, _ = model.model.patcher(x["ids"].to(dev), patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                            max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy(); n = x["ids"].shape[1]
        s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
        return {"entropy": s, "nov": x["nov"]}

    def bits(x, starts):
        ids = x["ids"].to(dev); n = ids.shape[1]
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(sorted(starts), n).to(dev),
                                         use_cache=False).logits[0].float(), -1)
        bt = np.r_[0.0, (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()]   # bits of token t
        ok = {}
        for k, s, e in x["spans"]:
            ok[(k, s)] = bool((lp[s - 1:e - 1].argmax(-1) == ids[0, s:e]).all())
        return bt, ok

    train = [x for x in (prep(json.loads(l), tok, tab) for l in list(open(GSM_TRAIN))[:N]) if x]
    test = [x for x in (prep(json.loads(l), tok, tab) for l in list(open(GSM))[:N]) if x]
    fit = blt_layouts.Fit(tuple(sorted({r for _, _, r in LAYOUTS})))
    for x in train:
        fit.add(entropy_sig(x), x["forced"])
    fit.done()
    rng = np.random.default_rng(0)
    rowsB, rowsA = [], []
    for i, x in enumerate(test):
        sig = entropy_sig(x); n = x["ids"].shape[1]; q = x["q_end"] + 1          # tokens 1..q-1 are the question
        layout_bits = {}
        for name, kind, R in LAYOUTS:
            st = set(blt_layouts.layout(kind, R, sig, fit, x["forced"]).tolist())
            bt, ok = bits(x, st)
            layout_bits[name] = (st, bt)
            rowsB.append({"problem": i, "layout": name, "rate": len(st) / n,
                          "n_res": sum(k == "res" for k, _ in ok), "ok_res": sum(v for (k, _), v in ok.items() if k == "res"),
                          "n_fin": sum(k == "fin" for k, _ in ok), "ok_fin": sum(v for (k, _), v in ok.items() if k == "fin"),
                          "bits_all": float(bt[1:].sum()), "n_all": n - 1, "bits_q": float(bt[1:q].sum()), "n_q": q - 1,
                          "bits_sol": float(bt[q:].sum()), "n_sol": n - q})
        if i < N_A:
            for name in ("entropy@0.1", "table@0.1"):
                st, full = layout_bits[name]
                S = sorted(t for t in st if t >= 2); nxt = dict(zip(S, S[1:] + [n]))
                grp = rng.integers(0, GROUPS, len(S))
                for g in range(GROUPS):
                    rem = [s for s, gg in zip(S, grp) if gg == g]
                    if not rem:
                        continue
                    bt, _ = bits(x, st - set(rem))
                    for s in rem:
                        rowsA.append({"problem": i, "layout": name, "start": int(s), "type": pos_type(x, s),
                                      "patch_len": int(nxt[s] - s), "value": float((bt[s:nxt[s]] - full[s:nxt[s]]).sum())})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} problems", flush=True)
    res = analyse(rowsA, rowsB)
    res["n_problems"] = len(test); res["n_problems_A"] = min(N_A, len(test))
    json.dump({**res, "rowsA": rowsA, "rowsB": rowsB}, open(os.path.join(_seg.HERE, "results", "blt_waste.json"), "w"))
    report(res)


def analyse(rowsA, rowsB):
    rng = np.random.default_rng(1)
    res = {"A": {}, "B": {}, "C": {}}
    # A
    for name in ("entropy@0.1", "table@0.1"):
        R = [r for r in rowsA if r["layout"] == name]
        P = np.array([r["problem"] for r in R]); V = np.array([r["value"] for r in R]); T = np.array([r["type"] for r in R])
        probs = np.unique(P); idx = {p: np.flatnonzero(P == p) for p in probs}
        bs_share, bs_mean = [], []
        for _ in range(BOOT):
            j = np.concatenate([idx[p] for p in rng.choice(probs, len(probs))])
            bs_share.append(float((V[j] <= 0).mean())); bs_mean.append(float(V[j].mean()))
        d = {"n_starts": int(len(V)), "share_value_le_0": float((V <= 0).mean()),
             "share_ci": [float(np.nanpercentile(bs_share, 2.5)), float(np.nanpercentile(bs_share, 97.5))],
             "mean_value": float(V.mean()), "mean_ci": [float(np.percentile(bs_mean, 2.5)), float(np.percentile(bs_mean, 97.5))],
             "by_type": {}}
        for t in sorted(set(T)):
            m = T == t
            d["by_type"][t] = {"share_of_starts": float(m.mean()), "share_value_le_0": float((V[m] <= 0).mean()),
                               "mean_value": float(V[m].mean()), "n": int(m.sum())}
        res["A"][name] = d
    lo, hi = res["A"]["entropy@0.1"]["share_ci"]
    res["A"]["verdict"] = "HOLDS" if lo >= 0.20 else ("FAILS" if hi < 0.10 else "INCONCLUSIVE")

    # B and C: per-problem sums, resampled by problem
    names = sorted({r["layout"] for r in rowsB}); probs = sorted({r["problem"] for r in rowsB})
    M = {(r["layout"], r["problem"]): r for r in rowsB}
    keys = ("rate", "n_res", "ok_res", "n_fin", "ok_fin", "bits_all", "n_all", "bits_q", "n_q", "bits_sol", "n_sol")
    arr = {k: np.array([[M[(nm, p)][k] for p in probs] for nm in names], float) for k in keys}

    def curves(pick):
        s = {k: v[:, pick].sum(1) for k, v in arr.items()}
        rate = arr["rate"][:, pick].mean(1)
        return {nm: {"rate": rate[i], "acc_res": s["ok_res"][i] / s["n_res"][i], "acc_fin": s["ok_fin"][i] / s["n_fin"][i],
                     "bpb_all": s["bits_all"][i] / s["n_all"][i], "bpb_q": s["bits_q"][i] / s["n_q"][i],
                     "bpb_sol": s["bits_sol"][i] / s["n_sol"][i]} for i, nm in enumerate(names)}

    def ks(c):
        ent = [c[f"entropy@{r}"] for r in ENT_R]
        out = {}
        for ref, metric in (("table@0.1", "acc_res"), ("eq@0.1", "acc_res"), ("table+marker@0.1", "acc_fin"), ("eq@0.1", "acc_fin")):
            need = interp_rate([e["rate"] for e in ent], [e[metric] for e in ent], c[ref][metric])
            out[f"{ref} {metric}"] = (need / c[ref]["rate"]) if need is not None else float("inf")
        tabs = [c[f"table@{r}"] for r in TAB_R]          # C: rate the table needs to match entropy@10's whole-text bpb
        need = interp_rate([t["rate"] for t in tabs], [-t["bpb_all"] for t in tabs], -c["entropy@0.1"]["bpb_all"])
        out["C table rate to match entropy@0.1 bpb_all / entropy rate"] = (need / c["entropy@0.1"]["rate"]) if need is not None else float("inf")
        return out
    full = curves(np.arange(len(probs)))
    res["curves"] = full
    k0 = ks(full)
    bs = {k: [] for k in k0}
    for _ in range(BOOT):
        kk = ks(curves(rng.integers(0, len(probs), len(probs))))
        for k, v in kk.items():
            bs[k].append(v)
    for k, v in k0.items():
        res["B" if not k.startswith("C ") else "C"][k] = {"k": v, "ci": [float(np.nanpercentile(bs[k], 2.5)), float(np.nanpercentile(bs[k], 97.5))]}
    lo, hi = res["B"]["table@0.1 acc_res"]["ci"]
    res["B"]["verdict"] = "HOLDS" if lo >= 1.5 else ("FAILS" if hi < 1.2 else "INCONCLUSIVE")
    return res


def report(res):
    for name in ("entropy@0.1", "table@0.1"):
        d = res["A"][name]
        print(f"A {name}: starts {d['n_starts']} | value<=0 {d['share_value_le_0']:.3f} {[round(x, 3) for x in d['share_ci']]}"
              f" | mean {d['mean_value']:+.3f} bits {[round(x, 3) for x in d['mean_ci']]}")
        for t, v in d["by_type"].items():
            print(f"    {t:11s} {v['share_of_starts']:.3f} of starts | value<=0 {v['share_value_le_0']:.3f} | mean {v['mean_value']:+.3f}")
    print("A verdict:", res["A"]["verdict"])
    for nm, c in sorted(res["curves"].items(), key=lambda kv: (kv[0].split("@")[0], kv[1]["rate"])):
        print(f"  {nm:18s} rate {c['rate']:.3f} | computed {c['acc_res']:.3f} final {c['acc_fin']:.3f} | bpb all {c['bpb_all']:.3f} q {c['bpb_q']:.3f} sol {c['bpb_sol']:.3f}")
    for part in ("B", "C"):
        for k, v in res[part].items():
            if k != "verdict":
                print(f"{part} k[{k}] = {v['k']:.2f} {[round(x, 2) for x in v['ci']]}")
    print("B verdict:", res["B"]["verdict"])


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
