"""R10 (paper3/R10_LEARNED_RULES.md): does a learned signal find computed values where the '=' cue is gone?
Detectors of "a computed value starts at the next byte", read at causal decision positions (previous byte not
alphanumeric) from BLT-1B's patcher state or Qwen2.5-1.5B's layer-14 state, trained on equation text (orig) or on
prose (data/prose_rewrites, R9); then real BLT-1B layouts at 10% with each detector's top 3% forced, rest entropy.

    .venv-blt/bin/python realblt_learned.py          -> results/blt_learned.json   (R10_SMOKE=1: 40 train / 20 test)
"""
import json
import os

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import blt_layouts
import blt_load
import reuse_screen
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts
from scale.rules import table_lookup

DATA = os.path.join(_seg.HERE, "data", "prose_rewrites")
SMOKE = os.environ.get("R10_SMOKE") == "1"
QWEN, QLAYER = "Qwen/Qwen2.5-1.5B", 14
R, Q, NEG, PCA, LAM, NBOOT = 0.10, 0.03, 0.25, 32, 10.0, 2000
FORMATS = ("orig", "prose")
ROUTERS = ("rp_orig", "rp_prose", "rq_orig", "rq_prose")          # r{p: patcher, q: qwen}_{trained on}
HYBRIDS = ("table",) + ROUTERS


def isal(c):
    return (48 <= c <= 57) or (65 <= c <= 90) or (97 <= c <= 122)


def build(r, f):
    """Bytes, spans (token positions), candidate byte positions and their labels for one problem in format f."""
    t = r[f]; b = t.encode()
    bpos = lambda c: len(t[:c].encode())
    head = len((r["question"] + "\n").encode())
    vstart = [bpos(s) for s in r[f + "_spans"]]
    spans = [("res", s + 1, s + len(v.encode()) + 1) for s, v in zip(vstart, r["values"])]
    fs = bpos(r["final_start_" + f]); spans.append(("fin", fs + 1, len(b) + 1))
    line = len(b) - len(("\nThe final answer is " + r["answer"]).encode())     # candidates stop before the final line
    cand = np.array([i for i in range(max(head, 1), line) if not isal(b[i - 1]) and b[i - 1] < 128 and b[i] < 128], int)   # ASCII only (smoke fix)
    V = set(vstart)
    return {"b": b, "spans": spans, "cand": cand, "y": np.array([i in V for i in cand], float), "head": head,
            "vtok": [s + 1 for s in vstart]}


def fit_logit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A = (X - mu) / sd
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    V = Vt[:PCA].T
    A1 = torch.tensor(np.c_[A @ V, np.ones(len(A))]); yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(A1.shape[1], dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([w], max_iter=200)

    def closure():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(A1 @ w, yt) + LAM * (w[:-1] ** 2).sum() / len(yt)
        l.backward(); return l
    opt.step(closure)
    return mu, sd, V, w.detach().numpy()


def apply(m, X):
    mu, sd, V, w = m
    return np.c_[((X - mu) / sd) @ V, np.ones(len(X))] @ w


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or not y.any():
        return np.nan
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    rk = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    recs = {sp: [json.loads(l) for l in open(os.path.join(DATA, f"{sp}.jsonl"))] for sp in ("train", "test")}
    if SMOKE:
        recs = {"train": recs["train"][:40], "test": recs["test"][:20]}
    btok = AutoTokenizer.from_pretrained(MODEL)
    P = {}                                                        # (format, split) -> list of problem dicts
    for f in FORMATS:
        for sp, rs in recs.items():
            out = []
            for r in rs:
                x = build(r, f)
                if len(btok(x["b"].decode()).input_ids) <= MAX_TOKENS:
                    x["idx"] = r["idx"]; out.append(x)
            P[(f, sp)] = out

    # stage 1: Qwen2.5-1.5B layer-14 state at the last token ending before each candidate byte
    qtok = AutoTokenizer.from_pretrained(QWEN)
    qm = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    for key, xs in P.items():
        for k, x in enumerate(xs):
            text = x["b"].decode(); enc = qtok(text, return_offsets_mapping=True)
            with torch.no_grad():
                hs = qm(torch.tensor([enc["input_ids"]], device=dev), output_hidden_states=True).hidden_states[QLAYER][0]
            hs = hs.float().cpu().numpy()
            ends = np.array([e for _, e in enc["offset_mapping"]])
            cpos = np.array([len(x["b"][:i].decode()) for i in x["cand"]])
            tq = np.array([int(np.flatnonzero(ends <= c - 1).max()) if (ends <= c - 1).any() else 0 for c in cpos])
            # c - 1: the token ending at c would be the separator itself, which Qwen splits off only before a digit
            # (a tokenizer leak of the next byte; smoke test 2026-10-27)
            x["Hq"] = hs[tq].astype(np.float16)
            if dev == "mps" and k % 50 == 0:
                torch.mps.empty_cache()
        print(f"qwen states {key}: {len(xs)} problems", flush=True)
    del qm
    if dev == "mps":
        torch.mps.empty_cache()

    # stage 2: BLT-1B patcher entropy, hidden state, table score
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    cap = {}
    hook = model.model.patcher.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    tabs = {}
    for f in FORMATS:
        fb = np.frombuffer(b"\n\n".join(x["b"] for x in P[(f, "train")]), np.uint8)
        T2, T1, glob = reuse_screen.tab_fit(fb, -reuse_screen.match_signals(fb)[0])
        tabs[f] = (T2, T1, float(glob))
    for (f, sp), xs in P.items():
        for k, x in enumerate(xs):
            ids = btok(x["b"].decode(), return_tensors="pt").input_ids.to(dev); n = ids.shape[1]
            with torch.no_grad():
                ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                max_patch_length=cfg.max_patch_length)
            e = ent[0].float().cpu().numpy(); h = cap["h"][0].float().cpu().numpy()
            s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
            v = table_lookup(*tabs[f], np.frombuffer(x["b"], np.uint8))
            x.update({"ids": ids.cpu(), "sig": {"entropy": s}, "Hp": h[x["cand"]].astype(np.float16),
                      "score": {"entropy": e[x["cand"]], "table": v[x["cand"]]}})
            if dev == "mps" and k % 50 == 0:
                torch.mps.empty_cache()
        print(f"patcher {f}/{sp}: {len(xs)} problems", flush=True)
    hook.remove()

    # routers
    rng = np.random.default_rng(0)
    for x in P[("orig", "train")] + P[("prose", "train")]:
        x["keep"] = (x["y"] > 0) | (rng.random(len(x["y"])) < NEG)
    stack = lambda xs, fld, sel=None: np.concatenate([(x[fld] if sel is None else x[fld][x[sel]]) for x in xs]).astype(np.float32)
    for rname in ROUTERS:
        fld = "Hp" if rname[1] == "p" else "Hq"; ftr = rname.split("_")[1]
        tr = P[(ftr, "train")]
        m = fit_logit(stack(tr, fld, "keep"), stack(tr, "y", "keep"))
        for (f, sp), xs in P.items():
            if f == ftr and sp == "train":                          # out-of-fold on its own train problems
                fold = {i: j % 5 for j, i in enumerate(rng.permutation(len(xs)))}
                for fo in range(5):
                    trx = [x for i, x in enumerate(xs) if fold[i] != fo]
                    mf = fit_logit(stack(trx, fld, "keep"), stack(trx, "y", "keep"))
                    for i, x in enumerate(xs):
                        if fold[i] == fo:
                            x["score"][rname] = apply(mf, x[fld].astype(np.float32))
            else:
                for x in xs:
                    x["score"][rname] = apply(m, x[fld].astype(np.float32))
        print(f"router {rname} trained ({int(stack(tr, 'keep').sum())} candidates)", flush=True)

    res = {"n": {f"{f}/{sp}": len(xs) for (f, sp), xs in P.items()}, "auc": {}, "thresholds": {}, "layouts": {},
           "verdicts": {}}
    # detection AUCs (test), bootstrap over problems
    brng = np.random.default_rng(1)
    for f in FORMATS:
        xs = P[(f, "test")]
        for sname in ("entropy", "table") + ROUTERS:
            val = auc(np.concatenate([x["y"] for x in xs]), np.concatenate([x["score"][sname] for x in xs]))
            bs = []
            for _ in range(NBOOT):
                pick = [xs[i] for i in brng.integers(0, len(xs), len(xs))]
                bs.append(auc(np.concatenate([x["y"] for x in pick]), np.concatenate([x["score"][sname] for x in pick])))
            res["auc"][f"{f}|{sname}"] = {"value": val, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}
            print(f"AUC {f:5s} {sname:9s} {val:.3f} [{res['auc'][f'{f}|{sname}']['ci'][0]:.3f}, {res['auc'][f'{f}|{sname}']['ci'][1]:.3f}]", flush=True)
    for f in FORMATS:                                               # paired AUC difference rq_orig - rp_orig
        xs = P[(f, "test")]; d = []
        for _ in range(NBOOT):
            pick = [xs[i] for i in brng.integers(0, len(xs), len(xs))]
            yy = np.concatenate([x["y"] for x in pick])
            d.append(auc(yy, np.concatenate([x["score"]["rq_orig"] for x in pick])) - auc(yy, np.concatenate([x["score"]["rp_orig"] for x in pick])))
        res["auc"][f"{f}|rq_orig-rp_orig"] = {"value": res["auc"][f"{f}|rq_orig"]["value"] - res["auc"][f"{f}|rp_orig"]["value"],
                                               "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}

    # hybrids: thresholds on the format's own train problems (label-free), 3% of positions forced
    rows = []
    for f in FORMATS:
        tr = P[(f, "train")]
        ntok = sum(x["ids"].shape[1] for x in tr)
        for hname in HYBRIDS:
            allsc = np.sort(np.concatenate([x["score"][hname] for x in tr]))[::-1]
            thr = float(allsc[min(len(allsc) - 1, int(Q * ntok))])
            res["thresholds"][f"{f}|{hname}"] = thr
            for x in P[(f, "train")] + P[(f, "test")]:
                x.setdefault("forced", {})[hname] = (x["cand"][x["score"][hname] > thr] + 1).tolist()   # token = byte + 1
        fit = blt_layouts.Fit((R,))
        for x in tr:
            fit.add(x["sig"], x["forced"])
        fit.done()
        for k, x in enumerate(P[(f, "test")]):
            ids = x["ids"].to(dev); n = ids.shape[1]
            ent = set(blt_layouts.layout("entropy", R, x["sig"], fit).tolist())
            lay = {"entropy": sorted(ent), "with": sorted(ent | set(x["vtok"]))}
            for hname in HYBRIDS:
                lay[hname] = sorted(set(blt_layouts.layout(f"forced[{hname}]:entropy", R, x["sig"], fit, x["forced"]).tolist()))
            for lname, st in lay.items():
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                bt = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()     # bt[j]: token j+1
                S = set(st); h0 = x["head"] + 1
                for j, (kind, s, e) in enumerate(x["spans"]):
                    rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": kind, "j": j,
                                 "covered": s in S, "ok": bool((lp[s - 1:e - 1].argmax(-1) == ids[0, s:e]).all())})
                rows.append({"format": f, "problem": x["idx"], "layout": lname, "kind": "bits", "j": -1,
                             "bits": float(bt[h0 - 1:].sum()), "nbytes": int(n - h0), "rate": len(st) / n,
                             "forced_share": len(x["forced"].get(lname, [])) / n if lname in HYBRIDS else 0.0})
            if dev == "mps" and k % 20 == 0:
                torch.mps.empty_cache()
            if (k + 1) % 50 == 0:
                print(f"  layouts {f}: {k + 1}/{len(P[(f, 'test')])}", flush=True)

    # summaries and verdicts
    def sel(f, lay, kind="res"):
        return {(r["problem"], r["j"]): r for r in rows if r["format"] == f and r["layout"] == lay and r["kind"] == kind}

    def boot(keys, d):
        Pk = np.array([k[0] for k in keys]); d = np.asarray(d, float)
        probs = np.unique(Pk); by = {p: d[Pk == p] for p in probs}
        bs = [np.concatenate([by[p] for p in brng.choice(probs, len(probs))]).mean() for _ in range(NBOOT)]
        return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    for f in FORMATS:
        E = sel(f, "entropy"); keys = sorted(E)
        W = sel(f, "with")
        gap = np.mean([W[k]["ok"] for k in keys]) - np.mean([E[k]["ok"] for k in keys])
        bitsE = {k: r for k, r in sel(f, "entropy", "bits").items()}
        for lname in ("entropy", "with") + HYBRIDS:
            L = sel(f, lname); Fn = sel(f, lname, "fin"); B = sel(f, lname, "bits")
            out = {"exact_res": float(np.mean([L[k]["ok"] for k in keys])), "covered_res": float(np.mean([L[k]["covered"] for k in keys])),
                   "exact_fin": float(np.mean([r["ok"] for r in Fn.values()])),
                   "bpb_solution": float(sum(r["bits"] for r in B.values()) / sum(r["nbytes"] for r in B.values())),
                   "rate": float(np.mean([r["rate"] for r in B.values()])), "forced_share": float(np.mean([r["forced_share"] for r in B.values()]))}
            if lname != "entropy":
                out["minus_entropy_res"] = boot(keys, [float(L[k]["ok"]) - float(E[k]["ok"]) for k in keys])
                out["gap_closed"] = float((out["exact_res"] - np.mean([E[k]["ok"] for k in keys])) / gap) if gap > 0 else None
                bk = sorted(B)
                out["minus_entropy_bpb"] = boot([(k[0], 0) for k in bk], [(B[k]["bits"] - bitsE[k]["bits"]) / B[k]["nbytes"] for k in bk])
            res["layouts"][f"{f}|{lname}"] = out
            print(f"LAYOUT {f:5s} {lname:9s} exact res {out['exact_res']:.3f} cov {out['covered_res']:.3f} fin {out['exact_fin']:.3f} "
                  f"bpb {out['bpb_solution']:.4f} rate {out['rate']:.3f} forced {out['forced_share']:.3f}"
                  + (f" | vs entropy {100 * out['minus_entropy_res']['value']:+.1f} [{100 * out['minus_entropy_res']['ci'][0]:+.1f}, "
                     f"{100 * out['minus_entropy_res']['ci'][1]:+.1f}]" if lname != "entropy" else ""), flush=True)
    L = res["layouts"]
    res["verdicts"] = {
        "L1_router_patcher_prose": {**L["prose|rp_prose"]["minus_entropy_res"], "holds": L["prose|rp_prose"]["minus_entropy_res"]["ci"][0] > 0},
        "L2_router_patcher_orig_on_prose": {**L["prose|rp_orig"]["minus_entropy_res"], "holds": L["prose|rp_orig"]["minus_entropy_res"]["ci"][0] > 0},
        "L3_qwen_orig_auc_on_prose": {**res["auc"]["prose|rq_orig"], "holds": res["auc"]["prose|rq_orig"]["ci"][0] > 0.75,
                                      "patcher_orig_auc_on_prose": res["auc"]["prose|rp_orig"], "difference": res["auc"]["prose|rq_orig-rp_orig"]},
        "L4_bpb_and_gap": {h: {"bpb_minus_entropy": L[f"prose|{h}"]["minus_entropy_bpb"], "gap_closed": L[f"prose|{h}"]["gap_closed"]} for h in HYBRIDS}}
    json.dump({**res, "rows": rows}, open(os.path.join(_seg.HERE, "results", "blt_learned_smoke.json" if SMOKE else "blt_learned.json"), "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
