"""R22 (paper3/R22_CUE_BUDGET_RULES.md): what R9 measured, and where the detector's final-answer cost comes from, on
R20's fresh problems (data/prose_rewrites_fresh). Per computed value: the cue just before it ('= ', '$', plain space),
the patcher's next-byte distribution at its first digit (entropy, P(digit), entropy within the digits, max-prob),
coverage and exact match. Budget per value: entropy starts in the solution body per computed value, matched across
writings on train problems. Writings: orig, prose, orig_is (orig with each '= ' before a value replaced by 'is ').

BLT-1B layouts (thresholds, the detector, the dependence tables fitted on R9's 1,382 train problems):
    10%          E, D3, F2, D3F / D3R (DET-1: forced starts displace entropy starts only before the final-answer line /
                 only inside the solution body), VU / MP (entropy fused with value uncertainty / 1 - max-prob), with,
                 eq (orig only), dep, R3 (random 3% of the detector's candidates)
    15%, 20%     E, D3, F2, D3F, D3R
    matched      E at the rate that matches the other writing's budget per value at 10%
    orig_is      E at 10%
R9's test problems are a patcher-only pilot.

    .venv-blt/bin/python realblt_cue_budget.py   -> results/blt_cue_budget.json   (R22_SMOKE=1: 40 train / 20 of R9's test)
"""
import json
import os
import re
import subprocess
import time
from collections import defaultdict

import numpy as np

import _seg  # noqa: F401
import torch
import transformers
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import realblt_learned as rl
import typed_screen
from realblt_autolabel import autolabel
from realblt_budget import MAX_TOKENS, MODEL, lengths_from_starts

SMOKE = os.environ.get("R22_SMOKE") == "1"
FRESH = os.path.join(_seg.HERE, "data", "prose_rewrites_fresh", "test.jsonl")
RES = os.path.join(_seg.HERE, "results")
OUT = os.path.join(RES, "blt_cue_budget_smoke.json" if SMOKE else "blt_cue_budget.json")
R20 = os.path.join(RES, "blt_matched_share_smoke.json" if SMOKE else "blt_matched_share.json")
R, NBOOT, NEG = rl.R, rl.NBOOT, rl.NEG
W = 2.0                                                    # F2: entropy + 2 x the detector's probability (R19's L*)
HIGH = (0.15, 0.20)                                        # M2
GRID = np.round(np.arange(0.02, 0.40 + 1e-9, 0.0025), 4)   # rates searched for the matched budget per value
CURVE = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25)
NDEP = 40 if SMOKE else 400                                # dependence tables: first NDEP train problems per format
QR3 = 0.03                                                 # R3: forced share of the random-candidate null
DIG, DOL = np.arange(52, 62), 40                           # token ids of '0'-'9' and '$' (token id = byte + 4)
LN2 = float(np.log(2))
WRITINGS = ("orig", "prose", "orig_is")
PM = ("H", "Pd", "Pdol", "Hd", "maxp", "T", "U")           # patcher measures at a unit's first byte
FKEY = {"D3": "D3", "D3R": "D3", "D3F": "D3", "R3": "R3", "eq": "eq", "with": "comp"}
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def cue(text, c):
    """The bytes just before a value starting at character c, as r9_cue_split.py."""
    pre = text[:c]
    if pre.endswith("= ") or pre.endswith("="):
        return "= "
    if pre.endswith("$"):
        return "$"
    return "space" if pre.endswith(" ") else "other"


def swap_is(r):
    """orig with every '= ' right before a computed value (or before its '$') replaced by 'is ', spans shifted."""
    t, parts, last, sh, spans, sw = r["orig"], [], 0, 0, [], []
    for c in r["orig_spans"]:
        k = c - 2 if t[:c].endswith("= ") else c - 3 if t[:c].endswith("= $") else None
        if k is not None:
            parts.append(t[last:k] + "is "); last = k + 2; sh += 1
        spans.append(c + sh); sw.append(k is not None)
    parts.append(t[last:])
    r2 = dict(r, orig="".join(parts), orig_spans=spans, final_start_orig=r["final_start_orig"] + sh)
    assert all(r2["orig"][c:c + len(v)] == v for c, v in zip(spans, r["values"])), r["idx"]
    return r2, sw


def text_of(r, w):
    """(record, format key for rl.build, cue of each computed value) in writing w."""
    if w == "orig_is":
        r2, sw = swap_is(r)
        return r2, "orig", ["is " if s else cue(r2["orig"], c) for c, s in zip(r2["orig_spans"], sw)]
    return r, w, [cue(r[w], c) for c in r[w + "_spans"]]


def body(x, which="body"):
    """Token mask of the solution body, after the question line and before the final-answer line (bytes head..line-1);
    which='qbody': everything before the final-answer line (tokens 2..line; tokens 0 and 1 always start)."""
    m = np.zeros(x["ids"].shape[1], bool); m[x["h0"] if which == "body" else 2:x["line"] + 1] = True
    return m


def patcher_pass(model, cfg, btok, recs, w, dev, cap, keep_h):
    xs = []
    for r in recs:
        r2, f, cues = text_of(r, w)
        x = rl.build(r2, f)
        ids = btok(x["b"].decode(), return_tensors="pt").input_ids
        if ids.shape[1] > MAX_TOKENS:
            continue
        n = ids.shape[1]
        assert n == len(x["b"]) + 1
        x.update({"idx": r["idx"], "w": w, "cues": cues + ["fin"], "nval": len(r["values"]), "h0": x["head"] + 1,
                  "line": len(x["b"]) - len(("\nThe final answer is " + r["answer"]).encode())})
        if keep_h:
            x["ya"] = autolabel(x)
        with torch.no_grad():
            ent, dl, logits = model.model.patcher(ids.to(dev), patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                  max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy(); lg = logits[0].float()
        lpb = torch.log_softmax(lg[:, 4:260], -1).cpu().numpy()             # bytes only, as realblt_typed.py
        T = typed_screen.typed(lpb)[3]; U = 1 - np.exp(lpb.max(-1))
        sig = {}
        for name, v in (("entropy", e), ("T", T), ("U", U)):
            s = np.full(n, -np.inf); s[2:] = v[1:n - 1]; sig[name] = s          # token t starts from the prediction at t-1
        at = np.array([s - 1 for _, s, _ in x["spans"]])
        p = torch.softmax(lg[at], -1).cpu().numpy()                           # the full vocabulary, as the entropy
        pd = p[:, DIG].sum(1); q = p[:, DIG] / np.maximum(pd[:, None], 1e-30)
        x["pm"] = {"H": e[at] / LN2, "Pd": pd, "Pdol": p[:, DOL], "Hd": -(q * np.log2(np.maximum(q, 1e-30))).sum(1),
                   "maxp": p.max(1), "T": T[at], "U": U[at]}
        x.update({"ids": ids, "sig": sig, "default": np.r_[0, np.cumsum(dl[0].cpu().numpy())[:-1]]})
        if keep_h:
            x["Hp"] = cap["h"][0].float().cpu().numpy()[x["cand"]].astype(np.float16)
        xs.append(x)
    return xs


def losses(model, ids, st, n, dev):
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                     use_cache=False).logits[0].float(), -1)
    return -lp[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()


def dep_table(model, xs, dev):
    """blt_screens/dependence_screen.py's boundary dependence, refitted on these train problems: BLT-1B's loss on the next
    4 bytes under the top 10% by entropy minus under its default layout, averaged per preceding 2-byte context."""
    s2, s1 = defaultdict(list), defaultdict(list)
    for k, x in enumerate(xs):
        ids = x["ids"].to(dev); n = ids.shape[1]; b = x["b"]
        tight = np.r_[0, 1, np.argsort(-x["sig"]["entropy"])[:max(int(0.10 * n) - 2, 1)]]
        gain = losses(model, ids, tight, n, dev) - losses(model, ids, x["default"], n, dev)
        for i in range(2, len(b) - 4):
            d = float(gain[i:i + 4].mean())                                   # tokens i+1..i+4 = bytes i..i+3
            s2[(b[i - 2], b[i - 1])].append(d); s1[b[i - 1]].append(d)
        if dev == "mps" and k % 20 == 0:
            torch.mps.empty_cache()
    T2 = {k: float(np.mean(v)) for k, v in s2.items() if len(v) >= 20}
    T1 = {k: float(np.mean(v)) for k, v in s1.items() if len(v) >= 20}
    return T2, T1, float(np.mean([d for v in s1.values() for d in v]))


def dep_signal(tab, b, n):
    T2, T1, glob = tab
    s = np.full(n, -np.inf)                                                   # token t: context bytes t-3, t-2 (realblt_typed.py)
    s[3:] = [T2.get((b[t - 3], b[t - 2]), T1.get(b[t - 2], glob)) for t in range(3, n)]
    return s


def fit_region(tr, fit, r_, which):
    """The region's fill threshold (D3R: the body; D3F: everything before the final-answer line): forced starts plus an
    entropy fill give the region as many starts as E gives it (train)."""
    target = nf = 0; vals = []
    for x in tr:
        bm = body(x, which); n = len(bm)
        st = blt_layouts.layout("entropy", r_, x["sig"], fit)
        target += int(bm[st].sum())
        fz = np.zeros(n, bool); fz[[t for t in x["forced"]["D3"] if 2 <= t < n]] = True
        nf += int((bm & fz).sum()); vals.append(x["sig"]["entropy"][bm & ~fz])
    v = np.sort(np.concatenate(vals))[::-1]; k = target - nf
    if k < 1:
        return float("inf"), 0.0, target, nf
    t = v[k - 1]
    return float(t), float((k - (v > t).sum()) / (v == t).sum()), target, nf


def region_layout(x, r_, fit, reg, which):
    """D3R / D3F: every start of E outside the region; inside it, D3's forced starts and the fitted entropy fill."""
    t_s, q = reg[:2]
    bm = body(x, which); n = len(bm); e = x["sig"]["entropy"]
    st = blt_layouts.layout("entropy", r_, x["sig"], fit)
    fz = np.zeros(n, bool); fz[[t for t in x["forced"]["D3"] if 2 <= t < n]] = True
    on = bm & (fz | (e > t_s) | ((e == t_s) & (blt_layouts._tiebreak(n) < q)))
    on[st[~bm[st]]] = True
    on[[0, 1]] = True
    return np.flatnonzero(on)


def per_value(tr, fit):
    """Budget per value at every grid rate: entropy starts in the solution body per computed value (train problems)."""
    xs = [x for x in tr if x["nval"]]
    be = np.sort(np.concatenate([x["sig"]["entropy"][body(x)] for x in xs])); nv = sum(x["nval"] for x in xs)
    out = []
    for g in GRID:
        t, q = fit._threshold("entropy", float(g))
        out.append(((be > t).sum() + q * (be == t).sum()) / nv)
    return np.array(out)


def layouts_for(f, fit, reg, r_m):
    def lays(x):
        sig, fo = x["sig"], x["forced"]
        L = lambda kind, r_=R: blt_layouts.layout(kind, r_, sig, fit, fo)
        out = {"E": L("entropy"), "D3": L("forced[D3]:entropy"), "F2": L("F2"),
               "D3R": region_layout(x, R, fit, reg[("body", R)], "body"), "D3F": region_layout(x, R, fit, reg[("qbody", R)], "qbody"),
               "VU": L("max:entropy,T"), "MP": L("max:entropy,U")}
        out["with"] = np.array(sorted(set(out["E"].tolist()) | set(x["vtok"])), int)
        if f == "orig":
            out["eq"] = L("forced[eq]:entropy")
        out["dep"] = L("dep"); out["R3"] = L("forced[R3]:entropy")
        for r_ in HIGH:
            tag = f"@{round(r_ * 100)}"
            out["E" + tag] = L("entropy", r_); out["D3" + tag] = L("forced[D3]:entropy", r_)
            out["F2" + tag] = L("F2", r_); out["D3R" + tag] = region_layout(x, r_, fit, reg[("body", r_)], "body")
            out["D3F" + tag] = region_layout(x, r_, fit, reg[("qbody", r_)], "qbody")
        out["E@m"] = L("entropy", r_m)
        return out
    return lays


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    train = [json.loads(l) for l in open(os.path.join(rl.DATA, "train.jsonl"))]
    r9test = [json.loads(l) for l in open(os.path.join(rl.DATA, "test.jsonl"))]
    fresh = [json.loads(l) for l in open(FRESH)]
    if SMOKE:
        train, r9test = train[:40], r9test[:20]; fresh = r9test
    btok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    cap = {}
    hook = model.model.patcher.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    P = {}
    for w in WRITINGS:
        for sp, rs in (("train", train), ("pilot", r9test), ("fresh", fresh)):
            if SMOKE and sp == "pilot":
                continue
            P[(w, sp)] = patcher_pass(model, cfg, btok, rs, w, dev, cap, keep_h=(w != "orig_is" and sp != "pilot"))
            log(f"patcher {w}/{sp}: {len(P[(w, sp)])} problems")
    if SMOKE:
        P.update({(w, "pilot"): P[(w, "fresh")] for w in WRITINGS})
    hook.remove()

    res = {"meta": {"smoke": SMOKE, "device": dev, "torch": torch.__version__, "transformers": transformers.__version__,
                    "window": blt_load.WINDOW, "model": MODEL,
                    "commit": subprocess.run(["git", "-C", _seg.HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
           "n": {}, "thresholds_bits": {}, "budget_per_value": {}, "matched": {}, "region": {}, "rates_patcher": {}}
    # the detector, exactly as R20 (out-of-fold scores on train, full fit on the fresh problems), then R3, eq, comp
    rng = np.random.default_rng(0)
    for f in rl.FORMATS:
        tr, te = P[(f, "train")], P[(f, "fresh")]
        for x in tr:
            x["keep"] = (x["ya"] > 0) | (np.random.default_rng(x["idx"]).random(len(x["ya"])) < NEG)
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
        ntok = sum(x["ids"].shape[1] for x in tr); ncand = sum(len(x["cand"]) for x in tr)
        allsc = np.sort(np.concatenate([x["dscore"] for x in tr]))[::-1]
        thr = float(allsc[min(len(allsc) - 1, int(0.03 * ntok))])
        p_inc = QR3 * ntok / ncand
        res["n"][f] = {"train": len(tr), "fresh": len(te), "values": sum(x["nval"] for x in te), "d3_threshold": thr, "r3_inclusion": p_inc}
        for x in tr + te:
            n = x["ids"].shape[1]
            p = 1 / (1 + np.exp(-(x["dscore"] + np.log(NEG))))               # calibrated for the 25% negative sampling
            x["forced"] = {"D3": (x["cand"][x["dscore"] > thr] + 1).tolist(),
                           "R3": (x["cand"][np.random.default_rng(1_000_003 + x["idx"]).random(len(x["cand"])) < p_inc] + 1).tolist(),
                           "comp": list(x["vtok"])}
            if f == "orig":
                x["forced"]["eq"] = [mt.start(1) + 1 for mt in re.finditer(rb"= \$?(\d)", x["b"])]
            pt = np.zeros(n); pt[x["cand"] + 1] = p
            x["sig"]["F2"] = x["sig"]["entropy"] + W * pt
        for x in tr + te:
            del x["Hp"]
        log(f"detector {f}: threshold {thr:.3f}, R3 inclusion {p_inc:.4f}")
    for xs in P.values():                                                    # orig_is and the pilot force nothing
        for x in xs:
            x.setdefault("forced", {})

    # boundary dependence per format, refitted on the first NDEP train problems
    for f in rl.FORMATS:
        tab = dep_table(model, P[(f, "train")][:NDEP], dev)
        for x in P[(f, "train")] + P[(f, "fresh")]:
            x["sig"]["dep"] = dep_signal(tab, x["b"], x["ids"].shape[1])
        res["n"][f]["dep_contexts"] = len(tab[0])
        log(f"dependence {f}: {len(tab[0])} two-byte contexts, after '= ' {tab[0].get((61, 32), float('nan')):.3f}, global {tab[2]:.3f}")

    # fits: thresholds, budget per value and the matched rates, D3R's body thresholds
    fits = {}
    for w in WRITINGS:
        fits[w] = blt_layouts.Fit((R,) + HIGH)
        for x in P[(w, "train")]:
            fits[w].add(x["sig"], x["forced"])
        fits[w].done()
    sv = {f: per_value(P[(f, "train")], fits[f]) for f in rl.FORMATS}
    i10 = int(np.flatnonzero(GRID == 0.10)[0])
    r_o = float(np.interp(sv["prose"][i10], sv["orig"], GRID)); r_p = float(np.interp(sv["orig"][i10], sv["prose"], GRID))
    bpv = {f: sum(len(x["b"]) for x in P[(f, "train")] if x["nval"]) / sum(x["nval"] for x in P[(f, "train")]) for f in rl.FORMATS}
    r_o2, r_p2 = R * bpv["prose"] / bpv["orig"], R * bpv["orig"] / bpv["prose"]
    res["budget_per_value"] = {f: {"grid": GRID.tolist(), "per_value": sv[f].tolist(), "at_10": float(sv[f][i10])} for f in rl.FORMATS}
    res["matched"] = {"R_o": r_o, "R_p": r_p, "clamped": bool(not (GRID[0] < r_o < GRID[-1] and GRID[0] < r_p < GRID[-1])),
                      "bytes_per_value": bpv, "R_o_bytes": r_o2, "R_p_bytes": r_p2}
    r_m = {"orig": r_o, "prose": r_p}
    log(f"budget per value at 10%: orig {sv['orig'][i10]:.2f}, prose {sv['prose'][i10]:.2f}; R_o* {r_o:.4f}, R_p* {r_p:.4f}; "
        f"bytes-matched {r_o2:.4f} / {r_p2:.4f}")
    reg = {}
    for f in rl.FORMATS:
        reg[f] = {(wh, r_): fit_region(P[(f, "train")], fits[f], r_, wh) for wh in ("body", "qbody") for r_ in (R,) + HIGH}
        res["region"][f] = {f"{wh}|{r_}": {"threshold_bits": v[0] / LN2, "q": v[1], "region_starts_E": v[2], "region_forced": v[3]}
                            for (wh, r_), v in reg[f].items()}
        log(f"D3R/D3F {f}: " + ", ".join(f"{wh} {r_:.2f}: {v[2]} starts, {v[3]} forced, fill > {v[0] / LN2:.3f} bits"
                                         for (wh, r_), v in reg[f].items()))
    rates_of = {"orig": list(CURVE) + [r_o, r_o2], "prose": list(CURVE) + [r_p, r_p2], "orig_is": list(CURVE)}
    res["rates_patcher"] = rates_of
    for w in WRITINGS:
        res["thresholds_bits"][w] = {str(r_): fits[w]._threshold("entropy", r_)[0] / LN2 for r_ in rates_of[w]}
        res["n"].setdefault(w, {})["train_rate_E10"] = fits[w].rate("entropy", R)

    # patcher-only rows (pilot and fresh): the patcher measures and E's coverage at every rate
    vrows = []
    for sp in ("pilot", "fresh"):
        for w in WRITINGS:
            for x in P[(w, sp)]:
                cov = [set(blt_layouts.layout("entropy", r_, x["sig"], fits[w]).tolist()) for r_ in rates_of[w]]
                for j, ((kind, s, _), cu) in enumerate(zip(x["spans"], x["cues"])):
                    vrows.append([sp, w, x["idx"], j, kind, cu] + [round(float(x["pm"][k][j]), 5) for k in PM]
                                 + ["".join("1" if s in c else "0" for c in cov)])
    # the pilot gate: R9's entropy coverage by cue must reproduce r9_cue_split.py exactly
    want = json.load(open(os.path.join(RES, "blt_prose_cue_split.json")))
    got = defaultdict(lambda: [0, 0])
    for sp, w, p, j, kind, cu, *_, cov in vrows:
        if sp == "pilot" and kind == "res" and w in ("orig", "prose"):
            g = got[f"{w}|{cu}"]; g[0] += 1; g[1] += cov[CURVE.index(0.10)] == "1"
    res["pilot_gate"] = {k: {"n": v[0], "covered": v[1] / v[0], "want": want.get(k)} for k, v in sorted(got.items())}
    ok_gate = all(k in want and want[k]["n"] == v[0] and abs(want[k]["entropy_covered"] - v[1] / v[0]) < 1e-9 for k, v in got.items()) \
        and len(got) == len(want)
    log("pilot gate (R9's cue split): " + ", ".join(f"{k} {v[1] / v[0]:.3f} (n {v[0]})" for k, v in sorted(got.items())) + f" -> {'OK' if ok_gate else 'MISMATCH'}")
    if not ok_gate and not SMOKE:
        json.dump({**res, "vrows": vrows}, open(OUT.replace(".json", "_gate_failed.json"), "w"))
        raise SystemExit("pilot gate failed: R9's coverage by cue does not reproduce; nothing fresh was scored")

    # BLT-1B forward passes on the fresh problems
    r20 = json.load(open(R20))["rows"]
    want20 = {(r_["format"], r_["problem"], r_["layout"], r_["kind"], r_["j"]): r_.get("ok", r_.get("bits")) for r_ in r20
              if r_["layout"] in ("E", "D3", "F2")}
    rows, brows = [], []
    for w in WRITINGS:
        f = "orig" if w == "orig_is" else w
        lays = (lambda x: {"E": blt_layouts.layout("entropy", R, x["sig"], fits[w])}) if w == "orig_is" else \
            layouts_for(f, fits[w], reg[f], r_m[f])
        xs = P[(w, "fresh")]; t1 = time.time()
        for k, x in enumerate(xs):
            ids = x["ids"].to(dev); n = ids.shape[1]; memo = {}
            for lname, st in lays(x).items():
                key = tuple(int(t) for t in st)
                if key not in memo:
                    with torch.no_grad():
                        lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(key, n).to(dev),
                                                     use_cache=False).logits[0].float(), -1)
                    am = lp.argmax(-1)
                    bt = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / LN2).cpu().numpy()
                    memo[key] = ([bool((am[s - 1:e - 1] == ids[0, s:e]).all()) for _, s, e in x["spans"]], bt)
                oks, bt = memo[key]; S = set(key)
                for j, ((kind, s, _), cu) in enumerate(zip(x["spans"], x["cues"])):
                    rows.append((w, x["idx"], lname, kind, j, cu, s in S, oks[j]))
                h0, L_ = x["h0"], x["line"]; fk = FKEY.get(lname.split("@")[0])
                nf = len(S & set(x["forced"].get(fk, ()))) if fk else 0
                brows.append((w, x["idx"], lname, round(float(bt[:h0 - 1].sum()), 5), round(float(bt[h0 - 1:L_].sum()), 5),
                              round(float(bt[L_:].sum()), 5), h0 - 1, L_ - h0 + 1, n - 1 - L_, round(len(key) / n, 6), round(nf / n, 6)))
            if dev == "mps" and k % 20 == 0:
                torch.mps.empty_cache()
            if k == 9 and w != "orig_is":                                    # R20's code path: E, D3, F2 must reproduce
                bad = [(r_[1], r_[2], r_[4]) for r_ in rows if r_[0] == w and r_[2] in ("E", "D3", "F2")
                       and want20.get((w, r_[1], r_[2], r_[3], r_[4])) != r_[7]]
                bb = [abs(b_[4] + b_[5] - want20[(w, b_[1], b_[2], "bits", -1)]) for b_ in brows
                      if b_[0] == w and b_[2] in ("E", "D3", "F2")]
                log(f"R20 check {w}, first 10 problems: {len(bad)} exact-match mismatches, max bits difference {max(bb):.5f}")
                if bad or max(bb) > 1e-3:
                    raise SystemExit(f"E/D3/F2 do not reproduce R20 on {w}: {bad[:5]}")
            if (k + 1) % 50 == 0:
                el = time.time() - t1
                log(f"  {w}: {k + 1}/{len(xs)} problems, {el / (k + 1):.1f} s per problem, ~{el / (k + 1) * (len(xs) - k - 1) / 60:.0f} min left")
        log(f"layouts {w}: {len(xs)} problems")

    res.update(analyse(rows, brows, vrows, rates_of, want20))
    res["rows"] = {"cols": ["w", "problem", "layout", "kind", "j", "cue", "covered", "ok"], "data": rows}
    res["bits_rows"] = {"cols": ["w", "problem", "layout", "bits_question", "bits_body", "bits_final_line", "bytes_question",
                                 "bytes_body", "bytes_final_line", "rate", "forced_share"], "data": brows}
    res["patcher_rows"] = {"cols": ["split", "w", "problem", "j", "kind", "cue"] + list(PM) + ["covered_at_rates"], "data": vrows}
    json.dump(res, open(OUT, "w"))
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))
    log("done")


def analyse(rows, brows, vrows, rates_of, want20):
    U, B = defaultdict(dict), defaultdict(dict)
    for w, p, lay, kind, j, cu, cov, ok in rows:
        U[(w, lay)][(p, j)] = (kind, cu, cov, ok)
    for w, p, lay, *v in brows:
        B[(w, lay)][p] = v
    probs = sorted(set(B[("orig", "E")]) & set(B[("prose", "E")]))
    pi = {p: i for i, p in enumerate(probs)}
    Wm = np.random.default_rng(1).multinomial(len(probs), np.full(len(probs), 1 / len(probs)), size=NBOOT).astype(float)
    ci = lambda bs: [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    def sums(pairs):
        S, N = np.zeros(len(probs)), np.zeros(len(probs))
        for p, v in pairs:
            if p in pi:
                S[pi[p]] += v; N[pi[p]] += 1
        return S, N

    def mean_(S, N):
        return {"value": float(S.sum() / max(N.sum(), 1e-12)), "ci": ci((Wm @ S) / np.maximum(Wm @ N, 1e-12)), "n": int(N.sum())}

    def two(a, b):                                                          # difference of two means on different units
        (S1, N1), (S2, N2) = a, b
        return {"value": float(S1.sum() / N1.sum() - S2.sum() / N2.sum()),
                "ci": ci((Wm @ S1) / np.maximum(Wm @ N1, 1e-12) - (Wm @ S2) / np.maximum(Wm @ N2, 1e-12))}

    def units(w, lay, kind="res", cues=None):
        return {k: v for k, v in U[(w, lay)].items() if v[0] == kind and (cues is None or v[1] in cues)}

    def diff(w, a, b, kind="res", cues=None, fld=3, wb=None):
        A, Bb = units(w, a, kind, cues), units(wb or w, b, kind, cues)
        return mean_(*sums((k[0], float(A[k][fld]) - float(Bb[k][fld])) for k in sorted(set(A) & set(Bb))))

    def share(w, lay, kind="res", cues=None, fld=3):
        return mean_(*sums((k[0], float(v[fld])) for k, v in units(w, lay, kind, cues).items()))

    def bpb_diff(w, a, b, part):
        A, Bb = B[(w, a)], B[(w, b)]; keys = sorted(set(A) & set(Bb))
        idx = {"question": [0], "body": [1], "final_line": [2], "solution": [1, 2], "whole": [0, 1, 2]}[part]
        S, N = np.zeros(len(probs)), np.zeros(len(probs))
        for p in keys:
            if p in pi:
                S[pi[p]] += sum(A[p][i] - Bb[p][i] for i in idx); N[pi[p]] += sum(A[p][i + 3] for i in idx)
        return mean_(S, N)

    out = {"layouts": {}, "diff": {}, "by_cue": {}, "fin_decomposition": {}, "patcher": {}, "verdicts": {}}
    for (w, lay) in sorted(U):
        u, b = list(U[(w, lay)].values()), np.array(list(B[(w, lay)].values()), float)
        m = lambda xs: float(np.mean(xs)) if len(xs) else None
        res_, fin = [v for v in u if v[0] == "res"], [v for v in u if v[0] == "fin"]
        out["layouts"][f"{w}|{lay}"] = o = {
            "exact_res": m([v[3] for v in res_]), "covered_res": m([v[2] for v in res_]),
            "exact_res_given_covered": m([v[3] for v in res_ if v[2]]), "exact_res_given_not": m([v[3] for v in res_ if not v[2]]),
            "exact_fin": m([v[3] for v in fin]), "covered_fin": m([v[2] for v in fin]),
            "exact_fin_given_covered": m([v[3] for v in fin if v[2]]), "exact_fin_given_not": m([v[3] for v in fin if not v[2]]),
            "bpb": {"question": b[:, 0].sum() / b[:, 3].sum(), "body": b[:, 1].sum() / b[:, 4].sum(), "final_line": b[:, 2].sum() / b[:, 5].sum(),
                    "solution": (b[:, 1] + b[:, 2]).sum() / (b[:, 4] + b[:, 5]).sum(), "whole": b[:, :3].sum() / b[:, 3:6].sum()},
            "rate": float(b[:, 6].mean()), "forced_share": float(b[:, 7].mean()), "n_problems": len(b), "n_values": len(res_),
            "by_cue": {c: {"n": len(vs), "covered": m([v[2] for v in vs]), "exact": m([v[3] for v in vs])}
                       for c in sorted({v[1] for v in res_}) for vs in [[v for v in res_ if v[1] == c]]}}
        print(f"LAYOUT {w:7s} {lay:7s} exact {o['exact_res'] or 0:.3f} cov {o['covered_res'] or 0:.3f} fin {o['exact_fin']:.3f} "
              f"(cov {o['covered_fin']:.3f}) bpb sol {o['bpb']['solution']:.4f} whole {o['bpb']['whole']:.4f} q {o['bpb']['question']:.4f} "
              f"rate {o['rate']:.4f} forced {o['forced_share']:.4f} | " + " ".join(f"{c} {v['covered']:.3f}/{v['exact']:.3f}" for c, v in o["by_cue"].items()), flush=True)
    # R20 reproduction on every fresh problem
    bad = [k for (w, lay), d in U.items() if w != "orig_is" and lay in ("E", "D3", "F2")
           for (p, j), v in d.items() if want20.get((w, p, lay, v[0], j)) != v[3]]
    bdiff = max(abs(v[1] + v[2] - want20[(w, p, lay, "bits", -1)]) for (w, lay), d in B.items() if w != "orig_is" and lay in ("E", "D3", "F2") for p, v in d.items())
    out["reproduce_R20"] = {"exact_mismatches": len(bad), "max_bits_difference": float(bdiff)}

    # differences against E (and the comparators), overall, by cue, on final answers and on bits
    for w in ("orig", "prose"):
        lays = sorted({lay for (w_, lay) in U if w_ == w})
        for lay in lays:
            base = "E@15" if lay.endswith("@15") else "E@20" if lay.endswith("@20") else "E"
            if lay == base:
                continue
            for meas, kind in (("exact", "res"), ("fin", "fin")):
                out["diff"][f"{w}|{lay}-{base}|{meas}"] = diff(w, lay, base, kind)
            for part in ("question", "solution", "whole"):
                out["diff"][f"{w}|{lay}-{base}|bpb_{part}"] = bpb_diff(w, lay, base, part)
            for c in sorted({v[1] for v in units(w, "E").values()}):
                out["by_cue"][f"{w}|{lay}-{base}|{c}"] = diff(w, lay, base, "res", (c,))
        comps = [("D3", "with"), ("D3", "dep"), ("D3", "R3"), ("D3R", "D3"), ("D3F", "D3"), ("VU", "D3"), ("F2", "D3")] \
            + ([("D3", "eq")] if w == "orig" else [])
        for a, b in comps:
            for meas, kind in (("exact", "res"), ("fin", "fin")):
                out["diff"][f"{w}|{a}-{b}|{meas}"] = diff(w, a, b, kind)
            for c in sorted({v[1] for v in units(w, "E").values()}):
                out["by_cue"][f"{w}|{a}-{b}|{c}"] = diff(w, a, b, "res", (c,))
        for lay in [l for l in lays if l != "E" and not l.startswith("E@")]:   # final answers by what happened to the start
            base = "E@15" if lay.endswith("@15") else "E@20" if lay.endswith("@20") else "E"
            E_, A = units(w, base, "fin"), units(w, lay, "fin"); keys = sorted(set(E_) & set(A))
            dec = {}
            for g, (ce, ca) in {"kept": (1, 1), "lost": (1, 0), "gained": (0, 1), "neither": (0, 0)}.items():
                sel = {k for k in keys if (E_[k][2], A[k][2]) == (bool(ce), bool(ca))}
                dec[g] = {**mean_(*sums((k[0], float(A[k][3]) - float(E_[k][3]) if k in sel else 0.0) for k in keys)), "problems": len(sel),
                          "exact_base": float(np.mean([E_[k][3] for k in sel])) if sel else None, "exact_layout": float(np.mean([A[k][3] for k in sel])) if sel else None}
            dec["total"] = mean_(*sums((k[0], float(A[k][3]) - float(E_[k][3])) for k in keys))
            dec["not_lost"] = mean_(*sums((k[0], float(A[k][3]) - float(E_[k][3]) if (E_[k][2], A[k][2]) != (True, False) else 0.0) for k in keys))
            out["fin_decomposition"][f"{w}|{lay}-{base}"] = dec

    # the patcher measures (fresh and pilot) by writing, cue and coverage at 10%
    V = [dict(zip(["split", "w", "problem", "j", "kind", "cue"] + list(PM) + ["cov"], r_)) for r_ in vrows]
    i10 = CURVE.index(0.10)
    for sp in ("fresh", "pilot"):
        for w in WRITINGS:
            for kind in ("res", "fin"):
                vs = [v for v in V if v["split"] == sp and v["w"] == w and v["kind"] == kind]
                for c in sorted({v["cue"] for v in vs}):
                    g = [v for v in vs if v["cue"] == c]
                    a = {k: np.array([v[k] for v in g]) for k in PM}
                    covs = np.array([[ch == "1" for ch in v["cov"]] for v in g])
                    out["patcher"][f"{sp}|{w}|{kind}|{c}"] = {
                        "n": len(g), **{f"{k}_mean": float(a[k].mean()) for k in PM},
                        **{f"{k}_median": float(np.median(a[k])) for k in ("H", "Pd", "Hd", "maxp")},
                        "Pd_ge_0.9": float((a["Pd"] >= 0.9).mean()), "Pd_or_dollar_ge_0.9": float((a["Pd"] + a["Pdol"] >= 0.9).mean()),
                        "H_p99": float(np.percentile(a["H"], 99)), "missed_10_Pd_ge_0.9": float((a["Pd"][~covs[:, i10]] >= 0.9).mean()) if (~covs[:, i10]).any() else None,
                        "covered_at_rates": dict(zip([str(r_) for r_ in rates_of[w]], covs.mean(0).tolist()))}
    for k, v in out["patcher"].items():
        if k.startswith("fresh") and "|res|" in k:
            print(f"PATCHER {k:26s} n {v['n']:5d} H {v['H_median']:.2f} Pd {v['Pd_median']:.3f} Hd {v['Hd_median']:.2f} maxp {v['maxp_median']:.3f} "
                  f"Pd>=.9 {v['Pd_ge_0.9']:.3f} cov@10 {v['covered_at_rates']['0.1']:.3f}", flush=True)

    # verdicts
    VR, D_ = out["verdicts"], out["diff"]
    Vf = [v for v in V if v["split"] == "fresh" and v["kind"] == "res"]
    miss = [v for v in Vf if ((v["w"] == "orig" and v["cue"] in ("= ", "$")) or (v["w"] == "prose" and v["cue"] == "$")) and v["cov"][i10] == "0"]
    c1 = mean_(*sums((v["problem"], float(v["Pd"] >= 0.9)) for v in miss))
    VR["C1"] = {**c1, "verdict": "HOLDS" if c1["ci"][0] >= 0.70 else "DOES NOT HOLD"}
    q1 = {w: share(w, "E", "res", ("$",), 2) for w in ("orig", "prose")}
    VR["Q1"] = {**{w: q1[w] for w in q1}, "verdict": "HOLDS" if all(q1[w]["ci"][1] < 0.05 for w in q1) else "DOES NOT HOLD"}
    cov = lambda w, lay, c: sums((k[0], float(v[2])) for k, v in units(w, lay, "res", (c,)).items())
    q2 = two(cov("prose", "E", "space"), cov("prose", "E", "$"))
    VR["Q2"] = {**q2, "verdict": "HOLDS" if q2["ci"][0] > 0.40 else "DOES NOT HOLD"}
    uw = units("prose", "with")
    gain = lambda c: sums((k[0], float(uw[k][3]) - float(v[3])) for k, v in units("prose", "E", "res", (c,)).items())
    q3 = two(gain("$"), gain("space"))
    VR["Q3"] = {**q3, "verdict": "HOLDS" if q3["ci"][0] > 0.15 else "DOES NOT HOLD"}
    po, pp = units("orig", "E"), units("prose", "E"); om, pm_ = units("orig", "E@m"), units("prose", "E@m")
    keys = sorted(set(po) & set(pp))
    g10 = sums((k[0], float(pp[k][2]) - float(po[k][2])) for k in keys)[0]
    gA = sums((k[0], float(pp[k][2]) - float(om[k][2])) for k in keys)[0]
    gB = sums((k[0], float(pm_[k][2]) - float(po[k][2])) for k in keys)[0]
    nv = len(keys)
    den = Wm @ g10; den = np.where(den == 0, np.nan, den)
    shr = lambda g: {"value": float(1 - g.sum() / g10.sum()),
                     "ci": [float(np.nanpercentile(1 - (Wm @ g) / den, 2.5)), float(np.nanpercentile(1 - (Wm @ g) / den, 97.5))]}
    sA, sB = shr(gA), shr(gB)
    read = "budget" if sA["value"] >= 0.67 and sB["value"] >= 0.67 else "cue" if sA["value"] < 0.33 and sB["value"] < 0.33 else "mixed"
    VR["D1"] = {"G_10": float(g10.sum() / nv), "G_A": float(gA.sum() / nv), "G_B": float(gB.sum() / nv), "s_A": sA, "s_B": sB, "verdict": read}
    dec = out["fin_decomposition"]["prose|D3-E"]
    tot, lost = dec["total"]["value"], dec["lost"]["value"]
    VR["F1"] = {"total": dec["total"], "lost": dec["lost"], "not_lost": dec["not_lost"], "lost_share": lost / tot if tot else None,
                "verdict": "HOLDS" if tot < 0 and lost <= 0.5 * tot else "DOES NOT HOLD"}
    for tag, lay in (("F2", "D3F"), ("F2_D3R", "D3R")):                     # F2 reads D3F; D3R is reported (amendment A1)
        f2a, f2b = D_[f"prose|{lay}-E|fin"], diff("prose", lay, "D3", "res")
        VR[tag] = {"layout": lay, f"{lay}-E_fin": f2a, f"{lay}-D3_exact": f2b,
                   "verdict": "HOLDS" if f2a["ci"][0] > -0.02 and f2b["ci"][0] > -0.03 else "DOES NOT HOLD"}
    w1 = D_["prose|D3-E|bpb_whole"]
    VR["W1"] = {**w1, "verdict": "HOLDS" if w1["ci"][1] < 0 else "DOES NOT HOLD"}
    for r_ in HIGH:
        g = D_[f"prose|D3@{round(r_ * 100)}-E@{round(r_ * 100)}|exact"]
        VR[f"G1@{round(r_ * 100)}"] = {**g, "verdict": "HOLDS" if g["ci"][0] > 0 else "DOES NOT HOLD"}
    k1 = D_["prose|D3-R3|exact"]
    VR["K1"] = {**k1, "verdict": "HOLDS" if k1["ci"][0] > 0 else "DOES NOT HOLD"}
    A_, B_ = units("orig_is", "E", "res", ("is ",)), units("orig", "E", "res", ("= ",))
    ks = sorted(set(A_) & set(B_))
    s1 = mean_(*sums((k[0], float(A_[k][2]) - float(B_[k][2])) for k in ks))
    pd_is = {(v["problem"], v["j"]): v["Pd"] for v in Vf if v["w"] == "orig_is" and v["cue"] == "is "}
    med = float(np.median(list(pd_is.values())))
    read_s1 = "unblinds" if s1["value"] >= 0.20 and s1["ci"][0] > 0 else "no change" if s1["value"] < 0.10 else "partial"
    hi = {k for k in ks if pd_is[k] >= 0.9}
    VR["S1"] = {**s1, "median_Pd_after_is": med, "cue_kept": med >= 0.9,
                "covered_after_is_Pd_ge_0.9": float(np.mean([A_[k][2] for k in hi])) if hi else None,
                "covered_after_is_Pd_lt_0.9": float(np.mean([A_[k][2] for k in ks if k not in hi])) if len(hi) < len(ks) else None,
                "verdict": read_s1 + (" (is fixes the type too: not a test of the cue)" if med >= 0.9 else "")}
    v1a, v1b = diff("prose", "VU", "E", "res", ("$",)), D_["prose|VU-E|fin"]
    VR["V1"] = {"VU-E_dollar": v1a, "VU-E_fin": v1b,
                "verdict": "HOLDS" if v1a["value"] >= 0.20 and v1a["ci"][0] > 0 and v1b["ci"][0] > -0.02 else "DOES NOT HOLD"}
    for k, v in VR.items():
        print(f"VERDICT {k:6s} {v['verdict']}", flush=True)
    return out


if __name__ == "__main__":
    main()
