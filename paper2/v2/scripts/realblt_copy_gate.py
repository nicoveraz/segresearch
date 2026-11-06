"""O6 step 1, the gate (followup/O6_COPY_VETO_RULES.md): is BLT-1B's 14% exact copy of the quadrant ticket caused by
its default patch starts at or inside the copy, rather than by the conversion, the template or the window?

Texts: the 293 quadrant problems (realblt_quadrant.build, same seeded order: train first, then test). The copy's digits
are tokens c1..c4 (token t = byte t-1); onset = a start at c1, inside = a start at c2..c4. Layouts per problem:
FULL (patch_lengths=None), D (the default layout given explicitly, read from the patcher's own patch lengths), D-all
(no start at c1..c4), D-onset, D-inside, D+onset (a start at c1, none at c2..c4). Exact = all four digits right under
teacher-forced argmax; bits = the copy's summed -log2 p. Also dumped: D's starts in the copy and in the 8 bytes before
it, the patcher's entropy (bits) deciding each copy byte, and the patcher's own argmax copy.

    ../segresearch/.venv-blt/bin/python realblt_copy_gate.py [N_PROBLEMS=300]     -> results/blt_copy_gate.json
    SEGR_BLT_WINDOW=0 (the conversion as released): FULL, D and D-all only          -> results/blt_copy_gate_w0.json
    O6_SMOKE=1: the first 8 test problems                                            -> results/blt_copy_gate*_smoke.json
"""
import json
import os
import sys
import time

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import torch
from transformers import AutoTokenizer

import blt_light
import blt_load
from realblt_budget import GSM, GSM_TRAIN, MAX_TOKENS, MODEL, lengths_from_starts
from realblt_quadrant import build

SMOKE = os.environ.get("O6_SMOKE") == "1"
WINDOWED = blt_load.WINDOW == 512
LN2 = np.log(2)


def boot(x, seed=6, n=2000):
    """Mean of a per-problem array and its 95% bootstrap interval over problems."""
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(n_problems=300):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_light.load(MODEL, dev).eval()
    cfg = model.config
    rng = np.random.default_rng(0)
    for l in list(open(GSM_TRAIN))[:300]:                       # the quadrant draws the train tickets first
        build(json.loads(l), rng)
    test = []
    for l in list(open(GSM))[:n_problems]:
        b, units = build(json.loads(l), rng)
        ids = tok(b.decode(), return_tensors="pt").input_ids
        if ids.shape[1] <= MAX_TOKENS and ids.shape[1] == len(b) + 1:
            test.append((b, units, ids))
    if SMOKE:
        test = test[:8]
    saved = {r["problem"]: r for r in json.load(open(os.path.join(_seg.HERE, "results", "blt_quadrant.json")))["units"]
             if r["kind"] == "id_copy"}
    names = ["FULL", "D", "D-all", "D-onset", "D-inside", "D+onset"] if WINDOWED else ["FULL", "D", "D-all"]
    thr = cfg.patching_threshold
    print(f"{len(test)} test problems; window {blt_load.WINDOW}; default threshold {thr:.4f} nats", flush=True)

    rows = []
    t0 = time.time()
    for i, (b, units, ids) in enumerate(test):
        ids = ids.to(dev); n = ids.shape[1]
        a = units["id_copy"][0][0]; s = units["id_new"][0][0]
        c = [a + 1, a + 2, a + 3, a + 4]                          # the copy's digit tokens
        with torch.no_grad():
            ent, plen, plog = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=thr,
                                                  max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()                           # nats; e[t-1] decides a start at token t
        D = np.r_[0, np.cumsum(plen[0].cpu().numpy())[:-1]]
        D = set(int(t) for t in D if t < n)                         # a start past the last byte changes nothing
        d_float = {0, 1} | {t for t in range(2, n) if e[t - 1] > thr}
        p_ok = [bool(plog[0, t - 1].float().argmax() == ids[0, t]) for t in c]
        lay = {"D": D, "D-all": D - set(c), "D-onset": D - {c[0]}, "D-inside": D - set(c[1:]),
               "D+onset": (D - set(c[1:])) | {c[0]}}
        done = {}

        def score(st):
            key = None if st is None else tuple(sorted(st))
            if key not in done:
                with torch.no_grad():
                    out = model(input_ids=ids, use_cache=False) if st is None else \
                        model(input_ids=ids, patch_lengths=lengths_from_starts(key, n).to(dev), use_cache=False)
                lp = torch.log_softmax(out.logits[0].float(), -1)
                tgt = ids[0, c[0]:c[-1] + 1]; pred = lp[c[0] - 1:c[-1]]
                ok = (pred.argmax(-1) == tgt).cpu().numpy()
                done[key] = {"bits": float(-pred.gather(1, tgt[:, None]).sum() / LN2), "exact": bool(ok.all()),
                             "digits": [bool(x) for x in ok]}
            return done[key]

        res = {name: score(None if name == "FULL" else lay[name]) for name in names}
        row = {"problem": i, "distance": a - s, "within_window": a - s < 512, "n_tokens": n,
               "rate_D": len(D) / n, "D_vs_float_threshold": len(D ^ d_float),
               "D_starts_copy": [t in D for t in c], "D_starts_context": [t in D for t in range(a - 7, a + 1)],
               "D_start_after": (a + 5) in D,
               "ent_bits_copy": [float(e[t - 1] / LN2) for t in c + [a + 5]],
               "ent_bits_source": [float(e[t - 1] / LN2) for t in range(s + 1, s + 5)],
               "patcher_ok": p_ok, **{name: res[name] for name in names}}
        if WINDOWED and i in saved:
            row["saved_full"] = {"bits": saved[i]["bits_full"], "exact": saved[i]["ok_full"]}
        rows.append(row)
        if (i + 1) % 25 == 0 or SMOKE:
            print(f"  {i + 1} problems, {time.time() - t0:.0f}s", flush=True)

    out = summarize(rows, names)
    tag = ("" if WINDOWED else "_w0") + ("_smoke" if SMOKE else "")
    path = os.path.join(_seg.HERE, "results", f"blt_copy_gate{tag}.json")
    json.dump({**out, "window": blt_load.WINDOW, "default_threshold_nats": thr, "rows": rows}, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "descriptive"}, indent=1))
    print(json.dumps(out["descriptive"], indent=1))
    print("wrote", path)


def summarize(rows, names):
    ex = {k: np.array([r[k]["exact"] for r in rows], float) for k in names}
    bits = {k: np.array([r[k]["bits"] for r in rows]) for k in names}
    win = np.array([r["within_window"] for r in rows])
    onset = np.array([r["D_starts_copy"][0] for r in rows]); inside = np.array([any(r["D_starts_copy"][1:]) for r in rows])
    out = {"n": len(rows), "exact": {k: float(v.mean()) for k, v in ex.items()},
           "bits": {k: float(v.mean()) for k, v in bits.items()}}
    # G0: D reproduces the default path, and (windowed) FULL reproduces the saved quadrant rows
    same = [r["FULL"]["exact"] == r["D"]["exact"] and abs(r["FULL"]["bits"] - r["D"]["bits"]) < 0.05 for r in rows]
    g0 = {"FULL_equals_D": int(sum(same)), "of": len(rows),
          "D_vs_float_threshold_mismatched_starts": int(sum(r["D_vs_float_threshold"] for r in rows))}
    if "saved_full" in rows[0]:
        g0["saved_exact"] = float(np.mean([r["saved_full"]["exact"] for r in rows]))
        g0["FULL_bits_within_0.05_of_saved"] = float(np.mean([abs(r["FULL"]["bits"] - r["saved_full"]["bits"]) < 0.05
                                                              for r in rows]))
        g0["met"] = bool(all(same) and abs(out["exact"]["FULL"] - g0["saved_exact"]) <= 0.01
                         and g0["FULL_bits_within_0.05_of_saved"] >= 0.95)
    out["G0"] = g0
    d, ci = boot(ex["D-all"] - ex["D"])
    out["G1" if "saved_full" in rows[0] else "conversion_check"] = {
        "exact_D": float(ex["D"].mean()), "exact_D-all": float(ex["D-all"].mean()), "diff": d, "ci": ci,
        "bits_saved": boot(bits["D"] - bits["D-all"])}
    if "saved_full" in rows[0]:
        out["G1"]["holds"] = bool(d >= 0.20 and ci[0] >= 0.10)
        dw, ciw = boot((ex["D-all"] - ex["D"])[win])
        out["G2"] = {"n": int(win.sum()), "exact_D": float(ex["D"][win].mean()), "exact_D-all": float(ex["D-all"][win].mean()),
                     "diff": dw, "ci": ciw, "holds": bool(dw >= 0.15 and ciw[0] > 0)}
        out["gate_passes"] = bool(g0["met"] and out["G1"]["holds"] and out["G2"]["holds"])
    else:
        out["conversion_check"]["hazard_without_window_fix"] = bool(d >= 0.10)
    desc = {"share_onset_start": float(onset.mean()), "share_inside_start": float(inside.mean()),
            "share_onset_or_inside": float((onset | inside).mean()),
            "share_start_in_context_bytes": [float(np.mean([r["D_starts_context"][j] for r in rows])) for j in range(8)],
            "share_start_after_copy": float(np.mean([r["D_start_after"] for r in rows])),
            "rate_D": float(np.mean([r["rate_D"] for r in rows])),
            "cells": {}, "first_failing_digit": {},
            "patcher_entropy_bits_copy_digits_and_after": np.mean([r["ent_bits_copy"] for r in rows], 0).round(3).tolist(),
            "patcher_entropy_bits_source_digits": np.mean([r["ent_bits_source"] for r in rows], 0).round(3).tolist(),
            "patcher_own_copy_exact": float(np.mean([all(r["patcher_ok"]) for r in rows])),
            "patcher_own_digit_ok": np.mean([r["patcher_ok"] for r in rows], 0).round(3).tolist(),
            "distance_bytes": {"min": int(min(r["distance"] for r in rows)), "median": float(np.median([r["distance"] for r in rows])),
                               "max": int(max(r["distance"] for r in rows)), "within_window": int(win.sum())}}
    for o in (False, True):
        for ins in (False, True):
            m = (onset == o) & (inside == ins)
            if m.any():
                desc["cells"][f"onset={int(o)} inside={int(ins)}"] = {
                    "n": int(m.sum()), **{f"exact_{k}": float(ex[k][m].mean()) for k in names if k != "FULL"}}
    for k in names:
        f = [next((j for j, ok in enumerate(r[k]["digits"]) if not ok), None) for r in rows]
        desc["first_failing_digit"][k] = {str(j): int(sum(x == j for x in f)) for j in (0, 1, 2, 3)}
    if "D+onset" in names:
        desc["onset_alone_D+onset_minus_D-all"] = {"exact": boot(ex["D+onset"] - ex["D-all"]),
                                                   "bits": boot(bits["D-all"] - bits["D+onset"])}
    if (~win).any():
        desc["beyond_window"] = {"n": int((~win).sum()), **{f"exact_{k}": float(ex[k][~win].mean()) for k in names}}
    out["descriptive"] = desc
    return out


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))
