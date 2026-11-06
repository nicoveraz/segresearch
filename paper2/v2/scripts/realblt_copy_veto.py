"""O6 step 2 (followup/O6_COPY_VETO_RULES.md): a causal suffix-match veto against patch starts inside copies, at a
matched rate, on the quadrant texts; and a no-harm check on computed values in plain GSM8K equation text.

Veto V_k: token t (byte t-1) may not start a patch if the k bytes ending at byte t-2 (the last bytes known when the
start at t is decided from the entropy at t-1) also occur earlier, ending at byte t-3 or before (overlap allowed).

Layouts on the 293 quadrant problems (thresholds fitted on the 295 quadrant train problems, blt_layouts train mode):
    D           BLT-1B's default layout (the patcher's own patch lengths, ~25% of bytes)
    D.V_k       the veto, with the entropy threshold lowered so that the pooled train rate equals D's (matched)
    D-V_k       the veto without refill (descriptive)
    E10, E10.V_k, E10-V_k   the same at entropy@10% (the quadrant's layout)
Readouts: exact and bits per unit kind (realblt_quadrant's kinds), bits per byte over the whole text and without the
copied ticket's digits, patch rate, and where the vetoed starts fall. No-harm: on 300 GSM8K test equation texts
(realblt_budget's format, no ticket) the starts that V_k removes from D and E10 at computed-value starts.

    ../segresearch/.venv-blt/bin/python realblt_copy_veto.py     -> results/blt_copy_veto.json
    O6_SMOKE=1: 30 train / 6 test problems of each set            -> results/blt_copy_veto_smoke.json
"""
import json
import os
import re
import time

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_light
import blt_load
import realblt_budget
from realblt_budget import GSM, GSM_TRAIN, MAX_TOKENS, MODEL, lengths_from_starts
from realblt_quadrant import KINDS, build

SMOKE = os.environ.get("O6_SMOKE") == "1"
KS = (6, 8)
LN2 = np.log(2)
assert blt_load.WINDOW == 512 and blt_layouts.MODE == "train"


def vetoed(b, n, k):
    """Boolean array over tokens: V_k forbids a start at token t."""
    v = np.zeros(n, bool)
    for t in range(k + 1, n):
        v[t] = b.find(b[t - 1 - k:t - 1], 0, t - 2) != -1
    return v


def ends_of(b, w, end):
    """End bytes of every occurrence of w inside b[:end], overlapping ones included."""
    out, i = [], b.find(w, 0, end)
    while i != -1:
        out.append(i + len(w) - 1)
        i = b.find(w, i + 1, end)
    return out


def boot_ratio(num_a, num_b, den, seed=6, n=2000):
    """(sum num_a - sum num_b) / sum den over problems, with a 95% bootstrap interval over problems."""
    num_a, num_b, den = (np.asarray(x, float) for x in (num_a, num_b, den))
    r = np.random.default_rng(seed)
    st = []
    for _ in range(n):
        i = r.integers(0, len(den), len(den))
        st.append((num_a[i].sum() - num_b[i].sum()) / den[i].sum())
    return float((num_a.sum() - num_b.sum()) / den.sum()), [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_light.load(MODEL, dev).eval()
    cfg = model.config
    n_tr, n_te = (30, 6) if SMOKE else (300, 300)

    def patcher(ids):
        with torch.no_grad():
            ent, plen, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                               max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy(); n = ids.shape[1]
        s = np.full(n, -np.inf); s[2:] = e[1:n - 1]                # token t starts a patch from the entropy at t-1
        D = np.r_[0, np.cumsum(plen[0].cpu().numpy())[:-1]]
        return s, np.array(sorted(int(t) for t in set(D.tolist()) if t < n))

    def signals(b, s):
        sig = {"entropy": s}
        veto = {k: vetoed(b, len(s), k) for k in KS}
        for k in KS:
            sig[f"entv{k}"] = np.where(veto[k], -np.inf, s)
        return sig, veto

    # ---- quadrant texts (same seeded order as realblt_quadrant: train tickets first)
    rng = np.random.default_rng(0)
    quad = {}
    for split, path, cap in (("train", GSM_TRAIN, n_tr), ("test", GSM, n_te)):
        quad[split] = []
        for l in list(open(path))[:300]:
            b, units = build(json.loads(l), rng)
            ids = tok(b.decode(), return_tensors="pt").input_ids
            if ids.shape[1] <= MAX_TOKENS and ids.shape[1] == len(b) + 1:
                quad[split].append((b, units, ids.to(dev)))
        quad[split] = quad[split][:cap]
    fit = blt_layouts.Fit((0.10,))
    d_starts = d_tokens = 0
    for b, units, ids in quad["train"]:
        s, D = patcher(ids)
        fit.add(signals(b, s)[0])
        d_starts += len(D); d_tokens += ids.shape[1]
    fit.done()
    r_d = d_starts / d_tokens
    print(f"quadrant: {len(quad['train'])} train / {len(quad['test'])} test; default train rate {r_d:.4f}; train rates "
          + ", ".join(f"{kd}@{R:.3f} {fit.rate(kd, R):.4f}" for kd in ["entropy"] + [f"entv{k}" for k in KS]
                      for R in (0.10, r_d)), flush=True)

    saved = {r["problem"]: r for r in json.load(open(os.path.join(_seg.HERE, "results", "blt_quadrant.json")))["units"]
             if r["kind"] == "id_copy"}
    names = ["D", "E10"] + [f"{base}{op}V{k}" for base in ("D", "E10") for op in (".", "-") for k in KS]
    rows = []
    t0 = time.time()
    for i, (b, units, ids) in enumerate(quad["test"]):
        n = ids.shape[1]
        s, D = patcher(ids)
        sig, veto = signals(b, s)
        E = blt_layouts.layout("entropy", 0.10, sig, fit)
        lay = {"D": D, "E10": E}
        for k in KS:
            lay[f"D.V{k}"] = blt_layouts.layout(f"entv{k}", r_d, sig, fit)
            lay[f"E10.V{k}"] = blt_layouts.layout(f"entv{k}", 0.10, sig, fit)
            lay[f"D-V{k}"] = D[~veto[k][D]]
            lay[f"E10-V{k}"] = E[~veto[k][E]]
        where = {}                                                  # token -> "kind start" / "kind inside"
        for kd, v in units.items():
            for a, e in v:
                for t in range(a + 1, e + 1):
                    where[t] = f"{kd} {'start' if t == a + 1 else 'inside'}"
        ac = units["id_copy"][0][0]
        done = {}

        def score(st):
            key = tuple(int(x) for x in st)
            if key not in done:
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(key, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                tgt = ids[0, 1:]
                bits = (-lp[:-1].gather(1, tgt[:, None])[:, 0] / LN2).cpu().numpy()     # index t-1 = token t = byte t-1
                ok = (lp[:-1].argmax(-1) == tgt).cpu().numpy()
                per = {kd: [len(v), int(sum(ok[a:e].all() for a, e in v)), float(sum(bits[a:e].sum() for a, e in v))]
                       for kd, v in units.items()}
                done[key] = {"rate": len(key) / n, "bits": float(bits.sum()), "bits_copy": float(bits[ac:ac + 4].sum()),
                             "n_bytes": n - 1, "kinds": per}
            return done[key]

        row = {"problem": i, "layouts": {name: score(lay[name]) for name in names}, "vetoed_starts": {},
               "n_starts": {base: len(lay[base]) for base in ("D", "E10")},
               "share_tokens_vetoed": {k: float(veto[k][2:].mean()) for k in KS}}
        for base in ("D", "E10"):
            for k in KS:
                rm = [int(t) for t in lay[base] if veto[k][t]]
                row["vetoed_starts"][f"{base}-V{k}"] = [where.get(t, "other") for t in rm]
        if i in saved:
            row["saved_entropy"] = {"bits": saved[i]["bits_entropy"], "exact": saved[i]["ok_entropy"]}
        rows.append(row)
        if (i + 1) % 25 == 0 or SMOKE:
            print(f"  {i + 1} problems, {time.time() - t0:.0f}s", flush=True)

    # ---- no-harm: 300 GSM8K equation texts, no ticket (realblt_budget.problem's format)
    def eq_texts(path, cap):
        out = []
        for l in list(open(path))[:cap]:
            pr = realblt_budget.problem(json.loads(l), tok, dev)
            if pr is not None:
                out.append(pr)
        return out
    eq_fit = blt_layouts.Fit((0.10,))
    for b, ids, _, _ in eq_texts(GSM_TRAIN, n_tr):
        eq_fit.add(signals(b, patcher(ids)[0])[0])
    eq_fit.done()
    noharm = {f"{base}-V{k}": {"computed_value_starts_in_layout": 0, "removed": 0, "removed_same_value": 0, "cases": []}
              for base in ("D", "E10") for k in KS}
    n_eq = n_values = 0
    for j, (b, ids, _, _) in enumerate(eq_texts(GSM, n_te)):
        n_eq += 1
        s, D = patcher(ids)
        sig, veto = signals(b, s)
        lay = {"D": set(D.tolist()), "E10": set(blt_layouts.layout("entropy", 0.10, sig, eq_fit).tolist())}
        for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b):
            n_values += 1
            value = m.group(1)
            for t in sorted({m.start() + 3, m.start(1) + 1}):          # the byte after '= ' and the first digit
                for base, st in lay.items():
                    for k in KS:
                        c = noharm[f"{base}-V{k}"]
                        if t not in st:
                            continue
                        c["computed_value_starts_in_layout"] += 1
                        if not veto[k][t]:
                            continue
                        # a true copy: an earlier occurrence of the k bytes before t goes on with the same value
                        same = any(b[e + 1:].startswith(b[t - 1:m.end(1)]) for e in ends_of(b, b[t - 1 - k:t - 1], t - 2))
                        c["removed"] += 1; c["removed_same_value"] += int(same)
                        c["cases"].append({"text": j, "token": t, "context": b[max(0, t - 41):t - 1].decode(errors="replace"),
                                           "value": value.decode(), "same_value_earlier": bool(same)})
    out = {"n_test": len(rows), "train_rate_default": r_d,
           "thresholds_nats": {f"{kd}@{R:.4f}": fit._threshold(kd, R) for kd in ["entropy"] + [f"entv{k}" for k in KS]
                               for R in (0.10, r_d)}}
    out.update(summarize(rows, names))
    out["V4_noharm"] = {"n_texts": n_eq, "n_values": n_values, **noharm}
    out["verdicts"] = verdicts(out)
    path = os.path.join(_seg.HERE, "results", f"blt_copy_veto{'_smoke' if SMOKE else ''}.json")
    json.dump({**out, "rows": rows}, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("V4_noharm",)}, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "cases"})
                      for k, v in out["V4_noharm"].items()}, indent=1))
    print("wrote", path)


def summarize(rows, names):
    L = {name: [r["layouts"][name] for r in rows] for name in names}
    res = {"by_layout": {}, "pairs": {}}
    for name in names:
        x = L[name]
        d = {"rate": float(np.mean([y["rate"] for y in x])),
             "bpb": sum(y["bits"] for y in x) / sum(y["n_bytes"] for y in x),
             "bpb_without_copy": sum(y["bits"] - y["bits_copy"] for y in x) / sum(y["n_bytes"] - 4 for y in x)}
        for kd in KINDS:
            nu = sum(y["kinds"][kd][0] for y in x)
            if nu:
                d[kd] = {"n": nu, "exact": sum(y["kinds"][kd][1] for y in x) / nu,
                         "bits_per_unit": sum(y["kinds"][kd][2] for y in x) / nu}
        res["by_layout"][name] = d
    for base in ("D", "E10"):
        for op in (".", "-"):
            for k in KS:
                a = f"{base}{op}V{k}"
                p = {}
                for kd in KINDS:
                    nu = [y["kinds"][kd][0] for y in L[base]]
                    if sum(nu):
                        p[kd] = {"exact": boot_ratio([y["kinds"][kd][1] for y in L[a]], [y["kinds"][kd][1] for y in L[base]], nu),
                                 "bits_per_unit": boot_ratio([y["kinds"][kd][2] for y in L[a]], [y["kinds"][kd][2] for y in L[base]], nu)}
                p["bpb"] = boot_ratio([y["bits"] for y in L[a]], [y["bits"] for y in L[base]], [y["n_bytes"] for y in L[base]])
                p["bpb_without_copy"] = boot_ratio([y["bits"] - y["bits_copy"] for y in L[a]],
                                                   [y["bits"] - y["bits_copy"] for y in L[base]],
                                                   [y["n_bytes"] - 4 for y in L[base]])
                p["rate"] = float(np.mean([y["rate"] for y in L[a]]) - np.mean([y["rate"] for y in L[base]]))
                res["pairs"][f"{a} vs {base}"] = p
    vet = {}
    for key in rows[0]["vetoed_starts"]:
        allw = [w for r in rows for w in r["vetoed_starts"][key]]
        vet[key] = {"per_text": len(allw) / len(rows),
                    "share_of_starts": len(allw) / sum(r["n_starts"][key.split("-")[0]] for r in rows), "where": dict(sorted(
            {w: allw.count(w) for w in set(allw)}.items(), key=lambda kv: -kv[1]))}
    res["vetoed_starts"] = vet
    res["share_tokens_vetoed"] = {k: float(np.mean([r["share_tokens_vetoed"][k] for r in rows])) for k in KS}
    have = [r for r in rows if "saved_entropy" in r]
    if have:                                                         # E10 should reproduce the quadrant's layout
        res["check_E10_vs_saved"] = {
            "exact_now": float(np.mean([r["layouts"]["E10"]["kinds"]["id_copy"][1] for r in have])),
            "exact_saved": float(np.mean([r["saved_entropy"]["exact"] for r in have])),
            "bits_within_0.05": float(np.mean([abs(r["layouts"]["E10"]["kinds"]["id_copy"][2] - r["saved_entropy"]["bits"]) < 0.05
                                               for r in have]))}
    return res


def verdicts(out):
    v = {}
    for k in KS:
        pd, pe = out["pairs"][f"D.V{k} vs D"], out["pairs"][f"E10.V{k} vs E10"]
        v1 = pd["id_copy"]["exact"]; v1e = pe["id_copy"]["exact"]
        v2 = (pd["bpb_without_copy"][0], pe["bpb_without_copy"][0])
        v3 = (pd["computed"]["exact"][0], pe["computed"]["exact"][0])
        v4 = (out["V4_noharm"][f"D-V{k}"]["removed"], out["V4_noharm"][f"E10-V{k}"]["removed"])
        v[f"k={k}"] = {"V1": {"value": v1, "holds": bool(v1[0] >= 0.15 and v1[1][0] > 0)},
                       "V1e": {"value": v1e, "holds": bool(v1e[1][0] > 0)},
                       "V2": {"value": v2, "holds": bool(max(v2) <= 0.01)},
                       "V3": {"value": v3, "holds": bool(min(v3) >= -0.02)},
                       "V4": {"removed": v4, "holds": bool(sum(v4) == 0)}}
        v[f"k={k}"]["fix_claim"] = all(x["holds"] for x in v[f"k={k}"].values())
    return v


if __name__ == "__main__":
    main()
