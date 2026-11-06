"""O6b (followup/O6b_COPY_LATCH_RULES.md): L8, a gated copy latch against patch starts inside verbatim copies, on
O6's dev texts. Exploratory; nothing here is confirmatory, and nothing goes into paper2v2.

Token t is byte t-1. h(j) and a(j) are the patcher's entropy (bits) and argmax at output token j, which has read bytes
0..j-1. L8 forbids a patch start at an onset token t when all of these hold:
    1. byte t-2 is ASCII and not a letter or digit;
    2. bytes t-9..t-2 also occur ending at byte t-3 or earlier (V_8's test); e = the end of the latest occurrence;
    3. T, the run of letters and digits from byte e+1 (a single . , - / : may stand between two), ends at byte t-10
       or earlier;
    4. h summed over T's bytes is at least 10 bits;
    5. a(t-1) is T's first byte;
    6. h(t-1) >= log2 10, or h(e+1) - h(t-1) >= 1 bit;
and it latches token t+m (m = 1..|T|-1) while bytes t-1..t+m-2 equal T[:m]. L8c: condition 6 is h(t-1) >= log2 10.

    ../segresearch/.venv-blt/bin/python realblt_copy_latch.py step1    patcher outputs on every dev text, CPU checks
                                    and counts -> results/blt_copy_latch_step1.json (outputs: blt_copy_latch_patcher.npz)
    ../segresearch/.venv-blt/bin/python realblt_copy_latch.py analyze  step 1's CPU part again from the stored outputs
    ../segresearch/.venv-blt/bin/python realblt_copy_latch.py step2    BLT-1B on the 293 quadrant test texts
                                    -> results/blt_copy_latch_step2.json
    ../segresearch/.venv-blt/bin/python realblt_copy_latch.py step3    E10.V8 with a start at the copy's `k` (reported)
                                    -> results/blt_copy_latch_step3.json
    O6_SMOKE=1: a few texts of each set -> *_smoke.*
"""
import collections
import importlib.util
import json
import os
import re
import subprocess
import sys
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
from realblt_copy_veto import boot_ratio, ends_of, vetoed
from realblt_quadrant import KINDS, build

SMOKE = os.environ.get("O6_SMOKE") == "1"
TAG = "_smoke" if SMOKE else ""
RES = os.path.join(_seg.HERE, "results")
STATUS = "exploratory (dev texts); rules followup/O6b_COPY_LATCH_RULES.md at 267de39"
LN2 = np.log(2)
L10 = float(np.log2(10))
S0 = 10.0                                       # condition 4: bits summed over the source run
OFFSET = 4                                      # the BLT token id of byte x is x + 4
THR_D = 1.335442066192627                       # BLT-1B's default threshold, nats (results/blt_copy_gate.json)
RUN = re.compile(rb"[A-Za-z0-9]+(?:[.,\-/:][A-Za-z0-9]+)*")
ALNUM = frozenset(range(48, 58)) | frozenset(range(65, 91)) | frozenset(range(97, 123))
VALUE = re.compile(rb"= \$?(\d[\d,]*(?:\.\d+)?)")
NUMR = re.compile(rb"\d[\d,]*(?:\.\d+)?")
DOMAINS = ("math", "algebra", "code", "prose", "clinical", "traces")
assert blt_load.WINDOW == 512 and blt_layouts.MODE == "train"

_spec = importlib.util.spec_from_file_location(              # the bytes designer's SIV, imported unchanged
    "final_rules", os.path.join(_seg.HERE, "analysis", "2026-10-30_copyfix", "bytes", "final_rules.py"))
final_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(final_rules)


# ---------------------------------------------------------------- the rule (CPU; bytes and patcher outputs only)
def onset(b, hb, arg, t, agree="real"):
    """(code, T, strict, e) for token t: code 0 if t is an L8 onset, else the first of conditions 1-6 that fails; T is
    the source run once condition 3 holds; strict says whether L8c's condition 6 holds too. Reads bytes <= t-2 and the
    outputs at tokens <= t-1 only. agree="up" passes condition 5 always and agree="oracle" compares the actual byte
    t-1 (a peek): both are descriptive bounds, never the rule."""
    c = b[t - 2]
    if c >= 128 or c in ALNUM:                                         # 1. boundary
        return 1, None, False, None
    i = b.rfind(b[t - 9:t - 1], 0, t - 2)                              # 2. recurrence: V_8's test, latest occurrence
    if i == -1:
        return 2, None, False, None
    e = i + 7
    m = RUN.match(b, e + 1, t - 1)                                     # 3. source run, scanned up to byte t-2 only,
    if not m or m.end() > t - 9:                                       #    ending at byte t-10 or earlier
        return 3, None, False, e
    T = m.group(0)
    if hb[e + 1:m.end()].sum() < S0:                                   # 4. unpredictable when first read
        return 4, T, False, e
    ok = arg[t - 1] == T[0] + OFFSET if agree == "real" else (agree == "up" or b[t - 1] == T[0])
    if not ok:                                                         # 5. agreement
        return 5, T, False, e
    strict = bool(hb[t - 1] >= L10)                                    # 6. induction evidence
    if not (strict or hb[e + 1] - hb[t - 1] >= 1.0):
        return 6, T, False, e
    return 0, T, strict, e


def latch(b, hb, arg, agree="real", upto=None):
    """L8 and L8c masks over tokens (True = no patch start), decided for tokens < upto."""
    n = len(b) + 1
    upto = n if upto is None else upto
    v, vc = np.zeros(n, bool), np.zeros(n, bool)
    for t in range(9, upto):
        code, T, strict, _ = onset(b, hb, arg, t, agree)
        if code:
            continue
        for w in ((v, vc) if strict else (v,)):
            w[t] = True
            for m in range(1, len(T)):                                 # the latch: bytes t-1..t+m-2 still equal T[:m]
                if t + m >= upto or b[t + m - 2] != T[m - 1]:
                    break
                w[t + m] = True
    return v, vc


def causal_changes(b, hb, arg, seed):
    """For every cut p: replace bytes p.. and the outputs at tokens p+1.. (draws from the same text, then the reversed
    suffix) and count the L8 and L8c decisions at tokens 0..p+1 that change. Returns (changed, decisions checked)."""
    rng = np.random.default_rng(seed)
    n = len(b) + 1
    v0, vc0 = latch(b, hb, arg)
    bb = np.frombuffer(b, np.uint8)
    bad = checked = 0
    for p in range(1, n - 1):
        for trial in (0, 1):
            if trial == 0:
                tail = bytes(rng.choice(bb, len(b) - p).tolist())
                h2 = np.r_[hb[:p + 1], rng.choice(hb, n - p - 1)]
                a2 = np.r_[arg[:p + 1], rng.choice(arg, n - p - 1)]
            else:
                tail = b[p:][::-1]
                h2 = np.r_[hb[:p + 1], hb[p + 1:][::-1]]
                a2 = np.r_[arg[:p + 1], arg[p + 1:][::-1]]
            v, vc = latch(b[:p] + tail, h2, a2, upto=p + 2)
            bad += int((v[:p + 2] != v0[:p + 2]).sum() + (vc[:p + 2] != vc0[:p + 2]).sum())
            checked += 2 * (p + 2)
    return bad, checked


def sig_of(r, masks=()):
    """Signals over tokens: the entropy deciding token t (from output t-1), and the entropy with each mask at -inf."""
    n = len(r["b"]) + 1
    s = np.full(n, -np.inf)
    s[2:] = r["ent"][1:n - 1]
    sig = {"entropy": s}
    for name in masks:
        sig["ent" + name.lower()] = np.where(r[name], -np.inf, s)
    return sig


# ---------------------------------------------------------------- texts, the patcher pass and its stored outputs
def alone():
    """Exit if another BLT-1B job runs (a 16 GB Mac cannot hold two); this process and its parents do not count."""
    mine, pid = set(), os.getpid()
    while pid > 1:
        mine.add(pid)
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        pid = int(out) if out else 1
    others = [l for l in subprocess.run(["pgrep", "-fl", "realblt"], capture_output=True, text=True).stdout.splitlines()
              if l.strip() and int(l.split()[0]) not in mine]
    if others:
        sys.exit("another BLT-1B job is running: " + " | ".join(others))


def build_texts(tok):
    """O6's texts in O6's order (realblt_copy_veto.py), plus the xdomain documents of results/xdomain_gains_*.npz."""
    keep = lambda b: len(b) + 1 <= MAX_TOKENS and tok(b.decode(), return_tensors="pt").input_ids.shape[1] == len(b) + 1
    rng = np.random.default_rng(0)
    C = {}
    for split, path in (("quad_train", GSM_TRAIN), ("quad_test", GSM)):        # train tickets are drawn first
        C[split] = []
        for l in list(open(path))[:300]:
            b, units = build(json.loads(l), rng)
            if keep(b):
                C[split].append({"b": b, "units": units})
    for split, path in (("eq_train", GSM_TRAIN), ("eq_test", GSM)):
        C[split] = []
        for l in list(open(path))[:300]:
            pr = realblt_budget.problem(json.loads(l), tok, "cpu")
            if pr is not None:
                C[split].append({"b": pr[0]})
    C["xd"] = []
    for dom in DOMAINS:
        z = np.load(os.path.join(RES, f"xdomain_gains_{dom}.npz"), allow_pickle=True)
        for key in sorted(k.rsplit("_bytes", 1)[0] for k in z.files if k.endswith("_bytes")):
            b = bytes(z[f"{key}_bytes"])
            if keep(b):
                C["xd"].append({"b": b, "dom": dom, "key": key})
    if SMOKE:
        for g, cap in (("quad_train", 30), ("quad_test", 6), ("eq_train", 30), ("eq_test", 6)):
            C[g] = C[g][:cap]
        C["xd"] = [r for r in C["xd"] if r["key"] in ("fit0", "test0")]
    return C


def patcher_pass(C, tok, model, dev):
    cfg = model.config
    t0 = time.time()
    for g, rows in C.items():
        for r in rows:
            ids = tok(r["b"].decode(), return_tensors="pt").input_ids.to(dev)
            n = ids.shape[1]
            assert n == len(r["b"]) + 1
            assert (ids[0, 1:].cpu().numpy() == np.frombuffer(r["b"], np.uint8).astype(int) + OFFSET).all()
            with torch.no_grad():
                ent, plen, plog = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
            lp = torch.log_softmax(plog[0].float(), -1)
            r["ent"] = ent[0, :n].float().cpu().numpy()                      # nats; bf16 values
            r["arg"] = plog[0].float().argmax(-1).cpu().numpy().astype(np.int16)   # as realblt_copy_gate.py:79
            r["sur"] = (-lp[:-1].gather(1, ids[0, 1:, None])[:, 0] / LN2).cpu().numpy()   # bits of byte j at output j
            D = np.r_[0, np.cumsum(plen[0].cpu().numpy())[:-1]]
            r["D"] = np.array(sorted(int(t) for t in set(D.tolist()) if t < n), int)
        print(f"  patcher: {g} {len(rows)} texts, {time.time() - t0:.0f}s", flush=True)


def save_cache(C, path):
    meta, cols = [], collections.defaultdict(list)
    for g, rows in C.items():
        for r in rows:
            meta.append({"group": g, "n": len(r["b"]) + 1, "nD": len(r["D"]),
                         **{k: r[k] for k in ("units", "dom", "key") if k in r}})
            cols["b"].append(np.frombuffer(r["b"], np.uint8)); cols["D"].append(r["D"])
            for k in ("ent", "arg", "sur"):
                cols[k].append(r[k])
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **{k: np.concatenate(v) for k, v in cols.items()})


def load_cache(path, groups=None):
    z = np.load(path)
    x = {k: z[k] for k in ("b", "ent", "arg", "sur", "D")}             # read each array once
    C = collections.defaultdict(list)
    ob = oe = osr = od = 0
    for m in json.loads(str(z["meta"])):
        n, nd = m["n"], m["nD"]
        if groups is None or m["group"] in groups:
            r = {k: m[k] for k in ("units", "dom", "key") if k in m}
            r.update(b=x["b"][ob:ob + n - 1].tobytes(), ent=x["ent"][oe:oe + n], arg=x["arg"][oe:oe + n],
                     sur=x["sur"][osr:osr + n - 1], D=x["D"][od:od + nd])
            C[m["group"]].append(r)
        ob += n - 1; oe += n; osr += n - 1; od += nd
    return dict(C)


def add_masks(C, siv_groups=("quad_train", "quad_test")):
    for g, rows in C.items():
        for r in rows:
            n = len(r["b"]) + 1
            r["hb"] = r["ent"].astype(np.float64) / LN2
            r["L8"], r["L8c"] = latch(r["b"], r["hb"], r["arg"])
            r["V8"] = vetoed(r["b"], n, 8)
            assert not (r["L8"] & ~r["V8"]).any(), f"L8 is not within V_8 ({g})"
            assert not (r["L8c"] & ~r["L8"]).any(), f"L8c is not within L8 ({g})"
            if g in siv_groups:
                r["SIV"] = final_rules.siv(r["b"], n, kmin=5)


def quad_fit(C, masks):
    fit = blt_layouts.Fit((0.10,))
    d_starts = d_tokens = 0
    for r in C["quad_train"]:
        fit.add(sig_of(r, masks))
        d_starts += len(r["D"]); d_tokens += len(r["b"]) + 1
    fit.done()
    return fit, d_starts / d_tokens


def where_map(units):
    w = {}
    for kd, v in units.items():
        for a, e in v:
            for t in range(a + 1, e + 1):
                w[t] = f"{kd} {'start' if t == a + 1 else 'inside'}"
    return w


# ---------------------------------------------------------------- step 1: CPU checks and counts
def analyze(C):
    t0 = time.time()
    out = {"status": STATUS, "n_texts": {g: len(v) for g, v in C.items()}}
    add_masks(C)
    out["subset_check"] = "passed: L8 within V_8 and L8c within L8 on every text"
    changed, checked = {}, {}
    for g, rows in C.items():
        x = [causal_changes(r["b"], r["hb"], r["arg"], seed=i) for i, r in enumerate(rows)]
        changed[g], checked[g] = sum(c for c, _ in x), sum(k for _, k in x)
        print(f"  causality: {g} {changed[g]} changed of {checked[g]} decisions, {time.time() - t0:.0f}s", flush=True)
    out["causality_check"] = {"changed_decisions": changed, "decisions_checked": checked,
                              "passed": all(v == 0 for v in changed.values())}

    # ---- thresholds on the quadrant train texts
    fit, r_d = quad_fit(C, ("L8", "L8c", "SIV", "V8"))
    thr = {kd: {f"{R:.4f}": fit._threshold(kd, R) for R in (r_d, 0.10)} for kd in ("entropy", "entl8", "entl8c", "entsiv", "entv8")}
    step = lambda x: 2.0 ** (np.floor(np.log2(x)) - 7)
    out["train_rate_default"] = r_d
    out["thresholds_nats"] = thr
    out["threshold_shift"] = {kd: {R: {"nats": thr[kd][R][0] - thr["entropy"][R][0],
                                       "bf16_steps": (thr[kd][R][0] - thr["entropy"][R][0]) / step(thr["entropy"][R][0])}
                                   for R in thr[kd]} for kd in thr if kd != "entropy"}

    # ---- (a) recall at the copy, (c) D starts removed outside it, and the ticket's source entropy
    copies, outside = [], {rule: collections.Counter() for rule in ("L8", "L8c", "V8")}
    per_text = {rule: [] for rule in outside}
    for i, r in enumerate(C["quad_test"]):
        b, hb, a = r["b"], r["hb"], r["units"]["id_copy"][0][0]
        c = [a + 1, a + 2, a + 3, a + 4]
        E = set(blt_layouts.layout("entropy", 0.10, sig_of(r), fit).tolist())
        code, T, strict, e = onset(b, hb, r["arg"], c[0])
        s = r["units"]["id_new"][0][0]
        copies.append({"problem": i, "ticket": b[a:a + 4].decode(), "distance": a - s, "code_c1": code,
                       "T": T.decode() if T else None, "S_bits": float(hb[e + 1:e + 1 + len(T)].sum()) if T else None,
                       "h_c1": float(hb[a]), "h_src1": float(hb[s]), "argmax_ok_c1": bool(r["arg"][a] == b[a] + OFFSET),
                       "L8": [bool(r["L8"][t]) for t in c], "L8c": [bool(r["L8c"][t]) for t in c],
                       "D_c1": bool(c[0] in set(r["D"].tolist())), "E10_c1": c[0] in E})
        w = where_map(r["units"])
        for rule in outside:
            rm = [int(t) for t in r["D"] if r[rule][t] and t not in c]
            per_text[rule].append(len(rm))
            outside[rule].update(w.get(t, "other") for t in rm)
    e10 = [x for x in copies if x["E10_c1"]]
    out["a"] = {rule: {"fires_c1": float(np.mean([x[rule][0] for x in copies])),
                       "fires_c1_at_E10_starts": float(np.mean([x[rule][0] for x in e10])) if e10 else None,
                       "latched_c1_to_c4": float(np.mean([all(x[rule]) for x in copies])),
                       "misses_by_condition": dict(collections.Counter(
                           x["code_c1"] if x["code_c1"] else "6 (L8c)" for x in copies if not x[rule][0]))}
                for rule in ("L8", "L8c")}
    out["a"]["n_copies"] = len(copies); out["a"]["n_E10_starts_c1"] = len(e10)
    out["a"]["n_D_starts_c1"] = int(sum(x["D_c1"] for x in copies))
    S = np.array([x["S_bits"] for x in copies if x["S_bits"] is not None])
    out["ticket_source_bits"] = {"n": len(S), "T_is_ticket": int(sum(x["T"] == x["ticket"] for x in copies)),
                                 "min": float(S.min()), "p1": float(np.percentile(S, 1)),
                                 "median": float(np.median(S)), "max": float(S.max())} if len(S) else None
    out["c"] = {rule: {"per_text": float(np.mean(per_text[rule])), "max_per_text": int(max(per_text[rule])),
                       "where": dict(outside[rule].most_common())} for rule in outside}
    out["copies"] = copies

    # ---- (b) computed-value starts in the equation texts (O6's V4 count)
    eq_fit = blt_layouts.Fit((0.10,))
    for r in C["eq_train"]:
        eq_fit.add(sig_of(r))
    eq_fit.done()
    nh = {f"{base}-{rule}": {"in_layout": 0, "removed": 0, "same_value": 0, "new_value": 0, "cases": []}
          for base in ("D", "E10") for rule in ("L8", "L8c", "V8")}
    positions = {"all": 0, "L8": 0, "L8c": 0, "V8": 0}
    v8_cases = []
    for j, r in enumerate(C["eq_test"]):
        b = r["b"]
        lay = {"D": set(r["D"].tolist()), "E10": set(blt_layouts.layout("entropy", 0.10, sig_of(r), eq_fit).tolist())}
        for m in VALUE.finditer(b):
            for t in sorted({m.start() + 3, m.start(1) + 1}):          # the byte after '= ' and the value's first digit
                positions["all"] += 1
                for rule in ("L8", "L8c", "V8"):
                    positions[rule] += int(r[rule][t])
                same = any(b[e + 1:].startswith(b[t - 1:m.end(1)]) for e in ends_of(b, b[t - 9:t - 1], t - 2))
                code, T, _, _ = onset(b, r["hb"], r["arg"], t)
                for base, st in lay.items():
                    if t not in st:
                        continue
                    for rule in ("L8", "L8c", "V8"):
                        x = nh[f"{base}-{rule}"]
                        x["in_layout"] += 1
                        if not r[rule][t]:
                            continue
                        x["removed"] += 1; x["same_value" if same else "new_value"] += 1
                        x["cases"].append({"text": j, "token": t, "context": b[max(0, t - 41):t - 1].decode(errors="replace"),
                                           "value": m.group(1).decode(), "same_value_earlier": bool(same),
                                           "how": "onset" if code == 0 else "latched", "T": T.decode() if T else None})
                    if base == "D" and r["V8"][t]:
                        v8_cases.append({"text": j, "token": t, "value": m.group(1).decode(), "same_value_earlier": bool(same),
                                         "L8": "removed" if r["L8"][t] else f"stopped by condition {code}",
                                         "T": T.decode() if T else None})
    nd = nh["D-L8"]["in_layout"]
    out["b"] = {"computed_value_positions": positions, **nh, "D_allowed_new_value": int(round(0.0025 * nd))}
    out["V8_computed_cases"] = {"n": len(v8_cases),
                                "by_outcome": dict(collections.Counter(x["L8"] for x in v8_cases).most_common()),
                                "cases": v8_cases}

    # ---- xdomain: E10 cost by [S2]'s method, with the real argmax and the upper / oracle bounds
    xd = {}
    stored = {dom: np.load(os.path.join(RES, f"xdomain_gains_{dom}.npz"), allow_pickle=True) for dom in DOMAINS}
    same_ent = []
    for r in C["xd"]:
        z = stored[r["dom"]]; n = len(r["b"]) + 1
        sn = z[f"{r['key']}_entropy"]
        same_ent.append(float(np.max(np.abs(sn[2:] - r["ent"][1:n - 1]))))
    out["xd_entropy_vs_stored"] = {"docs": len(same_ent), "identical": int(sum(x == 0 for x in same_ent)),
                                   "max_abs_diff_nats": max(same_ent) if same_ent else None}
    for dom in DOMAINS:
        z = stored[dom]
        docs = [r for r in C["xd"] if r["dom"] == dom]
        for subset in ("all", "test"):
            ds = docs if subset == "all" else [r for r in docs if r["key"].startswith("test")]
            if not ds:
                continue
            for agree in ("real", "up", "oracle"):
                xd[f"{dom}/{subset}/{agree}"] = xd_cost(ds, z, agree, dom == "math")
    out["xdomain_E10"] = xd
    out["decision"] = decide(out)
    out["seconds"] = time.time() - t0
    return out


def xd_cost(docs, z, agree, math):
    """[S2]'s method (judge_cost/xd.py): stored E10 base starts and per-position gains; net bits = the gain at masked
    base starts minus the gain of the refill, the top-K unmasked non-base positions by entropy, pooled per domain."""
    nbytes = sum(len(r["b"]) for r in docs)
    fp = npos = vb = vD = 0; vb_gain = 0.0; pool = []; comp = [0, 0]; cop = [0, 0]; fin = [0, 0]
    for r in docs:
        b, key = r["b"], r["key"]
        v = latch(b, r["hb"], r["arg"], agree)[0] if agree != "real" else r["L8"].copy()
        v[:2] = False
        sn = z[f"{key}_entropy"]; base = z[f"{key}_base"].copy(); base[:2] = False
        gb = np.nan_to_num(z[f"{key}_gain_bits"])
        fp += int(v.sum()); npos += len(b) - 1
        hit = v & base
        vb += int(hit.sum()); vb_gain += float(gb[hit].sum())
        vD += int((v & (sn > THR_D)).sum())
        free = np.flatnonzero(~base & ~v)
        free = free[free >= 2]
        pool.append(np.c_[sn[free], gb[free]])
        if math:                                                       # [S2]'s unit counts (mathcv.py)
            cset = set()
            for m in VALUE.finditer(b):
                for t in sorted({m.start() + 3, m.start(1) + 1}):
                    cset.add(t); comp[0] += 1; comp[1] += int(v[t])
            seen = set(); f0 = b.rfind(b"The final answer is ")
            for m in NUMR.finditer(b):
                a = m.start(); w = m.group(0)
                if a > 0 and 48 <= b[a - 1] <= 57:
                    continue
                t = a + 1
                if f0 >= 0 and a == f0 + 20:
                    fin[0] += 1; fin[1] += int(v[t])
                elif t not in cset and w in seen:
                    cop[0] += 1; cop[1] += int(v[t])
                seen.add(w)
    P = np.concatenate(pool)
    top = P[np.argsort(-P[:, 0])[:vb]] if vb else np.zeros((0, 2))
    net = vb_gain - float(top[:, 1].sum())
    rec = {"docs": len(docs), "kB": nbytes / 1000, "footprint": fp / npos, "E10_masked_per_kB": vb / (nbytes / 1000),
           "net_bpb_E10": net / nbytes, "no_refill_bpb_E10": vb_gain / nbytes, "Dproxy_masked_per_kB": vD / (nbytes / 1000)}
    if math:
        rec.update({"computed_positions_masked": comp[::-1], "copied_number_starts_masked": cop[::-1],
                    "final_answers_masked": fin[::-1]})
    return rec


def decide(out):
    """The go criteria and stop rules of dev step 1 (rules: section 3, K1-K3)."""
    a, b, c = out["a"], out["b"], out["c"]
    ok_a = {rule: a[rule]["fires_c1"] >= 0.95 and (a[rule]["fires_c1_at_E10_starts"] or 0) >= 0.95 for rule in ("L8", "L8c")}
    ok_a_E10 = {rule: (a[rule]["fires_c1_at_E10_starts"] or 0) >= 0.95 for rule in ("L8", "L8c")}
    ok_bE = {rule: b[f"E10-{rule}"]["new_value"] == 0 for rule in ("L8", "L8c")}
    ok_bD = {rule: b[f"D-{rule}"]["new_value"] <= b["D_allowed_new_value"] for rule in ("L8", "L8c")}
    ok_c = {rule: c[rule]["per_text"] <= 1.0 for rule in ("L8", "L8c")}
    d = {"checks_passed": out["causality_check"]["passed"], "a": ok_a, "b_E10": ok_bE, "b_D": ok_bD, "c": ok_c,
         "switch_to_L8c": False, "kill": [], "rule_for_step2": None}
    if not ok_a["L8"]:
        d["kill"].append("K1")
    elif not ok_bE["L8"]:
        d["kill"].append("K2")
    else:
        rule = "L8"
        if not ok_bD["L8"]:
            d["switch_to_L8c"] = True
            rule = "L8c" if ok_a_E10["L8c"] and ok_bE["L8c"] and ok_bD["L8c"] else None
            if rule is None:
                d["kill"].append("K2")
        if rule and not ok_c[rule]:
            d["kill"].append("K3"); rule = None
        d["rule_for_step2"] = rule if d["checks_passed"] else None
    return d


def step1():
    alone()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    C = build_texts(tok)
    print("texts: " + ", ".join(f"{g} {len(v)}" for g, v in C.items()), flush=True)
    model = blt_light.load(MODEL, dev).eval()
    patcher_pass(C, tok, model, dev)
    del model
    save_cache(C, os.path.join(RES, f"blt_copy_latch_patcher{TAG}.npz"))
    report(analyze(C))


def report(out):
    path = os.path.join(RES, f"blt_copy_latch_step1{TAG}.json")
    json.dump(out, open(path, "w"), indent=1)
    brief = {k: v for k, v in out.items() if k not in ("b", "V8_computed_cases", "xdomain_E10", "copies")}
    brief["b"] = {k: ({kk: vv for kk, vv in v.items() if kk != "cases"} if isinstance(v, dict) else v) for k, v in out["b"].items()}
    brief["V8_computed_cases"] = out["V8_computed_cases"]["by_outcome"]
    brief["xdomain_E10"] = {k: {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()}
                            for k, v in out["xdomain_E10"].items() if "/all/" in k or k.endswith("real")}
    print(json.dumps(brief, indent=1))
    print("wrote", path, "(exploratory, dev)")


# ---------------------------------------------------------------- steps 2 and 3: BLT-1B scoring
def scorer(model, dev, ids, units):
    n = ids.shape[1]
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
    return score


def summarize(rows, names, pairs):
    """O6 step 2's readouts (realblt_copy_veto.summarize) for the given layouts and pairs."""
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
    for a, base in pairs:
        p = {}
        for kd in KINDS:
            nu = [y["kinds"][kd][0] for y in L[base]]
            if sum(nu):
                p[kd] = {"exact": boot_ratio([y["kinds"][kd][1] for y in L[a]], [y["kinds"][kd][1] for y in L[base]], nu),
                         "bits_per_unit": boot_ratio([y["kinds"][kd][2] for y in L[a]], [y["kinds"][kd][2] for y in L[base]], nu)}
        p["bpb"] = boot_ratio([y["bits"] for y in L[a]], [y["bits"] for y in L[base]], [y["n_bytes"] for y in L[base]])
        p["bpb_without_copy"] = boot_ratio([y["bits"] - y["bits_copy"] for y in L[a]],
                                           [y["bits"] - y["bits_copy"] for y in L[base]], [y["n_bytes"] - 4 for y in L[base]])
        p["rate"] = float(np.mean([y["rate"] for y in L[a]]) - np.mean([y["rate"] for y in L[base]]))
        res["pairs"][f"{a} vs {base}"] = p
    return res


def bars(res, R):
    """K4's bars at dev step 2 (rules: Implementation, step 2 definitions)."""
    pd, pe = res["pairs"][f"D.{R} vs D"], res["pairs"][f"E10.{R} vs E10"]
    both = lambda kd: [pd[kd]["exact"][0], pe[kd]["exact"][0]]
    b = {"id_copy_D": {"value": pd["id_copy"]["exact"], "holds": pd["id_copy"]["exact"][0] >= 0.60},
         "id_copy_E10": {"value": pe["id_copy"]["exact"], "holds": pe["id_copy"]["exact"][0] >= 0.12},
         "C2_bpb_without_copy": {"value": [pd["bpb_without_copy"][0], pe["bpb_without_copy"][0]]},
         "C3_computed": {"value": both("computed")}, "C3f_final": {"value": both("final")},
         "C3c_copied": {"value": both("copied")}, "rate": {"value": [pd["rate"], pe["rate"]]}}
    b["C2_bpb_without_copy"]["holds"] = max(b["C2_bpb_without_copy"]["value"]) <= 0.01
    for k in ("C3_computed", "C3f_final", "C3c_copied"):
        b[k]["holds"] = min(b[k]["value"]) >= -0.02
    b["rate"]["holds"] = max(abs(x) for x in b["rate"]["value"]) <= 0.005
    b["all"] = all(x["holds"] for x in b.values())
    return b


def load_quad(dev, tok, model, masks):
    """The quadrant texts with their stored patcher outputs; the test texts' patcher is rerun and compared."""
    C = load_cache(os.path.join(RES, f"blt_copy_latch_patcher{TAG}.npz"), groups=("quad_train", "quad_test"))
    add_masks(C)
    cfg = model.config
    mism = 0
    for r in C["quad_test"]:
        r["ids"] = tok(r["b"].decode(), return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            ent, plen, plog = model.model.patcher(r["ids"], patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                  max_patch_length=cfg.max_patch_length)
        n = r["ids"].shape[1]
        D = np.r_[0, np.cumsum(plen[0].cpu().numpy())[:-1]]
        same = (np.array_equal(ent[0, :n].float().cpu().numpy(), r["ent"])
                and np.array_equal(plog[0].float().argmax(-1).cpu().numpy(), r["arg"])
                and sorted(int(t) for t in set(D.tolist()) if t < n) == r["D"].tolist())
        mism += int(not same)
    fit, r_d = quad_fit(C, masks)
    return C, fit, r_d, mism


def step2():
    s1 = json.load(open(os.path.join(RES, f"blt_copy_latch_step1{TAG}.json")))
    R = s1["decision"]["rule_for_step2"] or ("L8" if SMOKE else None)      # a smoke run only exercises the code
    if not R:
        sys.exit(f"dev step 1 did not pass ({s1['decision']}): no BLT-1B scoring")
    alone()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_light.load(MODEL, dev).eval()
    C, fit, r_d, mism = load_quad(dev, tok, model, (R, "SIV"))
    thr = {(kd, Rr): fit._threshold(kd, Rr) for kd in ("entropy", "ent" + R.lower(), "entsiv") for Rr in (r_d, 0.10)}
    assert all(list(v) == s1["thresholds_nats"][kd][f"{Rr:.4f}"] for (kd, Rr), v in thr.items()), "thresholds differ from step 1"
    print(f"rule {R}; {len(C['quad_train'])} train / {len(C['quad_test'])} test; default train rate {r_d:.4f}; patcher "
          f"rerun differs from the stored outputs on {mism} test texts", flush=True)
    names = ["D", "E10", "D-fit", f"D.{R}", f"E10.{R}", f"D-{R}", f"E10-{R}", "D.SIV", "E10.SIV"]
    pairs = [(f"D.{R}", "D"), (f"E10.{R}", "E10"), (f"D-{R}", "D"), (f"E10-{R}", "E10"), ("D.SIV", "D"),
             ("E10.SIV", "E10"), ("D-fit", "D")]
    rows = []
    t0 = time.time()
    for i, r in enumerate(C["quad_test"]):
        sig = sig_of(r, (R, "SIV"))
        D = r["D"]
        E = blt_layouts.layout("entropy", 0.10, sig, fit)
        lay = {"D": D, "E10": E, "D-fit": blt_layouts.layout("entropy", r_d, sig, fit),
               f"D.{R}": blt_layouts.layout("ent" + R.lower(), r_d, sig, fit),
               f"E10.{R}": blt_layouts.layout("ent" + R.lower(), 0.10, sig, fit),
               f"D-{R}": D[~r[R][D]], f"E10-{R}": E[~r[R][E]],
               "D.SIV": blt_layouts.layout("entsiv", r_d, sig, fit), "E10.SIV": blt_layouts.layout("entsiv", 0.10, sig, fit)}
        score = scorer(model, dev, r["ids"], r["units"])
        w = where_map(r["units"])
        rows.append({"problem": i, "layouts": {name: score(lay[name]) for name in names},
                     "D_fit_identical": bool(np.array_equal(lay["D-fit"], D)),
                     "removed": {f"{base}-{R}": [w.get(int(t), "other") for t in lay[base] if r[R][t]] for base in ("D", "E10")}})
        if (i + 1) % 25 == 0 or SMOKE:
            print(f"  {i + 1} problems, {time.time() - t0:.0f}s", flush=True)
    out = {"status": STATUS, "rule": R, "n_test": len(rows), "train_rate_default": r_d,
           "thresholds_nats": {f"{k[0]}@{k[1]:.4f}": v for k, v in thr.items()},
           "patcher_rerun_mismatched_texts": mism}
    out.update(summarize(rows, names, pairs))
    saved = json.load(open(os.path.join(RES, "blt_copy_veto.json")))["rows"]
    g0 = {}
    for name in ("D", "E10"):
        now = [r["layouts"][name]["kinds"]["id_copy"] for r in rows]
        old = [s["layouts"][name]["kinds"]["id_copy"] for s in saved[:len(rows)]]
        g0[name] = {"exact_equal": int(sum(x[1] == y[1] for x, y in zip(now, old))), "of": len(rows),
                    "bits_within_0.05": float(np.mean([abs(x[2] - y[2]) < 0.05 for x, y in zip(now, old)]))}
    g0["met"] = all(g0[k]["exact_equal"] == len(rows) and g0[k]["bits_within_0.05"] >= 0.95 for k in ("D", "E10"))
    out["G0_reproduction"] = g0
    out["D_fit_identical_to_D"] = int(sum(r["D_fit_identical"] for r in rows))
    out["removed_starts"] = {k: dict(collections.Counter(x for r in rows for x in r["removed"][k]).most_common())
                             for k in rows[0]["removed"]}
    out["bars"] = bars(out, R)
    out["go"] = {"G0": g0["met"], "bars": out["bars"]["all"], "all": bool(g0["met"] and out["bars"]["all"]),
                 "kill": [] if g0["met"] and out["bars"]["all"] else (["K4"] if g0["met"] else ["stop: G0 not met"])}
    path = os.path.join(RES, f"blt_copy_latch_step2{TAG}.json")
    json.dump({**out, "rows": rows}, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_layout", "pairs")}, indent=1))
    for name, v in out["by_layout"].items():
        print(f"{name:9s} rate {v['rate']:.3f} bpb-c {v['bpb_without_copy']:.4f} "
              + " ".join(f"{kd} {v[kd]['exact'] * 100:.1f}" for kd in KINDS if kd in v))
    for k, p in out["pairs"].items():
        print(k, "| rate %+.4f" % p["rate"], "| bpb-c %+.4f" % p["bpb_without_copy"][0],
              "| " + " ".join(f"{kd} {p[kd]['exact'][0] * 100:+.1f} [{p[kd]['exact'][1][0] * 100:+.1f}, "
                              f"{p[kd]['exact'][1][1] * 100:+.1f}]" for kd in KINDS if kd in p))
    print("wrote", path, "(exploratory, dev)")


def step3():
    s2 = json.load(open(os.path.join(RES, f"blt_copy_latch_step2{TAG}.json")))
    if not s2["go"]["all"] and not SMOKE:
        sys.exit("dev step 2 did not pass: step 3 is not run")
    alone()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_light.load(MODEL, dev).eval()
    C, fit, r_d, mism = load_quad(dev, tok, model, ("V8",))
    names = ["E10", "E10.V8", "E10.V8+k"]
    rows = []
    t0 = time.time()
    for i, r in enumerate(C["quad_test"]):
        sig = sig_of(r, ("V8",))
        a = r["units"]["id_copy"][0][0]
        assert r["b"][a - 4:a] == b"ket "
        E = blt_layouts.layout("entropy", 0.10, sig, fit)
        EV = blt_layouts.layout("entv8", 0.10, sig, fit)
        lay = {"E10": E, "E10.V8": EV, "E10.V8+k": np.array(sorted(set(EV.tolist()) | {a - 3}), int)}  # token a-3 = the `k`
        score = scorer(model, dev, r["ids"], r["units"])
        rows.append({"problem": i, "E10_c1": bool(a + 1 in set(E.tolist())),
                     "layouts": {name: score(lay[name]) for name in names}})
        if (i + 1) % 25 == 0 or SMOKE:
            print(f"  {i + 1} problems, {time.time() - t0:.0f}s", flush=True)
    out = {"status": STATUS + "; step 3 is reported only and never changes L8", "n_test": len(rows),
           "patcher_rerun_mismatched_texts": mism}
    out.update(summarize(rows, names, [("E10.V8", "E10"), ("E10.V8+k", "E10"), ("E10.V8+k", "E10.V8")]))
    out["id_copy_by_E10_c1"] = {f"{'with' if cov else 'without'} E10 start at c1": {
        "n": int(sum(r["E10_c1"] == cov for r in rows)),
        **{name: (float(np.mean([r["layouts"][name]["kinds"]["id_copy"][1] for r in rows if r["E10_c1"] == cov]))
                  if any(r["E10_c1"] == cov for r in rows) else None) for name in names}} for cov in (True, False)}
    path = os.path.join(RES, f"blt_copy_latch_step3{TAG}.json")
    json.dump({**out, "rows": rows}, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    for k, p in out["pairs"].items():
        print(k, "| rate %+.4f" % p["rate"], "| bpb-c %+.4f" % p["bpb_without_copy"][0],
              "| " + " ".join(f"{kd} {p[kd]['exact'][0] * 100:+.1f}" for kd in KINDS if kd in p))
    print("wrote", path, "(exploratory, dev)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "step1":
        step1()
    elif cmd == "analyze":
        report(analyze(load_cache(os.path.join(RES, f"blt_copy_latch_patcher{TAG}.npz"))))
    elif cmd == "step2":
        step2()
    elif cmd == "step3":
        step3()
    else:
        sys.exit(__doc__)
