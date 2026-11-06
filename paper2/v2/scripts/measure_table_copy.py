"""Bytes-only check for paper 2 v2 (PLAN.md:112-113): does the novelty table start patches on copy onsets the way
entropy does? No model is loaded and nothing is scored; the BLT tokenizer is loaded only for O6's text filter.

Texts: O6's dev F1 texts (realblt_quadrant.build, rng default_rng(0), train tickets drawn first; GSM8K train idx
0-299 as fit texts, GSM8K test idx 0-299 as test texts), rebuilt here with the repository's builder and filter and
checked byte for byte against the stored dev patcher outputs (results/blt_copy_latch_patcher.npz), which give the
entropy and BLT-1B's default layout D. Train rows are read with itertools.islice (first 300 lines only).

Layouts (blt_layouts, train mode, thresholds fitted on the 295 fit texts with the same insertion), at D's pooled fit
rate (r_d, as O6's D.X layouts) and at 10%:
    entropy   D itself at the default rate (O6's D); entropy@10% (O6's E10). entropy@r_d (D-fit) as a check.
    table     the novelty table ~/.cache/segresearch-blt/novtab_gsm.npz as stored, looked up as realblt_waste.prep /
              realblt_cues.signals do (scale.rules.table_lookup; token t = byte t-1); kind "nov" (R4, R8: Tables 2, 3)
    eq        entropy + a start right after every '= ' (the '=' rule; realblt_budget.problem's starts; kind
              forced[eq]:entropy, as realblt_waste / realblt_cues)
    r8b       entropy + the table's top 3% of training bytes (R8b; realblt_hybrid: forced[q30]:entropy, the top-3%
              value from realblt_hybrid's own training texts, checked against results/blt_hybrid.json "thresholds");
              q15 and q50 (1.5%, 5%) as in Table 3, secondary
Readouts per layout and rate: share of copies with a start at c1, at each c_k and at any byte c1..cL of the copied
ID; the same at the source occurrence; share of all starts at c1..cL, at s1..sL, at L8's and V_8's positions, and by
unit kind. Problem-level 95% intervals: realblt_copy_veto.boot_ratio (2,000 resamples, seed 6), plus Wilson / exact
bounds where the bootstrap interval has zero width.

Also: the table's rate sweep on the F1 texts; the table thresholds fitted R4's way (plain texts, no ticket); other
templates on the same dev problems, bytes only (table only): 6-digit and mixed IDs (cell F3, as requested), and
letters, doses, F2a and F2b (secondary); the dev smoke patcher outputs (GSM8K train idx < 128) for D at c1 there.

    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 <segresearch>/.venv-blt/bin/python -B measure_table_copy.py
"""
import sys

sys.dont_write_bytecode = True

import collections
import hashlib
import itertools
import json
import math
import os
import re
import time

os.environ["HF_HUB_OFFLINE"] = "1"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper2/v2 (was the private repo's absolute path)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import realblt_copy_latch as L
import realblt_copy_latch_fresh as F
from transformers import AutoTokenizer

import blt_layouts
import realblt_budget
from realblt_budget import GSM, GSM_TRAIN, MODEL
from realblt_copy_veto import boot_ratio
from realblt_quadrant import build
from scale.rules import table_lookup

assert blt_layouts.MODE == "train"
NOVTAB_GSM = os.path.expanduser("~/.cache/segresearch-blt/novtab_gsm.npz")      # realblt_waste.py:29
PATCHER = os.path.join(REPO, "results", "blt_copy_latch_patcher.npz")
SMOKE_NPZ = os.path.join(REPO, "results", "blt_copy_latch_fresh_patcher_smoke.npz")
STEP1 = os.path.join(REPO, "results", "blt_copy_latch_step1.json")
HYBRID = os.path.join(REPO, "results", "blt_hybrid.json")
QS = (0.015, 0.03, 0.05)                                          # realblt_hybrid.HYB's forced shares
LAYOUTS = {"entropy": None, "table": "nov", "eq": "forced[eq]:entropy", "r8b": "forced[q30]:entropy",
           "r8b_q15": "forced[q15]:entropy", "r8b_q50": "forced[q50]:entropy"}
SWEEP = (0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.34, 0.35, 0.36, 0.40)
TEMPLATES = (("F3", "digits6", "requested"), ("F3", "mixed", "requested"), ("F3", "letters4", "secondary"),
             ("F3", "dose", "secondary"), ("F2a", "digits4", "secondary"), ("F2b", "digits4", "secondary"))
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def dev_rows(path, n=300):
    """the first n lines only (islice): never reads GSM8K train beyond its first 300 lines"""
    with open(path) as f:
        return [json.loads(line) for line in itertools.islice(f, n)]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = max(0.0, c - h), min(1.0, c + h)
    if k == 0:
        hi = max(hi, 1 - 0.025 ** (1 / n))                         # exact (Clopper-Pearson) bound at k = 0
    if k == n:
        lo = min(lo, 0.025 ** (1 / n))
    return [lo, hi]


def share(x, den=None):
    x = np.asarray(x, float)
    den = np.ones(len(x)) if den is None else np.asarray(den, float)
    v, ci = boot_ratio(x, np.zeros(len(x)), den)
    rec = {"value": v, "ci_boot": ci, "num": float(x.sum()), "den": float(den.sum())}
    if np.all(den == 1):
        rec["ci_wilson"] = wilson(int(x.sum()), len(x))
    return rec


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v, ci = boot_ratio(a, b, np.ones(len(a)))
    return {"diff": v, "ci_boot": ci, "n": len(a)}


def ratio_diff(a_n, a_d, b_n, b_d, seed=6, n=2000):
    """sum a_n / sum a_d - sum b_n / sum b_d, resampling problems jointly"""
    a_n, a_d, b_n, b_d = (np.asarray(x, float) for x in (a_n, a_d, b_n, b_d))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(a_n), len(a_n))
        bs.append(a_n[i].sum() / a_d[i].sum() - b_n[i].sum() / b_d[i].sum())
    return {"diff": float(a_n.sum() / a_d.sum() - b_n.sum() / b_d.sum()),
            "ci_boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def nov_signal(tab, b):
    """realblt_waste.prep / realblt_cues.signals: table value of byte t-1 from bytes t-3, t-2, as the score of token t"""
    n = len(b) + 1
    v = table_lookup(tab["T2"], tab["T1"], float(tab["glob"]), np.frombuffer(b, np.uint8))
    s = np.full(n, -np.inf)
    s[2:] = v[1:n - 1]
    return s


def forced_of(b, nov, thr_q):
    """eq: realblt_budget.problem's in-line result starts (the byte after '= ', token m.start() + 3); qNN: positions whose
    table value is at least the top-q value (realblt_hybrid.py:46-48)"""
    f = {"eq": [m.start() + 2 + 1 for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b)]}
    for q, th in thr_q.items():
        f[f"q{int(round(q * 1000))}"] = np.flatnonzero(nov >= th).tolist()
    return f


def tokens_of(span):
    a, e = span
    return np.arange(a + 1, e + 1)                                  # token t = byte t-1


def ctx_value(tab, ctx):
    """the table value (float32, as table_lookup returns it) of a byte preceded by the two bytes ctx"""
    return float(table_lookup(tab["T2"], tab["T1"], float(tab["glob"]), np.frombuffer(ctx + b"0", np.uint8))[2])


# ---------------------------------------------------------------- texts
def rebuild_f1(tok):
    """O6's quadrant texts exactly as realblt_copy_latch.build_texts makes them (train tickets drawn first), with the
    train rows read by islice instead of list(open(...))[:300]"""
    keep = F.keeper(tok)
    rng = np.random.default_rng(0)
    out = {}
    for split, path in (("quad_train", GSM_TRAIN), ("quad_test", GSM)):
        out[split] = []
        for p in dev_rows(path):
            b, units = build(p, rng)
            if keep(b):
                out[split].append((b, units))
    return out


def check_against_store(C, rebuilt):
    rec = {}
    for g in ("quad_train", "quad_test"):
        assert len(C[g]) == len(rebuilt[g]), (g, len(C[g]), len(rebuilt[g]))
        same = 0
        for r, (b, units) in zip(C[g], rebuilt[g]):
            su = {k: [tuple(x) for x in v] for k, v in r["units"].items()}
            bu = {k: [tuple(x) for x in v] for k, v in units.items()}
            same += int(r["b"] == b and su == bu)
        assert same == len(C[g]), (g, same)
        rec[g] = f"{same} of {len(C[g])} texts identical (bytes and units)"
    return rec


def plain_train(tok, tab):
    """realblt_budget.problem's texts (no ticket) of the first 300 train rows: R4's and R8b's training texts"""
    out = []
    for p in dev_rows(GSM_TRAIN):
        pr = realblt_budget.problem(p, tok, "cpu")
        if pr is not None:
            out.append(nov_signal(tab, pr[0]))
    return out


def template_texts(tok, cell, kind, seed=0):
    """cell's template (realblt_copy_latch_fresh.make) on the dev problems: fit = train idx 0-299, test = test idx
    0-299, IDs drawn with one generator, fit texts first (as O6's dev draws)"""
    keep = F.keeper(tok)
    rng = np.random.default_rng(seed)
    out = {}
    for split, path in (("fit", GSM_TRAIN), ("test", GSM)):
        out[split] = []
        for p in dev_rows(path):
            x = F.make(p, rng, cell, kind)
            if x is None or not keep(x[0]):
                continue
            b, units, info = x
            out[split].append({"b": b, "units": {k: [tuple(u) for u in v] for k, v in units.items()}, "info": info})
    return out


# ---------------------------------------------------------------- layouts and readouts
def layouts_for(r, sig, forced, fit, r_d):
    lay = {}
    for R, tag in ((r_d, "D"), (0.10, "10")):
        for name, kind in LAYOUTS.items():
            if name == "entropy":
                lay[f"entropy@{tag}"] = r["D"] if tag == "D" else blt_layouts.layout("entropy", 0.10, sig, fit)
            else:
                lay[f"{name}@{tag}"] = blt_layouts.layout(kind, R, sig, fit, forced)
    lay["entropy_fit@D"] = blt_layouts.layout("entropy", r_d, sig, fit)            # check: D-fit = D
    return lay


def readout(r, st, n):
    S = set(int(t) for t in st)
    c = tokens_of(r["units"]["id_copy"][0])
    s = tokens_of(r["units"]["id_new"][0])
    where = L.where_map(r["units"])
    w = collections.Counter(where.get(int(t), "other") for t in st)
    rec = {"n": n, "starts": len(S), "c1": int(c[0]) in S, "c_any": any(int(t) in S for t in c),
           "c_inside": any(int(t) in S for t in c[1:]), "c_n": sum(int(t) in S for t in c),
           "s1": int(s[0]) in S, "s_any": any(int(t) in S for t in s), "s_inside": any(int(t) in S for t in s[1:]),
           "s_n": sum(int(t) in S for t in s), "c_k": [int(t) in S for t in c], "s_k": [int(t) in S for t in s],
           "where": dict(w)}
    if "L8" in r:
        rec["on_L8"] = int(r["L8"][st].sum()); rec["on_V8"] = int(r["V8"][st].sum())
    rec["units"] = {k: [len(v), sum(a + 1 in S for a, _ in v)] for k, v in r["units"].items()}   # (n, covered at start)
    return rec


def summarize(per, names, base_of):
    """per[name] = list of readouts over texts (same order). Shares with intervals, and paired differences against the
    entropy layout at the same rate."""
    res = {}
    for name in names:
        X = per[name]
        st = np.array([x["starts"] for x in X], float)
        rec = {"n_texts": len(X), "rate_mean": float(np.mean([x["starts"] / x["n"] for x in X])),
               "rate_pooled": float(st.sum() / sum(x["n"] for x in X)),
               "copy_c1": share([x["c1"] for x in X]), "copy_inside": share([x["c_inside"] for x in X]),
               "copy_any": share([x["c_any"] for x in X]),
               "source_s1": share([x["s1"] for x in X]), "source_inside": share([x["s_inside"] for x in X]),
               "source_any": share([x["s_any"] for x in X]),
               "copy_by_position": [float(np.mean([x["c_k"][k] for x in X if len(x["c_k"]) > k]))
                                    for k in range(max(len(x["c_k"]) for x in X))],      # doses vary in length
               "source_by_position": [float(np.mean([x["s_k"][k] for x in X if len(x["s_k"]) > k]))
                                      for k in range(max(len(x["s_k"]) for x in X))],
               "starts_on_copy_share": share([x["c_n"] for x in X], st),
               "starts_on_source_share": share([x["s_n"] for x in X], st),
               "copy_starts_per_text": float(np.mean([x["c_n"] for x in X])),
               "s1_and_c1_agree": float(np.mean([x["c1"] == x["s1"] for x in X])),
               "s_and_c_agree_all_positions": float(np.mean([x["c_k"] == x["s_k"] for x in X])),
               "c1_given_s1": None}
        withs = [x for x in X if x["s1"]]
        if withs:
            rec["c1_given_s1"] = {"k": int(sum(x["c1"] for x in withs)), "n": len(withs),
                                  "value": float(np.mean([x["c1"] for x in withs]))}
        if "on_L8" in X[0]:
            rec["starts_on_L8_share"] = share([x["on_L8"] for x in X], st)
            rec["starts_on_V8_share"] = share([x["on_V8"] for x in X], st)
        tot = collections.Counter()
        for x in X:
            tot.update(x["where"])
        rec["where_share"] = {k: v / st.sum() for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}
        rec["unit_start_coverage"] = {k: share([x["units"][k][1] for x in X], [x["units"][k][0] for x in X])
                                      for k in X[0]["units"] if sum(x["units"][k][0] for x in X)}
        b = base_of(name)
        if b and b != name and b in per:
            B = per[b]
            bst = np.array([y["starts"] for y in B], float)
            rec["vs_" + b] = {k: paired([x[k] for x in X], [y[k] for y in B])
                              for k in ("c1", "c_inside", "c_any", "s1", "s_any")}
            rec["vs_" + b]["starts_on_copy_share"] = ratio_diff([x["c_n"] for x in X], st, [y["c_n"] for y in B], bst)
            if "on_L8" in X[0]:
                rec["vs_" + b]["starts_on_L8_share"] = ratio_diff([x["on_L8"] for x in X], st,
                                                                  [y["on_L8"] for y in B], bst)
                rec["vs_" + b]["starts_on_V8_share"] = ratio_diff([x["on_V8"] for x in X], st,
                                                                  [y["on_V8"] for y in B], bst)
        res[name] = rec
    return res


def fmt(rec):
    v = rec["value"]; lo, hi = rec["ci_boot"]
    s = f"{100 * v:5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]"
    if "ci_wilson" in rec and (lo == hi):
        wl, wh = rec["ci_wilson"]
        s += f" (exact [{100 * wl:.1f}, {100 * wh:.1f}])"
    return s


def main():
    t0 = time.time()
    res = {"status": "exploratory, bytes only, dev texts (GSM8K test idx 0-299; train idx 0-299); no model loaded",
           "inputs_sha256": {}}
    for p in (PATCHER, NOVTAB_GSM, STEP1, HYBRID, SMOKE_NPZ, os.path.abspath(__file__)):
        res["inputs_sha256"][p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tab = np.load(NOVTAB_GSM)

    # ---- O6's dev F1 texts: stored outputs, rebuilt and checked
    C = L.load_cache(PATCHER, groups=("quad_train", "quad_test"))
    rebuilt = rebuild_f1(tok)
    res["rebuild_check"] = check_against_store(C, rebuilt)
    say("rebuild:", res["rebuild_check"])
    L.add_masks(C)                                                  # L8, L8c, V8 (and SIV) from the stored outputs

    # ---- R8b's top-q values (realblt_hybrid.py:42-45), and R4's plain training texts
    plain = plain_train(tok, tab)
    allv = np.concatenate([nv[np.isfinite(nv)] for nv in plain])
    thr_q = {q: float(np.quantile(allv, 1 - q)) for q in QS}
    stored = json.load(open(HYBRID))["thresholds"]
    res["r8b_thresholds"] = {"recomputed": {str(q): v for q, v in thr_q.items()}, "stored_blt_hybrid_json": stored,
                             "n_train_texts": len(plain),
                             "equal": all(thr_q[q] == stored[str(q)] for q in QS)}
    say("R8b thresholds:", res["r8b_thresholds"])

    # ---- signals, forced sets, fit on the F1 fit texts
    for g in ("quad_train", "quad_test"):
        for r in C[g]:
            r["sig"] = {"entropy": L.sig_of(r)["entropy"], "nov": nov_signal(tab, r["b"])}
            r["forced"] = forced_of(r["b"], r["sig"]["nov"], thr_q)
    tr, test = C["quad_train"], C["quad_test"]
    r_d = sum(len(r["D"]) for r in tr) / sum(len(r["b"]) + 1 for r in tr)
    step1 = json.load(open(STEP1))
    assert abs(r_d - step1["train_rate_default"]) < 1e-12, (r_d, step1["train_rate_default"])
    fit = blt_layouts.Fit((r_d, 0.10))
    for r in tr:
        fit.add(r["sig"], r["forced"])
    fit.done()
    thr = {name: {f"{R:.4f}": list(fit._threshold(kind or "entropy", R)) for R in (r_d, 0.10)}
           for name, kind in LAYOUTS.items()}
    res["r_d"] = r_d
    res["thresholds"] = thr
    res["thresholds_match_step1_entropy"] = {R: thr["entropy"][R] == step1["thresholds_nats"]["entropy"][R]
                                             for R in thr["entropy"]}
    res["fit_train_rates"] = {f"{name}@{R:.4f}": fit.rate(kind or "entropy", R)
                              for name, kind in LAYOUTS.items() for R in (r_d, 0.10)}
    say(f"r_d = {r_d:.5f}; thresholds", json.dumps(thr))
    say("entropy thresholds equal step1's:", res["thresholds_match_step1_entropy"])

    # the table's value at the ticket's onset context, against its thresholds and the fit tokens
    trainv = np.concatenate([r["sig"]["nov"][np.isfinite(r["sig"]["nov"])] for r in tr])
    n_tok, n_txt = sum(len(r["b"]) + 1 for r in tr), len(tr)
    v_t = ctx_value(tab, b"t ")
    res["table_at_ticket_onset"] = {
        "context": "'t ' (the byte after 'Ticket ', at the source and at the copy)", "value_float32": v_t,
        "fit_tokens_above": float((trainv > v_t).mean()), "fit_tokens_equal": float((trainv == v_t).mean()),
        "rate_needed_to_start_there": float(((trainv >= v_t).sum() + 2 * n_txt) / n_tok),
        "thr_rd": thr["table"][f"{r_d:.4f}"], "thr_10": thr["table"]["0.1000"], "r8b_top3_value": thr_q[0.03]}
    say("table at 't ':", res["table_at_ticket_onset"])

    # ---- per text
    per = collections.defaultdict(list)
    vals = []
    dfit_same = 0
    for r in test:
        n = len(r["b"]) + 1
        lay = layouts_for(r, r["sig"], r["forced"], fit, r_d)
        dfit_same += int(np.array_equal(lay["entropy_fit@D"], r["D"]))
        for name, st in lay.items():
            per[name].append(readout(r, st, n))
        c = tokens_of(r["units"]["id_copy"][0]); s = tokens_of(r["units"]["id_new"][0])
        vals.append({"nov_c": r["sig"]["nov"][c].tolist(), "nov_s": r["sig"]["nov"][s].tolist(),
                     "ent_c1_nats": float(r["sig"]["entropy"][c[0]]), "ent_s1_nats": float(r["sig"]["entropy"][s[0]])})
    res["dfit_equals_D"] = dfit_same
    res["table_value_copy_equals_source"] = int(sum(v["nov_c"] == v["nov_s"] for v in vals))
    res["n_test"] = len(test)
    names = list(per)
    base_of = lambda nm: "entropy@" + nm.split("@")[1]
    res["F1"] = summarize(per, names, base_of)
    res["reproduces_step1"] = {   # results/blt_copy_latch_step1.json a.n_D_starts_c1 / a.n_E10_starts_c1
        "D_c1": [int(res["F1"]["entropy@D"]["copy_c1"]["num"]), step1["a"]["n_D_starts_c1"]],
        "E10_c1": [int(res["F1"]["entropy@10"]["copy_c1"]["num"]), step1["a"]["n_E10_starts_c1"]]}
    res["F1_per_text"] = {nm: [{k: x[k] for k in ("c1", "c_inside", "s1", "s_inside", "c_n", "s_n", "starts", "n")}
                               for x in per[nm]] for nm in names}
    res["F1_values"] = vals
    say("reproduces step1 (mine, stored):", res["reproduces_step1"])

    say(f"\nF1 dev texts: {len(test)} test texts; D-fit = D on {dfit_same}; table value at c_k equals s_k on "
        f"{res['table_value_copy_equals_source']} texts")
    say(f"{'layout':14s} {'rate':>6s} | {'copy c1':>22s} | {'copy c2..cL':>22s} | {'copy any':>22s} | "
        f"{'source s1':>22s} | {'source any':>22s} | {'starts on copy (%)':>22s}")
    for nm in names:
        x = res["F1"][nm]
        say(f"{nm:14s} {x['rate_mean']:6.3f} | {fmt(x['copy_c1']):>22s} | {fmt(x['copy_inside']):>22s} | "
            f"{fmt(x['copy_any']):>22s} | {fmt(x['source_s1']):>22s} | {fmt(x['source_any']):>22s} | "
            f"{fmt(x['starts_on_copy_share'])}")
    say("\nby position (share of copies / sources with a start at c_k / s_k):")
    for nm in names:
        x = res["F1"][nm]
        say(f"  {nm:14s} copy " + " ".join(f"{100 * v:5.1f}" for v in x["copy_by_position"])
            + " | source " + " ".join(f"{100 * v:5.1f}" for v in x["source_by_position"])
            + f" | c1 given s1 {x['c1_given_s1']}")
    say("\npaired against entropy at the same rate (points; problem bootstrap):")
    for nm in names:
        for k, v in res["F1"][nm].items():
            if k.startswith("vs_"):
                say(f"  {nm:14s} " + "  ".join(
                    f"{m} {100 * d['diff']:+6.1f} [{100 * d['ci_boot'][0]:+6.1f}, {100 * d['ci_boot'][1]:+6.1f}]"
                    for m, d in v.items()))
    # the table's source / copy disagreements: only at positions whose value ties the threshold (_tiebreak)?
    res["table_source_copy_disagreements"] = {}
    for R, tag in ((r_d, "D"), (0.10, "10")):
        v_thr = fit._threshold("nov", R)[0]
        dis = tie = 0
        for r, x in zip(test, per[f"table@{tag}"]):
            c = tokens_of(r["units"]["id_copy"][0]); s = tokens_of(r["units"]["id_new"][0])
            for k in range(len(c)):
                if x["c_k"][k] != x["s_k"][k]:
                    dis += 1; tie += int(r["sig"]["nov"][c[k]] == v_thr == r["sig"]["nov"][s[k]])
        res["table_source_copy_disagreements"][tag] = {"positions": dis, "at_threshold_ties": tie}
    say("table source/copy disagreements (positions; all at threshold ties?):", res["table_source_copy_disagreements"])
    say("\nstart coverage by unit kind (share of units with a start at their first byte, %):")
    for nm in names:
        x = res["F1"][nm]["unit_start_coverage"]
        say(f"  {nm:14s} " + ", ".join(f"{k} {fmt(v)}" for k, v in x.items()))
    say("\nstarts on L8 / V_8 positions, and where starts fall (share of all starts, %):")
    for nm in names:
        x = res["F1"][nm]
        top = ", ".join(f"{k} {100 * v:.1f}" for k, v in list(x["where_share"].items())[:8])
        say(f"  {nm:14s} L8 {fmt(x['starts_on_L8_share'])} V8 {fmt(x['starts_on_V8_share'])} | {top}")

    # ---- the table's rate sweep on the F1 texts (thresholds on the F1 fit texts)
    sw = blt_layouts.Fit(SWEEP)
    for r in tr:
        sw.add({"nov": r["sig"]["nov"]})
    sw.done()
    res["table_sweep"] = {}
    for R in SWEEP:
        X = [readout(r, blt_layouts.layout("nov", R, {"nov": r["sig"]["nov"]}, sw), len(r["b"]) + 1) for r in test]
        res["table_sweep"][f"{R:.3f}"] = {"rate": float(np.mean([x["starts"] / x["n"] for x in X])),
                                          "c1": float(np.mean([x["c1"] for x in X])),
                                          "c_any": float(np.mean([x["c_any"] for x in X])),
                                          "s1": float(np.mean([x["s1"] for x in X])),
                                          "threshold": list(sw._threshold("nov", R))}
    say("\ntable rate sweep (F1 test texts):", json.dumps({k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                                                               for kk, vv in v.items() if kk != "threshold"}
                                                           for k, v in res["table_sweep"].items()}))

    # ---- R4's way: table thresholds fitted on the plain training texts (no ticket)
    pf = blt_layouts.Fit((r_d, 0.10))
    for nv in plain:
        pf.add({"nov": nv})
    pf.done()
    res["table_thresholds_plain_fit"] = {}
    for R, tag in ((r_d, "D"), (0.10, "10")):
        X = [readout(r, blt_layouts.layout("nov", R, {"nov": r["sig"]["nov"]}, pf), len(r["b"]) + 1) for r in test]
        res["table_thresholds_plain_fit"][tag] = {
            "threshold": list(pf._threshold("nov", R)), "rate": float(np.mean([x["starts"] / x["n"] for x in X])),
            "copy_c1": share([x["c1"] for x in X]), "copy_any": share([x["c_any"] for x in X]),
            "source_s1": share([x["s1"] for x in X])}
    say("table, thresholds fitted R4's way (plain texts):", json.dumps(
        {k: {"thr": v["threshold"], "rate": round(v["rate"], 3), "c1": v["copy_c1"]["value"],
             "any": round(v["copy_any"]["value"], 3), "s1": v["source_s1"]["value"]}
         for k, v in res["table_thresholds_plain_fit"].items()}))

    # ---- other templates on the same dev problems (bytes only; table only)
    res["templates"] = {}
    kfits = {}
    for cell, kind, why in TEMPLATES:
        T = template_texts(tok, cell, kind)
        kf = blt_layouts.Fit((r_d, 0.10))
        for r in T["fit"]:
            kf.add({"nov": nov_signal(tab, r["b"])})
        kf.done()
        kfits[(cell, kind)] = kf
        perk = collections.defaultdict(list)
        same = 0
        for r in T["test"]:
            n = len(r["b"]) + 1
            nv = nov_signal(tab, r["b"])
            c = tokens_of(r["units"]["id_copy"][0]); s = tokens_of(r["units"]["id_new"][0])
            same += int(nv[c].tolist() == nv[s].tolist())
            for R, tag in ((r_d, "D"), (0.10, "10")):
                perk[f"table@{tag}"].append(readout(r, blt_layouts.layout("nov", R, {"nov": nv}, kf), n))
        kres = summarize(perk, list(perk), lambda nm: None)
        r0 = T["test"][0]
        a0, a1 = r0["units"]["id_new"][0][0], r0["units"]["id_copy"][0][0]
        kres.update({"why": why, "n_fit": len(T["fit"]), "n_test": len(T["test"]),
                     "thresholds": {f"{R:.4f}": list(kf._threshold("nov", R)) for R in (r_d, 0.10)},
                     "table_value_copy_equals_source": same, "example_id": r0["info"]["id"],
                     "source_onset_context": r0["b"][a0 - 2:a0].decode(), "copy_onset_context": r0["b"][a1 - 2:a1].decode(),
                     "source_onset_value": ctx_value(tab, r0["b"][a0 - 2:a0]),
                     "copy_onset_value": ctx_value(tab, r0["b"][a1 - 2:a1])})
        res["templates"][f"{cell}/{kind}"] = kres
        say(f"\n{cell}/{kind} ({why}): {len(T['fit'])} fit / {len(T['test'])} test; onset contexts source "
            f"{kres['source_onset_context']!r} {kres['source_onset_value']:.3f}, copy {kres['copy_onset_context']!r} "
            f"{kres['copy_onset_value']:.3f}; table thresholds {kres['thresholds']}; value at c_k = s_k on {same}")
        for nm in perk:
            x = kres[nm]
            say(f"  {nm:9s} rate {x['rate_mean']:.3f} | copy c1 {fmt(x['copy_c1'])} | c2..cL {fmt(x['copy_inside'])} | "
                f"any {fmt(x['copy_any'])} | source s1 {fmt(x['source_s1'])} | any {fmt(x['source_any'])} | "
                f"starts on copy {fmt(x['starts_on_copy_share'])}")

    # ---- dev smoke patcher outputs (GSM8K train idx < 128): D at the onsets, beside the table (descriptive)
    S = F.load(SMOKE_NPZ, groups=("F2a.fit", "F2a.test", "F2b.fit", "F2b.test", "F3.fit", "F3.test"))
    sm = {}
    for g, rows in S.items():
        cell = g.split(".")[0]
        for r in rows:
            assert r["idx"] < 300
            key = f"{cell}/{r['kind']}"
            c = tokens_of(r["units"]["id_copy"][0]); s = tokens_of(r["units"]["id_new"][0])
            Dset = set(int(t) for t in r["D"])
            nv = nov_signal(tab, r["b"])
            x = sm.setdefault(key, {"n": 0, "D_c1": 0, "D_any": 0, "D_s1": 0, "table@D_c1": 0, "table@D_any": 0,
                                    "table@10_c1": 0, "idx_max": 0})
            x["n"] += 1; x["idx_max"] = max(x["idx_max"], r["idx"])
            x["D_c1"] += int(int(c[0]) in Dset); x["D_any"] += int(any(int(t) in Dset for t in c))
            x["D_s1"] += int(int(s[0]) in Dset)
            kf = kfits[(cell, r["kind"])]
            for R, tag in ((r_d, "D"), (0.10, "10")):
                st = set(int(t) for t in blt_layouts.layout("nov", R, {"nov": nv}, kf))
                x[f"table@{tag}_c1"] += int(int(c[0]) in st)
                if tag == "D":
                    x["table@D_any"] += int(any(int(t) in st for t in c))
    res["smoke_dev_D"] = sm
    say("\nsmoke (dev, descriptive; counts of texts):", json.dumps(sm))
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "results_table_copy.json"), "w"), indent=1)
    open(os.path.join(OUT, "results_table_copy.log"), "w").write("\n".join(LOG) + "\n")
    say(f"\nwrote {os.path.join(OUT, 'results_table_copy.json')} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
