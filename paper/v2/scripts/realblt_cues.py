"""Structural-cue controls for the novelty table of arXiv:2610.22100, on BLT-1B: is the table more than a detector of
the cue before a value? Paper 1 v2 does not use this experiment; it uses only signals(), which realblt_prose.py
imports. (Docstring rewritten for the public copy; the code is unchanged.)

Part A, GSM8K (stripped, 300 test / 300 train problems), layouts at 10% and 15%, per-unit results kept per problem
for paired problem-bootstrap intervals:
    entropy    BLT's rule
    eq         a patch start right after every '= ' (paper 1's hand-written rule), rest by entropy
    syntax     after every '= ' and after "The final answer is " (the stripped analogue of paper 2's syntax+entropy)
    nov        the novelty table fitted on stripped GSM8K solutions (paper 2)
    space      SpaceByte-style: entropy, restricted to positions after a non-alphanumeric byte
Part B, cue-free text: generated problems whose computed values are introduced by varied prose ("gives", "comes to",
"leaves him", "makes", at a sentence start, ...) and never by '='. A novelty table is fitted on 3,000 generated
training problems (same LZ target and procedure as paper 2). Layouts at 10%:
    entropy, nov_cf (the table fitted on this text), nov_gsm (the GSM8K table, transfer), space,
    lz_oracle  the true LZ novelty (forward match length, non-causal: the principle without a predictor),
    oracle     a patch start at every computed value (labels), rest by entropy
Prediction (before the run): if the table is a cue detector, its coverage of computed values in Part B falls to
about the base rate and its gain over entropy disappears; if it captures novelty, it keeps both. Part A: the table's
lead over the '=' rule (56.5 vs 51.8% exact at 10%) gets a paired interval.

    python paper/v2/scripts/realblt_cues.py [N=300]      -> paper/v2/results/blt_cues.json (not released)
"""
import json
import os
import re
import sys

import numpy as np

import _seg  # noqa: F401
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import reuse_screen
from realblt_budget import GSM, GSM_TRAIN, MAX_TOKENS, MODEL, lengths_from_starts, problem
from scale.rules import table_lookup

NOVTAB_GSM = os.path.expanduser("~/.cache/segresearch-blt/novtab_gsm.npz")
ITEMS = ["apples", "pencils", "boxes", "stickers", "marbles", "books", "cookies", "coins", "cards", "bottles"]
NAMES = ["Ana", "Ben", "Carla", "Diego", "Elena", "Felix", "Grace", "Hugo", "Iris", "Jon"]


# ---------------------------------------------------------------- Part B text
def gen_problem(rng):
    """(text, solution start, [(kind, start, end)]) with computed values never after '='."""
    n, it = rng.choice(NAMES), rng.choice(ITEMS)
    a, b = int(rng.integers(12, 60)), int(rng.integers(3, 30))
    q = f"{n} has {a} {it}. A friend gives {n} {b} more, and later {n} buys some bags of {it}."
    k = int(rng.integers(2, 6)); m = int(rng.integers(3, 12))
    q += f" Each bag holds {m} {it} and {n} buys {k} bags. Then {n} gives away {int(rng.integers(5, 20))} {it}."
    d = int(re.findall(r"gives away (\d+)", q)[0])
    q += f" How many {it} does {n} have now?"
    c1 = a + b; c2 = k * m; c3 = c1 + c2; c4 = c3 - d
    say = [
        [f"Adding {b} to {a} gives {c1}.", f"{a} plus {b} comes to {c1}.", f"With the gift, {n} has {c1} {it}.",
         f"Together that makes {c1}.", f"So far that is {c1} {it}."],
        [f"The bags hold {k} times {m}, which is {c2} {it}.", f"{k} bags of {m} add up to {c2}.",
         f"Buying the bags brings in {c2} more.", f"In the bags there are {c2} {it}.", f"{c2} {it} come in the bags."],
        [f"Now the total reaches {c3}.", f"Altogether {n} holds {c3} {it}.", f"Both amounts sum to {c3}.",
         f"Counting everything gives {c3}.", f"That brings it up to {c3} {it}."],
        [f"Giving away {d} leaves {c4}.", f"After that {n} keeps {c4} {it}.", f"Taking {d} away leaves {n} with {c4}.",
         f"Only {c4} remain.", f"{n} ends up with {c4} {it}."],
    ]
    sol = " ".join(str(rng.choice(s)) for s in say)
    text = q + "\n" + sol + f"\nThe final answer is {c4}"
    s0 = len(q) + 1
    units = []
    for mt in re.finditer(r"\d+", text):
        st, en = mt.span()
        if st < s0:
            continue
        if st == len(text) - len(str(c4)):
            units.append(("fin", st, en))
        elif int(mt.group()) in (c1, c2, c3, c4) and mt.group() not in text[:st]:
            units.append(("res", st, en))                      # a computed value, first occurrence
        elif mt.group() in text[:st]:
            units.append(("copy", st, en))
    return text, s0, units


def cuefree_problems(n, seed):
    rng = np.random.default_rng(seed)
    return [gen_problem(rng) for _ in range(n)]


# ---------------------------------------------------------------- shared
def signals(model, cfg, ids, b, tabs, dev):
    n = ids.shape[1]
    with torch.no_grad():
        ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                        max_patch_length=cfg.max_patch_length)
    e = ent[0].float().cpu().numpy()
    tokscore = lambda v: np.r_[-np.inf, -np.inf, v[1:n - 1]]   # token t = byte t-1
    sig = {"entropy": tokscore(e[:n] if len(e) >= n else np.r_[e, np.zeros(n - len(e))])}
    sig["entropy"] = np.full(n, -np.inf); sig["entropy"][2:] = e[1:n - 1]
    arr = np.frombuffer(b, np.uint8)
    for k, t in tabs.items():
        v = table_lookup(t["T2"], t["T1"], float(t["glob"]), arr)
        s = np.full(n, -np.inf); s[2:] = v[1:n - 1]; sig[k] = s
    prev_ok = np.zeros(n, bool)                                # token t starts at byte t-1; previous byte is t-2
    prev_ok[2:] = [not chr(c).isalnum() for c in arr[:n - 2]]
    sig["space"] = np.where(prev_ok, sig["entropy"], -np.inf)
    fm = reuse_screen.match_signals(arr)[0]                    # forward LZ match length (non-causal)
    s = np.full(n, -np.inf); s[2:] = -fm[1:n - 1]; sig["lz_oracle"] = s
    return sig


def run_part(name, items, train_items, layouts, budgets, model, cfg, tok, dev, tabs):
    """items: [(text, spans as (kind, start_tok, end_tok), forced dict)] -> per-unit rows."""
    def prep(text, spans, forced):
        b = text.encode()
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] > MAX_TOKENS or ids.shape[1] != len(b) + 1:
            return None
        return b, ids, spans, forced, signals(model, cfg, ids, b, tabs, dev)

    train = [x for x in (prep(*it) for it in train_items) if x]
    test = [x for x in (prep(*it) for it in items) if x]
    fit = blt_layouts.Fit(budgets)
    for _, _, _, forced, sig in train:
        fit.add(sig, forced)
    fit.done()
    rows, rates = [], {}
    for i, (b, ids, spans, forced, sig) in enumerate(test):
        n = ids.shape[1]
        for R in budgets:
            for lname, kind in layouts.items():
                st = blt_layouts.layout(kind, R, sig, fit, forced)
                S = set(st.tolist())
                rates.setdefault(f"{lname}@{int(R * 100)}", []).append(len(st) / n)
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                                 use_cache=False).logits[0].float(), -1)
                for kind_, s, e_ in spans:
                    tgt = ids[0, s:e_]; pred = lp[s - 1:e_ - 1]
                    rows.append({"part": name, "problem": i, "layout": f"{lname}@{int(R * 100)}", "kind": kind_,
                                 "start": int(s), "covered": s in S, "ok": bool((pred.argmax(-1) == tgt).all())})
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i + 1} problems", flush=True)
    return rows, {k: float(np.mean(v)) for k, v in rates.items()}


def paired(rows, kind, la, lb, rng, n=2000):
    """exact(la) - exact(lb) on units of `kind`, bootstrap over problems."""
    A = {(r["problem"], r["start"]): r["ok"] for r in rows if r["kind"] == kind and r["layout"] == la}
    B = {(r["problem"], r["start"]): r["ok"] for r in rows if r["kind"] == kind and r["layout"] == lb}
    keys = sorted(set(A) & set(B))
    P = np.array([k[0] for k in keys]); d = np.array([float(A[k]) - float(B[k]) for k in keys])
    probs = np.unique(P); by = {p: d[P == p] for p in probs}
    bs = [np.concatenate([by[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(n)]
    return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(keys)}


def summarize(rows):
    out = {}
    for r in rows:
        k = (r["layout"], r["kind"]); o = out.setdefault(k, [0, 0, 0])
        o[0] += 1; o[1] += r["covered"]; o[2] += r["ok"]
    return {f"{l}|{k}": {"n": v[0], "covered": v[1] / v[0], "exact": v[2] / v[0]} for (l, k), v in sorted(out.items())}


def main(N=300):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval(); cfg = model.config
    rng = np.random.default_rng(0)
    res = {}

    # Part A: GSM8K
    def gsm_item(p):
        pr = problem(p, tok, "cpu")
        if pr is None:
            return None
        b, ids, spans, res_starts = pr
        f0 = [s for k, s, _ in spans if k == "fin"]
        return (b.decode(), spans, {"eq": res_starts.tolist(), "syn": res_starts.tolist() + f0})
    gtab = {"nov": np.load(NOVTAB_GSM)}
    A_items = [x for x in (gsm_item(json.loads(l)) for l in list(open(GSM))[:N]) if x]
    A_train = [x for x in (gsm_item(json.loads(l)) for l in list(open(GSM_TRAIN))[:N]) if x]
    layoutsA = {"entropy": "entropy", "eq": "forced[eq]:entropy", "syntax": "forced[syn]:entropy", "nov": "nov", "space": "space"}
    rowsA, ratesA = run_part("gsm8k", A_items, A_train, layoutsA, (0.10, 0.15), model, cfg, tok, dev, gtab)
    res["gsm8k"] = {"rates": ratesA, "summary": summarize(rowsA), "paired": {}}
    for R in (10, 15):
        for kind in ("res", "fin"):
            for la, lb in (("nov", "eq"), ("nov", "syntax"), ("nov", "space"), ("space", "entropy"), ("eq", "entropy")):
                res["gsm8k"]["paired"][f"{kind}: {la}-{lb}@{R}"] = paired(rowsA, kind, f"{la}@{R}", f"{lb}@{R}", rng)

    # Part B: cue-free text
    fit_text = "\n\n".join(t for t, _, _ in cuefree_problems(3000, seed=1))
    fb = np.frombuffer(fit_text.encode(), np.uint8)
    T2, T1, glob = reuse_screen.tab_fit(fb, -reuse_screen.match_signals(fb)[0])
    tabs = {"nov_cf": {"T2": T2, "T1": T1, "glob": glob}, "nov_gsm": np.load(NOVTAB_GSM)}

    def cf_item(t, s0, units):
        spans = [(k, a + 1, e + 1) for k, a, e in units]       # token = byte + 1 (ASCII text)
        return (t, spans, {"comp": [s for k, s, _ in spans if k == "res"]})
    B_items = [cf_item(*x) for x in cuefree_problems(N, seed=2)]
    B_train = [cf_item(*x) for x in cuefree_problems(N, seed=3)]
    layoutsB = {"entropy": "entropy", "nov_cf": "nov_cf", "nov_gsm": "nov_gsm", "space": "space",
                "lz_oracle": "lz_oracle", "oracle": "forced[comp]:entropy"}
    rowsB, ratesB = run_part("cuefree", B_items, B_train, layoutsB, (0.10,), model, cfg, tok, dev, tabs)
    res["cuefree"] = {"rates": ratesB, "summary": summarize(rowsB), "paired": {},
                      "base_rate_computed_starts": float(np.mean([sum(k == "res" for k, _, _ in u) / len(t)
                                                                  for t, _, u in cuefree_problems(N, seed=2)]))}
    for la, lb in (("nov_cf", "entropy"), ("lz_oracle", "entropy"), ("oracle", "entropy"), ("space", "entropy"),
                   ("nov_cf", "oracle")):
        for kind in ("res", "copy"):
            res["cuefree"]["paired"][f"{kind}: {la}-{lb}@10"] = paired(rowsB, kind, f"{la}@10", f"{lb}@10", rng)

    out = os.path.join(_seg.HERE, "results", "blt_cues.json")
    json.dump({**res, "rows": rowsA + rowsB}, open(out, "w"))
    for part in ("gsm8k", "cuefree"):
        print(f"== {part} rates", json.dumps({k: round(v, 3) for k, v in res[part]["rates"].items()}))
        for k, v in res[part]["summary"].items():
            if k.endswith("|res") or k.endswith("|fin"):
                print(f"RESULT {part} {k:22s} n {v['n']:5d} covered {v['covered']:.3f} exact {v['exact']:.3f}")
        for k, v in res[part]["paired"].items():
            print(f"PAIRED {part} {k:28s} {100 * v['diff']:+6.1f} [{100 * v['ci'][0]:+6.1f}, {100 * v['ci'][1]:+6.1f}] n {v['n']}")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))
