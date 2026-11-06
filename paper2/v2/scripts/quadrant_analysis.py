"""Verdicts for P1-P2 from results/blt_quadrant.json, by the decision rules fixed before the run (general_rule.md,
commit 36886d3).

P1: paired differences gain(computed) - gain(K) with a bootstrap over problems (the same resampled problems for both
    kinds), and the 25% / 50% thresholds on each kind's interval.
P2: an INDEPENDENT learnability measure for each new word: Qwen2.5-1.5B's surprisal per byte of the word in the same
    context (the full quadrant text up to the word; fp16 on MPS). Texts are rebuilt exactly as realblt_quadrant.py built them
    (same rng seed and call order, same filter), and aligned to the stored rows by problem, kind, order and length.
    Thirds by that measure; gain per third; easy - hard with a problem bootstrap.

    .venv-blt/bin/python quadrant_analysis.py      -> results/blt_quadrant_verdicts.json
"""
import json
import os

import numpy as np
import torch

import _seg  # noqa: F401
import realblt_quadrant as q

HERE = os.path.dirname(os.path.abspath(__file__))
COPYLIKE = ("id_new", "id_copy", "copied", "word_rep")


def boot_diff(P, x, ma, mb, rng, n=2000):
    """Mean of x over units in ma minus over units in mb, problems resampled (same problems for both)."""
    probs = np.unique(P)
    ia = {p: x[ma & (P == p)] for p in probs}; ib = {p: x[mb & (P == p)] for p in probs}
    out = []
    for _ in range(n):
        pick = rng.choice(probs, len(probs))
        a = np.concatenate([ia[p] for p in pick]); b = np.concatenate([ib[p] for p in pick])
        out.append(a.mean() - b.mean())
    return float(x[ma].mean() - x[mb].mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def boot_mean(P, x, m, rng, n=2000):
    probs = np.unique(P[m]); i = {p: x[m & (P == p)] for p in probs}
    out = [np.concatenate([i[p] for p in rng.choice(probs, len(probs))]).mean() for _ in range(n)]
    return float(x[m].mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def rebuild():
    """[(bytes, units)] for the kept test problems, in the order realblt_quadrant.py used."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(q.MODEL)
    rng = np.random.default_rng(0)
    for l in list(open(q.GSM_TRAIN))[:300]:
        q.build(json.loads(l), rng)                                 # consume the rng exactly as the run did
    kept = []
    for l in list(open(q.GSM))[:300]:
        b, units = q.build(json.loads(l), rng)
        n = len(tok(b.decode()).input_ids)
        if n <= q.MAX_TOKENS and n == len(b) + 1:
            kept.append((b, units))
    return kept


def qwen_bits(kept, rows):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    want = {}                                                       # (problem, order) -> row index for word_new
    order = {}
    for j, r in enumerate(rows):
        if r["kind"] == "word_new":
            o = order.get(r["problem"], 0); order[r["problem"]] = o + 1
            want[(r["problem"], o)] = j
    out = np.full(len(rows), np.nan)
    for i, (b, units) in enumerate(kept):
        text = b.decode()
        enc = tok(text, return_offsets_mapping=True); ids, offs = enc["input_ids"], enc["offset_mapping"]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids], device=dev)).logits[0].float(), -1).cpu()
        s = np.r_[np.nan, (-lp[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0] / np.log(2)).numpy()]
        for o, (a, e) in enumerate(units["word_new"]):
            j = want.get((i, o))
            if j is None:
                continue
            assert rows[j]["n_bytes"] == e - a, (i, o)
            ca, ce = len(b[:a].decode()), len(b[:e].decode())
            toks = [t for t, (x, y) in enumerate(offs) if y > ca and x < ce and t > 0]
            out[j] = float(np.nansum(s[toks]) / (e - a))
        if (i + 1) % 50 == 0:
            print(f"  qwen {i + 1}/{len(kept)}", flush=True)
    return out


def main():
    d = json.load(open(os.path.join(HERE, "results", "blt_quadrant.json")))
    rows = d["units"]
    P = np.array([r["problem"] for r in rows]); K = np.array([r["kind"] for r in rows])
    gain = np.array([r["bits_without"] - r["bits_forced"] for r in rows])
    nb = np.array([r["n_bytes"] for r in rows], float)
    rng = np.random.default_rng(0)
    res = {"P1": {}, "P2": {}}
    gc = gain[K == "computed"].mean()
    verdicts = []
    for k in COPYLIKE:
        diff, ci = boot_diff(P, gain, K == "computed", K == k, rng)
        m, mci = boot_mean(P, gain, K == k, rng)
        below25 = mci[1] < 0.25 * gc; fail50 = mci[0] >= 0.5 * gc
        res["P1"][k] = {"gain": m, "gain_ci": mci, "computed_minus_k": diff, "diff_ci": ci,
                        "share_of_computed": m / gc, "upper_below_25pct": bool(below25), "lower_at_or_above_50pct": bool(fail50),
                        "gain_per_byte": float((gain / nb)[K == k].mean())}
        verdicts.append((ci[0] > 0, below25, fail50))
    res["P1"]["computed"] = {"gain": float(gc), "gain_per_byte": float((gain / nb)[K == "computed"].mean())}
    a_ok = all(v[0] for v in verdicts); b_ok = all(v[1] for v in verdicts); fails = any(v[2] for v in verdicts)
    res["P1"]["verdict"] = "FAILS" if fails else ("HOLDS" if a_ok and b_ok else "INCONCLUSIVE")
    res["P1"]["verdict_detail"] = {"a_all_differences_above_0": a_ok, "b_all_upper_bounds_below_25pct": b_ok}
    # P2 with the independent measure
    qb = qwen_bits(rebuild(), rows)
    m = (K == "word_new") & np.isfinite(qb)
    t = np.quantile(qb[m], [1 / 3, 2 / 3])
    thirds = {"easy": m & (qb <= t[0]), "middle": m & (qb > t[0]) & (qb <= t[1]), "hard": m & (qb > t[1])}
    for name, mm in thirds.items():
        g, gci = boot_mean(P, gain, mm, rng)
        res["P2"][name] = {"n": int(mm.sum()), "qwen_bits_per_byte": float(qb[mm].mean()), "gain": g, "gain_ci": gci}
    diff, ci = boot_diff(P, gain, thirds["easy"], thirds["hard"], rng)
    res["P2"]["easy_minus_hard"] = {"diff": diff, "ci": ci}
    res["P2"]["verdict"] = "HOLDS" if ci[0] > 0 else ("FAILS" if ci[1] <= 0 else "INCONCLUSIVE")
    bfull = np.array([r["bits_full"] / r["n_bytes"] for r in rows])
    rk = lambda x: np.argsort(np.argsort(x))
    res["P2"]["spearman_qwen_vs_blt_full"] = float(np.corrcoef(rk(qb[m]), rk(bfull[m]))[0, 1])
    json.dump(res, open(os.path.join(HERE, "results", "blt_quadrant_verdicts.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
