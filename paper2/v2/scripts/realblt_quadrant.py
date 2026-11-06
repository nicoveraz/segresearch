"""Quadrant test of the general rule (general_rule.md, P1-P2): what does a patch start buy, by kind of content,
within the SAME texts? (local; follows realblt_compress.py; review 2, W1/W3)

Text: GSM8K problems in the stripped format, with one random 4-digit ticket number inserted twice: "Ticket NNNN. "
at the start of the solution (new and unpredictable) and "Ticket NNNN closed." before the final line (a copy of
it). Unit kinds (starts in the solution only):
    computed   number right after '= '                         new, learnable (if BLT-1B can compute it)
    id_new     the first ticket number                          new, not learnable
    id_copy    the second ticket number                         a copy of unpredictable text
    copied     other number that occurs earlier in the text     a copy
    word_new   word whose whole token does not occur earlier    new; learnable or not (split by full-path loss)
    word_rep   word that occurs earlier                         a copy
    final      the number after "The final answer is "          a copy that ends a segment
Value of a patch start at a unit (the gain): start from BLT's entropy layout at a 10% patch rate (threshold fitted
on train problems with the same insertion); for each kind K, layout "without" = that layout with every start at a
K unit's first byte removed, layout "with" = with one added at each. Gain = per-unit bits without minus with (and
exact match), the rest of the layout held fixed. (A budget-matched "forced" comparison, as in paper 1, confounds the
value of a start with whether entropy already placed one: the ticket and the final answer are almost always
covered by entropy, so forcing changes nothing there; smoke test of 2026-10-23.) BLT-1B's default layout (~25%) is
reported as the full path. Intervals: bootstrap over problems.

Predictions (written before the run): gain(computed) is the largest; gain(id_new) ~ 0; gains at copies (copied,
id_copy, word_rep) small; among new words, the gain is concentrated in those the full path predicts well (P2).

    .venv-blt/bin/python realblt_quadrant.py [N_PROBLEMS=300] [N_TRAIN=300]
"""
import json
import os
import re
import sys

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
from realblt_budget import GSM, GSM_TRAIN, MAX_TOKENS, MODEL, lengths_from_starts

R = 0.10
KINDS = ("computed", "id_new", "id_copy", "copied", "word_new", "word_rep", "final")
NUM = re.compile(rb"\d[\d,]*(?:\.\d+)?")
TOK = re.compile(rb"[A-Za-z]+|[0-9]+(?:[.,][0-9]+)*")


def build(p, rng):
    """(bytes, {kind: [(start_byte, end_byte)]}) for one problem with the ticket inserted twice."""
    sol, ans = p["answer"].rsplit("#### ", 1)
    sol = re.sub(r"<<[^>]*>>", "", sol).strip(); ans = ans.strip()
    while True:
        tid = str(rng.integers(1000, 10000))
        if tid not in p["question"] + sol + ans:
            break
    head = p["question"] + "\n"
    text = head + f"Ticket {tid}. " + sol + f"\nTicket {tid} closed.\nThe final answer is " + ans
    b = text.encode()
    s0 = len(head.encode())
    f0 = len(b) - len(ans.encode())
    units = {k: [] for k in KINDS}
    ids_at = [m.start() for m in re.finditer(re.escape(tid.encode()), b)]
    tail = b.rindex(f"\nTicket {tid} closed.".encode())
    seen = set()
    for m in TOK.finditer(b):                                   # tokens before the solution count as seen context
        a, e = m.span(); w = m.group(0)
        if a >= s0:
            if w[:1].isdigit():
                if a == f0:
                    units["final"].append((a, e))
                elif a in ids_at:
                    units["id_new" if a == ids_at[0] else "id_copy"].append((a, e))
                elif re.search(rb"= \$?$", b[max(0, a - 3):a]):
                    units["computed"].append((a, e))
                elif w in seen:
                    units["copied"].append((a, e))
            elif a > 0 and b[a - 1:a] == b" " and len(w) > 1 and a < tail:      # not the inserted / template lines
                units["word_new" if w not in seen else "word_rep"].append((a, e))
        seen.add(w)
    return b, units


def main(n_problems=300, n_train=300):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    rng = np.random.default_rng(0)

    def prep(p):
        b, units = build(p, rng)
        ids = tok(b.decode(), return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] > MAX_TOKENS or ids.shape[1] != len(b) + 1:
            return None
        with torch.no_grad():
            ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                            max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy(); n = ids.shape[1]
        s = np.full(n, -np.inf); s[2:] = e[1:n - 1]               # token t starts a patch from the entropy at t-1
        forced = {k: [a + 1 for a, _ in v] for k, v in units.items()}   # token = byte + 1
        return b, ids, units, {"entropy": s}, forced

    train = [x for x in (prep(json.loads(l)) for l in list(open(GSM_TRAIN))[:n_train]) if x]
    test = [x for x in (prep(json.loads(l)) for l in list(open(GSM))[:n_problems]) if x]
    fit = blt_layouts.Fit((R,))
    for _, _, _, sig, forced in train:
        fit.add(sig, forced)
    fit.done()
    print(f"{len(train)} train / {len(test)} test; entropy train rate {fit.rate('entropy', R):.3f}", flush=True)

    rows = []                                                       # one row per unit
    rates = {"entropy": [], "covered_by_entropy": {k: [] for k in KINDS}}
    for i, (b, ids, units, sig, forced) in enumerate(test):
        n = ids.shape[1]

        def score(lengths):
            with torch.no_grad():
                out = model(input_ids=ids, patch_lengths=lengths, use_cache=False) if lengths is not None \
                    else model(input_ids=ids, use_cache=False)
            lp = torch.log_softmax(out.logits[0].float(), -1)
            res = {}
            for k, v in units.items():
                for a, e in v:
                    s_, e_ = a + 1, e + 1
                    tgt = ids[0, s_:e_]; pred = lp[s_ - 1:e_ - 1]
                    res[(k, a)] = (float(-pred.gather(1, tgt[:, None]).sum() / np.log(2)), bool((pred.argmax(-1) == tgt).all()))
            return res

        full = score(None)
        st = set(blt_layouts.layout("entropy", R, sig, fit).tolist())
        rates["entropy"].append(len(st) / n)
        ent = score(lengths_from_starts(sorted(st), n).to(dev))
        for k, v in units.items():
            if not v:
                continue
            f = set(forced[k])
            rates["covered_by_entropy"][k] += [t in st for t in f]
            wo = score(lengths_from_starts(sorted(st - f), n).to(dev))
            wi = score(lengths_from_starts(sorted(st | f), n).to(dev))
            for a, e in v:
                rows.append({"problem": i, "kind": k, "n_bytes": e - a, "covered": (a + 1) in st,
                             "bits_full": full[(k, a)][0], "ok_full": full[(k, a)][1],
                             "bits_entropy": ent[(k, a)][0], "ok_entropy": ent[(k, a)][1],
                             "bits_forced": wi[(k, a)][0], "ok_forced": wi[(k, a)][1],
                             "bits_without": wo[(k, a)][0], "ok_without": wo[(k, a)][1]})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} problems", flush=True)

    res = {"n_problems": len(test), "rate_entropy": float(np.mean(rates["entropy"])),
           "covered_by_entropy": {k: float(np.mean(v)) for k, v in rates["covered_by_entropy"].items() if v},
           "by_kind": {}, "word_new_by_learnability": {}}
    P = np.array([r["problem"] for r in rows]); K = np.array([r["kind"] for r in rows])
    gain = np.array([r["bits_without"] - r["bits_forced"] for r in rows])      # value of a start at the unit
    okg = np.array([float(r["ok_forced"]) - float(r["ok_without"]) for r in rows])
    bfull = np.array([r["bits_full"] / r["n_bytes"] for r in rows])
    brng = np.random.default_rng(1)
    probs = np.unique(P)

    def boot(mask, x):
        by = {p: x[mask & (P == p)] for p in probs}
        by = {p: v for p, v in by.items() if len(v)}
        keys = list(by)
        stats = []
        for _ in range(1000):
            pick = brng.choice(keys, len(keys))
            stats.append(np.concatenate([by[p] for p in pick]).mean())
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    for k in KINDS:
        m = K == k
        if not m.any():
            continue
        res["by_kind"][k] = {"n": int(m.sum()), "gain_bits": float(gain[m].mean()), "gain_bits_ci": boot(m, gain),
                             "gain_exact": float(okg[m].mean()), "gain_exact_ci": boot(m, okg),
                             "exact_full": float(np.mean([r["ok_full"] for r, mm in zip(rows, m) if mm])),
                             "exact_entropy": float(np.mean([r["ok_entropy"] for r, mm in zip(rows, m) if mm])),
                             "exact_with": float(np.mean([r["ok_forced"] for r, mm in zip(rows, m) if mm])),
                             "exact_without": float(np.mean([r["ok_without"] for r, mm in zip(rows, m) if mm])),
                             "bits_per_unit_without": float(np.mean([r["bits_without"] for r, mm in zip(rows, m) if mm])),
                             "bits_per_byte_full": float(bfull[m].mean())}
    m = K == "word_new"                                             # P2: split new words by full-path loss
    if m.any():
        q = np.quantile(bfull[m], [1 / 3, 2 / 3])
        for name, mm in (("easy (full-path bits/byte, lowest third)", m & (bfull <= q[0])),
                         ("middle third", m & (bfull > q[0]) & (bfull <= q[1])),
                         ("hard (highest third)", m & (bfull > q[1]))):
            res["word_new_by_learnability"][name] = {"n": int(mm.sum()), "gain_bits": float(gain[mm].mean()),
                                                     "gain_bits_ci": boot(mm, gain), "gain_exact": float(okg[mm].mean()),
                                                     "bits_per_byte_full": float(bfull[mm].mean())}
    out = os.path.join(_seg.HERE, "results", "blt_quadrant.json")
    json.dump({**res, "units": rows}, open(out, "w"))
    for k, v in res["by_kind"].items():
        print(f"RESULT {k:9s} n {v['n']:5d} | gain bits {v['gain_bits']:+.3f} [{v['gain_bits_ci'][0]:+.3f}, {v['gain_bits_ci'][1]:+.3f}]"
              f" | exact gain {v['gain_exact']:+.3f} (without {v['exact_without']:.3f}, with {v['exact_with']:.3f}, full {v['exact_full']:.3f})"
              f" | covered by entropy {res['covered_by_entropy'].get(k, float('nan')):.2f}", flush=True)
    for k, v in res["word_new_by_learnability"].items():
        print(f"RESULT word_new {k}: n {v['n']} gain bits {v['gain_bits']:+.3f} [{v['gain_bits_ci'][0]:+.3f}, {v['gain_bits_ci'][1]:+.3f}]"
              f" exact {v['gain_exact']:+.3f}", flush=True)
    print("entropy rate", res["rate_entropy"])


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
