"""Tight-budget test on Meta's trained BLT-1B: under a tight patch budget, does forcing a boundary at each
computed result (the number right after '= ') improve how well BLT-1B predicts those results?

The ranking analysis predicted it: at 6-15% budgets BLT-1B's own entropy covers in-line results after '= '
less often than ordinary word starts, because a number's TYPE is predictable there even though its VALUE must
be computed. Patch boundaries are changed at inference time (no training). Layouts at budget R (15%, 10%):

    default        BLT's own entropy patches (about 30% of bytes)
    entropy@R      positions with the highest BLT-1B entropy
    results@R      a boundary forced right after every '= ', the rest by entropy (label-based reference)
    dep@R          the top R by a label-free "boundary dependence" score: how much BLT-1B's own loss on the next
                   4 bytes rises when its patches are cut from ~30% to 10%, averaged per preceding 2-byte context
                   on GSM8K TRAIN problems (a causal lookup; fitted by blt_screens/dependence_screen.py)
    entdep@R       the top R by entropy + dependence (each standardized)

How the budget is applied (blt_layouts.py, SEGR_BLT_THRESH): by default one threshold per layout, fitted on
N_TRAIN GSM8K train problems so the mean patch rate there is R, then applied position by position (causal; the
rate varies per problem). SEGR_BLT_THRESH=problem reproduces the original top-R-per-problem selection (#26).

Scored on every in-line computed result (the number right after '= ' in the worked solution) and on the final
answer: bits per byte (teacher forced) and exact match (every byte is the model's top choice).
BLT-1B was trained at about 30%, so tighter budgets are out of distribution for every layout; compare layouts
at equal R.

    <env>/bin/python realblt_budget.py [N_PROBLEMS] [N_TRAIN]
    SEGR_BLT_THRESH=problem <env>/bin/python realblt_budget.py [N_PROBLEMS]
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, BltForCausalLM

import blt_layouts

MODEL = "itazap/blt-1b-hf"
GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
GSM_TRAIN = os.path.expanduser("~/.cache/segresearch-gsm8k/train.jsonl")
BUDGETS = (0.15, 0.10)
DEP = os.path.expanduser("~/.cache/segresearch-blt/dependence_table.npy")
MAX_TOKENS = 1000


def lengths_from_starts(starts, n):
    s = sorted(set(int(x) for x in starts if 0 <= x < n))
    return torch.tensor([[b - a for a, b in zip(s, s[1:] + [n])]], dtype=torch.long)


def problem(p, tok, dev):
    """GSM8K problem -> (bytes, token ids, spans, in-line result starts); spans and starts are token positions."""
    sol, ans = p["answer"].rsplit("#### ", 1)
    sol = re.sub(r"<<[^>]*>>", "", sol).strip(); ans = ans.strip()
    text = p["question"] + "\n" + sol + "\nThe final answer is " + ans
    b = text.encode()
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    n = ids.shape[1]
    if n > MAX_TOKENS or n != len(b) + 1:
        return None
    # spans as token positions (token t = byte t-1): in-line results right after '= ', and the final answer
    spans = []
    for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b):
        spans.append(("res", m.start() + 2 + 1, m.end(1) + 1))          # from the byte after '= ' (a '$' or digit)
    f0 = len(b) - len(ans.encode())
    spans.append(("fin", f0 + 1, len(b) + 1))
    return b, ids, spans, np.array([s for k, s, _ in spans if k == "res"], int)


def main(n_problems, n_train):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = BltForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    layouts = ["default"] + [f"{k}@{int(r * 100)}" for r in BUDGETS for k in ("entropy", "results", "dep", "entdep")]
    kind_of = {"entropy": "entropy", "results": "forced", "dep": "dep", "entdep": "entropy+dep"}
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]                # BLT: token t starts a patch if entropy[t-1] is high
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]   # token t = byte t-1
        return {"entropy": score, "dep": dep}, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    fit = None
    if blt_layouts.MODE == "train":
        # budget thresholds fitted once on TRAIN problems, applied position by position at test time (#26)
        fit = blt_layouts.Fit(BUDGETS)
        for p in [json.loads(l) for l in open(GSM_TRAIN)][:n_train]:
            pr = problem(p, tok, dev)
            if pr is not None:
                fit.add(signals(pr[0], pr[1])[0], pr[3])
        fit.done()
        print(f"thresholds fitted on {len(fit.sigs)} GSM8K train problems; train patch rates: " + ", ".join(
            f"{l} {fit.rate(kind_of[l.split('@')[0]], int(l.split('@')[1]) / 100):.3f}" for l in layouts[1:]), flush=True)
    stats = {l: {"res_bits": [], "res_exact": [], "fin_bits": [], "fin_exact": [], "rate": []} for l in layouts}
    covered = {l: [] for l in layouts}
    t0 = time.time(); used = 0
    for i, p in enumerate([json.loads(l) for l in open(GSM)][:n_problems]):
        pr = problem(p, tok, dev)
        if pr is None:
            continue
        b, ids, spans, res_starts = pr
        n = ids.shape[1]
        used += 1
        sig, default = signals(b, ids)
        lay = {"default": default}
        for name in layouts[1:]:
            k, R = name.split("@")
            lay[name] = blt_layouts.layout(kind_of[k], int(R) / 100, sig, fit, res_starts if k == "results" else ())
        for name, st in lay.items():
            st = np.array(sorted(set(int(x) for x in st if 0 <= x < n)))
            covered[name] += [s in set(st.tolist()) for s in res_starts]
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                             use_cache=False).logits[0].float(), -1)
            stats[name]["rate"].append(len(st) / n)
            for kind, s, e_ in spans:
                tgt = ids[0, s:e_]; pred = lp[s - 1:e_ - 1]
                bits = float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)); ex = bool((pred.argmax(-1) == tgt).all())
                stats[name][f"{kind}_bits"].append(bits); stats[name][f"{kind}_exact"].append(ex)
        if used % 25 == 0:
            print(f"  {used} problems, {time.time() - t0:.0f}s", flush=True)
    print(f"\nBLT-1B, {used} GSM8K test problems, budget thresholds: {blt_layouts.MODE}; {len(stats['default']['res_exact'])} in-line results, {len(stats['default']['fin_exact'])} final answers")
    for name in layouts:
        s = stats[name]
        print(f"RESULT {name:11s} patch rate {np.mean(s['rate']):.3f} | results covered {np.mean(covered[name]):5.1%} | "
              f"in-line results: {np.mean(s['res_bits']):.3f} bits, exact {np.mean(s['res_exact']):5.1%} | "
              f"final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
