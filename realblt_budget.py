"""Tight-budget test on Meta's trained BLT-1B: under a tight patch budget, does forcing a boundary at each
computed result (the number right after '= ') improve how well BLT-1B predicts those results?

The ranking analysis predicted it: at 6-15% budgets BLT-1B's own entropy covers in-line results after '= '
less often than ordinary word starts, because a number's TYPE is predictable there even though its VALUE must
be computed. Patch boundaries are changed at inference time (no training). Layouts, at equal patch counts:

    default        BLT's own entropy patches (about 30% of bytes)
    entropy@R      the top R of positions by BLT-1B's own entropy (R = 15%, 10%)
    results@R      the same count, with a boundary forced right after every '= ' (replacing the lowest-ranked)

Scored on every in-line computed result (the number right after '= ' in the worked solution) and on the final
answer: bits per byte (teacher forced) and exact match (every byte is the model's top choice).
BLT-1B was trained at about 30%, so tighter budgets are out of distribution for every layout; compare layouts
at equal R.

    <env>/bin/python realblt_budget.py [N_PROBLEMS]
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, BltForCausalLM

MODEL = "itazap/blt-1b-hf"
GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
BUDGETS = (0.15, 0.10)
MAX_TOKENS = 1000


def lengths_from_starts(starts, n):
    s = sorted(set(int(x) for x in starts if 0 <= x < n))
    return torch.tensor([[b - a for a, b in zip(s, s[1:] + [n])]], dtype=torch.long)


def main(n_problems):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = BltForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    layouts = ["default"] + [f"{k}@{int(r * 100)}" for r in BUDGETS for k in ("entropy", "results")]
    stats = {l: {"res_bits": [], "res_exact": [], "fin_bits": [], "fin_exact": [], "rate": []} for l in layouts}
    covered = {l: [] for l in layouts}
    t0 = time.time(); used = 0
    for i, p in enumerate([json.loads(l) for l in open(GSM)][:n_problems]):
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip(); ans = ans.strip()
        text = p["question"] + "\n" + sol + "\nThe final answer is " + ans
        b = text.encode()
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        n = ids.shape[1]
        if n > MAX_TOKENS or n != len(b) + 1:
            continue
        used += 1
        # spans as token positions (token t = byte t-1): in-line results right after '= ', and the final answer
        spans = []
        for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b):
            spans.append(("res", m.start() + 2 + 1, m.end(1) + 1))          # from the byte after '= ' (a '$' or digit)
        f0 = len(b) - len(ans.encode())
        spans.append(("fin", f0 + 1, len(b) + 1))
        res_starts = np.array([s for k, s, _ in spans if k == "res"], int)
        with torch.no_grad():
            ent, default_len, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]                # BLT: token t starts a patch if entropy[t-1] is high
        default = np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]
        lay = {"default": default}
        for r in BUDGETS:
            k = max(int(round(r * n)) - 2, 1)
            top = np.argsort(-score)[:k]
            lay[f"entropy@{int(r * 100)}"] = np.r_[0, 1, top]
            forced = [s for s in res_starts if s >= 2]
            keep = [t for t in np.argsort(-score) if t not in set(forced)][:max(k - len(forced), 0)]
            lay[f"results@{int(r * 100)}"] = np.r_[0, 1, forced, keep]
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
    print(f"\nBLT-1B, {used} GSM8K test problems; {len(stats['default']['res_exact'])} in-line results, {len(stats['default']['fin_exact'])} final answers")
    for name in layouts:
        s = stats[name]
        print(f"RESULT {name:11s} patch rate {np.mean(s['rate']):.3f} | results covered {np.mean(covered[name]):5.1%} | "
              f"in-line results: {np.mean(s['res_bits']):.3f} bits, exact {np.mean(s['res_exact']):5.1%} | "
              f"final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
