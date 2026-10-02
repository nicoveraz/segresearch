"""End-to-end GSM8K accuracy on BLT-1B under tight patch budgets (issue #17).

The other BLT-1B scripts score each target given the TRUE text before it (teacher forcing). Here BLT-1B writes the
whole solution itself, by greedy decoding from a one-shot prompt, and is scored on the number after "The final
answer is". Patch boundaries are decided online, byte by byte, from the bytes generated so far: every layout is
causal (blt_layouts.py train mode: thresholds fitted on GSM8K train problems and applied position by position).

    default      BLT's own entropy patches
    entropy@R    BLT-1B's entropy, threshold fitted to R of bytes
    results@R    a boundary after every '= ' (hand-written), the rest by entropy
    entdep@R     entropy + boundary dependence (label-free)

No KV cache (patch lengths change as bytes arrive), so each byte costs a full forward pass.

    <env>/bin/python realblt_e2e.py [N_PROBLEMS] [N_TRAIN] [BUDGET]
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
import registry
from realblt_budget import DEP, GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem

MAX_NEW = 320
SHOT = 0                                     # GSM8K train problem used as the one-shot example


def clean(sol):
    return re.sub(r"<<[^>]*>>", "", sol).strip()


def number(s):
    m = re.search(r"-?\$?\d[\d,]*(?:\.\d+)?", s)
    return None if m is None else m.group().replace("$", "").replace(",", "").rstrip(".")


def main(n_problems, n_train, R):
    assert blt_layouts.MODE == "train", "online decoding needs causal (train-fitted) thresholds"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = BltForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf
    train = [json.loads(l) for l in open(GSM_TRAIN)]
    shot = train[SHOT]
    sol, ans = shot["answer"].rsplit("#### ", 1)
    prompt = f"Question: {shot['question']}\nAnswer: {clean(sol)}\nThe final answer is {ans.strip()}\n\n"

    def signals(b):
        ids = torch.tensor([[1] + [x + 4 for x in b]], device=dev)
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        return ids, {"entropy": score, "dep": dep}, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    def after_eq(b):
        """Token positions right after '= ' (token t = byte t-1): the hand-written result boundary."""
        return np.array([i + 2 + 1 for i in range(len(b) - 1) if b[i:i + 2] == b"= "], int)

    fit = blt_layouts.Fit((R,))
    for p in train[1:n_train + 1]:                                       # train problems other than the shot
        pr = problem(p, tok, dev)
        if pr is not None:
            fit.add(signals(pr[0])[1], pr[3])
    fit.done()
    tag = int(round(R * 100))
    layouts = {"default": None, f"entropy@{tag}": "entropy", f"results@{tag}": "forced", f"entdep@{tag}": "entropy+dep"}
    print(f"thresholds fitted on {len(fit.sigs)} train problems; one-shot prompt {len(prompt)} bytes", flush=True)

    def next_byte(b, kind):
        """Greedy next byte given bytes b, with patch starts from the layout (causal: decided from b alone)."""
        ids, sig, default = signals(b + b"\0")                           # the new byte's position must exist to place a start
        n = ids.shape[1] - 1
        if kind is None:
            st = default
        else:
            st = blt_layouts.layout(kind, R, sig, fit, after_eq(b + b"\0") if kind == "forced" else ())
        st = np.array(sorted({int(x) for x in st if 0 <= x < n + 1}))
        with torch.no_grad():
            lg = model(input_ids=ids, patch_lengths=lengths_from_starts(st, n + 1).to(dev), use_cache=False).logits[0, n - 1]
        return int(lg[4:260].float().argmax()), len(st) / (n + 1)

    res = {l: {"correct": [], "rate": [], "pred": []} for l in layouts}
    tests = [json.loads(l) for l in open(GSM)][:n_problems]
    t0 = time.time()
    for j, p in enumerate(tests):
        gold = number(p["answer"].rsplit("#### ", 1)[1])
        ctx = (prompt + f"Question: {p['question']}\nAnswer:").encode()
        for name, kind in layouts.items():
            gen, rates = b"", []
            for _ in range(MAX_NEW):
                nb, rate = next_byte(ctx + gen, kind)
                gen += bytes([nb]); rates.append(rate)
                if gen.endswith(b"\n\n") or re.search(rb"The final answer is [^\n]*\n", gen):
                    break
            m = re.search(rb"The final answer is ([^\n]*)", gen)
            pred = number(m.group(1).decode(errors="replace")) if m else None
            res[name]["correct"].append(pred is not None and gold is not None and float(pred) == float(gold))
            res[name]["rate"].append(float(np.mean(rates))); res[name]["pred"].append(pred)
        if (j + 1) % 5 == 0:
            print(f"  {j + 1}/{len(tests)} problems, {time.time() - t0:.0f}s; " +
                  ", ".join(f"{l} {np.mean(r['correct']):.0%}" for l, r in res.items()), flush=True)
    print(f"\nBLT-1B end-to-end on {len(tests)} GSM8K test problems (one-shot, greedy, online patching)")
    for name, r in res.items():
        registry.emit("realblt_e2e", f"RESULT {name:11s} patch rate {np.mean(r['rate']):.3f} | final: exact {np.mean(r['correct']):5.1%}",
                      experiment="blt_e2e", thresh=blt_layouts.MODE, n_problems=len(tests))
    registry.save_items(f"blt_e2e_{tag}", {"layouts": res, "gold": [number(p["answer"].rsplit("#### ", 1)[1]) for p in tests]},
                        experiment="blt_e2e", budget=R, n_problems=len(tests))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 100, int(a[1]) if len(a) > 1 else 300, float(a[2]) if len(a) > 2 else 0.15)
