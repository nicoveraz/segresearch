"""Dump BLT-1B patch starts for one GSM8K test solution under entropy@10 and entdep@10 (for paper figure 1).

Thresholds are fitted on GSM8K train problems exactly as in realblt_budget.py (train mode). Writes
results/fig1_layouts.json: the text, the patch-start byte offsets per layout, and the in-line result spans.

    <env>/bin/python blt_fig1_dump.py [TEST_INDEX]
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, BltForCausalLM

import blt_layouts
import blt_load
from realblt_budget import DEP, GSM, GSM_TRAIN, MODEL, problem


def main(idx):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        return {"entropy": score, "dep": dep}, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    fit = blt_layouts.Fit((0.10,))
    for p in [json.loads(l) for l in open(GSM_TRAIN)][:300]:
        pr = problem(p, tok, dev)
        if pr is not None:
            fit.add(signals(pr[0], pr[1])[0], pr[3])
    fit.done()
    p = [json.loads(l) for l in open(GSM)][idx]
    b, ids, spans, res_starts = problem(p, tok, dev)
    sig, default = signals(b, ids)
    to_bytes = lambda st: sorted(int(t) - 1 for t in st if t >= 1)              # token t = byte t-1
    out = {"index": idx, "mode": blt_layouts.MODE, "window": blt_load.WINDOW, "text": b.decode(), "default": to_bytes(default),
           "entropy@10": to_bytes(blt_layouts.layout("entropy", 0.10, sig, fit)),
           "entdep@10": to_bytes(blt_layouts.layout("entropy+dep", 0.10, sig, fit)),
           "results": [[int(s) - 1, int(e) - 1] for k, s, e in spans if k == "res"]}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"fig1_layouts{blt_load.SUFFIX}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path, {k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
