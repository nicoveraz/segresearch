"""Real-model check: does Meta's trained BLT-1B answer better with a patch boundary at the answer start?

No training: the patch boundaries of a trained byte model can be changed at inference time (BltModel accepts
patch_lengths). For GSM8K test problems the model reads the question and the worked solution up to '#### ' and
is scored on the final answer, under four patch layouts:

    default      BLT's own entropy patches (global threshold on its entropy model)
    +answer      default plus a boundary at the first answer byte (decided from the preceding phrase)
    -answer      default with any boundary at the first answer byte removed
    jump         patches where the entropy model's entropy rises most, with the same number of patches as default

Metrics: bits per answer byte (teacher forced) and exact match (every answer byte is the model's top choice).
Runs in its own environment (PyTorch + transformers), not the project's MLX one:

    <env>/bin/python realblt.py [N_PROBLEMS]
"""
import json
import os
import re
import sys
import time

import numpy as np
import blt_load
import registry
import torch
from transformers import AutoTokenizer, BltForCausalLM

MODEL = "itazap/blt-1b-hf"
GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")


def lengths_from_starts(starts, n):
    s = sorted(set(int(x) for x in starts if 0 <= x < n))
    return torch.tensor([[b - a for a, b in zip(s, s[1:] + [n])]], dtype=torch.long)


def main(n_problems):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    problems = [json.loads(l) for l in open(GSM)][:n_problems]
    res = {k: {"bits": [], "exact": []} for k in ("default", "+answer", "-answer", "jump")}
    had_boundary = []
    t0 = time.time()
    for i, p in enumerate(problems):
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()                       # calculator annotations are a GSM8K artifact
        ans = ans.strip()
        # score the answer as the solution last wrote it (thousands separators, a leading $)
        pat = "".join(c + ",?" if c.isdigit() else re.escape(c) for c in ans).rstrip("?").rstrip(",")
        hits = list(re.finditer(r"(\$?)(" + pat + r")(?![\d])", sol))
        dollar = ""
        if hits:
            dollar, ans = hits[-1].group(1), hits[-1].group(2)
        prefix = p["question"] + "\n" + sol + "\nThe final answer is " + dollar
        ids = tok(prefix + ans, return_tensors="pt").input_ids.to(dev)
        n_pre = tok(prefix, return_tensors="pt").input_ids.shape[1]
        n = ids.shape[1]; a0 = n_pre                                         # first answer token
        with torch.no_grad():
            ent, default_len, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
        starts = np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]
        starts = starts[starts < n]
        had_boundary.append(a0 in set(starts.tolist()))
        e = ent[0].float().cpu().numpy()
        # jump rule with the same number of patches: entropy rise before token t (t >= 2, as BLT always starts 0, 1)
        rise = np.full(n, -np.inf); rise[2:] = e[1:n - 1] - e[0:n - 2]
        k = max(len(starts) - 2, 0)
        jump = np.r_[0, 1, np.sort(np.argsort(-rise)[:k])]
        layouts = {"default": starts, "+answer": np.r_[starts, a0],
                   "-answer": starts[starts != a0], "jump": jump}
        for name, st in layouts.items():
            pl = lengths_from_starts(st, n).to(dev)
            with torch.no_grad():
                logits = model(input_ids=ids, patch_lengths=pl, use_cache=False).logits[0].float()
            lp = torch.log_softmax(logits, -1)
            tgt = ids[0, a0:]                                                # answer tokens, predicted from position t-1
            pred = lp[a0 - 1:n - 1]
            res[name]["bits"].append(float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)))
            res[name]["exact"].append(bool((pred.argmax(-1) == tgt).all()))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(problems)} problems, {time.time() - t0:.0f}s", flush=True)
    print(f"\nBLT-1B on {len(problems)} GSM8K test problems (final answer, given the worked solution)")
    print(f"default patches already start at the answer in {np.mean(had_boundary):.0%} of problems")
    for name, r in res.items():
        registry.emit("realblt", f"RESULT layout={name:8s} bits/answer byte {np.mean(r['bits']):.3f} | exact match {np.mean(r['exact']):.1%}", window=blt_load.WINDOW, experiment="blt_answer_boundary", n_problems=len(problems))
    m = np.array(had_boundary)
    for name in ("default", "+answer", "-answer"):
        r = res[name]
        print(f"  {name:8s} where default had NO boundary at the answer: exact {np.mean(np.array(r['exact'])[~m]):.1%}, "
              f"bits {np.mean(np.array(r['bits'])[~m]):.3f} (n={int((~m).sum())})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
