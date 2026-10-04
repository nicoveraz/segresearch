"""Beyond patching (issue #22): does confidence-driven compute skip computed outputs too?

Many speed-ups spend less compute where a cheap predictor is confident: speculative decoding (a small draft model
proposes, the large model verifies), confidence-based early exit, adaptive depth. The byte-patching result says
entropy misses positions whose TYPE is predictable but whose VALUE must be computed. This tests the same split with
two public token models on GSM8K worked solutions, teacher forced:

    draft   Qwen2.5-0.5B          target   Qwen2.5-1.5B

Tokens are grouped by what they are: computed results (the number right after '= '), copied numbers (a number that
already appeared earlier in the problem or solution), other numbers, and text. Per group:

    agree       draft's top token = target's top token  (greedy speculative decoding's acceptance rate)
    conf>0.9    share of tokens where the draft is confident (top probability > 0.9)
    wrong|conf  among confident draft tokens, share where the draft is wrong vs the true text: the errors a
                confidence-based early exit (accept the cheap prediction when confident) would let through

    <env>/bin/python confidence_exit.py [N_PROBLEMS] [gsm8k|trace]     (trace: generated programs, values not memorizable)
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import registry

GSM = os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl")
DRAFT, TARGET = "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def char_groups(text, q_len):
    """Group label per character: 'computed' (number after '= '), 'copied' (number seen earlier), 'number', 'text'."""
    g = ["text"] * len(text); seen = set()
    for m in NUM.finditer(text):
        v = m.group().replace(",", "")
        label = "computed" if m.start() >= q_len and text[max(0, m.start() - 3):m.start()].rstrip("$").endswith("= ") \
            else ("copied" if v in seen else "number")
        for i in range(m.start(), m.end()):
            g[i] = label
        seen.add(v)
    return g


def texts(n, data):
    """(text, scored-from offset, per-char groups) for GSM8K solutions or generated program traces."""
    if data == "trace":
        import random, realblt_reason
        rng = random.Random(2); out = []
        for _ in range(n):
            text, spans = realblt_reason.trace_problem(rng)
            g = ["text"] * len(text)
            for kind, a, b in spans:
                g[a:b] = [kind] * (b - a)
            out.append((text, 0, g))
        return out
    out = []
    for p in [json.loads(l) for l in open(GSM)][:n]:
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        q = p["question"] + "\n"
        text = q + sol + "\nThe final answer is " + ans.strip()
        out.append((text, len(q), char_groups(text, len(q))))
    return out


def main(n, data="gsm8k"):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TARGET)                            # the two models share a tokenizer
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.bfloat16).to(dev).eval()
    target = AutoModelForCausalLM.from_pretrained(TARGET, dtype=torch.bfloat16).to(dev).eval()
    stats = defaultdict(lambda: {"agree": [], "conf": [], "wrong_conf": [], "draft_right": [], "target_right": []})
    for text, q0, groups in texts(n, data):
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc.input_ids.to(dev); offs = enc.offset_mapping[0].tolist()
        with torch.no_grad():
            pd = torch.softmax(draft(ids).logits[0].float(), -1); pt = target(ids).logits[0].float()
        dconf, dtop = pd.max(-1); ttop = pt.argmax(-1)
        for i in range(1, ids.shape[1]):                                    # token i predicted at position i-1
            s, e = offs[i]
            if s >= e or s < q0:                                            # score the solution only
                continue
            grp = groups[s] if groups[s] != "text" or not text[s:e].strip() else "text"
            truth = ids[0, i].item(); dt = dtop[i - 1].item()
            st = stats[grp]
            st["agree"].append(dt == ttop[i - 1].item()); st["draft_right"].append(dt == truth)
            st["target_right"].append(ttop[i - 1].item() == truth)
            c = dconf[i - 1].item() > 0.9; st["conf"].append(c)
            if c:
                st["wrong_conf"].append(dt != truth)
    print(f"\n{DRAFT} (draft) vs {TARGET} (target), {n} {data} texts, teacher forced")
    for grp in ("computed", "copied", "copy", "number", "text"):
        s = stats[grp]
        if not s["agree"]:
            continue
        registry.emit("confidence_exit", f"RESULT {grp:9s} tokens {len(s['agree'])} | agree {np.mean(s['agree']):5.1%} | "
                      f"draft right {np.mean(s['draft_right']):5.1%} | target right {np.mean(s['target_right']):5.1%} | "
                      f"conf>0.9 {np.mean(s['conf']):5.1%} | wrong|conf {np.mean(s['wrong_conf']) if s['wrong_conf'] else float('nan'):5.1%}",
                      experiment="confidence_exit", data=data, draft=DRAFT, target=TARGET, n_problems=n)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, sys.argv[2] if len(sys.argv) > 2 else "gsm8k")
