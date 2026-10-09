"""Does a learned chunker (H-Net) have the same blind spot as entropy patching? (#21)

H-Net (Hwang, Wang and Gu, 2025) learns its byte-chunk boundaries end to end: a routing module starts a new chunk
where adjacent encoder states are dissimilar. This runs the public checkpoints on GSM8K test solutions, written as
in realblt_budget.py (calculator annotations removed, "The final answer is N" appended), and measures, at H-Net's
native chunk rate, how often a chunk starts at:

    res        the first byte of each in-line computed result (right after '= ')
    fin        the first byte of the final answer
    num_other  the first byte of other numbers (mostly copied from the question)
    word       other word starts (alphanumeric after a space or newline)
    all        every byte (the overall chunk rate)

for every stage of the hierarchy (stage 0 = byte -> chunk; stage 1 of a 2-stage model = chunk -> outer chunk,
mapped back to bytes). Also reported: H-Net's own next-byte entropy at those positions (how predictable the byte
is to the model) and teacher-forced exact match on results and final answers, split by whether a stage-0 chunk
starts there (correlational only: boundaries are not changed).

BLT's entropy patcher covers in-line results less often than word starts under tight budgets; if H-Net does too,
the blind spot is about uncertainty-driven allocation in general, not BLT's rule.

    /workspace/hnetenv/bin/python scale/cloud/hnet_check.py [N_PROBLEMS]     (after scale/cloud/hnet_setup.sh)
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import torch

W = os.environ.get("WORK", "/workspace")
OUT = os.environ.get("HNET_OUT", f"{W}/work/logs/hnet_check.json")
MODELS = os.environ.get("HNET_MODELS", "hnet_2stage_XL hnet_1stage_XL").split()
GSM_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
MAX_BYTES = 1000


def problems(n):
    path = f"{W}/gsm8k_test.jsonl"
    if not os.path.exists(path):
        urllib.request.urlretrieve(GSM_URL, path + ".tmp"); os.replace(path + ".tmp", path)
    out = []
    for line in open(path):
        p = json.loads(line)
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip(); ans = ans.strip()
        b = (p["question"] + "\n" + sol + "\nThe final answer is " + ans).encode()
        if len(b) > MAX_BYTES:
            continue
        res = [(m.start() + 2, m.end(1)) for m in re.finditer(rb"= \$?(\d[\d,]*(?:\.\d+)?)", b)]
        fin = (len(b) - len(ans.encode()), len(b))
        res_set = {s for s, _ in res} | {fin[0]}
        num = [m.start() for m in re.finditer(rb"(?<![\d.,$])\$?\d", b) if m.start() not in res_set]
        skip = res_set | set(num)
        word = [i for i in range(1, len(b)) if b[i - 1] in b" \n" and chr(b[i]).isalnum() and i not in skip]
        out.append({"b": b, "res": res, "fin": fin, "num_other": num, "word": word})
        if len(out) == n:
            break
    return out


def load(name):
    sys.path.insert(0, f"{W}/hnet")
    from generate import load_from_pretrained
    return load_from_pretrained(f"{W}/hnet_ckpt/{name}.pt", f"{W}/hnet/configs/{name}.json")


def run(model, probs):
    BOS = 254
    stats = {}
    add = lambda k, v: stats.setdefault(k, []).append(v)
    t0 = time.time()
    for i, p in enumerate(probs):
        b = p["b"]; n = len(b)
        ids = torch.tensor([[BOS] + list(b)], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            out = model(ids)                                   # packed mode (mask=None), batch of one
        logits = out.logits[0].float()
        logp = torch.log_softmax(logits, -1)
        H = (-(logp.exp() * logp).sum(-1) / np.log(2)).cpu().numpy()   # H[t]: entropy of the prediction of input t+1
        pred = logits.argmax(-1).cpu().numpy()
        # stage 0: boundary_mask over input positions (input t = byte t-1); deeper stages are over the previous
        # stage's chunks, mapped back to bytes through the chunk start positions
        starts = np.arange(n + 1)
        stage_starts = []
        for bp in out.bpred_output:
            m = bp.boundary_mask.reshape(-1).cpu().numpy().astype(bool)[:len(starts)]
            starts = starts[m]
            stage_starts.append(set(int(t) - 1 for t in starts if t >= 1))   # byte index
        targets = {"res": [s for s, _ in p["res"]], "fin": [p["fin"][0]], "num_other": p["num_other"],
                   "word": p["word"], "all": list(range(n))}
        for st, S in enumerate(stage_starts):
            for k, pos in targets.items():
                for x in pos:
                    add(f"s{st}_{k}", x in S)
        for k, pos in targets.items():
            for x in pos:
                add(f"H_{k}", float(H[x]))                     # entropy of the prediction of byte x (input x+1)
        for kind, spans in (("res", p["res"]), ("fin", [p["fin"]])):
            for s, e in spans:
                ok = bool((pred[s:e] == np.frombuffer(b[s:e], np.uint8)).all())   # logits at input t predict byte t
                add(f"exact_{kind}", ok)
                add(f"exact_{kind}_{'covered' if s in stage_starts[0] else 'uncovered'}", ok)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} problems, {time.time() - t0:.0f}s", flush=True)
    return {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(stats.items())}


def main(n):
    probs = problems(n)
    print(f"{len(probs)} GSM8K test problems (<= {MAX_BYTES} bytes)", flush=True)
    res = {"n_problems": len(probs), "hnet_commit": open(f"{W}/hnet_commit.txt").read().strip()
           if os.path.exists(f"{W}/hnet_commit.txt") else None, "models": {}}
    for name in MODELS:
        if not os.path.exists(f"{W}/hnet_ckpt/{name}.pt"):
            print(f"skip {name}: no checkpoint"); continue
        model = load(name)
        print(f"== {name}", flush=True)
        r = run(model, probs)
        res["models"][name] = r
        stages = sorted({k.split("_")[0] for k in r if k.startswith("s") and k[1].isdigit()})
        for st in stages:
            print(f"  stage {st[1:]}: chunk rate {r[f'{st}_all']['mean']:.3f} | chunk starts at: computed results "
                  f"{r[f'{st}_res']['mean']:.1%}, final answers {r[f'{st}_fin']['mean']:.1%}, other numbers "
                  f"{r[f'{st}_num_other']['mean']:.1%}, word starts {r[f'{st}_word']['mean']:.1%}")
        print(f"  next-byte entropy (bits): results {r['H_res']['mean']:.2f}, final {r['H_fin']['mean']:.2f}, "
              f"other numbers {r['H_num_other']['mean']:.2f}, words {r['H_word']['mean']:.2f}, all {r['H_all']['mean']:.2f}")
        cov, unc = r.get("exact_res_covered", {}), r.get("exact_res_uncovered", {})
        print(f"  exact match: results {r['exact_res']['mean']:.1%} (chunk start there {cov.get('mean', float('nan')):.1%}"
              f" of {cov.get('n', 0)}, not {unc.get('mean', float('nan')):.1%} of {unc.get('n', 0)}), "
              f"final {r['exact_fin']['mean']:.1%}", flush=True)
        del model; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1319)
