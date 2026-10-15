"""Does a learned chunker (H-Net) find structure in code? (local follow-up to hnet_check.py)

hnet_check.py found that H-Net's chunks start at in-line math results by themselves (no blind spot). Here the same
public checkpoints read Python source (the pod's standard library, chunks of <= 1000 bytes cut at line starts), and
we measure how often a chunk starts at:

    stmt       the first non-space byte of a line (statement start)
    rhs        right after '=', 'return ' or '(' (the value the code computes or passes)
    tok        other identifier starts (alphanumeric after a non-alphanumeric byte)
    mid        inside an identifier
    all        every byte (the chunk rate)

for every stage of the hierarchy, also counting a start at the preceding space (H-Net chunks like BPE tokens).
scale/novstruct.py and scale/learned_boundary.py measure the same categories for novelty and the learned predictors.

    /workspace/hnetenv/bin/python scale/cloud/hnet_code.py [N_CHUNKS]     (after scale/cloud/hnet_setup.sh)
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hnet_check import MODELS, W, load  # noqa: E402

OUT = os.environ.get("HNET_OUT", f"{W}/work/logs/hnet_code.json")
MAX_BYTES = 1000
ALNUM = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def chunks(n):
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "**", "*.py"), recursive=True))
    rng = np.random.default_rng(0); rng.shuffle(files)
    out = []
    for f in files:
        try:
            t = open(f, "rb").read()
        except OSError:
            continue
        lines = t.split(b"\n")
        if len(lines) < 40:
            continue
        s = int(rng.integers(0, len(lines) - 30))
        b = b""
        for ln in lines[s:]:
            if len(b) + len(ln) + 1 > MAX_BYTES:
                break
            b += ln + b"\n"
        if len(b) > 300:
            out.append(b)
        if len(out) >= n:
            break
    return out


def targets(b):
    n = len(b)
    stmt = [m.start(1) for m in re.finditer(rb"(?:^|\n)[ \t]*(\S)", b)]
    rhs = set()
    for tok in (b"=", b"return ", b"("):
        i = b.find(tok)
        while i >= 0:
            j = i + len(tok)
            while j < n and b[j] == 32:                      # the value itself, after any spaces
                j += 1
            if j < n:
                rhs.add(j)
            i = b.find(tok, i + 1)
    tok = [t for t in range(n) if b[t] in ALNUM and (t == 0 or b[t - 1] not in ALNUM)]
    mid = [t for t in range(n) if b[t] in ALNUM and t > 0 and b[t - 1] in ALNUM]
    S = set(stmt)
    return {"stmt": stmt, "rhs": sorted(rhs - S), "tok": [t for t in tok if t not in S and t not in rhs], "mid": mid,
            "all": list(range(n))}


def run(model, data):
    BOS = 254
    stats = {}
    add = lambda k, v: stats.setdefault(k, []).append(v)
    t0 = time.time()
    for i, b in enumerate(data):
        n = len(b)
        ids = torch.tensor([[BOS] + list(b)], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            out = model(ids)
        starts = np.arange(n + 1)
        stage_starts = []
        for bp in out.bpred_output:
            m = bp.boundary_mask.reshape(-1).cpu().numpy().astype(bool)[:len(starts)]
            starts = starts[m]
            stage_starts.append(set(int(t) - 1 for t in starts if t >= 1))
        for st, S in enumerate(stage_starts):
            for k, pos in targets(b).items():
                for x in pos:
                    add(f"s{st}_{k}", x in S)
                    add(f"s{st}_{k}_sp", x in S or (x >= 1 and b[x - 1] in b" \n" and (x - 1) in S))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} chunks, {time.time() - t0:.0f}s", flush=True)
    return {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(stats.items())}


def main(n):
    data = chunks(n)
    print(f"{len(data)} Python chunks (<= {MAX_BYTES} bytes)", flush=True)
    res = {"n_chunks": len(data), "models": {}}
    for name in MODELS:
        if not os.path.exists(f"{W}/hnet_ckpt/{name}.pt"):
            print(f"skip {name}: no checkpoint"); continue
        model = load(name)
        print(f"== {name}", flush=True)
        r = run(model, data)
        res["models"][name] = r
        for st in sorted({k.split("_")[0] for k in r if k.startswith("s") and k[1].isdigit()}):
            rate = r[f"{st}_all"]["mean"]
            print(f"  stage {st[1:]}: chunk rate {rate:.3f} | chunk starts at: " + ", ".join(
                f"{k} {r[f'{st}_{k}']['mean']:.0%} ({r[f'{st}_{k}_sp']['mean']:.0%} with space)" for k in ("stmt", "rhs", "tok", "mid")))
        del model; torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT + ".tmp", "w"), indent=1); os.replace(OUT + ".tmp", OUT)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
