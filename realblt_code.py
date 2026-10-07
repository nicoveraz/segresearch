"""Beyond math: the tight-budget test on Meta's BLT-1B with Python code.

Python standard-library source (train and test chunks from different files). Targets: identifiers (other than self and cls) that repeat a
name seen within the previous 120 bytes of the chunk (predictable type, hard value). Patch boundaries are changed at
inference time. Layouts at budget R = 15% and 10% of bytes (applied as in realblt_budget.py: thresholds fitted on the
TRAIN chunks by default, SEGR_BLT_THRESH=problem for the original top-R-per-chunk selection; see blt_layouts.py):

    default      BLT's own entropy patches
    entropy@R    top R by BLT-1B's entropy
    dep@R        top R by label-free boundary dependence: how much BLT-1B's loss on the next 4 bytes rises when its
                 patches are cut from default to 10%, averaged per preceding 2-byte context on TRAIN chunks
    entdep@R     top R by entropy + dependence (standardized)
    oracle@R     boundaries forced at every repeated-identifier start, the rest by entropy (uses the targets)

Scored on the repeated identifiers: bits per byte (teacher forced) and exact match.

    <env>/bin/python realblt_code.py [N_TRAIN_CHUNKS] [N_TEST_CHUNKS]
"""
import glob
import hashlib
import keyword
import os
import re
import sys
import sysconfig
import time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, BltForCausalLM

import blt_layouts
import blt_load
import registry

MODEL = "itazap/blt-1b-hf"
BUDGETS = (0.15, 0.10)
CHUNK = 800
IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


def chunks(val):
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(f for f in glob.glob(root + "/**/*.py", recursive=True)
                   if "/test" not in f and "site-packages" not in f and "/idlelib" not in f)
    out = []
    for f in files:
        if (int(hashlib.md5(os.path.relpath(f, root).encode()).hexdigest(), 16) % 12 == 0) != val:
            continue
        src = open(f, "rb").read()
        if not src.isascii():
            continue
        lines, cur = src.split(b"\n"), b""
        for line in lines:
            if len(cur) + len(line) + 1 > CHUNK and len(cur) > 200:
                out.append(cur); cur = b""
            cur += line + b"\n"
    rng = np.random.default_rng(0)
    return [out[i] for i in rng.permutation(len(out))]


def targets(b):
    """Starts and ends (byte offsets) of identifiers repeating a name seen within the previous 120 bytes."""
    last, spans = {}, []
    for m in IDENT.finditer(b):
        name = m.group()
        if keyword.iskeyword(name.decode()) or name in (b"self", b"cls"):
            continue
        if name in last and m.start() - last[name] <= 120:
            spans.append((m.start(), m.end()))
        last[name] = m.start()
    return spans


def main(n_train, n_test):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config

    def prep(b):
        ids = torch.tensor([[1] + [x + 4 for x in b]], device=dev)          # BOS + byte tokens (byte + 4)
        with torch.no_grad():
            ent, dl, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                             max_patch_length=cfg.max_patch_length)
        n = ids.shape[1]; e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]
        return ids, n, score, np.r_[0, np.cumsum(dl[0].cpu().numpy())[:-1]]

    def L(st, n):
        s = sorted(set(int(x) for x in st if 0 <= x < n))
        return torch.tensor([[b - a for a, b in zip(s, s[1:] + [n])]], device=dev)

    def logp(ids, st, n):
        with torch.no_grad():
            return torch.log_softmax(model(input_ids=ids, patch_lengths=L(st, n), use_cache=False).logits[0].float(), -1)

    t0 = time.time()
    print(f"fitting boundary dependence on {n_train} train chunks", flush=True)
    s2, s1 = defaultdict(list), defaultdict(list)
    train_seen = []
    for b in chunks(False)[:n_train]:
        ids, n, score, default = prep(b)
        train_seen.append((b, n, score))
        tight = np.r_[0, 1, np.argsort(-score)[:max(int(0.10 * n) - 2, 1)]]
        loss = lambda st: -logp(ids, st, n)[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
        gain = loss(tight) - loss(default)
        for i in range(2, len(b) - 4):
            t = i + 1; d = float(gain[t - 1:t + 3].mean())
            s2[(b[i - 2], b[i - 1])].append(d); s1[b[i - 1]].append(d)
    T2 = {k: float(np.mean(v)) for k, v in s2.items() if len(v) >= 20}
    T1 = {k: float(np.mean(v)) for k, v in s1.items() if len(v) >= 20}
    glob_ = float(np.mean([x for v in s1.values() for x in v]))
    top = sorted(T2.items(), key=lambda kv: -kv[1])[:10]
    print(f"  {len(T2)} contexts; top: " + ", ".join(f"{bytes(k)!r} {v:.2f}" for k, v in top), flush=True)
    dep_at = lambda b, i: T2.get((b[i - 2], b[i - 1]), T1.get(b[i - 1], glob_))

    def dep_of(b, n):
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        return dep

    layouts = ["default"] + [f"{k}@{int(r * 100)}" for r in BUDGETS for k in ("entropy", "dep", "entdep", "oracle")]
    kind_of = {"entropy": "entropy", "dep": "dep", "entdep": "entropy+dep", "oracle": "forced"}
    fit = None
    if blt_layouts.MODE == "train":
        # budget thresholds fitted on the same TRAIN chunks, applied position by position at test time (#26)
        fit = blt_layouts.Fit(BUDGETS)
        for b, n, score in train_seen:
            fit.add({"entropy": score, "dep": dep_of(b, n)}, [s + 1 for s, _ in targets(b)])
        fit.done()
        print("  train patch rates: " + ", ".join(
            f"{l} {fit.rate(kind_of[l.split('@')[0]], int(l.split('@')[1]) / 100):.3f}" for l in layouts[1:]), flush=True)
    st_ = {l: {"bits": [], "exact": [], "cov": [], "rate": []} for l in layouts}
    used, pid = 0, []                                                     # pid: chunk index of each identifier
    for b in chunks(True)[:n_test]:
        spans = targets(b)
        if not spans:
            continue
        used += 1
        pid += [used - 1] * len(spans)
        ids, n, score, default = prep(b)
        tstarts = [s + 1 for s, _ in spans]                                  # token = byte + 1
        sig = {"entropy": score, "dep": dep_of(b, n)}
        lay = {"default": default}
        for name in layouts[1:]:
            k, R = name.split("@")
            lay[name] = blt_layouts.layout(kind_of[k], int(R) / 100, sig, fit, tstarts if k == "oracle" else ())
        for name, st in lay.items():
            st = np.array(sorted(set(int(x) for x in st if 0 <= x < n)))
            lp = logp(ids, st, n); S = set(st.tolist()); st_[name]["rate"].append(len(st) / n)
            for (s, e_), t in zip(spans, tstarts):
                tgt = ids[0, s + 1:e_ + 1]; pred = lp[s:e_]
                st_[name]["bits"].append(float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)))
                st_[name]["exact"].append(bool((pred.argmax(-1) == tgt).all())); st_[name]["cov"].append(t in S)
        if used % 25 == 0:
            print(f"  {used} test chunks, {time.time() - t0:.0f}s", flush=True)
    print(f"\nBLT-1B on {used} Python test chunks, {len(st_['default']['exact'])} repeated identifiers, "
          f"budget thresholds: {blt_layouts.MODE}")
    for name in layouts:
        s = st_[name]
        registry.emit("realblt_code", f"RESULT {name:11s} patch rate {np.mean(s['rate']):.3f} | covered {np.mean(s['cov']):5.1%} | repeated identifiers: {np.mean(s['bits']):.3f} bits, "
              f"exact {np.mean(s['exact']):5.1%}", experiment="blt_code", thresh=blt_layouts.MODE, window=blt_load.WINDOW)
    registry.save_items(f"blt_code_{blt_layouts.MODE}{blt_load.SUFFIX}", {"layouts": st_, "pid": pid}, experiment="blt_code",
                        thresh=blt_layouts.MODE, n_chunks=used)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
