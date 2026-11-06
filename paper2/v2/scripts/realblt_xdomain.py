"""Cross-domain test (issue #30): the value of a patch start at every position, in six domains, on BLT-1B.
Rules: paper3/XDOMAIN_RULES.md (committed before any gain was measured). Documents: xdomain_data.py.

Base layout B: BLT-1B's entropy at a 10% patch rate (threshold fitted on the domain's FIT documents). Positions
t >= 2 are split into 48 groups by t mod 48; one forward pass per group with that group's starts toggled against B.
Gain at t, start present minus absent, the rest of B fixed:
    bits   bits on tokens t..t+15, without minus with
    unit   exact greedy match of the alphanumeric run starting at t (unit starts only, capped at 16), with minus without
Writes results/xdomain_gains_{domain}.npz (per-document arrays; the analysis is xdomain_analysis.py).

    .venv-blt/bin/python realblt_xdomain.py [N_FIT=50] [N_TEST=60] [domain ...]
"""
import json
import os
import sys
import time

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
from blt_novelty import novelty
from realblt_budget import MODEL, lengths_from_starts

DATA = os.path.expanduser("~/.cache/segresearch-xdomain")
DOMAINS = ("math", "algebra", "code", "prose", "clinical", "traces")
R, G, WIN, UCAP = 0.10, 48, 16, 16


def alnum(c):
    return (48 <= c <= 57) or (65 <= c <= 90) or (97 <= c <= 122)


def unit_lengths(b):
    """Per token t (byte t-1): length of the alphanumeric run starting at byte t-1 if it is a unit start, else 0."""
    n = len(b) + 1
    L = np.zeros(n, np.int64)
    for i, c in enumerate(b):
        if alnum(c) and (i == 0 or not alnum(b[i - 1])):
            j = i
            while j < len(b) and alnum(b[j]) and j - i < UCAP:
                j += 1
            L[i + 1] = j - i
    return L


def main(n_fit=50, n_test=60, domains=DOMAINS):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config

    def nll_ok(ids, starts):
        n = ids.shape[1]
        with torch.no_grad():
            out = model(input_ids=ids, patch_lengths=lengths_from_starts(sorted(starts), n).to(dev), use_cache=False)
        lp = torch.log_softmax(out.logits[0].float(), -1)
        tgt = ids[0, 1:]
        nll = np.r_[0.0, (-lp[:-1].gather(1, tgt[:, None])[:, 0] / np.log(2)).cpu().numpy()]   # token j from j-1
        ok = np.r_[True, (lp[:-1].argmax(-1) == tgt).cpu().numpy()]
        return nll, ok

    def window(nll, t):
        return float(nll[t:t + WIN].sum())

    for dom in domains:
        t0 = time.time()
        docs = {sp: [json.loads(l)["text"].encode() for l in open(os.path.join(DATA, f"{dom}_{sp}.jsonl"))][:n]
                for sp, n in (("fit", n_fit), ("test", n_test))}
        S = novelty.fit_scores(np.frombuffer(open(os.path.join(DATA, f"{dom}_pool.txt"), "rb").read(), np.uint8))
        prepped = {}
        for sp, ds in docs.items():
            prepped[sp] = []
            for b in ds:
                ids = tok(b.decode(), return_tensors="pt").input_ids.to(dev)
                n = ids.shape[1]
                assert n == len(b) + 1, (dom, n, len(b))
                with torch.no_grad():
                    ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                    max_patch_length=cfg.max_patch_length)
                e = ent[0].float().cpu().numpy()
                s = np.full(n, -np.inf); s[2:] = e[1:n - 1]           # token t starts a patch from the entropy at t-1
                nov = np.full(n, -np.inf); nov[1:] = novelty.scores_for_bytes(S, np.frombuffer(b, np.uint8))
                nov[:2] = -np.inf
                prepped[sp].append((b, ids, s, nov))
        fit = blt_layouts.Fit((R,))
        for _, _, s, _ in prepped["fit"]:
            fit.add({"entropy": s})
        fit.done()
        out = {}
        for sp, items in prepped.items():
            for di, (b, ids, s, nov) in enumerate(items):
                n = ids.shape[1]
                B = set(blt_layouts.layout("entropy", R, {"entropy": s}, fit).tolist())
                UL = unit_lengths(b)
                nll0, ok0 = nll_ok(ids, B)
                gb = np.full(n, np.nan); gu = np.full(n, np.nan)
                for g in range(G):
                    T = [t for t in range(2, n) if t % G == g]
                    nll1, ok1 = nll_ok(ids, B.symmetric_difference(T))
                    for t in T:
                        has = t in B                                  # B has the start: toggled = without it
                        gb[t] = (window(nll1, t) - window(nll0, t)) if has else (window(nll0, t) - window(nll1, t))
                        if UL[t]:
                            e0, e1 = bool(ok0[t:t + UL[t]].all()), bool(ok1[t:t + UL[t]].all())
                            gu[t] = (int(e0) - int(e1)) if has else (int(e1) - int(e0))
                starts = np.zeros(n, bool); starts[sorted(B)] = True
                key = f"{sp}{di}"
                out.update({f"{key}_bytes": np.frombuffer(b, np.uint8), f"{key}_entropy": s, f"{key}_novelty": nov,
                            f"{key}_base": starts, f"{key}_gain_bits": gb, f"{key}_gain_unit": gu,
                            f"{key}_unit_len": UL, f"{key}_bits_base": nll0, f"{key}_ok_base": ok0})
                if dev == "mps":
                    torch.mps.empty_cache()
                print(f"  {dom} {sp} {di + 1}/{len(items)} | n {n} | base rate {len(B) / n:.3f} | "
                      f"{time.time() - t0:.0f}s", flush=True)
        out["n_fit"], out["n_test"] = len(prepped["fit"]), len(prepped["test"])
        out["fit_rate"] = fit.rate("entropy", R)
        np.savez_compressed(os.path.join(_seg.HERE, "results", f"xdomain_gains_{dom}.npz"), **out)
        print(f"DONE {dom} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(int(x) for x in a[:2]), *([a[2:]] if a[2:] else []))
