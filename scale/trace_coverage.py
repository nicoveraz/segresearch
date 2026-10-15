"""Where do entropy and novelty start patches in execution traces? (local; explains the Kaggle v10 tie)

Rebuilds the trace rules on the Mac as scale/generic.py prep does (entropy model on the first 200 MB of the training
traces in prep; here the first 40 MB, same model size and steps; novelty table fitted on a sample; thresholds: entropy
top 10% of train, novelty matched to entropy's eval rate) and reports each rule's coverage of the first byte of
computed and copied values, of variable names, and of literals, on held-out traces.

    .venv-scale/bin/python -m scale.trace_coverage        # -> results/generic/trace/coverage.json
"""
import json
import os
import re

import numpy as np

from scale.generic import CTX, forward_match, tab_fit, traces
from scale.patchers import ENTROPY, score_stream, train_entropy_lm
from scale.rules import table_lookup
from scale.train import device


def main():
    dev = device()
    tr = np.frombuffer(traces(40 << 20, seed=0), np.uint8)
    va = np.frombuffer(traces(2 << 20, seed=1), np.uint8)
    ent = train_entropy_lm(tr, ENTROPY, dev=dev, log_every=0)
    rng = np.random.default_rng(0)
    sample = np.concatenate([tr[s:s + 200_000] for s in rng.integers(0, len(tr) - 200_000, 20)])
    Hs = score_stream(ent, sample, surprisal=False)[0]; Hv = score_stream(ent, va, surprisal=False)[0]
    thr_e = float(np.quantile(Hs, 0.9))
    tab = tab_fit(sample, -forward_match(sample))
    Nv = table_lookup(*tab, va)
    rate_e = float((Hv[CTX:] > thr_e).mean())
    cands = np.unique(table_lookup(*tab, sample))
    thr_n = float(min(cands, key=lambda t: abs((Nv[CTX:] > t).mean() - rate_e)))
    masks = {"entropy10": Hv > thr_e, "novelty": Nv > thr_n}
    s = va.tobytes(); n = len(s)
    cat = {k: np.zeros(n, bool) for k in ("computed value", "copied value", "variable name (line start)", "literal value", "operator/operand")}
    for m in re.finditer(rb"(?m)^(.*) -> (\d+)$", s):
        cat["computed value" if re.search(rb" [-+*] ", m.group(1)) else "copied value"][m.start(2)] = True
    for m in re.finditer(rb"(?m)^([a-z]\w*) = (\d+)$", s):
        cat["literal value"][m.start(2)] = True
    for m in re.finditer(rb"(?m)^(\w+)", s):
        cat["variable name (line start)"][m.start(1)] = True
    for m in re.finditer(rb" [-+*] (\w+) ->", s):
        cat["operator/operand"][m.start(1)] = True
    keep = np.zeros(n, bool); keep[CTX:] = True
    out = {}
    print(f"held-out traces {n / 1e6:.1f} MB")
    for r, m in masks.items():
        m = m & keep; row = {"rate": float(m[keep].mean())}
        line = f"  {r:10s} rate {row['rate']:.1%} |"
        for k, c in cat.items():
            c = c & keep
            cov, share = (m & c).sum() / max(c.sum(), 1), (m & c).sum() / max(m.sum(), 1)
            row[k] = {"covered": float(cov), "share": float(share), "chance": float(c[keep].mean())}
            line += f" {k}: {cov:.0%} (share {share:.0%})"
        print(line); out[r] = row
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results/generic/trace/coverage.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
