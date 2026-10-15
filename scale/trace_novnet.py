"""Can a context-aware novelty predictor find computed values in traces? (local follow-up to trace_coverage.py)

The 2-byte novelty table sees only '> ' before every printed value, so it cannot tell a computed value (usually a new
number) from a copied one (a number printed a line or two earlier) and covers neither. Here the same target (minus
the forward LZ match length in the last 128 bytes) is predicted by scale.learned_boundary's causal network (128-byte
receptive field). Reported at a 10% rate (top 10% of train): coverage of computed / copied value starts, line starts,
operands and literals, for the true target (the ceiling), the network, and the table.

    .venv-scale/bin/python -m scale.trace_novnet         # -> results/generic/trace/novnet.json
"""
import json
import os
import re

import numpy as np

from scale.generic import CTX, forward_match, tab_fit, traces
from scale.learned_boundary import fit_net, predict
from scale.rules import table_lookup
from scale.train import device


def cats(b):
    s = b.tobytes(); n = len(s)
    c = {k: np.zeros(n, bool) for k in ("computed value", "copied value", "line start", "operand", "literal")}
    for m in re.finditer(rb"(?m)^(.*) -> (\d+)$", s):
        c["computed value" if re.search(rb" [-+*] ", m.group(1)) else "copied value"][m.start(2)] = True
    for m in re.finditer(rb"(?m)^([a-z]\w*) = (\d+)$", s):
        c["literal"][m.start(2)] = True
    for m in re.finditer(rb"(?m)^(\w+)", s):
        c["line start"][m.start(1)] = True
    for m in re.finditer(rb" [-+*] (\w+) ->", s):
        c["operand"][m.start(1)] = True
    return c


def main():
    dev = device()
    tr = np.frombuffer(traces(6 << 20, seed=0), np.uint8).copy()
    va = np.frombuffer(traces(3 << 19, seed=1), np.uint8).copy()
    ytr, yva = -forward_match(tr).astype(np.float32), -forward_match(va).astype(np.float32)
    tab = tab_fit(tr, ytr)
    net = fit_net(tr, ytr, dev=dev)
    preds = {"true target (ceiling)": (ytr, yva), "network (128 bytes)": (predict(net, tr, dev), predict(net, va, dev)),
             "2-byte table": (table_lookup(*tab, tr), table_lookup(*tab, va))}
    C = cats(va); keep = np.zeros(len(va), bool); keep[CTX:] = True
    out = {}
    for name, (ptr, pva) in preds.items():
        # top 10% of train; ties (integer target) broken at random so the rate is exactly ~10%
        jtr = ptr + 1e-3 * np.random.default_rng(0).random(len(ptr)); jva = pva + 1e-3 * np.random.default_rng(1).random(len(pva))
        m = (jva > np.quantile(jtr, 0.9)) & keep
        row = {"rate": float(m[keep].mean())}
        line = f"  {name:22s} rate {row['rate']:.1%} |"
        for k, c in C.items():
            c = c & keep; cov = (m & c).sum() / max(c.sum(), 1); share = (m & c).sum() / max(m.sum(), 1)
            row[k] = {"covered": float(cov), "share": float(share)}
            line += f" {k}: {cov:.0%}"
        print(line, flush=True); out[name] = row
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results/generic/trace/novnet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
