"""Logic and state tracking with models trained here: does patch placement matter where the skill can be learned?

On BLT-1B the program-trace arithmetic (20% at default) and the logic questions (chance) were beyond the model,
so placement could not be tested. Here the same generated tasks (realblt_reason.py) are the training corpus, and
mathexp.py's pipeline (training, bits, greedy exact-match accuracy with online boundaries) is reused, with its
cache, syntax markers and answer spans swapped in memory:

    computed  program-trace values after an arithmetic line ('# a is now 11' after 'a = a + b')
    copy      program-trace values after a constant line ('a = 7') -- the no-computation control
    final     the logic answer (Yes/No) after the worked proof
    boxed     each proof step's conclusion ('Bob is red, so Bob is <kind>')
Hand-written "syntax" for this corpus: the byte right after 'is now ', ', so <name> is ' (any name) or 'Answer: '.
Rules as in mathexp.py: entropyR, depR (deptrigger.py, refit on this corpus), syntax+entropyR, ...

    uv run reasonexp.py prepare                    # generate the corpus, train its small entropy model
    SEGR_D=128 ... uv run reasonexp.py fit [STEPS] [N_WINDOWS]       # dependence table for this corpus
    uv run reasonexp.py rates
    SEGR_D=128 ... uv run reasonexp.py run RULE SEED STEPS
"""
import os
import random
import sys

import numpy as np

import mathexp
import prepare
import realblt_reason
from prepare import PATCHER, RI

mathexp.CACHE = mathexp.CACHE.replace("-math", "-reason")
mathexp.NPZ = os.path.join(mathexp.CACHE, "math.npz")
mathexp.SYNTAX = [b"is now ", b"Answer: "] + [f", so {n} is ".encode() for n in realblt_reason.NAMES]
ROLE = {"computed": "ANS_LOCAL", "copy": "VALUE", "answer": "ANS_LONG", "step": "VAR"}
_targets_math = mathexp._targets


def _targets(b, roles):
    """As mathexp._targets, but copies are scored too (the control)."""
    out = _targets_math(b, roles)
    m = roles == RI["VALUE"]
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]]); ends = np.flatnonzero(m & ~np.r_[m[1:], False]) + 1
    return out + [("copy", int(s), int(e)) for s, e in zip(starts, ends) if s > prepare.CTX]


mathexp._targets = _targets


def _corpus(n_logic, n_trace, seed):
    """Logic (with proof) and trace problems in a random order, separated by blank lines, with answer roles."""
    rng = random.Random(seed)
    probs = [realblt_reason.logic_problem(rng, False) for _ in range(n_logic)] + \
            [realblt_reason.trace_problem(rng) for _ in range(n_trace)]
    order = np.random.default_rng(seed).permutation(len(probs))
    b, r = bytearray(), []
    for i in order:
        text, spans = probs[i]
        roles = np.full(len(text) + 2, RI["TEXT"], np.int8); roles[-2:] = RI["STRUCT"]
        for kind, s, e in spans:
            roles[s:e] = RI[ROLE[kind]]
        b.extend((text + "\n\n").encode()); r.extend(roles.tolist())
    return np.frombuffer(bytes(b), np.uint8).copy(), np.array(r, np.int8)


def prepare_cache():
    os.makedirs(mathexp.CACHE, exist_ok=True)
    data = {"train": _corpus(12000, 18000, 1), "val": _corpus(1200, 1800, 2)}
    names = {"computed": "ANS_LOCAL", "copy": "VALUE", "answer": "ANS_LONG", "step": "VAR"}
    for sp, (bts, roles) in data.items():
        print(f"{sp}: {len(bts):,} bytes; " + ", ".join(f"{k} {np.mean(roles == RI[v]):.3f}" for k, v in names.items()))
    print("training the small entropy model", flush=True)
    pat = prepare.train_lm(data["train"][0], PATCHER, {PATCHER["steps"]}, seed=0)[PATCHER["steps"]]
    out = {f"pat_{'/'.join(map(str, k)) if isinstance(k, tuple) else k}": v for k, v in mathexp._flatten(pat)}
    for sp, (bts, roles) in data.items():
        H, _ = prepare.score_stream(pat, PATCHER["heads"], bts)
        out.update({f"{sp}_bytes": bts, f"{sp}_roles": roles, f"{sp}_H": H})
    np.savez_compressed(mathexp.NPZ, **out)
    print("cached", mathexp.NPZ)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prepare":
        prepare_cache()
    elif a[0] == "fit":
        import deptrigger
        deptrigger.OUT = os.path.join(mathexp.CACHE, "deptrigger.npz")
        deptrigger.main(int(a[1]) if len(a) > 1 else 16000, int(a[2]) if len(a) > 2 else 4096, "marginal")
    elif a[0] == "rates":
        z = np.load(mathexp.NPZ)
        starts = lambda role: (z["val_roles"] == RI[role]) & ~np.r_[False, z["val_roles"][:-1] == RI[role]]
        for n in a[1:] or ("entropy", "entropy10", "jump10", "syntax+entropy10", "dep10", "entdep10"):
            r = mathexp.Rule(n, z); m = r.mask(z["val_bytes"], z["val_H"])
            print(f"{n:17s} val {(m > 0).mean():.3f} | covered: " + ", ".join(
                f"{k} {(m[starts(v)] > 0).mean():.2f}" for k, v in ROLE.items()))
    else:
        mathexp.run(a[1], int(a[2]), int(a[3]))
