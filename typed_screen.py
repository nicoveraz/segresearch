"""Typed uncertainty: is the model sure WHAT KIND of byte comes next, but not WHICH one? (#31 follow-up; local)

Entropy is capped by the number of options: at a digit there are 10 bytes, so even complete ignorance of the value
is at most log2(10) = 3.3 bits, while an ordinary word start is often higher. Entropy therefore ranks a computed
result ("easy to anticipate, hard to compute") below routine word starts. Here the small model's next-byte
distribution p (the same prediction entropy uses; causal, no labels) is split over generic byte classes
(digits, lowercase, uppercase, whitespace, ASCII punctuation, other bytes):

    Hclass   entropy over the class (how unsure the model is about the KIND of byte)
    W        expected within-class ignorance: sum_c P(c) * H(p | c) / log2|c|   (0 = knows the item, 1 = no idea)
    T        typed ignorance: P(c*) * H(p | c*) / log2|c*| for the most likely class c* (sure of the kind AND
             ignorant of the item)

Coverage of computed / copy / final / step starts and other word starts at a 10% budget (threshold = quantile on
2M train bytes), each signal alone and combined with entropy, on the math and logic+trace corpora.

    uv run typed_screen.py
"""
import json
import os

import numpy as np

CORPORA = {"math": os.path.expanduser("~/.cache/segresearch-math/math.npz"),
           "reason": os.path.expanduser("~/.cache/segresearch-reason/math.npz")}
N_FIT = 2_000_000
R = 0.10


def byte_classes():
    c = np.full(256, 5, np.int64)                                  # 5: other bytes (control, non-ASCII)
    c[ord("0"):ord("9") + 1] = 0
    c[ord("a"):ord("z") + 1] = 1
    c[ord("A"):ord("Z") + 1] = 2
    c[[9, 10, 13, 32]] = 3
    for x in range(33, 127):
        if c[x] == 5:
            c[x] = 4                                               # ASCII punctuation and symbols
    return c


CLS = byte_classes()
ONEHOT = np.eye(6)[CLS]                                            # (256, 6)
LOGSIZE = np.log2(np.maximum(ONEHOT.sum(0), 2))                    # log2 |class| (whitespace etc. >= 1 bit)


def typed(lp):
    """lp: (..., 256) natural-log probabilities -> H, Hclass, W, T (bits; W, T in [0, 1])."""
    p = np.exp(lp)
    H = -(p * lp).sum(-1) / np.log(2)
    Pc = p @ ONEHOT                                                # (..., 6)
    Hc = -(Pc * np.log2(np.maximum(Pc, 1e-12))).sum(-1)
    # H(p | c) for every class: entropy of p restricted to c, renormalized
    plogp = (p * lp / np.log(2)) @ ONEHOT                          # sum_{x in c} p log2 p
    Hin = np.log2(np.maximum(Pc, 1e-12)) - plogp / np.maximum(Pc, 1e-12)
    norm = np.clip(Hin / LOGSIZE, 0, 1)
    W = (Pc * norm).sum(-1)
    cs = Pc.argmax(-1)
    T = np.take_along_axis(Pc, cs[..., None], -1)[..., 0] * np.take_along_axis(norm, cs[..., None], -1)[..., 0]
    return H, Hc, W, T


def score_typed(p, n_heads, stream):
    """As prepare.score_stream (byte i scored from bytes < i, CTX//2 context minimum), returning H, Hclass, W, T."""
    import mlx.core as mx
    import prepare
    from prepare import CTX, log_softmax_np
    out = [np.zeros(len(stream), np.float32) for _ in range(4)]
    s32 = stream.astype(np.int32); half = CTX // 2
    starts = np.arange(0, len(s32) - CTX, half)
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]
        X = np.stack([s32[s:s + CTX] for s in st])
        sig = typed(log_softmax_np(prepare.lm_forward(p, mx.array(X), n_heads)))
        for j, s in enumerate(st):
            for o, v in zip(out, sig):
                o[s + half + 1: s + CTX + 1] = v[j, half:]
    return out


def starts(roles, role):
    from prepare import RI
    m = roles == RI[role]
    return m & ~np.r_[False, m[:-1]]


def main():
    import mathexp
    import realtext
    from prepare import CTX, PATCHER
    out = {}
    for c, path in CORPORA.items():
        z = np.load(path)
        pat = mathexp._patcher(z)
        btr = z["train_bytes"][-N_FIT:]; bva = z["val_bytes"]; roles = z["val_roles"]
        tr = score_typed(pat, PATCHER["heads"], btr); va = score_typed(pat, PATCHER["heads"], bva)
        chk = float(np.abs(va[0][CTX:] - z["val_H"][CTX:]).max())
        print(f"\n== {c}: rescored {len(btr):,} train + {len(bva):,} val bytes; max |H - cached H| {chk:.1e}", flush=True)
        tgt = {k: starts(roles, v) for k, v in
               {"computed": "ANS_LOCAL", "copy": "VALUE", "final": "ANS_LONG", "step": "VAR"}.items()}
        ans = np.zeros(len(roles), bool)
        for v in tgt.values():
            ans |= v
        tgt["word"] = realtext._word_starts(bva) & ~ans
        names = ("entropy", "Hclass", "W", "T")
        print("  mean at each kind of start:   " + "  ".join(f"{n:>8s}" for n in names))
        for k, m in tgt.items():
            print(f"  {k:9s} n={m.sum():6d}          " + "  ".join(f"{v[m].mean():8.3f}" for v in va))
        print(f"  {'all':9s}                   " + "  ".join(f"{v[CTX:].mean():8.3f}" for v in va))
        z_ = lambda a, ref: (a - ref.mean()) / (ref.std() + 1e-9)
        S = {"entropy": (tr[0], va[0]), "W": (tr[2], va[2]), "T": (tr[3], va[3]),
             "T-Hclass": (z_(tr[3], tr[3]) - z_(tr[1], tr[1]), z_(va[3], tr[3]) - z_(va[1], tr[1]))}
        rows = {}
        for name, (a, v) in S.items():
            rows[f"{name}10"] = v > np.quantile(a, 1 - R)
            if name != "entropy":
                rows[f"entropy+{name}10"] = z_(va[0], tr[0]) + z_(v, a) > np.quantile(z_(tr[0], tr[0]) + z_(a, a), 1 - R)
        print(f"  coverage at {R:.0%}:            rate " + " ".join(f"{k:>8s}" for k in tgt))
        out[c] = {}
        for name, m in rows.items():
            cov = {k: float(m[v].mean()) for k, v in tgt.items()}
            out[c][name] = {"rate": float(m.mean()), **cov}
            print(f"  {name:28s} {m.mean():.3f} " + " ".join(f"{cov[k]:8.1%}" for k in tgt), flush=True)
    json.dump(out, open("results/typed_screen.json", "w"), indent=1)
    print("\n-> results/typed_screen.json")


if __name__ == "__main__":
    main()
