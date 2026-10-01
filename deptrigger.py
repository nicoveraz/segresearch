"""A label-free patch trigger from boundary dependence, fitted with the harness's own model.

1. Train a model with BLT entropy patches at the usual budget (rule "entropy", 25% of bytes).
2. On training windows, measure each byte's loss under those patches and under a tight layout (rule
   "entropy10", 10% of bytes). The rise is how much that byte depends on patch starts that the tight layout drops.
3. For each preceding two-byte context (b[t-2], b[t-1]), average the rise over bytes t..t+3: the "dependence"
   of a patch start at t. A causal lookup, no answer labels (as in blt_screens/dependence_screen.py for BLT-1B).
4. Save the table; mathexp.py's rule entdep10 ranks positions by entropy + dependence (standardized) to 10%.

    SEGR_D=128 SEGR_GLAYERS=4 SEGR_POOL=xattn SEGR_LOCAL=window uv run deptrigger.py [STEPS] [N_WINDOWS]
"""
import os
import sys

import mlx.core as mx
import numpy as np

import mathexp
from prepare import CTX, log_softmax_np

OUT = os.path.join(mathexp.CACHE, "deptrigger.npz")


def table_lookup(T2, T1, glob, b):
    """dep[t] for every position t of byte array b (context b[t-2], b[t-1]); T2 is a 65536 array (nan = unseen)."""
    b = b.astype(np.int64)
    p1 = np.r_[0, b[:-1]]; p2 = np.r_[0, 0, b[:-2]]
    d = T2[p2 * 256 + p1]
    d = np.where(np.isnan(d), T1[p1], d)
    return np.where(np.isnan(d), glob, d).astype(np.float32)


def main(steps, n_windows):
    import harness
    z = np.load(mathexp.NPZ)
    btr, Htr = z["train_bytes"], z["train_H"]
    base, tight = mathexp.Rule("entropy", z), mathexp.Rule("entropy10", z)
    print(f"1. training the reference model with entropy patches ({steps} steps)", flush=True)
    p = mathexp._train(btr, base.mask(btr, Htr), steps, seed=0)

    print(f"2. per-byte losses under 25% and 10% layouts on {n_windows} training windows", flush=True)
    rng = np.random.default_rng(1)
    starts = rng.integers(0, len(btr) - CTX - 1, n_windows)
    s2, c2, s1, c1 = (np.zeros(65536), np.zeros(65536), np.zeros(256), np.zeros(256))
    allg = []
    for k in range(0, n_windows, 64):
        W = starts[k:k + 64]
        X = np.stack([btr[w:w + CTX] for w in W]).astype(np.int32)
        Y = np.stack([btr[w + 1:w + CTX + 1] for w in W]).astype(np.int32)
        losses = []
        for rule in (base, tight):
            BD = np.stack([rule.mask(btr[w:w + CTX + 1], Htr[w:w + CTX + 1]) for w in W]).astype(np.int32); BD[:, 0] = 1
            lp = log_softmax_np(harness._model(p, mx.array(X), mx.array(BD)))
            losses.append(-np.take_along_axis(lp, Y[..., None], -1)[..., 0])   # [:, u] = loss on byte u+1 of the window
        g = losses[1] - losses[0]
        # boundary at window byte t (2 <= t <= CTX-4): mean rise on bytes t..t+3 = loss columns t-1..t+2
        cs = np.cumsum(np.c_[np.zeros(len(W)), g], 1)
        for j, w in enumerate(W):
            t = np.arange(CTX // 4, CTX - 3)                                     # skip the window start (little context)
            d = (cs[j, t + 3] - cs[j, t - 1]) / 4
            b = btr[w:w + CTX].astype(np.int64)
            key = b[t - 2] * 256 + b[t - 1]
            np.add.at(s2, key, d); np.add.at(c2, key, 1)
            np.add.at(s1, b[t - 1], d); np.add.at(c1, b[t - 1], 1)
            allg.append(d)
    glob = float(np.concatenate(allg).mean())
    T2 = np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan); T1 = np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan)
    print(f"   {int((c2 >= 20).sum())} contexts; mean rise {glob:.3f} nats", flush=True)
    top = np.argsort(-np.nan_to_num(T2, nan=-1e9))[:12]
    print("   top contexts: " + ", ".join(f"{bytes([int(k) // 256, int(k) % 256])!r} {T2[k]:.2f}" for k in top), flush=True)
    np.savez(OUT, T2=T2, T1=T1, glob=glob)
    print("saved", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16000, int(sys.argv[2]) if len(sys.argv) > 2 else 2048)
