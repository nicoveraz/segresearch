"""A label-free patch trigger from boundary dependence, fitted with the harness's own model.

Mode "marginal" (default):
1. Train a reference model with RANDOM patch starts (25% of bytes), so it learns to use a boundary anywhere.
2. On training windows with a random 10% layout, add ONE boundary at a candidate position t and measure the drop
   in loss on bytes t..t+3: the value of a patch start at t.
3. Average per preceding two-byte context (b[t-2], b[t-1]). A causal lookup, no answer labels.
Mode "remove" (v1, as for BLT-1B in blt_screens/dependence_screen.py): reference trained on entropy patches
(25%); the rise in loss on bytes t..t+3 when the layout is cut to entropy10. It failed here: the reference model
rarely had a boundary at results (13% of answer starts), so it never learned to use one and dropping it costs little.
Either way, the table is saved; mathexp.py's rule entdep10 ranks positions by entropy + dependence (standardized) to 10%.

    SEGR_D=128 SEGR_GLAYERS=4 SEGR_POOL=xattn SEGR_LOCAL=window uv run deptrigger.py [STEPS] [N_WINDOWS] [marginal|remove]
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


def main(steps, n_windows, mode):
    import harness
    z = np.load(mathexp.NPZ)
    btr, Htr = z["train_bytes"], z["train_H"]
    rng = np.random.default_rng(1)
    if mode == "remove":
        # v1: reference trained on entropy patches (25%); rise when cut to entropy10
        ref, base, tight = mathexp.Rule("entropy", z).mask(btr, Htr), None, mathexp.Rule("entropy10", z)
    else:
        # v2 (marginal): reference trained on RANDOM patches (25%) so it learns to use a boundary anywhere;
        # gain of adding ONE boundary at t to a random 10% layout
        ref = rng.random(len(btr)) < 0.25
    print(f"1. training the reference model ({mode}; {steps} steps; patch rate {ref.mean():.3f})", flush=True)
    p = mathexp._train(btr, ref, steps, seed=0)

    def losses(X, Y, BD):
        lp = log_softmax_np(harness._model(p, mx.array(X), mx.array(BD.astype(np.int32))))
        return -np.take_along_axis(lp, Y[..., None], -1)[..., 0]           # [:, u] = loss on window byte u+1

    print(f"2. measuring dependence on {n_windows} training windows", flush=True)
    starts = rng.integers(0, len(btr) - CTX - 1, n_windows)
    s2, c2, s1, c1 = (np.zeros(65536), np.zeros(65536), np.zeros(256), np.zeros(256))
    allg, samples = [], []                                                # samples: (window start, position, gain)

    def add(b, t, d):
        key = b[t - 2] * 256 + b[t - 1]
        np.add.at(s2, key, d); np.add.at(c2, key, 1); np.add.at(s1, b[t - 1], d); np.add.at(c1, b[t - 1], 1)
        allg.append(d)

    if mode == "remove":
        base = mathexp.Rule("entropy", z)
        for k in range(0, n_windows, 64):
            W = starts[k:k + 64]
            X = np.stack([btr[w:w + CTX] for w in W]).astype(np.int32)
            Y = np.stack([btr[w + 1:w + CTX + 1] for w in W]).astype(np.int32)
            L = []
            for rule in (base, tight):
                BD = np.stack([rule.mask(btr[w:w + CTX + 1], Htr[w:w + CTX + 1]) for w in W]); BD[:, 0] = 1
                L.append(losses(X, Y, BD))
            cs = np.cumsum(np.c_[np.zeros(len(W)), L[1] - L[0]], 1)
            t = np.arange(CTX // 4, CTX - 3)                                 # skip the window start (little context)
            for j, w in enumerate(W):                                        # boundary at t: rise on bytes t..t+3
                add(btr[w:w + CTX].astype(np.int64), t, (cs[j, t + 3] - cs[j, t - 1]) / 4)
    else:
        K = 32                                                               # candidate positions per window
        M = int(os.environ.get("SEGR_DEP_LAYOUTS", "1"))                     # >1: average each gain over M random layouts
        for w in starts if M > 1 else ():
            b = btr[w:w + CTX].astype(np.int64)
            cand = rng.choice(np.arange(CTX // 4, CTX - 3), K, replace=False)
            X = np.repeat(btr[w:w + CTX][None].astype(np.int32), M * (K + 1), 0)
            Y = np.repeat(btr[w + 1:w + CTX + 1][None].astype(np.int32), M * (K + 1), 0)
            BD = np.zeros((M * (K + 1), CTX + 1), bool)
            for m in range(M):
                bd = rng.random(CTX + 1) < 0.10; bd[0] = True; bd[cand] = False
                BD[m * (K + 1):(m + 1) * (K + 1)] = bd
                BD[m * (K + 1) + 1 + np.arange(K), cand] = True
            cs = np.cumsum(np.c_[np.zeros(len(BD)), losses(X, Y, BD)], 1)
            win = lambda r, t: (cs[r, t + 3] - cs[r, t - 1]) / 4
            g = np.mean([win(m * (K + 1), cand) - win(m * (K + 1) + 1 + np.arange(K), cand) for m in range(M)], 0)
            add(b, cand, g)
            samples.append(np.stack([np.full(K, w), cand, g], 1))
        for w in starts if M == 1 else ():
            b = btr[w:w + CTX].astype(np.int64)
            X = np.repeat(btr[w:w + CTX][None].astype(np.int32), K + 1, 0)
            Y = np.repeat(btr[w + 1:w + CTX + 1][None].astype(np.int32), K + 1, 0)
            bd = rng.random(CTX + 1) < 0.10; bd[0] = True
            cand = rng.choice(np.flatnonzero(~bd[CTX // 4:CTX - 3]) + CTX // 4, K, replace=False)
            BD = np.repeat(bd[None], K + 1, 0)
            BD[np.arange(1, K + 1), cand] = True
            L = losses(X, Y, BD)
            cs = np.cumsum(np.c_[np.zeros(K + 1), L], 1)
            win = lambda r, t: (cs[r, t + 3] - cs[r, t - 1]) / 4             # mean loss on bytes t..t+3
            g = win(0, cand) - win(np.arange(1, K + 1), cand)                 # drop from adding a boundary at t
            add(b, cand, g)
            samples.append(np.stack([np.full(K, w), cand, g], 1))
    allg = np.concatenate([np.atleast_1d(g) for g in allg])
    glob = float(allg.mean())
    T2 = np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan); T1 = np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan)
    print(f"   {int((c2 >= 20).sum())} contexts; mean {glob:.3f} nats", flush=True)
    top = np.argsort(-np.nan_to_num(T2, nan=-1e9))[:12]
    print("   top contexts: " + ", ".join(f"{bytes([int(k) // 256, int(k) % 256])!r} {T2[k]:.2f}" for k in top), flush=True)
    np.savez(OUT, T2=T2, T1=T1, glob=glob, mode=mode)
    if samples:                                                          # raw measurements, for a learned patcher (neural_patcher.py)
        np.save(OUT.replace(".npz", "_samples.npy"), np.concatenate(samples))
    print("saved", OUT)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 16000, int(a[1]) if len(a) > 1 else 2048, a[2] if len(a) > 2 else "marginal")
