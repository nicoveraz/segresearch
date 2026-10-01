"""A learned, label-free scratchpad trigger: put scratchpads where a fresh global step actually lowers the loss.

1. Train the scratchpad model (SEGR_SCRATCH=1) with random scratchpads, so it learns to use them anywhere.
2. On training windows, measure for each candidate position t the GAIN of a scratchpad there: the drop in loss
   on the rest of the patch (bytes t .. patch end) with a scratchpad at t, against none.
3. Fit a ridge regression predicting that gain from information available before byte t: the identities of
   bytes t-1 and t-2, the small model's entropy and its rise at t, and the offset within the patch.
4. Save the weights; mathexp.py's rule sp16:learned puts scratchpads at the top 6% of predicted gains.
No answer labels are used anywhere: whatever the trigger learns about '=' or '####' it learns from the loss.

    SEGR_SCRATCH=1 SEGR_POOL=xattn SEGR_LOCAL=window uv run gaintrigger.py [STEPS] [N_WINDOWS]
"""
import os
import sys

import mlx.core as mx
import numpy as np

import mathexp
from prepare import CTX

K = 16                                       # patch size, as in sp16:*
OUT = os.path.join(mathexp.CACHE, "gaintrigger.npz")


def features(b, H):
    """Causal features for each position t (to decide a scratchpad before predicting byte t)."""
    n = len(b); J = np.diff(H, prepend=H[0])
    prev1 = np.r_[0, b[:-1]].astype(np.int64); prev2 = np.r_[0, 0, b[:-2]].astype(np.int64)
    off = np.arange(n) % K
    X = np.zeros((n, 256 + 256 + K + 3), np.float32)
    X[np.arange(n), prev1] = 1; X[np.arange(n), 256 + prev2] = 1; X[np.arange(n), 512 + off] = 1
    X[:, 512 + K] = H; X[:, 513 + K] = J; X[:, 514 + K] = 1.0
    return X


def main(steps, n_windows):
    import harness
    assert harness.SCRATCH, "run with SEGR_SCRATCH=1"
    z = np.load(mathexp.NPZ)
    btr, Htr = z["train_bytes"], z["train_H"]
    rule = mathexp.Rule("sp16:random", z)
    print(f"1. training the scratchpad model with random scratchpads ({steps} steps)", flush=True)
    p = mathexp._train(btr, rule.mask(btr, Htr), steps, seed=0)

    print(f"2. measuring scratchpad gains on {n_windows} training windows", flush=True)
    rng = np.random.default_rng(1)
    starts = rng.integers(0, len(btr) - CTX - 1, n_windows); starts -= starts % K     # align patches with the stride
    feats, gains = [], []
    for w0 in starts:
        x = btr[w0:w0 + CTX].astype(np.int32); y = btr[w0 + 1:w0 + CTX + 1].astype(np.int32)
        base = np.zeros(CTX + 1, np.int32); base[::K] = 1
        cands = [t for t in range(1, CTX) if base[t] == 0]                            # positions inside a patch
        F = np.repeat(base[None], len(cands) + 1, 0)
        for i, t in enumerate(cands):
            F[i + 1, t] = 2                                                           # scratchpad before byte t
        X = np.repeat(x[None], len(F), 0)
        lg = mathexp.log_softmax_np(harness._model(p, mx.array(X), mx.array(F)))
        loss = -lg[np.arange(len(F))[:, None], np.arange(CTX)[None], y[None]]        # loss[v, u]: predicting x_{u+1}
        for i, t in enumerate(cands):
            end = (t // K + 1) * K                                                    # the patch containing byte t ends here
            # byte t is predicted at position t-1; the scratchpad affects bytes t .. end-1 (positions t-1 .. end-2)
            gains.append(float((loss[0, t - 1:end - 1] - loss[i + 1, t - 1:end - 1]).sum()))
        feats.append(features(btr[w0:w0 + CTX], Htr[w0:w0 + CTX])[cands])
    Xf, g = np.concatenate(feats), np.array(gains, np.float32)
    print(f"   {len(g)} positions; gain mean {g.mean():.3f}, p90 {np.quantile(g, .9):.3f} nats", flush=True)

    print("3. fitting the ridge regression", flush=True)
    lam = 1.0
    w = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ g)
    pred = Xf @ w
    print(f"   correlation of predicted and measured gain: {np.corrcoef(pred, g)[0, 1]:.3f}", flush=True)
    top = np.argsort(-w[:256])[:8]
    print("   previous bytes with the largest learned gain: "
          + ", ".join(f"{bytes([int(v)])!r} {w[v]:+.3f}" for v in top), flush=True)
    np.savez(OUT, w=w, K=K)
    print("saved", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16000, int(sys.argv[2]) if len(sys.argv) > 2 else 400)
