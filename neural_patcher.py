"""A learned patcher: predict the value of a patch start from the preceding bytes (issue #19).

deptrigger.py measures, for sampled positions t, how much adding one patch start at t lowers the loss on bytes
t..t+3 (marginal mode), then averages per preceding 2-byte context into a lookup table. The table cannot use more
context than 2 bytes and does not transfer across formats (GSM8K table -> program traces: 0% coverage). Here a small
causal network reads the last K bytes and predicts the measured gain directly:

    embed each of bytes t-K..t-1 (16 dims) -> concat -> MLP (256, 64) -> predicted gain at t

No labels: the target is the model's own loss change. Training and held-out samples come from different windows.
The screen compares, on held-out samples, how well the network and a 2-byte table fit on the same training samples
predict the gain (Pearson and Spearman correlation, and how many of the true top-10% positions land in the
predicted top 10%). The weights are saved for mathexp.py's rule ndepR (top R of bytes by predicted gain).

    uv run neural_patcher.py [K] [EPOCHS] [NAME=CORPUS_NPZ:SAMPLES_FILE ...]      (several pairs: one network on all)
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mathexp

SAMPLES = os.path.join(mathexp.CACHE, "deptrigger_big_samples.npy")
OUT = os.path.join(mathexp.CACHE, "neural_patcher.npz")


class Patcher(nn.Module):
    def __init__(self, K, e=16, h=256):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(257, e)                  # 256 = padding (before the start of the text)
        self.l1, self.l2, self.l3 = nn.Linear(K * e, h), nn.Linear(h, 64), nn.Linear(64, 1)

    def __call__(self, ctx):                             # ctx: (N, K) byte ids, oldest first
        x = self.emb(ctx).reshape(ctx.shape[0], -1)
        return self.l3(nn.gelu(self.l2(nn.gelu(self.l1(x)))))[:, 0]


def contexts(b, pos, K):
    """(N, K) array of the K bytes before each position (256 where the text has not started)."""
    b = np.r_[np.full(K, 256, np.int64), b.astype(np.int64)]
    return b[(np.asarray(pos)[:, None] + np.arange(K)[None])]          # padded index p..p+K-1 = bytes p-K..p-1


def predict(model, b, batch=65536):
    """Predicted gain for every position of byte array b (position t uses bytes t-K..t-1 only)."""
    out = []
    for s in range(0, len(b), batch):
        pos = np.arange(s, min(s + batch, len(b)))
        out.append(np.array(model(mx.array(contexts(b, pos, model.K)))))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def load(path=None):
    z = np.load(path or OUT)
    m = Patcher(int(z["K"]))
    m.update(nn.utils.tree_unflatten([(k[2:], mx.array(z[k])) for k in z.files if k.startswith("w.")]))
    return m


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def top_overlap(pred, true, q=0.10):
    k = max(1, int(q * len(true)))
    return len(set(np.argsort(-pred)[:k]) & set(np.argsort(-true)[:k])) / k


def corpus(npz, samples_path, K):
    """Contexts, gains, held-out mask and a 2-byte table baseline (fit on the training samples) for one corpus."""
    btr = np.load(npz)["train_bytes"]
    S = np.load(samples_path)
    w, t, g = S[:, 0].astype(np.int64), S[:, 1].astype(np.int64), S[:, 2].astype(np.float32)
    pos = w + t                                                          # absolute position in the train stream
    test = np.unique(w, return_inverse=True)[1] % 10 == 0                 # hold out every 10th window
    p1, p2 = btr[pos - 1].astype(np.int64), btr[pos - 2].astype(np.int64)
    key = p2 * 256 + p1
    s2, c2 = np.zeros(65536), np.zeros(65536)
    np.add.at(s2, key[~test], g[~test]); np.add.at(c2, key[~test], 1)
    s1, c1 = np.zeros(256), np.zeros(256)
    np.add.at(s1, p1[~test], g[~test]); np.add.at(c1, p1[~test], 1)
    T2 = np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan); T1 = np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan)
    tab = np.where(np.isnan(T2[key]), np.where(np.isnan(T1[p1]), g[~test].mean(), T1[p1]), T2[key])
    return contexts(btr, pos, K), g, test, tab


def main(K, epochs, pairs):
    """pairs: [(name, corpus npz, samples file)]; one network is trained on all of them together."""
    data = {name: corpus(npz, sp, K) for name, npz, sp in pairs}
    for name, (X, g, test, _) in data.items():
        print(f"{name}: {len(g)} samples; held out {test.mean():.0%}; gain mean {g.mean():.3f}, sd {g.std():.3f}", flush=True)
    X = np.concatenate([d[0] for d in data.values()]); g = np.concatenate([d[1] for d in data.values()])
    test = np.concatenate([d[2] for d in data.values()])
    model = Patcher(K)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    step = nn.value_and_grad(model, lambda m, x, y: ((m(x) - y) ** 2).mean())
    tr = np.flatnonzero(~test); rng = np.random.default_rng(0)
    for ep in range(epochs):
        rng.shuffle(tr); tot = 0.0
        for s in range(0, len(tr), 1024):
            idx = tr[s:s + 1024]
            l, gr = step(model, mx.array(X[idx]), mx.array(g[idx]))
            opt.update(model, gr); mx.eval(model.parameters(), opt.state); tot += l.item() * len(idx)
        print(f"  epoch {ep + 1}: train mse {tot / len(tr):.4f}", flush=True)
    for name, (Xc, gc, tc, tab) in data.items():
        pn = np.array(model(mx.array(Xc[tc]))); gt = gc[tc]
        for label, p in ((f"{name} 2-byte table", tab[tc]), (f"{name} network, {K} bytes", pn)):
            print(f"SCREEN {label:28s} pearson {np.corrcoef(p, gt)[0, 1]:.3f} | spearman {spearman(p, gt):.3f} | "
                  f"top-10% overlap {top_overlap(p, gt):.2f} (chance 0.10)", flush=True)
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    np.savez(OUT, K=K, **{f"w.{k}": np.array(v) for k, v in flat.items()})
    print("saved", OUT)


if __name__ == "__main__":
    a = sys.argv[1:]
    # pairs as NAME=CORPUS_NPZ:SAMPLES_FILE after K and EPOCHS; default: the math corpus and its samples
    pairs = [(x.split("=")[0], *x.split("=")[1].split(":")) for x in a[2:]] or [("math", mathexp.NPZ, SAMPLES)]
    main(int(a[0]) if a else 16, int(a[1]) if len(a) > 1 else 8, pairs)
