"""The two learned patch signals for the scaling pilot (#27), in PyTorch:

  - the small entropy model: trained as prepare.train_lm, scored as prepare.score_stream (entropy of byte i given
    the bytes before it, with at least CTX/2 bytes of context; the first CTX/2 + 1 entries are 0) and, during
    decoding, as mathexp._entropy_window;
  - boundary dependence, fitted as deptrigger.py "marginal" mode: a reference BLT-lite model trained with random
    patch starts (25%), then, for random training windows with a random 10% layout, the drop in mean loss on bytes
    t..t+3 from adding one patch start at t, averaged per preceding 2-byte context (contexts seen < 20 times fall
    back to 1 byte, then to the global mean).
"""
import numpy as np
import torch
import torch.nn.functional as F

from scale.model import EntropyLM
from scale.train import CTX, device, lr_at, make_model, train

ENTROPY = dict(d=32, layers=1, heads=2, lr=3e-3, steps=4000, bs=32)     # prepare.PATCHER
REFERENCE = dict(d=128, glayers=4, steps=16000, bs=32, lr=3e-3, seed=0)  # the reference used for deptrigger.npz


def train_entropy_lm(stream, cfg=ENTROPY, seed=0, dev=None, log_every=1000):
    dev = dev or device()
    torch.manual_seed(seed)
    model = EntropyLM(cfg["d"], cfg["layers"], cfg["heads"], CTX).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for k in range(cfg["steps"]):
        i = rng.integers(0, len(stream) - CTX - 1, cfg["bs"])
        x = torch.from_numpy(np.stack([stream[j:j + CTX] for j in i]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([stream[j + 1:j + CTX + 1] for j in i]).astype(np.int64)).to(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(k, cfg["lr"], cfg["steps"])
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (k + 1) % log_every == 0:
            print(f"    entropy model update {k + 1} loss {loss.item():.3f}", flush=True)
    return model.eval()


@torch.no_grad()
def score_stream(model, stream, dev=None, batch=512):
    """Entropy and surprisal (bits) of byte i given bytes < i, as prepare.score_stream."""
    dev = dev or next(model.parameters()).device
    H = np.zeros(len(stream), np.float32)
    S = np.zeros(len(stream), np.float32)
    half = CTX // 2
    starts = np.arange(0, len(stream) - CTX, half)
    for b in range(0, len(starts), batch):
        st = starts[b:b + batch]
        X = torch.from_numpy(np.stack([stream[s:s + CTX] for s in st]).astype(np.int64)).to(dev)
        lp = torch.log_softmax(model(X).double(), -1)
        h = (-(lp.exp() * lp).sum(-1) / np.log(2)).cpu().numpy()
        lp = lp.cpu().numpy()
        for r, s in enumerate(st):
            tgt = stream[s + half + 1: s + CTX + 1].astype(np.int64)
            H[s + half + 1: s + CTX + 1] = h[r, half:]
            S[s + half + 1: s + CTX + 1] = -lp[r, half:][np.arange(len(tgt)), tgt] / np.log(2)
    return H, S


@torch.no_grad()
def entropy_window(model, w):
    """Entropy of each byte of window w given the bytes before it (0 for the first byte), and of the next byte."""
    dev = next(model.parameters()).device
    lp = torch.log_softmax(model(torch.from_numpy(w[None].astype(np.int64)).to(dev)).double(), -1)[0]
    h = (-(lp.exp() * lp).sum(-1) / np.log(2)).cpu().numpy()
    return np.r_[0.0, h[:-1]].astype(np.float32), float(h[-1])


def fit_dependence(train_b, ref_cfg=REFERENCE, n_windows=4096, K=32, seed=1, dev=None, ref_bytes=None,
                   ref_model=None, log_every=2000):
    """Boundary-dependence table (T2, T1, glob) as deptrigger.py marginal mode (one random layout per window).
    ref_bytes: train the reference on the first ref_bytes bytes only (the random 25% mask over a multi-GB corpus
    would not fit in memory; the reference sees ref_cfg steps x bs x CTX bytes anyway)."""
    dev = dev or device()
    rng = np.random.default_rng(seed)
    sub = train_b if ref_bytes is None else train_b[:ref_bytes]
    if ref_model is None:
        ref = (rng.random(len(sub)) < 0.25).astype(np.uint8)
        print(f"  reference model: {ref_cfg['steps']} updates, random patch rate {ref.mean():.3f}", flush=True)
        ref_model = train(ref_cfg, sub, ref, dev=dev, log_every=log_every)
    ref_model.eval()
    starts = rng.integers(0, len(sub) - CTX - 1, n_windows)
    s2, c2, s1, c1 = (np.zeros(65536), np.zeros(65536), np.zeros(256), np.zeros(256))
    allg = []
    for w in starts:
        b = sub[w:w + CTX].astype(np.int64)
        X = torch.from_numpy(np.repeat(sub[w:w + CTX][None].astype(np.int64), K + 1, 0)).to(dev)
        Y = np.repeat(sub[w + 1:w + CTX + 1][None].astype(np.int64), K + 1, 0)
        bd = rng.random(CTX + 1) < 0.10
        bd[0] = True
        cand = rng.choice(np.flatnonzero(~bd[CTX // 4:CTX - 3]) + CTX // 4, K, replace=False)
        BD = np.repeat(bd[None], K + 1, 0)
        BD[np.arange(1, K + 1), cand] = True
        with torch.no_grad():
            lp = torch.log_softmax(ref_model(X, torch.from_numpy(BD.astype(np.int64)).to(dev)).double(), -1).cpu().numpy()
        L = -np.take_along_axis(lp, Y[..., None], -1)[..., 0]
        cs = np.cumsum(np.c_[np.zeros(K + 1), L], 1)
        win = lambda r, t: (cs[r, t + 3] - cs[r, t - 1]) / 4                 # mean loss on bytes t..t+3
        g = win(0, cand) - win(np.arange(1, K + 1), cand)                     # drop from adding a boundary at t
        key = b[cand - 2] * 256 + b[cand - 1]
        np.add.at(s2, key, g); np.add.at(c2, key, 1); np.add.at(s1, b[cand - 1], g); np.add.at(c1, b[cand - 1], 1)
        allg.append(g)
    glob = float(np.concatenate(allg).mean())
    T2 = np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan)
    T1 = np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan)
    top = np.argsort(-np.nan_to_num(T2, nan=-1e9))[:12]
    print(f"  {int((c2 >= 20).sum())} contexts; mean gain {glob:.3f} nats; top: "
          + ", ".join(f"{bytes([int(k) // 256, int(k) % 256])!r} {T2[k]:.2f}" for k in top), flush=True)
    return {"T2": T2, "T1": T1, "glob": glob}
