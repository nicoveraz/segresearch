"""Learned chunking baseline: an H-Net-style dynamic chunker, simplified to fit the BLT-lite harness.

Instead of a fixed boundary mask from boundary.py, the model decides its own patch starts and learns
them end to end from the prediction loss, with no labels and no entropy signals:

  - a byte encoder (one causal layer, 32-byte window) turns bytes into states h_t
  - a router gives the probability that byte t+1 starts a patch from the cosine similarity of
    neighbouring states, 0.5 * (1 - cos(q(h_t), k(h_{t-1}))), so the decision only uses bytes <= t and
    cannot leak the byte being predicted
  - patches start where that probability is >= 0.5; the rest of the model is the harness's BLT-lite
    (pooling, 2-layer global transformer, local decoder), with two H-Net devices that let gradients
    reach the router: each pooled patch is scaled by its boundary probability (straight-through: the
    forward value is 1) and the global outputs are smoothed by an EMA weighted by that probability
  - a ratio loss (H-Net) pulls the boundary rate toward SEGR_CHUNK_RATE (default 0.10)

Same data, steps, windows and metric as run.py, so results compare directly with boundary rules.

    uv run learned_chunking.py [--seed 0]                      # dev formats of the default suite
    SEGR_SUITE=hard uv run learned_chunking.py                  # hard corpus (formats D, E)
"""
import argparse
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import harness as H
import prepare
from prepare import CTX, MAIN_BS, MAIN_STEPS, RI, ROLES, _ln, adamw, log_softmax_np, make_step

D = H.D
TARGET = float(os.environ.get("SEGR_CHUNK_RATE", "0.10"))
ALPHA = 0.03            # ratio-loss weight (H-Net)
ENC_WINDOW = 32


def _init(key):
    p = H._init(key)
    ks = mx.random.split(mx.random.split(key, 18)[17], 6)
    n = lambda k, sh: mx.random.normal(sh, key=k) * 0.02
    p["enc"] = {"ln1": mx.ones(D), "ln2": mx.ones(D), "qkv": n(ks[0], (D, 3 * D)), "o": n(ks[1], (D, D)),
                "w1": n(ks[2], (D, 4 * D)), "w2": n(ks[3], (4 * D, D))}
    p["router"] = {"ln": mx.ones(D), "q": n(ks[4], (D, D)), "k": n(ks[5], (D, D))}
    return p


def _model(p, x):
    """x: (B,T) bytes. Returns logits (position t predicts x_{t+1}), the hard boundary fraction, the mean
    boundary probability, and the boundary flags (B, T+1) for bytes x_0..x_T."""
    B, T = x.shape
    ar = mx.arange(T)
    causal = (ar[:, None] >= ar[None])[None]
    one_hot = lambda i: (i[:, :, None] == ar).astype(mx.float32)

    h = H._block(p["enc"], p["emb"][x], causal & ((ar[:, None] - ar[None]) < ENC_WINDOW)[None])
    hn = _ln(h, p["router"]["ln"])
    q, k = hn @ p["router"]["q"], hn @ p["router"]["k"]
    qa, kb = q[:, 1:], k[:, :-1]
    cos = (qa * kb).sum(-1) / (mx.sqrt((qa * qa).sum(-1) * (kb * kb).sum(-1)) + 1e-6)
    pb = mx.concatenate([mx.ones((B, 1)), mx.zeros((B, 1)), 0.5 * (1 - cos)], axis=1)   # bytes 0..T
    bd = mx.stop_gradient((pb >= 0.5).astype(mx.int32))

    pid = mx.cumsum(bd[:, :T], axis=1) - 1
    start = mx.where(bd[:, :T] == 1, ar[None], 0)
    off = ar[None] - mx.cummax(start, axis=1)
    e = h + p["off"][off]
    oh = one_hot(pid)                                                     # (B, T, P)
    conf = (oh * (bd[:, :T] * pb[:, :T])[:, :, None]).sum(1)              # (B, P): prob of each patch's start
    conf_ste = conf + mx.stop_gradient(1 - conf)                          # forward 1, gradient to the router
    G = (oh.transpose(0, 2, 1) @ e) * conf_ste[:, :, None] + p["ppos"][None]
    for L in p["g"]:
        G = H._block(L, G, causal)

    # dechunk smoothing (H-Net): Gs_p = sum_{j<=p} P_j * prod_{j<i<=p} (1 - P_i) * G_j
    P = mx.clip(conf, 1e-4, 1.0)
    lc = mx.cumsum(mx.log(mx.clip(1 - P, 1e-6, 1.0)), axis=1)
    diff = mx.where(causal, lc[:, :, None] - lc[:, None, :], -1e9)
    Gs = (mx.exp(diff) * P[:, None, :]) @ G

    gidx = mx.where(bd[:, 1:T + 1] == 1, pid, pid - 1)                   # fresh context iff x_{t+1} starts a patch
    hh = e + (one_hot(gidx) @ Gs) @ p["gproj"]
    if H.LOCAL == "window":
        local = causal & ((ar[:, None] - ar[None]) < H.WINDOW)[None]
    else:
        local = (pid[:, :, None] == pid[:, None, :]) & causal
    hh = H._block(p["loc"], hh, local)
    logits = _ln(hh, p["lnf"]) @ p["emb"].T
    F = bd[:, 1:].astype(mx.float32).mean()
    return logits, F, pb[:, 1:].mean(), bd


def _loss(p, x, y):
    logits, F, G, _ = _model(p, x)
    N = 1.0 / TARGET
    ratio = N / (N - 1) * ((N - 1) * F * G + (1 - F) * (1 - G))
    return nn.losses.cross_entropy(logits, y, reduction="mean") + ALPHA * ratio


def train_eval(train_bytes, eval_bytes, eval_roles, seed=0):
    tr = train_bytes.astype(np.int32)
    p, step = make_step(_init(mx.random.key(seed)), _loss, adamw(3e-3, MAIN_STEPS, 3e-4))
    rng = np.random.default_rng(seed)
    for _ in range(MAIN_STEPS):
        i = rng.integers(0, len(tr) - CTX - 1, MAIN_BS)
        step(np.stack([tr[j:j + CTX] for j in i]), np.stack([tr[j + 1:j + CTX + 1] for j in i]))

    ev = eval_bytes.astype(np.int32)
    n = (len(ev) - 1) // CTX
    X = np.stack([ev[k * CTX:(k + 1) * CTX] for k in range(n)])
    Y = np.stack([ev[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    Rl = np.stack([eval_roles[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    lps, bds = [], []
    for b in range(0, n, 128):
        logits, _, _, bd = _model(p, mx.array(X[b:b + 128]))
        lps.append(log_softmax_np(logits)); bds.append(np.array(bd)[:, 1:])
    lp, BD = np.concatenate(lps), np.concatenate(bds).astype(bool)      # BD[:, t]: x_{t+1} starts a patch
    bits = -np.take_along_axis(lp, Y[..., None], -1)[..., 0] / np.log(2)
    acc = lp.argmax(-1) == Y
    keep = np.zeros_like(bits, bool); keep[:, CTX // 2:] = True
    out = {"bpb": float(bits[keep].mean()), "boundary_rate": float(BD.mean())}
    ans = keep & np.isin(Rl, [RI["ANS_LOCAL"], RI["ANS_LONG"]])
    out["ans_bits"], out["ans_acc"] = float(bits[ans].mean()), float(acc[ans].mean())
    for r in ROLES:
        m = keep & (Rl == RI[r])
        out[f"{r}_bits"], out[f"{r}_acc"] = float(bits[m].mean()), float(acc[m].mean())
        out[f"{r}_start_rate"] = float(BD[Rl == RI[r]].mean())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); t0 = time.time(); res = {}
    for fmt in (os.environ["FMT"],) if os.environ.get("FMT") else prepare.FORMATS_DEV:
        tr, _ = prepare.load(fmt, "train"); va, va_roles = prepare.load(fmt, "val")
        res[fmt] = o = train_eval(tr.bytes, va.bytes, va_roles, seed=args.seed)
        print(f"[{fmt}] ans_bits {o['ans_bits']:.4f} acc {o['ans_acc']:.3f} | local {o['ANS_LOCAL_bits']:.3f} "
              f"long {o['ANS_LONG_bits']:.3f} | bpb {o['bpb']:.4f} | rate {o['boundary_rate']:.3f}", flush=True)
        print(f"[{fmt}] patch-start rate by role: " +
              ", ".join(f"{r} {o[r + '_start_rate']:.2f}" for r in ROLES), flush=True)
    mean = lambda k: float(np.mean([res[f][k] for f in res]))
    print("---")
    print(f"val_ans_bits:      {mean('ans_bits'):.6f}")
    print(f"val_ans_local_bits:{mean('ANS_LOCAL_bits'):.6f}")
    print(f"val_ans_long_bits: {mean('ANS_LONG_bits'):.6f}")
    print(f"boundary_rate:     {mean('boundary_rate'):.4f}")
    print(f"seed:              {args.seed}")
    print(f"elapsed_seconds:   {time.time() - t0:.0f}")
