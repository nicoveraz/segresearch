"""Can a learned, label-free boundary predictor find structure that the 2-byte tables miss? (local screen)

The novelty table (nov) and the boundary-dependence table (dep) predict their targets from the previous 2 bytes only.
scale/novstruct.py showed that on code the novelty table never fires at statement starts or right-hand sides and
spends its boundaries inside hex/escape literals. Here the same two targets are predicted by a small causal network
that reads the previous K bytes (no markers, no labels, one recipe for every field):

    novelty   upcoming bytes that do not repeat the last 128 (minus the forward match length; reuse_screen)
    gain      how much one extra patch start at t lowers a reference BLT-lite's loss on bytes t..t+3
              (deptrigger marginal mode; the dep table's target)

network: byte embedding (32) -> causal dilated convolutions (64 channels, dilations 1..64, receptive field 128)
         -> value at t from bytes < t

For each domain (math: the scaling corpus; code: Python files installed on this machine, split by file) and target,
the screen fits the 2-byte table and the network on the same training bytes, picks each one's top 10% on train,
and reports on held-out bytes: rank correlation with the target, and where the boundaries fall (statement / line
starts, right after '=' / 'return ' / '(', token starts, mid-identifier) against chance.

    .venv-scale/bin/python -m scale.learned_boundary [math|code ...]       # SEGR_LB_REF_STEPS (default 6000)
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scale.generic import forward_match, tab_fit
from scale.patchers import REFERENCE, _hi
from scale.rules import syntax_starts, table_lookup
from scale.train import CTX, device, train

BUNDLE = os.path.expanduser("~/.cache/segresearch-scale/bundle")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "learned_boundary")
REF_STEPS = int(os.environ.get("SEGR_LB_REF_STEPS", 6000))
ALNUM = np.frombuffer(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", np.uint8)


def load(domain):
    if domain == "math":
        tr = np.fromfile(f"{BUNDLE}/math_train.u8", np.uint8)[:6_000_000]
        va = np.fromfile(f"{BUNDLE}/val.u8", np.uint8)[:1_500_000]
        return tr, va
    files = sorted(set(glob.glob(os.path.join(os.path.dirname(os.__file__), "**", "*.py"), recursive=True)))
    rng = np.random.default_rng(0); rng.shuffle(files)
    tr, va, n = [], [], 0
    for i, f in enumerate(files):
        t = open(f, "rb").read()
        if len(t) < 200:
            continue
        (va if i % 5 == 0 else tr).append(t[:30000]); n += min(len(t), 30000)
        if n > 7_500_000:
            break
    return np.frombuffer(b"\n\n".join(tr), np.uint8)[:6_000_000].copy(), np.frombuffer(b"\n\n".join(va), np.uint8)[:1_500_000].copy()


def marks(b):
    tok = np.isin(b, ALNUM) & np.r_[True, ~np.isin(b[:-1], ALNUM)]
    first = np.zeros(len(b), bool)
    for m in re.finditer(rb"\n[ \t]*(\S)", b.tobytes()):
        first[m.start(1)] = True
    return {"line/statement start": first, "after '='/'return '/'('": syntax_starts(b, [b"=", b"return ", b"("]),
            "token start": tok, "mid-identifier": np.isin(b, ALNUM) & ~tok}


class Net(nn.Module):
    def __init__(self, ch=64, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.emb = nn.Embedding(257, 32)                  # 256 = padding before the stream
        self.inp = nn.Conv1d(32, ch, 1)
        self.convs = nn.ModuleList(nn.Conv1d(ch, ch, 2, dilation=d) for d in dil)
        self.dil = dil
        self.out = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.GELU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x):                                 # x: (B, L) bytes; output (B, L): value at t from x[<t]
        x = F.pad(x, (1, 0), value=256)[:, :-1]           # shift: position t sees bytes up to t-1
        h = self.inp(self.emb(x).transpose(1, 2))
        for c, d in zip(self.convs, self.dil):
            h = h + F.gelu(c(F.pad(h, (d, 0))))
        return self.out(h)[:, 0]


def fit_net(b, y, w=None, steps=3000, L=512, bs=64, dev=None, seed=0):
    """Regress y (per byte; w = weight, 0 where no target) from the preceding bytes."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
    mu, sd = float(np.average(y, weights=w)), float(np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w)))
    w = np.ones(len(b), np.float32) if w is None else w.astype(np.float32)
    for k in range(steps):
        i = rng.integers(0, len(b) - L, bs)
        X = torch.from_numpy(np.stack([b[j:j + L] for j in i]).astype(np.int64)).to(dev)
        Y = torch.from_numpy(np.stack([(y[j:j + L] - mu) / sd for j in i]).astype(np.float32)).to(dev)
        W = torch.from_numpy(np.stack([w[j:j + L] for j in i])).to(dev)
        W[:, :CTX // 2] = 0                               # warm-up context
        for g in opt.param_groups:
            g["lr"] = 2e-3 * min(1, (k + 1) / 200) * 0.5 * (1 + np.cos(np.pi * k / steps))
        loss = ((net(X) - Y) ** 2 * W).sum() / W.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net, b, dev, L=4096):
    out = np.zeros(len(b), np.float32)
    for s in range(0, len(b), L):
        lo = max(0, s - CTX); seg = torch.from_numpy(b[lo:s + L].astype(np.int64))[None].to(dev)
        out[s:s + L] = net(seg)[0, s - lo:].float().cpu().numpy()
    return out


@torch.no_grad()
def predict_batched(net, b, dev, L=4096, bs=64):
    """predict() for long streams: rows of CTX context + L positions, bs rows per call (the stream is left-padded with
    the padding byte 256, so the first row sees the same empty context as predict())."""
    out = np.zeros(len(b), np.float32)
    p = np.r_[np.full(CTX, 256, np.int64), np.asarray(b, np.int64), np.full(L, 256, np.int64)]
    starts = np.arange(0, len(b), L)
    for i in range(0, len(starts), bs):
        st = starts[i:i + bs]
        X = torch.from_numpy(np.stack([p[s:s + CTX + L] for s in st])).to(dev)
        y = net(X)[:, CTX:].float().cpu().numpy()
        for s, row in zip(st, y):
            out[s:s + L] = row[:len(b) - s] if s + L > len(b) else row
    return out


def gain_samples(b, dev, n_windows=4096, K=32, seed=1):
    """Per-byte gain target (deptrigger marginal mode, as patchers.fit_dependence) at K sampled positions per window."""
    rng = np.random.default_rng(seed)
    ref = train(dict(REFERENCE, steps=REF_STEPS), b, (rng.random(len(b)) < 0.25).astype(np.uint8), dev=dev, log_every=0)
    ref.eval()
    y = np.zeros(len(b), np.float32); w = np.zeros(len(b), np.float32)
    for s in rng.integers(0, len(b) - CTX - 1, n_windows):
        X = torch.from_numpy(np.repeat(b[s:s + CTX][None].astype(np.int64), K + 1, 0)).to(dev)
        Y = np.repeat(b[s + 1:s + CTX + 1][None].astype(np.int64), K + 1, 0)
        bd = rng.random(CTX + 1) < 0.10; bd[0] = True
        cand = rng.choice(np.flatnonzero(~bd[CTX // 4:CTX - 3]) + CTX // 4, K, replace=False)
        BD = np.repeat(bd[None], K + 1, 0); BD[np.arange(1, K + 1), cand] = True
        with torch.no_grad():
            lp = torch.log_softmax(ref(X, torch.from_numpy(BD.astype(np.int64)).to(dev)).to(_hi(dev)), -1).cpu().numpy()
        Lo = -np.take_along_axis(lp, Y[..., None], -1)[..., 0]
        cs = np.cumsum(np.c_[np.zeros(K + 1), Lo], 1)
        win = lambda r, t: (cs[r, t + 3] - cs[r, t - 1]) / 4
        y[s + cand] = win(0, cand) - win(np.arange(1, K + 1), cand); w[s + cand] = 1
    return y, w


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def report(name, pred_tr, pred_va, va, y_va, w_va, mk, rate=0.10):
    thr = np.quantile(pred_tr, 1 - rate); m = pred_va > thr
    sel = w_va > 0
    row = {"rate": float(m.mean()), "spearman": spearman(pred_va[sel], y_va[sel])}
    line = f"    {name:22s} rate {m.mean():5.1%}  rank corr {row['spearman']:.2f} |"
    for k, s in mk.items():
        cov, share = (m & s).sum() / max(s.sum(), 1), (m & s).sum() / max(m.sum(), 1)
        row[k] = {"covered": float(cov), "share": float(share), "chance": float(s.mean())}
        line += f" {k}: {cov:4.0%} cov, {share:3.0%} of bd (chance {s.mean():3.0%}) |"
    print(line, flush=True)
    return row


def main(domains):
    dev = device(); os.makedirs(OUT, exist_ok=True)
    for dom in domains:
        t0 = time.time()
        tr, va = load(dom); mk = marks(va); res = {}
        print(f"== {dom}: {len(tr) / 1e6:.1f} MB train, {len(va) / 1e6:.1f} MB held out", flush=True)
        # novelty target
        ytr, yva = -forward_match(tr).astype(np.float32), -forward_match(va).astype(np.float32)
        tab = tab_fit(tr, ytr)
        print(f"  novelty target ({time.time() - t0:.0f}s)", flush=True)
        res["novelty_table"] = report("novelty, 2-byte table", table_lookup(*tab, tr), table_lookup(*tab, va), va, yva, np.ones(len(va)), mk)
        net = fit_net(tr, ytr, dev=dev)
        res["novelty_net"] = report("novelty, network", predict(net, tr, dev), predict(net, va, dev), va, yva, np.ones(len(va)), mk)
        res["novelty_oracle"] = report("novelty, true target", ytr, yva, va, yva, np.ones(len(va)), mk)
        # gain target
        print(f"  gain target: reference {REF_STEPS} updates + samples ({time.time() - t0:.0f}s)", flush=True)
        gtr, wtr = gain_samples(tr, dev)
        sel = wtr > 0
        # 2-byte gain table from the sampled positions (as fit_dependence)
        b64 = tr.astype(np.int64); key = np.r_[0, 0, b64[:-2]] * 256 + np.r_[0, b64[:-1]]
        s2, c2 = np.bincount(key[sel], gtr[sel], 65536), np.bincount(key[sel], None, 65536)
        p1 = np.r_[0, b64[:-1]]; s1, c1 = np.bincount(p1[sel], gtr[sel], 256), np.bincount(p1[sel], None, 256)
        gtab = (np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan), np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan), float(gtr[sel].mean()))
        # held-out check of rank correlation: split the sampled training positions by position (last 20% held out)
        cut = int(len(tr) * 0.8); fitw = wtr.copy(); fitw[cut:] = 0
        gnet = fit_net(tr, gtr, w=fitw, dev=dev)
        ptr_net, ptr_tab = predict(gnet, tr, dev), table_lookup(*gtab, tr)
        ho = sel.copy(); ho[:cut] = False
        print(f"  gain rank corr on held-out sampled positions: table {spearman(ptr_tab[ho], gtr[ho]):.2f} "
              f"(fitted on all, optimistic), network {spearman(ptr_net[ho], gtr[ho]):.2f}", flush=True)
        dummy_y, dummy_w = np.zeros(len(va), np.float32), np.ones(len(va))
        res["gain_table"] = report("gain, 2-byte table", ptr_tab, table_lookup(*gtab, va), va, dummy_y, dummy_w, mk)
        res["gain_net"] = report("gain, network", ptr_net, predict(gnet, va, dev), va, dummy_y, dummy_w, mk)
        json.dump(res, open(f"{OUT}/{dom}.json", "w"), indent=1)
        print(f"  {dom} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["code", "math"])
