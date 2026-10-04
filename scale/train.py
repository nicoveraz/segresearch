"""Training loop for the scaling pilot (#27): BLT-lite (scale/model.py) on a byte corpus with a fixed patch mask.

Mirrors mathexp._train: random windows of CTX bytes drawn with np.random.default_rng(seed), bd[:, 0] = 1, mean
cross-entropy, AdamW (weight decay 0.01) with linear warmup and cosine decay to lr/10, exactly as prepare.adamw.
Adds what the cloud needs: bf16 autocast on CUDA, checkpoints every few minutes, and resume.

The corpus and masks are numpy arrays or memmaps (uint8 bytes; uint8 0/1 mask or a bit-packed mask, see
scale/masks.py), so the same code runs on the Mac and on a rented GPU.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from scale.model import BLTLite

CTX = 128


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(k, lr, steps, warmup=None):
    """prepare.adamw: linear warmup from 0 to lr over min(100, steps // 4) updates, then cosine to lr / 10.
    k is the 0-based update index (MLX evaluates the schedule before incrementing its step)."""
    warm = min(100, steps // 4) if warmup is None else warmup
    if k < warm:
        return lr * k / warm
    s = min(k - warm, steps - warm)
    return lr / 10 + 0.5 * (1 + math.cos(math.pi * s / (steps - warm))) * (lr - lr / 10)


class Windows:
    """Draws training batches exactly as mathexp._train does."""

    def __init__(self, data, mask, seed, bs):
        self.data, self.mask, self.bs = data, mask, bs
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        i = self.rng.integers(0, len(self.data) - CTX - 1, self.bs)
        x = np.stack([self.data[j:j + CTX] for j in i]).astype(np.int64)
        y = np.stack([self.data[j + 1:j + CTX + 1] for j in i]).astype(np.int64)
        bd = np.stack([self.mask[j:j + CTX + 1] for j in i]).astype(np.int64)
        bd[:, 0] = 1
        return x, y, bd


def make_model(cfg):
    return BLTLite(d=cfg["d"], glayers=cfg["glayers"], ctx=CTX, pool=cfg.get("pool", "xattn"),
                   local=cfg.get("local", "window"), window=cfg.get("window", 32))


def train(cfg, data, mask, ckpt=None, ckpt_minutes=10.0, log_every=500, model=None, dev=None, losses=None):
    """cfg: d, glayers, steps, bs, lr, seed (+ pool, local, window, warmup). Returns the trained model.
    If ckpt is a path, saves model/optimizer/sampler state there every ckpt_minutes and resumes from it."""
    dev = dev or device()
    torch.manual_seed(cfg["seed"])
    model = (model or make_model(cfg)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    batches = Windows(data, mask, cfg["seed"], cfg["bs"])
    k0 = 0
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        batches.rng.bit_generator.state = st["rng"]; k0 = st["k"]
        print(f"  resumed from {ckpt} at update {k0}", flush=True)
    amp = dev.type == "cuda"
    t_save = time.time(); t0 = time.time(); run = 0.0
    for k in range(k0, cfg["steps"]):
        x, y, bd = (torch.from_numpy(a).to(dev) for a in batches())
        for g in opt.param_groups:
            g["lr"] = lr_at(k, cfg["lr"], cfg["steps"], cfg.get("warmup"))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x, bd)
        loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if losses is not None:
            losses.append(loss.item())
        if log_every and (k + 1) % log_every == 0:
            run = loss.item()
            rate = (k + 1 - k0) * cfg["bs"] * CTX / (time.time() - t0)
            print(f"  update {k + 1:7d}/{cfg['steps']} loss {run:.3f}  {rate / 1e6:.2f} MB/s", flush=True)
        if ckpt and (time.time() - t_save > ckpt_minutes * 60 or k + 1 == cfg["steps"]):
            tmp = ckpt + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "rng": batches.rng.bit_generator.state, "k": k + 1, "cfg": cfg}, tmp)
            os.replace(tmp, ckpt)
            t_save = time.time()
    return model
