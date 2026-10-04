"""Check that scale/model.py reproduces the MLX models exactly (Mac only: needs mlx and torch).

    .venv-scale/bin/python -m scale.check_port

MLX weights (harness._init, prepare.init_lm, scaled up so activations are not near zero) are copied into the
PyTorch models; logits must match on random bytes and random patch layouts (rates from one patch per window to every
byte), for every pooling / local-decoder setting. Exits 1 on any mismatch.
"""
import sys

import mlx.core as mx
import numpy as np
import torch

import harness
import prepare
from scale.model import BLTLite, EntropyLM

torch.set_grad_enabled(False)
RNG = np.random.default_rng(0)
TOL = 2e-4                 # max |difference| relative to max |logit|


def _to_torch(model, params, scale):
    """Copy a nested MLX parameter dict into a torch module with the same parameter names."""
    flat = {}

    def walk(t, prefix):
        if isinstance(t, dict):
            for k, v in t.items():
                walk(v, prefix + [k])
        elif isinstance(t, list):
            for i, v in enumerate(t):
                walk(v, prefix + [str(i)])
        else:
            flat[".".join(prefix)] = np.array(t)
    walk(params, [])
    sd = model.state_dict()
    assert set(flat) == set(sd), (set(flat) ^ set(sd))
    for k, v in flat.items():
        s = 1.0 if k.split(".")[-1].startswith("ln") else scale
        sd[k].copy_(torch.from_numpy(v * s))
    return {k: mx.array(v * (1.0 if k.split(".")[-1].startswith("ln") else scale)) for k, v in flat.items()}


def _nest_lists(flat):
    out = {}
    for k, v in flat.items():
        parts = k.split(".")
        cur = out
        for i, p in enumerate(parts[:-1]):
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v

    def fix(t):
        if isinstance(t, dict):
            t = {k: fix(v) for k, v in t.items()}
            if t and all(k.isdigit() for k in t):
                return [t[str(i)] for i in range(len(t))]
        return t
    return fix(out)


def layouts(B, T):
    out = []
    for rate in (0.0, 0.1, 0.3, 1.0):
        bd = (RNG.random((B, T + 1)) < rate).astype(np.int32)
        bd[:, 0] = 1
        out.append((rate, bd))
    return out


def check_training(steps=300):
    """Same initial weights, same batches, same schedule: MLX (as in mathexp._train) vs scale/train.py on CPU.

    Training at lr 3e-3 is chaotic: two PyTorch runs whose initial weights differ by 1e-6 drift apart by a few %
    after ~150 updates. So the checks are: (1) the learning-rate schedule equals MLX's at every update; (2) per-update
    losses agree over the first 50 updates (before chaos sets in); (3) the mean loss over the last 100 updates
    is within the chaos of PyTorch itself: its distance from the mean of five PyTorch runs (one unperturbed, four
    perturbed by 1e-6) is at most twice their spread, with a 1% floor. (MLX on the GPU is not bit-deterministic, so
    its trajectory is a different random draw each time.)"""
    import mathexp
    import mlx.nn as mnn
    from scale.train import lr_at, make_model, train
    harness.D, harness.GLAYERS, harness.POOL, harness.LOCAL = 128, 4, "xattn", "window"
    z = np.load(mathexp.NPZ)
    btr, Htr = z["train_bytes"][:3_000_000], z["train_H"][:3_000_000]
    mask = mathexp.Rule("entropy10", {"train_bytes": btr, "train_H": Htr}).mask(btr, Htr).astype(np.int32)
    p0 = harness._init(mx.random.key(3))
    cfg = dict(d=128, glayers=4, steps=steps, bs=32, lr=3e-3, seed=0)
    opt = prepare.adamw(3e-3, steps, 3e-4)
    sched = opt._schedulers["learning_rate"]
    sdiff = max(abs(float(sched(mx.array(k))) - lr_at(k, 3e-3, steps)) for k in range(steps))
    torch.set_grad_enabled(True)
    runs = []
    for eps in (0.0, 1e-6, 1e-6, 1e-6, 1e-6):
        tm = make_model(cfg)
        _to_torch(tm, p0, scale=1.0)
        g = torch.Generator().manual_seed(len(runs) + 1)
        with torch.no_grad():
            for p in tm.parameters():
                p.add_(eps * torch.randn(p.shape, generator=g))
        lt = []
        train(cfg, btr, mask, model=tm, dev=torch.device("cpu"), losses=lt, log_every=0)
        runs.append(np.array(lt))
    torch.set_grad_enabled(False)
    lossf = lambda p, x, y, bd: mnn.losses.cross_entropy(harness._model(p, x, bd), y, reduction="mean")
    pm, step = prepare.make_step(p0, lossf, opt)
    rng = np.random.default_rng(0); tr = btr.astype(np.int32); lm = []
    for _ in range(steps):
        i = rng.integers(0, len(tr) - T - 1, 32)
        bd = np.stack([mask[j:j + T + 1] for j in i]); bd[:, 0] = 1
        lm.append(step(np.stack([tr[j:j + T] for j in i]), np.stack([tr[j + 1:j + T + 1] for j in i]), bd).item())
    lm, lt = np.array(lm), runs[0]
    early = float((np.abs(lt - lm) / lm)[:50].max())
    tails = np.array([r[-100:].mean() for r in runs])              # the unperturbed run and four perturbed ones
    t_mlx = lm[-100:].mean()
    spread = (tails.max() - tails.min()) / tails.mean()
    d_mlx = abs(t_mlx - tails.mean()) / tails.mean()
    print(f"  schedule max |MLX - torch| {sdiff:.1e}; first 50 updates max rel diff {early:.1e}; mean loss over "
          f"last 100: MLX {t_mlx:.4f}, torch runs {tails.min():.4f}-{tails.max():.4f} (spread {spread:.2%}); "
          f"MLX vs torch mean {d_mlx:.2%}")
    return sdiff < 1e-8 and early < 1e-3 and d_mlx <= max(2 * spread, 0.01)


T = prepare.CTX


def main():
    worst, failed = 0.0, []
    for d, gl, pool, local in [(64, 2, "sum", "patch"), (64, 2, "xattn", "patch"), (64, 2, "sum", "window"),
                               (128, 4, "xattn", "window"), (96, 3, "xattn", "window")]:
        harness.D, harness.GLAYERS, harness.POOL, harness.LOCAL = d, gl, pool, local
        p = harness._init(mx.random.key(1))
        tm = BLTLite(d=d, glayers=gl, ctx=T, pool=pool, local=local, window=harness.WINDOW)
        flat = _to_torch(tm, p, scale=8.0)
        pm = _nest_lists(flat)
        for rate, bd in layouts(4, T):
            x = RNG.integers(0, 256, (4, T)).astype(np.int32)
            a = np.array(harness._model(pm, mx.array(x), mx.array(bd)))
            b = tm(torch.from_numpy(x).long(), torch.from_numpy(bd)).numpy()
            err = float(np.abs(a - b).max() / np.abs(a).max())
            worst = max(worst, err)
            ok = err < TOL
            failed += [] if ok else [f"BLTLite d={d} glayers={gl} {pool}/{local} rate={rate}"]
            print(f"  {'ok  ' if ok else 'FAIL'} BLTLite d={d:3d} glayers={gl} pool={pool:5s} local={local:6s} "
                  f"rate={rate:.1f}  rel err {err:.1e}")
    pl = prepare.init_lm(mx.random.key(2), prepare.PATCHER["d"], prepare.PATCHER["layers"])
    te = EntropyLM(prepare.PATCHER["d"], prepare.PATCHER["layers"], prepare.PATCHER["heads"], T)
    flat = _to_torch(te, pl, scale=8.0)
    x = RNG.integers(0, 256, (4, T)).astype(np.int32)
    a = np.array(prepare.lm_forward(_nest_lists(flat), mx.array(x), prepare.PATCHER["heads"]))
    b = te(torch.from_numpy(x).long()).numpy()
    err = float(np.abs(a - b).max() / np.abs(a).max())
    worst = max(worst, err)
    failed += [] if err < TOL else ["EntropyLM"]
    print(f"  {'ok  ' if err < TOL else 'FAIL'} EntropyLM                                          rel err {err:.1e}")
    print(f"worst relative error {worst:.1e} (tolerance {TOL:.0e})")

    # causality: logits for bytes <= t+1 must not change when later bytes or later patch flags change
    # (the compacted global model's size depends on the whole window's patch count; this checks it never leaks)
    leaks = 0
    for pool, local in (("xattn", "window"), ("sum", "patch")):
        tm = BLTLite(d=64, glayers=2, ctx=T, pool=pool, local=local)
        for _ in range(20):
            x = torch.from_numpy(RNG.integers(0, 256, (2, T))).long()
            bd = torch.from_numpy((RNG.random((2, T + 1)) < 0.15).astype(np.int64)); bd[:, 0] = 1
            t = int(RNG.integers(4, T - 4))
            x2, bd2 = x.clone(), bd.clone()
            x2[:, t + 1:] = torch.from_numpy(RNG.integers(0, 256, (2, T - t - 1)))
            bd2[:, t + 2:] = torch.from_numpy((RNG.random((2, T - t - 1)) < 0.5).astype(np.int64))
            leaks += not torch.allclose(tm(x, bd)[:, :t + 1], tm(x2, bd2)[:, :t + 1], atol=1e-5)
    print(f"  {'ok  ' if leaks == 0 else 'FAIL'} causality: {leaks}/40 perturbations changed earlier logits")
    failed += [] if leaks == 0 else ["causality"]

    ok = check_training()
    print(f"  {'ok  ' if ok else 'FAIL'} training equivalence (300 updates, D=128)")
    failed += [] if ok else ["training"]
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
