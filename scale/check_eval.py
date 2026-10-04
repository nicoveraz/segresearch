"""scale/evaluate.py must reproduce mathexp._bits and mathexp._accuracy (Mac only).

    .venv-scale/bin/python -m scale.check_eval

A small MLX model is trained briefly (1,500 updates, D=128) under each rule; its weights, the cached MLX entropy model
and the rule thresholds go to the PyTorch side; bits per byte and exact-match counts must agree.
"""
import sys

import mlx.core as mx
import numpy as np
import torch

import harness
import mathexp
import prepare
from scale.check_port import _to_torch
from scale.evaluate import accuracy, bits
from scale.model import EntropyLM
from scale.rules import Rule
from scale.train import make_model

torch.set_grad_enabled(False)
N_ACC = 60
harness.D, harness.GLAYERS, harness.POOL, harness.LOCAL = 128, 4, "xattn", "window"
mathexp.N_ACC = N_ACC
z = np.load(mathexp.NPZ)
data = {k: z[k] for k in ("train_bytes", "train_H", "val_bytes", "val_H", "val_roles")}
table = np.load(mathexp.CACHE + "/deptrigger.npz")
pat = mathexp._patcher(z)
ent = EntropyLM(prepare.PATCHER["d"], prepare.PATCHER["layers"], prepare.PATCHER["heads"], prepare.CTX)
sd = ent.state_dict()
for k in z.files:
    if k.startswith("pat_"):
        sd[k[4:].replace("/", ".")].copy_(torch.from_numpy(z[k]))
ent.eval()

failed = []
for name in ("entropy10", "dep10", "syntax+entropy10"):
    old = mathexp.Rule(name, data)
    new = Rule(name, data["train_bytes"], data["train_H"], table)
    m_tr = old.mask(data["train_bytes"], data["train_H"]); m_va = old.mask(data["val_bytes"], data["val_H"])
    p = mathexp._train(data["train_bytes"], m_tr, 1500, seed=0)
    tm = make_model(dict(d=128, glayers=4))
    _to_torch(tm, {k: v for k, v in p.items()}, scale=1.0)
    tm.eval()
    b_old = mathexp._bits(p, data["val_bytes"], m_va, data["val_roles"])
    b_new = bits(tm, data["val_bytes"], m_va, data["val_roles"])
    a_old = mathexp._accuracy(p, pat, old, data["val_bytes"], data["val_roles"], 0)
    a_new = accuracy(tm, ent, new, data["val_bytes"], data["val_roles"], 0, n_acc=N_ACC)
    db = abs(b_old["bpb"] - b_new["bpb"])
    hits = {k: (round(a_old[k] * N_ACC), round(a_new[k] * N_ACC)) for k in a_old if k.endswith("_acc")}
    ok = db < 1e-4 and all(abs(x - y) <= 1 for x, y in hits.values())
    failed += [] if ok else [name]
    print(f"  {'ok  ' if ok else 'FAIL'} {name:17s} bpb MLX {b_old['bpb']:.5f} torch {b_new['bpb']:.5f} | "
          + " ".join(f"{k[:-4]} {x}/{y}" for k, (x, y) in hits.items()) + f"  (hits MLX/torch of {N_ACC})")
if failed:
    print("FAILED: " + ", ".join(failed))
    sys.exit(1)
print("evaluation matches mathexp")
