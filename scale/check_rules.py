"""scale/rules.py must give the same masks as mathexp.Rule (Mac only: needs the MLX code and the math cache).

    .venv-scale/bin/python -m scale.check_rules
"""
import sys

import numpy as np

import mathexp
from scale.rules import Rule

z = np.load(mathexp.NPZ)
data = {k: z[k] for k in ("train_bytes", "train_H", "val_bytes", "val_H")}
table = np.load(mathexp.CACHE + "/deptrigger.npz")
failed = []
for name in ("entropy10", "dep10", "syntax+entropy10", "entropy15", "dep20", "syntax+entropy20", "jump10", "entropy20"):
    old = mathexp.Rule(name, data)
    new = Rule(name, data["train_bytes"], data["train_H"], table)
    re_new = Rule.from_state(new.state(), table)
    for split in ("train", "val"):
        b, H = data[f"{split}_bytes"], data[f"{split}_H"]
        a, c, d = old.mask(b, H), new.mask(b, H), re_new.mask(b, H)
        ok = np.array_equal(a, c) and np.array_equal(a, d)
        failed += [] if ok else [f"{name} {split}"]
        print(f"  {'ok  ' if ok else 'FAIL'} {name:17s} {split:5s} rate {a.mean():.4f}  differing {int((a != c).sum())}")
if failed:
    print("FAILED: " + ", ".join(failed))
    sys.exit(1)
print("all rules identical to mathexp.Rule")
