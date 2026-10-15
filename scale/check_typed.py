"""scale's typed uncertainty must match the MLX version (typed_screen.py / mathexp typed rules). Mac only.

    .venv-scale/bin/python -m scale.check_typed

The cached MLX entropy model is loaded into the PyTorch EntropyLM (as check_patchers.py). Compares score_typed with
typed_screen.score_typed on a val slice, typed_window with the short-window path of mathexp's typed rule, and the
typed10 masks (thresholds fitted on the same train bytes by each implementation).
"""
import sys

import numpy as np
import torch

import mathexp
import prepare
import typed_screen
from scale.model import EntropyLM
from scale.patchers import score_typed, typed_window
from scale.rules import Rule

z = np.load(mathexp.NPZ)
pat = mathexp._patcher(z)
model = EntropyLM(prepare.PATCHER["d"], prepare.PATCHER["layers"], prepare.PATCHER["heads"], prepare.CTX)
sd = model.state_dict()
for k in z.files:
    if k.startswith("pat_"):
        sd[k[4:].replace("/", ".")].copy_(torch.from_numpy(z[k]))
model.eval()

val = z["val_bytes"][:200_000]
T_mlx = typed_screen.score_typed(pat, prepare.PATCHER["heads"], val)[3]
T_pt = score_typed(model, val, dev=torch.device("cpu"))
eS = float(np.abs(T_mlx - T_pt).max())
old = mathexp.Rule("typed10", {"train_bytes": z["train_bytes"][-2_000_000:], "train_H": z["train_H"][-2_000_000:]})
worst = 0.0
rng = np.random.default_rng(0)
for _ in range(30):
    s = int(rng.integers(0, len(val) - 200)); w = val[s:s + int(rng.integers(20, 128))]
    worst = max(worst, float(np.abs(typed_window(model, w) - old.typed(w)[0]).max()))
trb = z["train_bytes"][-2_000_000:]
new = Rule("typed10", trb, None, train_T=score_typed(model, trb, dev=torch.device("cpu")))
a, c = old.mask(val, z["val_H"][:200_000]), new.mask(val, None, T_pt)
diff = int((a != c).sum())
print(f"  score_typed: max |dT| {eS:.1e}; typed_window: max |dT| {worst:.1e}; thresholds {float(old.thr):.6f} vs "
      f"{float(new.thr):.6f}; typed10 masks differ at {diff} of {len(a)} (rate {a.mean():.4f} vs {c.mean():.4f})")
old_f = mathexp.Rule("final+typed10", {"train_bytes": trb, "train_H": z["train_H"][-2_000_000:]})
new_f = Rule("final+typed10", trb, None, train_T=score_typed(model, trb, dev=torch.device("cpu")))
af, cf = old_f.mask(val, z["val_H"][:200_000]), new_f.mask(val, None, T_pt)
diff_f = int((af != cf).sum())
print(f"  final+typed10: thresholds {float(old_f.thr):.6f} vs {float(new_f.thr):.6f}; masks differ at {diff_f} (rate {af.mean():.4f})")
ok = eS < 1e-4 and worst < 1e-4 and diff <= len(a) * 1e-4 and diff_f <= len(a) * 1e-4
print("ok" if ok else "FAILED")
sys.exit(0 if ok else 1)
