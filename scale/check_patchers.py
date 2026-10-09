"""scale/patchers.py must score entropy exactly as the MLX code (Mac only).

    .venv-scale/bin/python -m scale.check_patchers

The cached MLX entropy model (pat_* in math.npz) is loaded into the PyTorch EntropyLM; score_stream must match
prepare.score_stream and entropy_window must match mathexp._entropy_window.
"""
import sys

import mlx.core as mx
import numpy as np
import torch

import mathexp
import prepare
from scale.model import EntropyLM
from scale.patchers import entropy_window, score_stream

z = np.load(mathexp.NPZ)
pat = mathexp._patcher(z)
model = EntropyLM(prepare.PATCHER["d"], prepare.PATCHER["layers"], prepare.PATCHER["heads"], prepare.CTX)
sd = model.state_dict()
for k in z.files:
    if k.startswith("pat_"):
        name = k[4:].replace("/", ".")
        sd[name].copy_(torch.from_numpy(z[k]))
model.eval()

stream = z["val_bytes"][:200_000]
H_mlx, S_mlx = prepare.score_stream(pat, prepare.PATCHER["heads"], stream)
H, S = score_stream(model, stream, dev=torch.device("cpu"))
eH, eS = float(np.abs(H - H_mlx).max()), float(np.abs(S - S_mlx).max())
# the cache was scored on the full split; the slice's last CTX/2 bytes have no window here, so compare the interior
cached = float(np.abs(H[1000:-prepare.CTX] - z["val_H"][1000:200_000 - prepare.CTX]).max())
print(f"  score_stream: max |dH| {eH:.1e} bits, max |dS| {eS:.1e} bits; vs cached val_H (interior) {cached:.1e}")

worst = 0.0
rng = np.random.default_rng(0)
for _ in range(20):
    s = int(rng.integers(0, len(stream) - 200)); w = stream[s:s + int(rng.integers(20, 127))]
    a, an = mathexp._entropy_window(pat, w)
    b, bn = entropy_window(model, w)
    worst = max(worst, float(np.abs(a - b).max()), abs(an - bn))
print(f"  entropy_window: max |dH| {worst:.1e} bits")
ok = eH < 1e-3 and eS < 1e-3 and worst < 1e-3 and cached < 1e-3
print("ok" if ok else "FAILED")
sys.exit(0 if ok else 1)
