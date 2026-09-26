"""FIXED. One experiment: evaluate boundary.py on the development formats (A, B), validation split.

    uv run run.py > run.log 2>&1          # default seed 0
    uv run run.py --seed 1 > run.log 2>&1 # replicate
Final summary lines are grep-able, e.g.  grep "^val_ans_bits:" run.log
"""
import argparse
import time

import numpy as np

import harness
import prepare

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
t0 = time.time()
harness.check_boundary_source("boundary.py")
import boundary  # noqa: E402  (imported only after the source check passes)

res = {}
for fmt in prepare.FORMATS_DEV:
    tr, _ = prepare.load(fmt, "train")
    va, va_roles = prepare.load(fmt, "val")
    m_tr, m_va = harness.masks(boundary, tr, va)
    res[fmt] = o = harness.train_eval(tr.bytes, m_tr, va.bytes, m_va, va_roles, seed=args.seed)
    print(f"[{fmt}] ans_bits {o['ans_bits']:.4f} acc {o['ans_acc']:.3f} | local {o['ANS_LOCAL_bits']:.3f} "
          f"long {o['ANS_LONG_bits']:.3f} | bpb {o['bpb']:.4f} | rate {o['boundary_rate']:.3f}", flush=True)
    print(f"[{fmt}] patch-start rate by role: " +
          ", ".join(f"{r} {o[r + '_start_rate']:.2f}" for r in prepare.ROLES), flush=True)

mean = lambda k: float(np.mean([res[f][k] for f in res]))
print("---")
print(f"val_ans_bits:      {mean('ans_bits'):.6f}")
print(f"val_ans_acc:       {mean('ans_acc'):.4f}")
print(f"val_ans_local_bits:{mean('ANS_LOCAL_bits'):.6f}")
print(f"val_ans_long_bits: {mean('ANS_LONG_bits'):.6f}")
print(f"val_bpb:           {mean('bpb'):.6f}")
print(f"boundary_rate:     {mean('boundary_rate'):.4f}")
print(f"seed:              {args.seed}")
print(f"elapsed_seconds:   {time.time() - t0:.0f}")
