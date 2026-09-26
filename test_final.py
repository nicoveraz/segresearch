"""HUMAN ONLY. The agent must never run this or read its output.

Evaluates the current boundary.py and all baselines on the TEST split of every format,
including the held-out format C that the agent never optimized on.

    uv run test_final.py
"""
import harness
import prepare
import baselines

harness.check_boundary_source("boundary.py")
import boundary  # noqa: E402

rules = {"boundary.py (agent)": boundary, **{k: v for k, v in baselines.ALL.items()}}
fmts = prepare.FORMATS_DEV + (prepare.FORMAT_HIDDEN,)
print(f"{'rule':24s} " + " ".join(f"{'ans_bits ' + f:>12s} {'long ' + f:>9s} {'rate':>5s}" for f in fmts))
for name, mod in rules.items():
    row = []
    for fmt in fmts:
        tr, _ = prepare.load(fmt, "train")
        te, te_roles = prepare.load(fmt, "test")
        m_tr, m_te = harness.masks(mod, tr, te)
        o = harness.train_eval(tr.bytes, m_tr, te.bytes, m_te, te_roles, seed=0)
        row.append(f"{o['ans_bits']:12.4f} {o['ANS_LONG_bits']:9.3f} {o['boundary_rate']:5.2f}")
    print(f"{name:24s} " + " ".join(row), flush=True)
