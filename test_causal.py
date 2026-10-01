"""Causality tests for every patching rule and model (issue #14).

    uv run test_causal.py            # exits 1 if any must-pass check fails

The contract. A mask entry m[t] means "byte t starts a patch" (or, for scratchpad rules, "a scratchpad runs
before predicting byte t"). It is used while predicting byte t, so it may depend on
  - bytes before t,
  - entropy signals up to t (entropy[i] is the uncertainty about byte i given the bytes before i),
and never on byte t or later, on entropy after t, or on surprisal at t or later (surprisal[i] uses byte i).
The models must respect the other half: the prediction of byte t+1 may use bytes <= t and mask entries
<= t+1, and nothing later.

How the rules are tested: take a real window from a cache, pick t, replace everything the contract forbids
(bytes[t:], entropy[t+1:], surprisal[t:]) with the same arrays from another place in the data (half the
windows) or with random values (the other half), and require m[:t+1] to be unchanged. Replacing with real
data keeps delimiters and number patterns in the perturbed part, so rules keyed to syntax get exercised. Negative controls (rules that read byte t or surprisal at t,
including the word-start rule fixed in 7338f28) must be caught, or the test is too weak and fails.

Three groups:
  rule        must pass: the rule itself is causal.
  split-level reported, not failed: a budget step that picks a threshold from the evaluated split itself
              (a quantile or top-k over it), so whether t starts a patch depends weakly on later positions.
  control     must be caught.

Not covered: realblt*.py and blt_screens/ (torch, BLT-1B). Their entropy@R / dep@R / entdep@R layouts take the
top R of positions per problem (and entdep z-scores per problem), which is split-level in the sense above.
"""
import os
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

mx.set_default_device(mx.cpu)            # small checks; leave the GPU to running experiments

import baselines  # noqa: E402
import boundary  # noqa: E402
import harness  # noqa: E402
import mathexp  # noqa: E402
import prepare  # noqa: E402
import realtext  # noqa: E402

N, TRIALS = 1024, 40
RNG = np.random.default_rng(0)
CACHE = os.path.expanduser("~/.cache/segresearch")
SINCE = {"byte": 0, "surprisal": 0, "entropy": 1}       # first index after t that may change, per signal kind
CTX_SKIP = prepare.CTX            # the first CTX//2+1 signal entries of a split are 0; skip them


# ----------------------------------------------------------------------------- data
def _load(path, keys):
    z = np.load(os.path.expanduser(path))
    return {k: z[k] for k in keys}


def _windows(arrays):
    """A random window of every array, plus a donor window from elsewhere, same length."""
    n = len(next(iter(arrays.values())))
    a, d = RNG.integers(CTX_SKIP, n - N, 2)
    return {k: v[a:a + N].copy() for k, v in arrays.items()}, {k: v[d:d + N].copy() for k, v in arrays.items()}


def _noise(win, kinds):
    """A donor of random bytes and random signal values (a harsher perturbation than real data)."""
    return {k: (RNG.integers(0, 256, N).astype(win[k].dtype) if kind == "byte"
                else RNG.uniform(0, 8, N).astype(win[k].dtype)) for k, kind in kinds.items()}


def _perturb(win, donor, kinds, t):
    out = {k: v.copy() for k, v in win.items()}
    for k, kind in kinds.items():
        out[k][t + SINCE[kind]:] = donor[k][t + SINCE[kind]:]
    return out


def check(fn, arrays, kinds, trials=TRIALS, per_window=8):
    """Out of trials x per_window perturbations (half with a real-data donor, half with random values), how many
    changed fn's output at positions <= t, and, for the first one, how far before t the earliest change was
    (0 = at t itself). Several t per window matter: a rule that leaks byte t often leaks only in some contexts
    (the old word-start rule only after a space)."""
    leaks, first, total = 0, None, 0
    for i in range(trials):
        win, donor = _windows(arrays)
        if i % 2:
            donor = _noise(win, kinds)
        base = np.asarray(fn(win))
        for t in RNG.integers(8, N - 8, per_window):
            a, b = base[:t + 1], np.asarray(fn(_perturb(win, donor, kinds, int(t))))[:t + 1]
            total += 1
            if not np.array_equal(a, b):
                leaks += 1
                if first is None:
                    first = int(t) - int(np.flatnonzero(a != b)[0])
    return leaks, first, total


# ----------------------------------------------------------------------------- rules on the synthetic corpus
def synthetic_cases():
    keys = ["bytes", "Hp_200", "Hp_1000", "Hp_4000", "Sp", "Hr", "Sr"]
    tr = _load(f"{CACHE}/format_A.npz", [f"train_{k}" for k in keys])
    tr = {k[len("train_"):]: v for k, v in tr.items()}
    kinds = {"bytes": "byte", "Hp_200": "entropy", "Hp_1000": "entropy", "Hp_4000": "entropy",
             "Sp": "surprisal", "Hr": "entropy", "Sr": "surprisal"}

    def sig(w):
        return prepare.Signals(bytes=w["bytes"], patcher_entropy={s: w[f"Hp_{s}"] for s in (200, 1000, 4000)},
                               patcher_surprisal=w["Sp"], reference_entropy=w["Hr"], reference_surprisal=w["Sr"])

    full = sig(tr)
    state = boundary.fit(full)
    cases = [("rule", "boundary.py (H12)", lambda w: boundary.score(sig(w), state))]
    for name, cls in baselines.ALL.items():
        cases.append(("rule", f"baselines.{name}", lambda w, c=cls: c.score(sig(w), None)))
    cases.append(("split-level", "harness.masks budget step (blt_entropy, eval split)",
                  lambda w: harness.masks(baselines.BLTEntropy, full, sig(w))[1]))
    cases.append(("control", "surprisal level (reads byte t)", lambda w: w["Sp"]))
    cases.append(("control", "reference surprisal jump", lambda w: np.diff(w["Sr"], prepend=0)))
    return [(g, n, f, tr, kinds) for g, n, f in cases]


# ----------------------------------------------------------------------------- rules on real text and math
def _legacy_word_starts(b):
    """The word-start rule before 7338f28: also required byte t itself not to be a space (leaks byte t)."""
    sp = (b == 32) | (b == 10)
    return np.r_[True, sp[:-1] & ~sp[1:]]


def realtext_cases():
    z = _load("~/.cache/segresearch-gsm8k/gsm8k.npz", ["train_bytes", "train_roles", "train_H"])
    arrays = {"bytes": z["train_bytes"], "H": z["train_H"]}
    kinds = {"bytes": "byte", "H": "entropy"}
    rest = {"train_bytes": z["train_bytes"][:400_000], "train_H": z["train_H"][:400_000],
            "train_roles": z["train_roles"][:400_000]}

    def masks(name):
        return lambda w: realtext.masks(name, {**rest, "val_bytes": w["bytes"], "val_H": w["H"],
                                               "val_roles": np.zeros(len(w["bytes"]), np.int8)}, "val")
    cases = [("rule", "realtext.words", masks("words"))]
    for name in ("words+jump20", "words+jump25"):
        cases.append(("rule", f"realtext.{name}", masks(name)))
    for name in ("entropy", "jump25"):
        cases.append(("split-level", f"realtext.{name} (refits on the split if over budget)", masks(name)))
    cases.append(("control", "word starts before 7338f28", lambda w: _legacy_word_starts(w["bytes"])))
    return [(g, n, f, arrays, kinds) for g, n, f in cases]


MATH_RULES = ["entropy", "jump", "syntax+jump", "words", "words+jump", "syntax", "stride6+syntax", "words+syntax",
              "words+syntax+digits", "words+syntax+ops", "words+syntax+rand07", "words+syntax+rand22",
              "entropy10", "jump10", "syntax+entropy10", "dep10", "entdep10",
              "sp:none", "sp:dense5", "sp:entropy", "sp:jump", "sp:syntax", "sp:random", "sp:learned",
              "sp16:none", "sp16:dense8", "sp16:entropy", "sp16:jump", "sp16:syntax", "sp16:random", "sp16:learned"]


def mathexp_cases():
    z = _load(mathexp.NPZ, ["train_bytes", "train_H"])
    arrays = {"bytes": z["train_bytes"], "H": z["train_H"]}
    kinds = {"bytes": "byte", "H": "entropy"}
    small = {"train_bytes": z["train_bytes"][:300_000], "train_H": z["train_H"][:300_000]}
    cases = []
    for name in MATH_RULES:
        try:
            rule = mathexp.Rule(name, small)          # thresholds from a train prefix; causality does not depend on them
        except (FileNotFoundError, KeyError) as e:
            print(f"  skip mathexp.{name}: {e}")
            continue
        cases.append(("rule", f"mathexp.{name}", lambda w, r=rule: r.mask(w["bytes"], w["H"])))
    return [(g, n, f, arrays, kinds) for g, n, f in cases]


# ----------------------------------------------------------------------------- models
def _model_check(forward, scratch, trials=12, T=prepare.CTX):
    """Logits at positions <= t (predicting bytes <= t+1) must not change when bytes after t or mask entries
    after t+1 change. Also checks the model does depend on byte t (else the test proves nothing)."""
    leaks = blind = 0
    for _ in range(trials):
        x = RNG.integers(0, 256, (2, T)).astype(np.int32)
        hi = 3 if scratch else 2
        bd = RNG.integers(0, hi, (2, T + 1)).astype(np.int32) * (RNG.random((2, T + 1)) < 0.3); bd[:, 0] = 1
        t = int(RNG.integers(4, T - 4))
        x2, bd2 = x.copy(), bd.copy()
        x2[:, t + 1:] = RNG.integers(0, 256, (2, T - t - 1)); bd2[:, t + 2:] = RNG.integers(0, hi, (2, T - t - 1))
        a = np.array(forward(mx.array(x), mx.array(bd)))[:, :t + 1]
        b = np.array(forward(mx.array(x2), mx.array(bd2)))[:, :t + 1]
        leaks += not np.allclose(a, b, atol=1e-5)
        x3 = x.copy(); x3[:, t] = (x3[:, t] + 1) % 256
        c = np.array(forward(mx.array(x3), mx.array(bd)))[:, t]
        blind += np.allclose(np.array(forward(mx.array(x), mx.array(bd)))[:, t], c, atol=1e-5)
    return leaks, blind


def model_results():
    out = []
    saved = (harness.POOL, harness.LOCAL, harness.SCRATCH)
    try:
        for scratch in (False, True):
            for pool in ("sum", "xattn"):
                for local in ("patch", "window"):
                    harness.POOL, harness.LOCAL, harness.SCRATCH = pool, local, scratch
                    p = harness._init(mx.random.key(0))
                    leaks, blind = _model_check(lambda x, bd: harness._model(p, x, bd), scratch)
                    out.append((f"harness model pool={pool} local={local} scratch={int(scratch)}", leaks, blind))
        harness.POOL, harness.LOCAL, harness.SCRATCH = "sum", "patch", False
        import learned_chunking as lc
        for local in ("patch", "window"):
            harness.LOCAL = local
            p = lc._init(mx.random.key(0))
            leaks, blind = _model_check(lambda x, bd: lc._model(p, x)[0], False)
            out.append((f"learned_chunking model local={local}", leaks, blind))
    finally:
        harness.POOL, harness.LOCAL, harness.SCRATCH = saved

    # the entropy scorers: H[i] may use bytes < i only; S[i] may also use byte i. Weights are scaled up so the
    # random model's predictions are peaked enough for the sensitivity check (at init they are nearly uniform)
    p = prepare.init_lm(mx.random.key(0), prepare.PATCHER["d"], prepare.PATCHER["layers"])
    p = tree_map(lambda a: a * 40 if a.ndim == 2 else a, p)
    leaks = blind = 0
    for _ in range(8):
        s = RNG.integers(0, 256, 1200).astype(np.uint8)
        i = int(RNG.integers(prepare.CTX, 1100))
        s2 = s.copy(); s2[i:] = RNG.integers(0, 256, len(s) - i)
        (H, S), (H2, S2) = prepare.score_stream(p, prepare.PATCHER["heads"], s), prepare.score_stream(p, prepare.PATCHER["heads"], s2)
        leaks += not (np.allclose(H[:i + 1], H2[:i + 1], atol=1e-5) and np.allclose(S[:i], S2[:i], atol=1e-5))
        blind += H[i + 1] == H2[i + 1]
    out.append(("prepare.score_stream (entropy scorer)", leaks, blind))
    return out


# ----------------------------------------------------------------------------- main
def main():
    failed = []
    print(f"Rules: {TRIALS} windows of {N} bytes x 8 positions t each; a leak = output at positions <= t changed")
    cases = synthetic_cases() + realtext_cases() + mathexp_cases()
    for group, name, fn, arrays, kinds in cases:
        leaks, first, total = check(fn, arrays, kinds)
        if group == "rule":
            ok = leaks == 0
            failed += [] if ok else [name]
            status = "PASS" if ok else "FAIL"
        elif group == "control":
            ok = leaks > 0
            failed += [] if ok else [f"control not caught: {name}"]
            status = "CAUGHT" if ok else "MISSED"
        else:
            status = "SPLIT" if leaks else "ok"
        print(f"  {status:6s} {group:11s} {name:55s} {leaks:3d}/{total}"
              + (f"  earliest change t-{first}" if first is not None else ""))

    print("Models: 12 trials each; leak = logits <= t changed; blind = logits at t ignore byte t")
    for name, leaks, blind in model_results():
        ok = leaks == 0 and blind == 0
        failed += [] if ok else [name]
        print(f"  {'PASS' if ok else 'FAIL':6s} {name:67s} leaks {leaks}  blind {blind}")

    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("all must-pass checks passed")


if __name__ == "__main__":
    main()
