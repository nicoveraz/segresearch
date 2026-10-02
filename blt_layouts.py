"""Tight-budget patch layouts for the BLT-1B scripts (realblt_budget.py, realblt_code.py, realblt_reason.py).

Positions are BLT token indices: token 0 is BOS, token t is byte t-1. Tokens 0 and 1 always start a patch, and a
score array holds -inf at t < 2. A layout is the sorted array of patch-start tokens.

Two ways to apply a budget R (SEGR_BLT_THRESH, issue #26):

    train    (default) thresholds and z-score parameters are fitted once on TRAIN problems (Fit), then applied
             position by position: token t starts a patch iff its score passes the threshold. Whether t starts a
             patch never depends on later positions. The patch rate varies per problem; on the train problems it
             averages R for every layout, so layouts stay matched in compute on average.
    problem  the original rule: the top R of positions within each problem, and z-scores computed per problem.
             Exact patch counts, but whether t is chosen depends on scores later in the same problem (answer
             bytes included). Kept for comparison.

Layout kinds:
    entropy           the score itself (BLT-1B's next-byte entropy)
    dep, ...          any other single signal (a boundary-dependence table)
    entropy+dep       sum of the z-scored signals named after '+' ("entdep" in the scripts)
    forced            label-based reference: the given starts (results, repeated identifiers, targets), filled up
                      to the budget with the top-entropy remaining positions
"""
import os

import numpy as np

MODE = os.environ.get("SEGR_BLT_THRESH", "train")
assert MODE in ("train", "problem"), MODE


def _signal(kind, sig, z):
    """The score array for a layout kind; z maps a signal name to (mean, std) or None (per-problem)."""
    if kind == "forced":
        kind = "entropy"
    parts = kind.split("+")
    if len(parts) == 1:
        return sig[kind]
    fin = np.all([np.isfinite(sig[p]) for p in parts], axis=0)
    total = np.zeros(len(fin))
    for p in parts:
        x = sig[p]
        m, s = z[p] if z is not None else (x[fin].mean(), x[fin].std())
        total = total + np.where(fin, (x - m) / s, 0.0)
    return np.where(fin, total, -np.inf)


def _tiebreak(n):
    """A fixed pseudo-random number in [0, 1) per token position: depends on the position only, so it is causal."""
    return (np.arange(n, dtype=np.uint64) * np.uint64(2654435761) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def _top(score, k):
    return np.argsort(-score)[:k]           # same call as the original scripts, so ties break the same way


class Fit:
    """Collect signals on train problems, then fit thresholds: Fit(budgets); add(sig, forced) per problem; done()."""
    def __init__(self, budgets):
        self.budgets, self.sigs, self.forced = budgets, [], []

    def add(self, sig, forced=()):
        sig = {k: np.asarray(v, np.float64) for k, v in sig.items()}
        for k, v in sig.items():                 # -inf marks "no score"; NaN means the model or table is broken
            if np.isnan(v).any():
                raise ValueError(f"signal {k!r} has NaN values (a broken model run or dependence table?)")
        self.sigs.append(sig)
        self.forced.append(np.array(sorted({int(t) for t in forced if t >= 2}), int))

    def done(self):
        names = self.sigs[0].keys()
        self.z = {}
        for name in names:
            x = np.concatenate([s[name][np.isfinite(s[name])] for s in self.sigs])
            self.z[name] = (float(x.mean()), float(x.std()))
        self.n_tokens = sum(len(next(iter(s.values()))) for s in self.sigs)
        self.thr = {}
        return self

    def _threshold(self, kind, R):
        """(value, q): positions above value start a patch, and positions equal to it with probability q, so that
        pooled over the train problems starts / tokens is R. q matters for dependence tables, which are lookups
        with many tied values; for continuous scores it is ~1."""
        key = (kind, R)
        if key in self.thr:
            return self.thr[key]
        target = R * self.n_tokens - 2 * len(self.sigs)                 # tokens 0 and 1 of every problem
        vals = []
        for s, f in zip(self.sigs, self.forced):
            x = _signal(kind, s, self.z)
            if kind == "forced":
                target -= len(f)
                x = x.copy(); x[f] = -np.inf
            vals.append(x[np.isfinite(x)])
        v = np.sort(np.concatenate(vals))[::-1]
        k = int(round(min(max(target, 1), len(v))))
        t = v[k - 1]
        gt, eq = int((v > t).sum()), int((v == t).sum())
        self.thr[key] = (float(t), (k - gt) / eq)
        return self.thr[key]

    def rate(self, kind, R):
        """Mean starts / tokens on the train problems (should be close to R)."""
        n = sum(len(layout(kind, R, s, self, f)) for s, f in zip(self.sigs, self.forced))
        return n / self.n_tokens


def layout(kind, R, sig, fit=None, forced=(), mode=None):
    """Patch-start tokens for one problem. sig: {signal name: array over tokens}; fit: a Fit (train mode only)."""
    mode = mode or MODE
    n = len(next(iter(sig.values())))
    forced = np.array(sorted({int(t) for t in forced if 2 <= t < n}), int)
    if mode == "problem":
        k = max(int(round(R * n)) - 2, 1)
        score = _signal(kind, sig, None)
        if kind == "forced":
            fs = set(forced.tolist())
            fill = [int(t) for t in _top(score, n) if t not in fs][:max(k - len(forced), 0)]
            return np.array(sorted({0, 1, *forced.tolist(), *fill}), int)
        return np.array(sorted({0, 1, *_top(score, k).tolist()}), int)
    thr, q = fit._threshold(kind, R)
    score = _signal(kind, sig, fit.z)
    if kind == "forced":
        score = score.copy(); score[forced] = -np.inf
    on = (score > thr) | ((score == thr) & (_tiebreak(n) < q))
    on &= np.isfinite(score)
    extra = forced.tolist() if kind == "forced" else []
    return np.array(sorted({0, 1, *np.flatnonzero(on).tolist(), *extra}), int)
