"""THE FILE THE AGENT EDITS.

Decide where patches start. The harness calls:

    state = fit(train_signals)          # optional; corpus statistics from the TRAIN split only
    s     = score(signals, state)       # one value per byte; called on train and on eval splits

score() returns either
  - a float array: the harness patches at the top 25% of bytes (threshold fitted on train), or
  - a bool array:  your own mask, which must use at most 25% of bytes (fewer is allowed and is
                   reported; spending less compute for the same result is a win).
Entry i means "a new patch starts AT byte i".

Available signals (see prepare.Signals): sig.bytes, sig.patcher_entropy[step],
sig.patcher_surprisal, sig.reference_entropy, sig.reference_surprisal.

Rules enforced by harness.check_boundary_source: no string/bytes literals, no ord/chr, no f-strings,
imports limited to numpy/math/collections/itertools/functools/heapq/typing. The rule must be
generic: it will be tested on a held-out corpus format with different delimiters and keywords.

Current rule: learned unit separator + the most learnable unit starts.
  - fit() learns the unit separator from the train split: the most frequent byte value that is almost
    always (>= 95%) followed by a jump in the small model's entropy of more than 1 bit. A patch starts
    right after every occurrence, so each unit (record) is one patch and its inputs stay together.
  - The most learnable unit starts also start a patch, giving computed outputs (answers) a fresh
    global step. A learnable start is a jump of more than 1 bit in the small model's entropy where the
    strong reference model is confident (< 1 bit). fit() splits the small-minus-reference entropy gap
    at those starts into two groups (Otsu threshold on train) and keeps only the upper group, so units
    the small model finds only mildly harder (e.g. a variable name picked from a few recent ones) do
    not get their own patch.
No other boundaries: splitting a unit's inputs across patches hurts both local and long-range answers.
"""
import numpy as np

JUMP = 1.0          # bits: an entropy rise this large marks the first byte of an unpredictable unit
CONFIDENT = 1.0     # bits: the reference model is confident below this (the byte is learnable)
ALWAYS = 0.95       # a separator is followed by a jump at least this often
MIN_FREQ = 0.005    # ignore rare byte values


def _jump(sig):
    h = sig.patcher_entropy[max(sig.patcher_entropy)]
    return np.diff(h, prepend=h[0])


def _learnable(sig):
    hp = sig.patcher_entropy[max(sig.patcher_entropy)]
    start = (_jump(sig) > JUMP) & (sig.reference_entropy < CONFIDENT)
    return start, hp - sig.reference_entropy


def _otsu(x, bins=64):
    """Threshold that best splits x into two groups (maximum between-group variance)."""
    counts, edges = np.histogram(x, bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(counts); mu = np.cumsum(counts * mids)
    w1, w2 = w[:-1], w[-1] - w[:-1]
    between = np.where((w1 > 0) & (w2 > 0),
                       (mu[-1] * w1 - mu[:-1] * w[-1]) ** 2 / np.maximum(w1 * w2, 1), 0.0)
    return edges[1:-1][np.argmax(between)]


def fit(sig):
    b = sig.bytes
    followed = np.r_[_jump(sig)[1:], 0.0] > JUMP
    count = np.bincount(b, minlength=256)
    hits = np.bincount(b, weights=followed, minlength=256)
    rate = hits / np.maximum(count, 1)
    ok = (rate >= ALWAYS) & (count >= MIN_FREQ * len(b))
    start, gap = _learnable(sig)
    return int(np.argmax(np.where(ok, count, -1))), _otsu(gap[start])


def score(sig, state):
    sep, cut = state
    after_sep = np.r_[False, sig.bytes[:-1] == sep]
    start, gap = _learnable(sig)
    return after_sep | (start & (gap > cut))
