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

Current rule: learned unit separator + learnable-unit starts.
  - fit() learns the unit separator from the train split: the most frequent byte value that is almost
    always (>= 95%) followed by a jump in the small model's entropy of more than 1 bit. A patch starts
    right after every occurrence, so each unit (record) is one patch and its inputs stay together.
  - Every learnable-unit start also starts a patch: a jump of more than 1 bit in the small model's
    entropy where the strong reference model is confident (< 1 bit). This gives computed outputs
    (answers) a fresh global step.
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


def fit(sig):
    b = sig.bytes
    followed = np.r_[_jump(sig)[1:], 0.0] > JUMP
    count = np.bincount(b, minlength=256)
    hits = np.bincount(b, weights=followed, minlength=256)
    rate = hits / np.maximum(count, 1)
    ok = (rate >= ALWAYS) & (count >= MIN_FREQ * len(b))
    return int(np.argmax(np.where(ok, count, -1)))


def score(sig, sep):
    after_sep = np.r_[False, sig.bytes[:-1] == sep]
    learnable = (_jump(sig) > JUMP) & (sig.reference_entropy < CONFIDENT)
    return after_sep | learnable
