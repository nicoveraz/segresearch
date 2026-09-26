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

Current rule: entropy jump. A patch starts where the small model's entropy rises relative to the
previous byte, i.e. at the first byte of each unpredictable field, not throughout it.
"""
import numpy as np


def fit(sig):
    return None


def score(sig, state):
    h = sig.patcher_entropy[max(sig.patcher_entropy)]
    return np.diff(h, prepend=h[0])
