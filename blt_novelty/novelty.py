"""Novelty patching as a lookup table over the previous two bytes (arXiv:2610.22100), in the form BLT's
lookup-table patcher expects.

Novelty of position t: minus the length of the longest prefix of b[t:] that also starts within the previous W bytes
(capped). It looks ahead, so it is predicted causally: the mean novelty per preceding two-byte context
(b[t-2], b[t-1]), falling back to the one-byte context and then the global mean for contexts seen fewer than
MIN_COUNT times. A patch starts at t when that score exceeds a threshold chosen for a target patch rate.

BLT's tokens are byte + OFFSET (4); ids 0-3 are special (BOE, BOS, EOS, BPE). Its lookup-table patcher
(bytelatent/data/patcher.py, find_lookup_table_patch_start_ids) starts a patch at token i + 2 when
table[token[i], token[i + 1]] is True, i.e. from the two tokens before the patch start, the same causal context as
here. Special tokens are mapped to byte 0, the padding this code uses before the first byte, so masks match the
paper's implementation exactly (see test_novelty.py).
"""
import numpy as np

W, CAP, MIN_COUNT = 128, 16, 20
OFFSET, VOCAB = 4, 260


def forward_match(b, w=W, cap=CAP):
    """Length of the longest prefix of b[t:] that also starts in b[t-w+1 : t] (capped at cap), for every t."""
    b = np.asarray(b, np.uint8).astype(np.int16)
    n = len(b); out = np.zeros(n, np.int16)
    for d in range(1, w):
        eq = np.zeros(n + 1, bool); eq[d:n] = b[d:] == b[:-d]
        r = np.zeros(n + 1, np.int16)                        # r[t]: matches at offset d from t on (capped)
        for _ in range(cap):
            r[:-1] = eq[:-1] * (1 + r[1:])
        out = np.maximum(out, np.minimum(r[:-1], cap))
    return out


def fit_scores(b, w=W, cap=CAP, min_count=MIN_COUNT):
    """Dense 256 x 256 novelty scores S[p2, p1] fitted on byte array b (no labels)."""
    b = np.asarray(b, np.uint8).astype(np.int64)
    target = -forward_match(b, w, cap).astype(np.float32)   # float32 as in the paper's code (same global mean)
    p1 = np.r_[0, b[:-1]]; p2 = np.r_[0, 0, b[:-2]]
    key = p2 * 256 + p1
    s2, c2 = np.bincount(key, target, 65536), np.bincount(key, None, 65536)
    s1, c1 = np.bincount(p1, target, 256), np.bincount(p1, None, 256)
    t1 = np.where(c1 >= min_count, s1 / np.maximum(c1, 1), float(target.mean()))
    t2 = np.where(c2 >= min_count, s2 / np.maximum(c2, 1), np.repeat(t1[None, :], 256, 0).reshape(-1))
    return t2.reshape(256, 256)


def scores_for_bytes(S, b):
    """Novelty score of every position of byte array b (context bytes before the start padded with 0)."""
    b = np.asarray(b, np.uint8).astype(np.int64)
    return S[np.r_[0, 0, b[:-2]], np.r_[0, b[:-1]]]


def threshold_for_rate(S, b, rate):
    """Threshold so that a fraction `rate` of the positions of b start a patch (ties counted as below)."""
    return float(np.quantile(scores_for_bytes(S, b), 1 - rate))


def blt_lookup_table(S, threshold):
    """Boolean VOCAB x VOCAB table over BLT token ids for bytelatent's lookup-table patcher (torch.bool)."""
    import torch
    m = S > threshold
    rows = np.zeros((VOCAB, VOCAB), bool)
    ids = np.r_[np.zeros(OFFSET, np.int64), np.arange(256)]  # token id -> byte (special tokens -> byte 0)
    rows[:, :] = m[ids[:, None], ids[None, :]]
    return torch.from_numpy(rows)


def start_mask(tokens, table):
    """Patch-start mask for BLT token ids (batch x seq): position t >= 2 starts a patch when
    table[tokens[t - 2], tokens[t - 1]]; positions 0 and 1 always start patches (as in bytelatent)."""
    import torch
    tokens = torch.as_tensor(tokens)
    m = torch.ones_like(tokens, dtype=torch.bool)
    if tokens.shape[1] > 2:
        m[:, 2:] = table[tokens[:, :-2], tokens[:, 1:-1]]
    return m
