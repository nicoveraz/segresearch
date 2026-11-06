"""Reference implementations of the three proposed causal copy vetoes (scratch; not part of the repo).
Token t = byte t-1. The decision for token t may read bytes b[0..t-2] only (what the entropy at t-1 reads).
Each function returns a boolean array over tokens: True = no patch start allowed at t."""
import re
import numpy as np

IDTOK = re.compile(rb"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")      # IDs, doses, decimals, dates with '-' or '.'


def idlike(w):
    """>= 4 bytes and contains a digit, or an upper-case letter after its first byte"""
    return len(w) >= 4 and (any(48 <= c <= 57 for c in w) or any(65 <= c <= 90 for c in w[1:]))


def match_len(b, t, cap=400):
    """longest m such that the m bytes ending at byte t-2 also occur earlier, ending at byte t-3 or before"""
    lo, hi = 0, min(cap, t - 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if b.find(b[t - 1 - mid:t - 1], 0, t - 2) != -1:
            lo = mid
        else:
            hi = mid - 1
    return lo


def siv(b, n, kmin=5):
    """Design 1, segment-initial veto. Veto t if the current line so far (from the byte after the last newline, or the
    text start, up to byte t-2) has >= kmin bytes, occurs earlier as the beginning of an earlier line, and every such
    earlier line continued with the same next byte."""
    starts = [0] + [i + 1 for i in range(len(b)) if b[i] == 10]
    sset = set(starts)
    v = np.zeros(n, bool)
    for t in range(2, n):
        s0 = b.rfind(b"\n", 0, t - 1) + 1                         # first byte of the current line (<= t-1)
        seg = b[s0:t - 1]
        if len(seg) < kmin:
            continue
        nxt = set()
        i = b.find(seg, 0, t - 2)
        while i != -1 and i < s0:
            if i in sset:
                nxt.add(b[i + len(seg)])
            i = b.find(seg, i + 1, t - 2)
        v[t] = len(nxt) == 1
    return v


def po(b, n, k=8):
    """Design 2's added path, pointer onset on a digit-free line, with its continuation. Onset at t: the longest
    recurring suffix has L >= k bytes; the byte after its latest earlier occurrence starts a rare ID-like token S (S
    occurs once in b[:t-1]); the current line has no digit before the match. Continuation: veto while every byte since
    the onset has extended that same match and its pointer is still inside S."""
    v = np.zeros(n, bool)
    state = None                                                 # (onset t0, L0, source token end)
    for t in range(k + 1, n):
        L = match_len(b, t)
        if state is not None:
            t0, L0, s_end = state
            e = b.rfind(b[t - 1 - L:t - 1], 0, t - 2) + L - 1 if L else -1
            if L >= L0 + (t - t0) and e + 1 < s_end:
                v[t] = True
                continue
            state = None
        if L < k:
            continue
        e = b.rfind(b[t - 1 - L:t - 1], 0, t - 2) + L - 1
        m = IDTOK.match(b, e + 1)
        if not m or m.end() > t - 1 or not idlike(m.group(0)):
            continue
        s = m.group(0)
        if sum(1 for x in IDTOK.finditer(b, 0, t - 1) if x.group(0) == s) != 1:
            continue
        s0 = b.rfind(b"\n", 0, t - 1) + 1
        if any(48 <= c <= 57 for c in b[s0:max(s0, t - 1 - L)]):
            continue
        v[t] = True
        state = (t, L, m.end())
    return v


def vc(b, n, m_free=3, k_ctx=6):
    """Design 3's added path, verified-copy inside veto, any context. r = the ID-token run in progress ending at byte
    t-2. Veto t if r is a proper prefix of an earlier rare ID-like token W (unanimous next byte), and |r| >= m_free,
    or |r| >= 1 and the k_ctx bytes before r equal the k_ctx bytes before W."""
    v = np.zeros(n, bool)
    for t in range(3, n):
        e = t - 2
        if e < 0 or not (48 <= b[e] <= 57 or 65 <= b[e] <= 90 or 97 <= b[e] <= 122):
            continue
        r0 = e
        while r0 - 1 >= 0:
            p = b[r0 - 1]
            if 48 <= p <= 57 or 65 <= p <= 90 or 97 <= p <= 122:
                r0 -= 1
            elif p in b"-_." and r0 - 2 >= 0 and (48 <= b[r0 - 2] <= 57 or 65 <= b[r0 - 2] <= 90 or 97 <= b[r0 - 2] <= 122):
                r0 -= 1
            else:
                break
        r = b[r0:e + 1]
        toks = [(x.start(), x.group(0)) for x in IDTOK.finditer(b, 0, r0) if x.end() <= r0]
        cnt = {}
        for _, w in toks:
            cnt[w] = cnt.get(w, 0) + 1
        cands = [(s, w) for s, w in toks if cnt[w] == 1 and idlike(w) and len(w) > len(r) and w.startswith(r)]
        if not cands or len({w[len(r)] for _, w in cands}) != 1:
            continue
        if len(r) >= m_free or (k_ctx and r0 >= k_ctx and any(s >= k_ctx and b[s - k_ctx:s] == b[r0 - k_ctx:r0] for s, _ in cands)):
            v[t] = True
    return v
