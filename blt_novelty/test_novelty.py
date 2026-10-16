"""Tests for blt_novelty (pytest). The first three run anywhere (numpy + torch); the paper-equivalence tests run
inside the segresearch repository; the bytelatent tests run when Meta's BLT code base is importable.

    python -m pytest blt_novelty -q
"""
import os
import sys

import numpy as np
import pytest
import torch

from blt_novelty.novelty import (OFFSET, blt_lookup_table, fit_scores, forward_match, scores_for_bytes, start_mask,
                                 threshold_for_rate)

ROOT = os.path.join(os.path.dirname(__file__), "..")
RNG = np.random.default_rng(0)


def _text(n=40_000):
    words = [b"the", b"sum", b"of", b"is", b"=", b"12", b"7", b"apples", b"\n", b"x", b"+", b"total", b"cost"]
    parts = [words[i] for i in RNG.integers(0, len(words), n // 3)]
    return np.frombuffer(b" ".join(parts)[:n], np.uint8)


def test_forward_match_brute_force():
    b = _text(3000)
    fm = forward_match(b, w=32, cap=8)
    s = b.tobytes()
    for t in RNG.integers(0, len(b), 300):
        best = 0
        for d in range(1, 32):
            if t - d < 0:
                break
            k = 0
            while k < 8 and t + k < len(s) and s[t + k] == s[t - d + k]:
                k += 1
            best = max(best, k)
        assert fm[t] == best


def test_valid_patches_and_rate():
    b = _text()
    S = fit_scores(b)
    thr = threshold_for_rate(S, b, 0.10)
    rate = (scores_for_bytes(S, b) > thr).mean()
    assert 0.05 < rate <= 0.10 + 1e-9
    table = blt_lookup_table(S, thr)
    assert table.shape == (260, 260) and table.dtype == torch.bool
    tokens = torch.from_numpy(b[:512].astype(np.int64) + OFFSET)[None]
    m = start_mask(tokens, table)
    assert m[0, 0] and m[0, 1]                                   # first two positions always start patches
    starts = torch.nonzero(m[0]).flatten()
    lengths = torch.diff(torch.cat([starts, torch.tensor([tokens.shape[1]])]))
    assert int(lengths.sum()) == tokens.shape[1] and bool((lengths > 0).all())


def test_causal():
    b = _text(2000)
    S = fit_scores(b); table = blt_lookup_table(S, threshold_for_rate(S, b, 0.2))
    tok = torch.from_numpy(b[:300].astype(np.int64) + OFFSET)[None]
    base = start_mask(tok, table)
    for k in RNG.integers(0, 298, 50):
        t2 = tok.clone(); t2[0, k] = OFFSET + (int(t2[0, k]) - OFFSET + 37) % 256
        diff = torch.nonzero(start_mask(t2, table)[0] != base[0]).flatten().tolist()
        assert all(p in (k + 1, k + 2) for p in diff)            # a token only affects the next two decisions


def _paper_modules():
    sys.path.insert(0, os.path.abspath(ROOT))
    try:
        import reuse_screen
        return reuse_screen
    except Exception:
        pytest.skip("segresearch's reuse_screen.py not importable (run inside the segresearch repository)")


def test_matches_paper_table():
    """Same scores as the paper's implementation (reuse_screen.match_signals + tab_fit + lookup)."""
    rs = _paper_modules()
    b = _text(60_000); fit, held = b[:40_000], b[40_000:]
    S = fit_scores(fit)
    T2, T1, glob = rs.tab_fit(fit, -rs.match_signals(fit)[0])
    hb = held.astype(np.int64); p1 = np.r_[0, hb[:-1]]; p2 = np.r_[0, 0, hb[:-2]]
    d = T2[p2 * 256 + p1]; d = np.where(np.isnan(d), T1[p1], d); d = np.where(np.isnan(d), glob, d)
    assert np.allclose(scores_for_bytes(S, held), d, rtol=0, atol=1e-9)


def test_matches_paper_blt1b_layout():
    """On BLT-1B token ids ([BOS] + byte + 4), the table's patch starts equal the paper's novelty layout
    (realblt_combo.py: token t >= 2 starts a patch when nov(bytes)[t - 1] > threshold)."""
    _paper_modules()
    b = _text(30_000); S = fit_scores(b[:20_000]); held = b[20_000:21_000]
    for thr in np.quantile(scores_for_bytes(S, held), [0.5, 0.85, 0.9]):
        tok = torch.cat([torch.tensor([1]), torch.from_numpy(held.astype(np.int64) + OFFSET)])[None]
        m = start_mask(tok, blt_lookup_table(S, thr))[0].numpy()
        paper = scores_for_bytes(S, held)[1:len(held)] > thr       # token t = byte t - 1
        assert (m[2:] == paper).all()


def _bytelatent():
    try:
        from bytelatent.data import patcher
        return patcher
    except Exception:
        pytest.skip("bytelatent (Meta's BLT code) not importable")


def test_bytelatent_lookup_function_agrees():
    patcher = _bytelatent()
    b = _text(5000); S = fit_scores(b); table = blt_lookup_table(S, threshold_for_rate(S, b, 0.1))
    tok = torch.from_numpy(b[:400].astype(np.int64) + OFFSET).reshape(2, 200)
    ids = patcher.find_lookup_table_patch_start_ids(tok, table, include_next_token=False)
    mine = start_mask(tok, table)
    for r in range(2):
        got = ids[r][ids[r] < 200].tolist()
        assert got == torch.nonzero(mine[r]).flatten().tolist()


def test_bytelatent_patcher_mode(tmp_path):
    patcher = _bytelatent()
    if not hasattr(patcher.PatchingModeEnum, "lookup_table"):
        pytest.skip("blt_lookup_table_mode.diff not applied")
    b = _text(5000); S = fit_scores(b); thr = threshold_for_rate(S, b, 0.1)
    torch.save({"lookup_table": blt_lookup_table(S, thr)}, tmp_path / "t.pt")
    p = patcher.PatcherArgs(patching_mode="lookup_table", lookup_table_path=str(tmp_path / "t.pt"),
                            device="cpu", patching_device="cpu").build()
    tok = torch.from_numpy(b[:256].astype(np.int64) + OFFSET)[None]
    lengths, _ = p.patch(tok, include_next_token=False)
    lengths = lengths[0][lengths[0] > 0]
    assert int(lengths.sum()) == 256
    starts = torch.cumsum(torch.cat([torch.tensor([0]), lengths[:-1]]), 0).tolist()
    assert starts == torch.nonzero(start_mask(tok, blt_lookup_table(S, thr))[0]).flatten().tolist()
