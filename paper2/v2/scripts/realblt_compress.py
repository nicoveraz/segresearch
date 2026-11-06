"""Compression-derived patch rules on BLT-1B: do classic compressors find computed results better than the novelty
table? (local; follows realblt_window_sweep.py / realblt_typed.py)

New signals, each causal (the score for a patch start at token t uses only bytes before it):
    lz     document-local LZ77: -L, L = length of the longest suffix of the text so far (up to 32 bytes) that also
           occurs earlier in the same problem (matches under 3 bytes count as none). No match -> new content -> patch. Copies from the question (an
           operand, a repeated result) have a match; a computed value does not.
    ppm    PPM-C escape probability u / (n + u) at the longest context (order <= 5) seen in the GSM8K training solutions
           written in this (stripped) format: the compressor's own estimate that the next byte is new in this context.
    lz+nov the LZ score with the novelty table (paper 2) added in z units (breaks LZ's many ties)
Baselines: entropy (BLT-1B's rule), typed (value uncertainty), nov (paper 2's two-byte table fitted on stripped GSM8K
train text, realblt_combo.NOVTAB), novc (the same table fitted on the raw training corpus).
Run 1 (2026-10-23) fitted nov and ppm on the annotated bundle text, a format mismatch (nov covered 0% of results);
its output is kept as results/blt_compress_annotated_tables.json.
Thresholds fitted on GSM8K train problems at the same patch rate; window 512; coverage and teacher-forced exact
match of in-line computed results and final answers, as realblt_typed.py.

    .venv-blt/bin/python realblt_compress.py [N_PROBLEMS] [N_TRAIN]
"""
import json
import os
import re
import sys

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import typed_screen
from realblt_budget import GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem
from scale.rules import table_lookup

BUNDLE = os.path.expanduser("~/.cache/segresearch-scale/bundle")
BUDGETS = (0.15, 0.10)
NOVTAB_GSM = os.path.expanduser("~/.cache/segresearch-blt/novtab_gsm.npz")
KINDS = {"entropy": "entropy", "typed": "T", "nov": "nov", "novc": "novc", "lz": "lz", "ppm": "ppm", "lz+nov": "lz+nov"}
MAXL, ORDER = 32, 5


def stripped_train_bytes():
    """All GSM8K train solutions as problem() writes them (no <<...>> annotations), as realblt_combo.novelty_table."""
    texts = []
    for l in open(GSM_TRAIN):
        p = json.loads(l)
        sol, ans = p["answer"].rsplit("#### ", 1)
        texts.append((p["question"] + "\n" + re.sub(r"<<[^>]*>>", "", sol).strip() + "\nThe final answer is "
                      + ans.strip()).encode())
    return np.frombuffer(b"\n\n".join(texts), np.uint8)


def lz_scores(b):
    """-L per byte position j: L = longest suffix of b[:j] (<= MAXL) occurring earlier, ending before j."""
    out = np.zeros(len(b))
    for j in range(1, len(b)):
        best = 0
        for L in range(1, min(MAXL, j) + 1):
            suf = b[j - L:j]
            if b.find(suf, 0, j - 1) == -1:              # an occurrence that ends before position j
                break
            best = L
        out[j] = -best if best >= 3 else 0.0           # 1-2 byte matches are everywhere: not a copy signal
    return out


class PPM:
    """PPM-C escape probabilities from corpus counts, orders 1..ORDER (vectorized counting with numpy)."""

    def __init__(self, corpus):
        c = corpus.astype(np.int64)
        self.tabs = {}
        for k in range(1, ORDER + 1):
            ctx = np.zeros(len(c) - k, np.int64)
            for i in range(k):
                ctx = ctx * 256 + c[i:len(c) - k + i]
            nxt = c[k:]
            pair = np.unique(ctx * 256 + nxt)
            uctx, u = np.unique(pair >> 8, return_counts=True)            # distinct next bytes per context
            nctx, n = np.unique(ctx, return_counts=True)                  # occurrences per context
            assert (uctx == nctx).all()
            self.tabs[k] = (uctx, n, u)

    def escape(self, b):
        """Escape probability for predicting byte j from the longest seen context b[j-k:j], per position j."""
        x = np.frombuffer(b, np.uint8).astype(np.int64)
        out = np.full(len(x), 1.0)                       # no context seen: everything is new
        done = np.zeros(len(x), bool)
        for k in range(ORDER, 0, -1):
            if len(x) <= k:
                continue
            ctx = np.zeros(len(x) - k, np.int64)
            for i in range(k):
                ctx = ctx * 256 + x[i:len(x) - k + i]
            keys, n, u = self.tabs[k]
            pos = np.searchsorted(keys, ctx)
            pos = np.minimum(pos, len(keys) - 1)
            hit = keys[pos] == ctx
            j = np.arange(k, len(x))
            sel = hit & ~done[j]
            out[j[sel]] = u[pos[sel]] / (n[pos[sel]] + u[pos[sel]])
            done[j[sel]] = True
        return out


def main(n_problems, n_train):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    tabs = {k: np.load(f) for k, f in (("nov", NOVTAB_GSM), ("novc", f"{BUNDLE}/novelty_corpus.npz"))}
    ppm = PPM(stripped_train_bytes())
    layouts = [f"{k}@{int(r * 100)}" for r in BUDGETS for k in KINDS]

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, _, logits = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                 max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        lg = torch.log_softmax(logits[0, :, 4:260].float(), -1).cpu().numpy()
        _, _, _, T = typed_screen.typed(lg)
        sig = {}
        for name, v in (("entropy", e), ("T", T)):
            s = np.full(n, -np.inf); s[2:] = v[1:n - 1]        # token t starts a patch from the prediction at t-1
            sig[name] = s
        arr = np.frombuffer(b, np.uint8)
        nov = {k: table_lookup(t["T2"], t["T1"], float(t["glob"]), arr) for k, t in tabs.items()}
        for name, v in (("nov", nov["nov"]), ("novc", nov["novc"]), ("lz", lz_scores(b)), ("ppm", ppm.escape(b))):
            s = np.full(n, -np.inf); s[2:] = v[1:n - 1]        # byte j = token j+1: score for byte j at token j+1
            sig[name] = s
        return sig

    train = [pr for pr in (problem(p, tok, dev) for p in [json.loads(l) for l in open(GSM_TRAIN)][:n_train]) if pr]
    test = [pr for pr in (problem(p, tok, dev) for p in [json.loads(l) for l in open(GSM)][:n_problems]) if pr]
    fit = blt_layouts.Fit(BUDGETS)
    for b, ids, spans, res in train:
        fit.add(signals(b, ids), res)
    fit.done()
    print(f"{len(train)} train / {len(test)} test; train rates: " + ", ".join(
        f"{l} {fit.rate(KINDS[l.split('@')[0]], int(l.split('@')[1]) / 100):.3f}" for l in layouts), flush=True)
    stats = {l: {"res_exact": [], "fin_exact": [], "rate": [], "cov_res": [], "cov_fin": []} for l in layouts}
    for i, (b, ids, spans, res) in enumerate(test):
        n = ids.shape[1]
        sig = signals(b, ids)
        fin0 = [s for k, s, _ in spans if k == "fin"]
        for name in layouts:
            k, R = name.split("@")
            st = blt_layouts.layout(KINDS[k], int(R) / 100, sig, fit, ())
            st = np.array(sorted(set(int(x) for x in st if 0 <= x < n))); S = set(st.tolist())
            stats[name]["rate"].append(len(st) / n)
            stats[name]["cov_res"] += [s in S for s in res]
            stats[name]["cov_fin"] += [s in S for s in fin0]
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                             use_cache=False).logits[0].float(), -1)
            for kind, s, e_ in spans:
                tgt = ids[0, s:e_]; pred = lp[s - 1:e_ - 1]
                stats[name][f"{kind}_exact"].append(bool((pred.argmax(-1) == tgt).all()))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} problems", flush=True)
    out = {}
    for name in layouts:
        s = stats[name]
        out[name] = {k: float(np.mean(v)) for k, v in s.items()}
        _seg.emit("realblt_compress", f"RESULT {name:12s} rate {out[name]['rate']:.3f} | covered results "
                      f"{out[name]['cov_res']:.3f} final {out[name]['cov_fin']:.3f} | exact results "
                      f"{out[name]['res_exact']:.3f} final {out[name]['fin_exact']:.3f}", experiment="blt_compress")
    json.dump({"summary": out, "items": stats}, open(os.path.join(_seg.HERE, "results", "blt_compress.json"), "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
