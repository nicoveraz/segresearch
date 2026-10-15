"""General ways to keep the final-answer boundary without a format cue (local; follows realblt_combo.py).

On BLT-1B, typed uncertainty (T, from the patcher) skips final answers: the patcher can predict them (they repeat the
last result), yet a patch start there matters. typed+final fixes it with the cue 'The final answer is '. Domain-free
alternatives, all at the same budget R (train-fitted, causal):

    typed+stale@R  staleness bonus: start a patch at t if z(T[t]) + (bytes since the last start) / G > thr, decided
                   left to right (G = 8: every 8 bytes without a start add one standard deviation)
    typed+cap@R    BLT-style maximum patch length: start where T[t] > thr, and always after K = 20 bytes without one
    typed+dec@R    ask the decoder: run BLT-1B with the typed layout at 0.8 R, compute the DECODER's typed
                   uncertainty at every byte (its own prediction under that layout), and add patch starts where it is
                   highest, up to R. One-step approximation of deciding with the decoder online.
    references: entropy@R, results@R ('= ', label-based), entdep@R, typed@R, typed+final@R

    .venv-blt/bin/python realblt_general.py [N_PROBLEMS] [N_TRAIN]
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import registry
import typed_screen
from realblt_budget import DEP, GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem

BUDGETS = (0.15, 0.10)
MARKER = b"The final answer is "
G, K, BASE = 8.0, 20, 0.8


def seq_layout(score, thr, mode):
    """Left-to-right causal layout: tokens 0, 1 always start; mode 'stale' adds gap / G to the score, 'cap' forces a
    start after K tokens without one."""
    n = len(score); starts = [0, 1]; last = 1
    for t in range(2, n):
        gap = t - last
        on = (score[t] + gap / G > thr) if mode == "stale" else (score[t] > thr or gap >= K)
        if on and np.isfinite(score[t]):
            starts.append(t); last = t
    return np.array(starts, int)


def fit_seq(scores, R, mode):
    """Threshold so that starts / tokens over the train problems is R (bisection; the rate falls as thr rises)."""
    n_tok = sum(len(s) for s in scores)
    lo, hi = -20.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = sum(len(seq_layout(s, mid, mode)) for s in scores) / n_tok
        lo, hi = (mid, hi) if rate > R else (lo, mid)
    return (lo + hi) / 2


def main(n_problems, n_train):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, logits = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                           max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        T = typed_screen.typed(torch.log_softmax(logits[0, :, 4:260].float(), -1).cpu().numpy())[3]
        sig = {}
        for name, v in (("entropy", e), ("T", T)):
            s = np.full(n, -np.inf); s[2:] = v[1:n - 1]; sig[name] = s
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        sig["dep"] = dep
        return sig, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    def run(ids, st, n):
        with torch.no_grad():
            return model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev), use_cache=False).logits[0].float()

    def dec_T(ids, st, n):
        """The decoder's typed uncertainty for every token t under layout st (row t-1 predicts token t)."""
        T = typed_screen.typed(torch.log_softmax(run(ids, st, n)[:, 4:260], -1).cpu().numpy())[3]
        out = np.full(n, -np.inf); out[2:] = T[1:n - 1]
        return out

    def marker_starts(b):
        out, i = [], b.find(MARKER)
        while i >= 0:
            out.append(i + len(MARKER) + 1); i = b.find(MARKER, i + 1)
        return out

    # ---- fit on train problems
    fit = blt_layouts.Fit(BUDGETS)
    train = []
    for p in [json.loads(l) for l in open(GSM_TRAIN)][:n_train]:
        pr = problem(p, tok, dev)
        if pr is not None:
            sig, _ = signals(pr[0], pr[1])
            fit.add(sig, {"default": pr[3], "final": marker_starts(pr[0])})
            train.append((pr, sig))
    fit.done()
    zT = lambda s: (s - fit.z["T"][0]) / fit.z["T"][1]
    seq_thr = {(m, R): fit_seq([zT(sig["T"]) if m == "stale" else sig["T"] for _, sig in train], R, m)
               for m in ("stale", "cap") for R in BUDGETS}
    dec_thr = {}
    for R in BUDGETS:                                                # added starts so that the total rate is R
        vals, n_tok, n_base = [], 0, 0
        for (b, ids, spans, rs), sig in train:
            n = ids.shape[1]; base = blt_layouts.layout("T", BASE * R, sig, fit)
            d = dec_T(ids, base, n); d[base] = -np.inf
            vals.append(d[np.isfinite(d)]); n_tok += n; n_base += len(base)
        v = np.sort(np.concatenate(vals))[::-1]
        dec_thr[R] = float(v[max(int(R * n_tok) - n_base, 1) - 1])
    print(f"fitted on {len(train)} train problems; staleness / cap / decoder thresholds: " +
          ", ".join(f"{k} {v:.3f}" for k, v in seq_thr.items()) + ", " + ", ".join(f"dec@{R} {v:.3f}" for R, v in dec_thr.items()), flush=True)

    kinds = {"entropy": "entropy", "results": "forced", "entdep": "entropy+dep", "typed": "T", "typed+final": "forced[final]:T"}
    layouts = ["default"] + [f"{k}@{int(R * 100)}" for R in BUDGETS for k in (*kinds, "typed+stale", "typed+cap", "typed+dec")]
    stats = {l: {"res_bits": [], "res_exact": [], "fin_bits": [], "fin_exact": [], "rate": []} for l in layouts}
    covered = {l: {"res": [], "fin": []} for l in layouts}
    t0 = time.time(); used = 0
    for p in [json.loads(l) for l in open(GSM)][:n_problems]:
        pr = problem(p, tok, dev)
        if pr is None:
            continue
        b, ids, spans, res_starts = pr
        n = ids.shape[1]; used += 1
        sig, default = signals(b, ids)
        forced = {"default": res_starts, "final": marker_starts(b)}
        lay = {"default": default}
        for R in BUDGETS:
            r = int(R * 100)
            for k, kind in kinds.items():
                lay[f"{k}@{r}"] = blt_layouts.layout(kind, R, sig, fit, forced)
            lay[f"typed+stale@{r}"] = seq_layout(zT(sig["T"]), seq_thr[("stale", R)], "stale")
            lay[f"typed+cap@{r}"] = seq_layout(sig["T"], seq_thr[("cap", R)], "cap")
            base = blt_layouts.layout("T", BASE * R, sig, fit)
            d = dec_T(ids, base, n)
            lay[f"typed+dec@{r}"] = np.union1d(base, np.flatnonzero(d > dec_thr[R]))
        fin0 = [s for k, s, _ in spans if k == "fin"]
        for name, st in lay.items():
            st = np.array(sorted(set(int(x) for x in st if 0 <= x < n))); S = set(st.tolist())
            covered[name]["res"] += [s in S for s in res_starts]
            covered[name]["fin"] += [s in S for s in fin0]
            lp = torch.log_softmax(run(ids, st, n), -1)
            stats[name]["rate"].append(len(st) / n)
            for kind, s, e_ in spans:
                tgt = ids[0, s:e_]; pred = lp[s - 1:e_ - 1]
                stats[name][f"{kind}_bits"].append(float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)))
                stats[name][f"{kind}_exact"].append(bool((pred.argmax(-1) == tgt).all()))
        if used % 25 == 0:
            print(f"  {used} problems, {time.time() - t0:.0f}s", flush=True)
    print(f"\nBLT-1B, {used} GSM8K test problems, general final-answer fixes; {len(stats['default']['res_exact'])} in-line results")
    for name in layouts:
        s, c = stats[name], covered[name]
        registry.emit("realblt_general", f"RESULT {name:14s} patch rate {np.mean(s['rate']):.3f} | covered: results "
              f"{np.mean(c['res']):5.1%}, final {np.mean(c['fin']):5.1%} | in-line results: {np.mean(s['res_bits']):.3f} bits, "
              f"exact {np.mean(s['res_exact']):5.1%} | final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}",
              experiment="blt_general", window=blt_load.WINDOW)
    registry.save_items(f"blt_general{blt_load.SUFFIX}", {"layouts": stats, "covered": covered},
                        experiment="blt_general", n_problems=used, window=blt_load.WINDOW)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
