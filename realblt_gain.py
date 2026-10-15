"""Must retrieve vs must compute: two domain-free signals for the positions typed uncertainty misses (local).

realblt_general.py: at final answers the patcher can predict the value (it repeats an earlier result) but the decoder,
without a fresh global step, is confidently wrong, so no uncertainty signal finds them. Two signals of the gap between
what the full-context model knows and what the decoder can produce:

    gain       context gain: BLT-1B's patcher entropy for byte t with only the last W = 32 tokens of context, minus
               with the full context (bits that come from far back: "must retrieve"). Causal; computable in advance.
    dis        disagreement: run BLT-1B with the typed layout at 0.8 R; where the patcher is confident about byte t
               and the decoder gives that byte little probability (p_patcher(x*) * (1 - p_decoder(x*)), x* = the
               patcher's top byte), add patch starts up to R. One-step approximation (needs the model's own output).

Layouts at R (train-fitted thresholds, as realblt_combo.py):
    gain@R, typed+gain@R (union of top sets: the larger of T's and gain's train percentiles), typed+dis@R
    references: entropy@R, results@R ('= ', label-based), entdep@R, typed@R, typed+final@R (format cue)

    .venv-blt/bin/python realblt_gain.py [N_PROBLEMS] [N_TRAIN]
"""
import json
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
W, BASE = 32, 0.8


def main(n_problems, n_train):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf
    patch = lambda x: model.model.patcher(x, patch_size=None, threshold=cfg.patching_threshold,
                                          max_patch_length=cfg.max_patch_length)

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, logits = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                           max_patch_length=cfg.max_patch_length)
            e = ent[0].float().cpu().numpy()
            lpP = torch.log_softmax(logits[0].float(), -1)                       # row t-1 predicts token t
            # short context: for each token t > W, the patcher's prediction from tokens t-W .. t-1 only
            hs = e.copy()
            if n - 1 > W:
                win = ids[0].unfold(0, W, 1)[:n - W]                              # win[j] = tokens j .. j+W-1, predicts j+W
                hs_w = torch.cat([patch(win[i:i + 256])[0][:, -1].float() for i in range(0, len(win), 256)])
                hs[W - 1:n - 1] = hs_w.cpu().numpy()                              # row t-1 = W-1 .. n-2
        T = typed_screen.typed(torch.log_softmax(logits[0, :, 4:260].float(), -1).cpu().numpy())[3]
        sig = {}
        for name, v in (("entropy", e), ("T", T), ("gain", hs - e)):
            s = np.full(n, -np.inf); s[2:] = v[1:n - 1]; sig[name] = s
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        sig["dep"] = dep
        return sig, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]], lpP

    def run(ids, st, n):
        with torch.no_grad():
            return model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev), use_cache=False).logits[0].float()

    def disagreement(ids, st, n, lpP):
        """p_patcher(x*) * (1 - p_decoder(x*)) per token t, x* = the patcher's top prediction (row t-1)."""
        lpD = torch.log_softmax(run(ids, st, n), -1)
        top = lpP.argmax(-1, keepdim=True)
        d = (lpP.gather(-1, top).exp() * (1 - lpD.gather(-1, top).exp()))[:, 0].cpu().numpy()
        out = np.full(n, -np.inf); out[2:] = d[1:n - 1]
        return out

    def marker_starts(b):
        out, i = [], b.find(MARKER)
        while i >= 0:
            out.append(i + len(MARKER) + 1); i = b.find(MARKER, i + 1)
        return out

    fit = blt_layouts.Fit(BUDGETS)
    train = []
    for p in [json.loads(l) for l in open(GSM_TRAIN)][:n_train]:
        pr = problem(p, tok, dev)
        if pr is not None:
            sig, _, lpP = signals(pr[0], pr[1])
            fit.add(sig, {"default": pr[3], "final": marker_starts(pr[0])})
            train.append((pr, sig, lpP))
    fit.done()
    dis_thr = {}
    for R in BUDGETS:
        vals, n_tok, n_base = [], 0, 0
        for (b, ids, spans, rs), sig, lpP in train:
            n = ids.shape[1]; base = blt_layouts.layout("T", BASE * R, sig, fit)
            d = disagreement(ids, base, n, lpP); d[base] = -np.inf
            vals.append(d[np.isfinite(d)]); n_tok += n; n_base += len(base)
        v = np.sort(np.concatenate(vals))[::-1]
        dis_thr[R] = float(v[max(int(R * n_tok) - n_base, 1) - 1])
    kinds = {"entropy": "entropy", "results": "forced", "entdep": "entropy+dep", "typed": "T",
             "typed+final": "forced[final]:T", "gain": "gain", "typed+gain": "max:T,gain"}
    layouts = ["default"] + [f"{k}@{int(R * 100)}" for R in BUDGETS for k in (*kinds, "typed+dis")]
    print(f"fitted on {len(train)} train problems; train rates: " + ", ".join(
        f"{l} {fit.rate(kinds[l.split('@')[0]], int(l.split('@')[1]) / 100):.3f}" for l in layouts[1:] if not l.startswith("typed+dis"))
        + "; disagreement thresholds " + ", ".join(f"@{R} {v:.3f}" for R, v in dis_thr.items()), flush=True)
    del train
    stats = {l: {"res_bits": [], "res_exact": [], "fin_bits": [], "fin_exact": [], "rate": []} for l in layouts}
    covered = {l: {"res": [], "fin": []} for l in layouts}
    t0 = time.time(); used = 0
    for p in [json.loads(l) for l in open(GSM)][:n_problems]:
        pr = problem(p, tok, dev)
        if pr is None:
            continue
        b, ids, spans, res_starts = pr
        n = ids.shape[1]; used += 1
        sig, default, lpP = signals(b, ids)
        forced = {"default": res_starts, "final": marker_starts(b)}
        lay = {"default": default}
        for R in BUDGETS:
            r = int(R * 100)
            for k, kind in kinds.items():
                lay[f"{k}@{r}"] = blt_layouts.layout(kind, R, sig, fit, forced)
            base = blt_layouts.layout("T", BASE * R, sig, fit)
            lay[f"typed+dis@{r}"] = np.union1d(base, np.flatnonzero(disagreement(ids, base, n, lpP) > dis_thr[R]))
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
    print(f"\nBLT-1B, {used} GSM8K test problems, retrieve vs compute signals; {len(stats['default']['res_exact'])} in-line results")
    for name in layouts:
        s, c = stats[name], covered[name]
        registry.emit("realblt_gain", f"RESULT {name:14s} patch rate {np.mean(s['rate']):.3f} | covered: results "
              f"{np.mean(c['res']):5.1%}, final {np.mean(c['fin']):5.1%} | in-line results: {np.mean(s['res_bits']):.3f} bits, "
              f"exact {np.mean(s['res_exact']):5.1%} | final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}",
              experiment="blt_gain", window=blt_load.WINDOW)
    registry.save_items(f"blt_gain{blt_load.SUFFIX}", {"layouts": stats, "covered": covered},
                        experiment="blt_gain", n_problems=used, window=blt_load.WINDOW)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
