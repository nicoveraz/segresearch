"""Typed uncertainty on Meta's BLT-1B (local; follows typed_screen.py): does "sure of the kind, ignorant of the item",
read from BLT-1B's own entropy model, place patches at computed results without labels or math syntax?

Same setup as realblt_budget.py (GSM8K test problems, patches changed at inference, train-fitted thresholds,
512-byte window restored), with BLT-1B's patcher distribution over bytes split into generic byte classes
(typed_screen.byte_classes):

    typed@R    T = P(likeliest class) x within-class entropy / log2|class|, top R
    typedc@R   z(T) - z(class entropy)
    entropy@R, results@R (a boundary after each '= ', label-based reference), entdep@R as in realblt_budget.py

Also reported: coverage of other number starts (numbers not right after '= ', mostly copied from the question or
earlier lines). BLT-1B's entropy model can copy, unlike the tiny model of the screen, so typed uncertainty may be
low at copies and high at computed results.

    .venv-blt/bin/python realblt_typed.py [N_PROBLEMS] [N_TRAIN]
"""
import json
import re
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


def main(n_problems, n_train):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    kinds = {"entropy": "entropy", "results": "forced", "entdep": "entropy+dep", "typed": "T", "typedc": "T+negHc"}
    layouts = ["default"] + [f"{k}@{int(r * 100)}" for r in BUDGETS for k in kinds]
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, logits = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                           max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        lg = torch.log_softmax(logits[0, :, 4:260].float(), -1).cpu().numpy()      # token id = byte + 4
        _, Hc, _, T = typed_screen.typed(lg)
        sig = {}
        for name, v in (("entropy", e), ("T", T), ("negHc", -Hc)):
            s = np.full(n, -np.inf); s[2:] = v[1:n - 1]                 # token t starts a patch from the prediction at t-1
            sig[name] = s
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        sig["dep"] = dep
        return sig, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    def number_starts(b):
        """Token positions of numbers NOT right after '= ' (mostly copies), for the coverage diagnostic."""
        res = {m.start() + 2 for m in re.finditer(rb"= \$?\d", b)}
        return np.array([m.start() + 1 for m in re.finditer(rb"(?<![\d.,$])\$?\d", b) if m.start() not in res], int)

    fit = blt_layouts.Fit(BUDGETS)
    for p in [json.loads(l) for l in open(GSM_TRAIN)][:n_train]:
        pr = problem(p, tok, dev)
        if pr is not None:
            assert int(pr[1][0, 1]) == pr[0][0] + 4, "BLT token ids are not bytes + 4"
            fit.add(signals(pr[0], pr[1])[0], pr[3])
    fit.done()
    print(f"thresholds fitted on {len(fit.sigs)} GSM8K train problems; train patch rates: " + ", ".join(
        f"{l} {fit.rate(kinds[l.split('@')[0]], int(l.split('@')[1]) / 100):.3f}" for l in layouts[1:]), flush=True)
    stats = {l: {"res_bits": [], "res_exact": [], "fin_bits": [], "fin_exact": [], "rate": []} for l in layouts}
    covered = {l: {"res": [], "fin": [], "num": []} for l in layouts}
    t0 = time.time(); used = 0
    for p in [json.loads(l) for l in open(GSM)][:n_problems]:
        pr = problem(p, tok, dev)
        if pr is None:
            continue
        b, ids, spans, res_starts = pr
        n = ids.shape[1]; used += 1
        sig, default = signals(b, ids)
        lay = {"default": default}
        for name in layouts[1:]:
            k, R = name.split("@")
            lay[name] = blt_layouts.layout(kinds[k], int(R) / 100, sig, fit, res_starts if k == "results" else ())
        fin0 = [s for k, s, _ in spans if k == "fin"]
        nums = number_starts(b)
        for name, st in lay.items():
            st = np.array(sorted(set(int(x) for x in st if 0 <= x < n))); S = set(st.tolist())
            covered[name]["res"] += [s in S for s in res_starts]
            covered[name]["fin"] += [s in S for s in fin0]
            covered[name]["num"] += [s in S for s in nums]
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev),
                                             use_cache=False).logits[0].float(), -1)
            stats[name]["rate"].append(len(st) / n)
            for kind, s, e_ in spans:
                tgt = ids[0, s:e_]; pred = lp[s - 1:e_ - 1]
                stats[name][f"{kind}_bits"].append(float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)))
                stats[name][f"{kind}_exact"].append(bool((pred.argmax(-1) == tgt).all()))
        if used % 25 == 0:
            print(f"  {used} problems, {time.time() - t0:.0f}s", flush=True)
    print(f"\nBLT-1B, {used} GSM8K test problems, typed-uncertainty layouts; {len(stats['default']['res_exact'])} in-line results")
    for name in layouts:
        s, c = stats[name], covered[name]
        registry.emit("realblt_typed", f"RESULT {name:10s} patch rate {np.mean(s['rate']):.3f} | covered: results "
              f"{np.mean(c['res']):5.1%}, final {np.mean(c['fin']):5.1%}, other numbers {np.mean(c['num']):5.1%} | "
              f"in-line results: {np.mean(s['res_bits']):.3f} bits, exact {np.mean(s['res_exact']):5.1%} | "
              f"final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}",
              experiment="blt_typed", thresh=blt_layouts.MODE, window=blt_load.WINDOW)
    registry.save_items(f"blt_typed{blt_load.SUFFIX}", {"layouts": stats, "covered": covered},
                        experiment="blt_typed", n_problems=used, window=blt_load.WINDOW)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
