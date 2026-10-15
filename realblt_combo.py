"""Combining typed uncertainty with an answer boundary and with novelty on BLT-1B (local; follows realblt_typed.py).

realblt_typed.py: typed uncertainty (T) lifts computed results far above entropy but skips final answers (BLT-1B can
predict them), and final-answer accuracy drops. Adding standardized scores hurt in training. Layouts here keep each
signal's choices intact instead of adding scores:

    typed+final@R      a boundary after the text marker 'The final answer is ' (format, not labels), rest by T
    nov@R              the self-supervised novelty table (reuse_screen.py), fitted on GSM8K train text in this format
    max@R              union of top sets: the larger of T's and novelty's train percentiles
    max+final@R        the marker boundary, rest by the union
    nov+final@R        the marker boundary, rest by novelty
SEGR_COMBO_LAYOUTS=a,b,... runs only those layout kinds (plus default), recorded as blt_combo_<NAME> with
SEGR_COMBO_NAME (so a follow-up run never overwrites the first one's items).
    entropy@R, typed@R, entdep@R, results@R as before (references, for paired tests)


Same setup as realblt_budget.py (GSM8K test problems, patches changed at inference, train-fitted thresholds,
512-byte window restored), with BLT-1B's patcher distribution over bytes split into generic byte classes
(typed_screen.byte_classes):

    typed@R    T = P(likeliest class) x within-class entropy / log2|class|, top R
    typedc@R   z(T) - z(class entropy)
    entropy@R, results@R (a boundary after each '= ', label-based reference), entdep@R as in realblt_budget.py

Also reported: coverage of other number starts (numbers not right after '= ', mostly copied from the question or
earlier lines). BLT-1B's entropy model can copy, unlike the tiny model of the screen, so typed uncertainty may be
low at copies and high at computed results.

    .venv-blt/bin/python realblt_combo.py [N_PROBLEMS] [N_TRAIN]
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

import blt_layouts
import blt_load
import registry
import reuse_screen
import typed_screen
from realblt_budget import DEP, GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem

NOVTAB = os.path.expanduser("~/.cache/segresearch-blt/novtab_gsm.npz")
MARKER = b"The final answer is "


def novelty_table():
    """Novelty table fitted on all GSM8K train solutions written as problem() writes them (no labels)."""
    if not os.path.exists(NOVTAB):
        texts = []
        for l in open(GSM_TRAIN):
            p = json.loads(l)
            sol, ans = p["answer"].rsplit("#### ", 1)
            sol = re.sub(r"<<[^>]*>>", "", sol).strip()
            texts.append((p["question"] + "\n" + sol + "\nThe final answer is " + ans.strip()).encode())
        b = np.frombuffer(b"\n\n".join(texts), np.uint8)
        np.savez(NOVTAB, **dict(zip(("T2", "T1", "glob"), reuse_screen.tab_fit(b, -reuse_screen.match_signals(b)[0]))))
        print(f"novelty table fitted on {len(b):,} GSM8K train bytes -> {NOVTAB}", flush=True)
    t = np.load(NOVTAB)
    return lambda b: reuse_screen_lookup(t, b)


def reuse_screen_lookup(t, b):
    b = b.astype(np.int64); p1 = np.r_[0, b[:-1]]; p2 = np.r_[0, 0, b[:-2]]
    d = t["T2"][p2 * 256 + p1]
    d = np.where(np.isnan(d), t["T1"][p1], d)
    return np.where(np.isnan(d), float(t["glob"]), d)

BUDGETS = (0.15, 0.10)


def main(n_problems, n_train):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    kinds = {"entropy": "entropy", "results": "forced", "entdep": "entropy+dep", "typed": "T",
             "typed+final": "forced[final]:T", "nov": "nov", "max": "max:T,nov", "max+final": "forced[final]:max:T,nov",
             "nov+final": "forced[final]:nov"}
    sel = os.environ.get("SEGR_COMBO_LAYOUTS")
    if sel:
        kinds = {k: kinds[k] for k in sel.split(",")}
    else:
        kinds.pop("nov+final")                                       # the first run's layout set
    tag = "blt_combo" + (f"_{os.environ['SEGR_COMBO_NAME']}" if os.environ.get("SEGR_COMBO_NAME") else "")
    nov = novelty_table()
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
        nv = np.full(n, -np.inf); nv[2:] = nov(np.frombuffer(b, np.uint8))[1:n - 1]    # token t = byte t-1
        sig["nov"] = nv
        return sig, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    def marker_starts(b):
        """Token positions right after the final-answer marker (a format cue; causal: depends on earlier bytes)."""
        out, i = [], b.find(MARKER)
        while i >= 0:
            out.append(i + len(MARKER) + 1); i = b.find(MARKER, i + 1)
        return out

    def number_starts(b):
        """Token positions of numbers NOT right after '= ' (mostly copies), for the coverage diagnostic."""
        res = {m.start() + 2 for m in re.finditer(rb"= \$?\d", b)}
        return np.array([m.start() + 1 for m in re.finditer(rb"(?<![\d.,$])\$?\d", b) if m.start() not in res], int)

    fit = blt_layouts.Fit(BUDGETS)
    for p in [json.loads(l) for l in open(GSM_TRAIN)][:n_train]:
        pr = problem(p, tok, dev)
        if pr is not None:
            assert int(pr[1][0, 1]) == pr[0][0] + 4, "BLT token ids are not bytes + 4"
            fit.add(signals(pr[0], pr[1])[0], {"default": pr[3], "final": marker_starts(pr[0])})
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
            lay[name] = blt_layouts.layout(kinds[k], int(R) / 100, sig, fit, {"default": res_starts, "final": marker_starts(b)})
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
    print(f"\nBLT-1B, {used} GSM8K test problems, combination layouts; {len(stats['default']['res_exact'])} in-line results")
    for name in layouts:
        s, c = stats[name], covered[name]
        registry.emit("realblt_combo", f"RESULT {name:10s} patch rate {np.mean(s['rate']):.3f} | covered: results "
              f"{np.mean(c['res']):5.1%}, final {np.mean(c['fin']):5.1%}, other numbers {np.mean(c['num']):5.1%} | "
              f"in-line results: {np.mean(s['res_bits']):.3f} bits, exact {np.mean(s['res_exact']):5.1%} | "
              f"final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}",
              experiment=tag, thresh=blt_layouts.MODE, window=blt_load.WINDOW)
    registry.save_items(f"{tag}{blt_load.SUFFIX}", {"layouts": stats, "covered": covered},
                        experiment=tag, n_problems=used, window=blt_load.WINDOW)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 300)
