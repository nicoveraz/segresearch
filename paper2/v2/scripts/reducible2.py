"""Reducible surprisal, redone per paper3/REVIEW_2026-10-22.md A1-A2: STRIPPED GSM8K format (the format of the
BLT-1B layout results) and an INDEPENDENT novelty axis.

Per GSM8K test problem (formatted as realblt_budget.problem, <= 1000 tokens), with BLT-1B (window 512):
    s_small, ent_small   BLT-1B's patcher: surprisal of the true byte and entropy (entropy patching's score)
    s_large              BLT-1B itself (default patching)
    reducible            s_small - s_large
    lz_true (lz_true4)   independent novelty: 1 - share of the next 8 (4) bytes that lie inside an LZ77 match
                         (>= 3 bytes) with text earlier in the same problem. Computed from the text, NOT from any
                         rule under test (non-causal: it looks ahead; used only as an axis and as an oracle rule)
    tok_new              finer axis (issue #18): 1 if the whole word or number at (or next after) the position does
                         not occur as a whole token earlier in the problem, else 0. Per value, not per 8-byte window.
                         At the kind level it is partly circular (copied units are numbers seen before), so the test
                         is the rules' SELECTIONS: share new among the positions each rule picks.
Columns are cached (COLS) so new axes need no BLT-1B forward: `reducible2.py axes` reanalyses the cache.
Rules ranked for a 10% budget: ent_small, nov (paper 2's GSM8K-fitted table), novc (the table fitted on the raw
training corpus), s_small, reducible (oracle), lz_true (oracle). Reported: means at unit starts by kind; Spearman
of each causal score with reducible and with lz_true; coverage of unit starts; the plane (lz_true percentile,
s_large percentile) of each kind and of each rule's selected positions.

    .venv-blt/bin/python reducible2.py [N_PROBLEMS] [OUT]
"""
import json
import os
import re
import sys

import numpy as np

import _seg  # noqa: F401  (segresearch submodule on sys.path)
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUNDLE = os.path.expanduser("~/.cache/segresearch-scale/bundle")
MAXT = 1000
COLS = os.path.expanduser("~/.cache/segresearch-scale/ncm_math/reducible2_cols.npz")
TOK = re.compile(rb"[A-Za-z]+|[0-9]+(?:[.,][0-9]+)*")


def tok_new(b):
    """Per byte: is the token covering it (gaps belong to the next token) new as a whole token in b so far?"""
    out = np.zeros(len(b)); seen = set(); prev = 0
    for m in TOK.finditer(b):
        w = m.group(0)
        out[prev:m.end()] = w not in seen
        seen.add(w); prev = m.end()
    return out


def tok_isnum(b):
    out = np.zeros(len(b), bool); prev = 0
    for m in TOK.finditer(b):
        out[prev:m.end()] = m.group(0)[:1].isdigit(); prev = m.end()
    return out


def lz_true(b, ahead=8, minlen=3, maxlen=16):
    n = len(b)
    L = np.zeros(n, int)
    for j in range(1, n):
        best = 0
        for l in range(minlen, min(maxlen, n - j) + 1):
            s = b.find(b[j:j + l], 0, j + l - 1)
            if s == -1 or s >= j:
                break
            best = l
        L[j] = best
    inm = np.zeros(n, bool)
    for j in np.flatnonzero(L):
        inm[j:j + L[j]] = True
    return np.array([1 - inm[j:j + ahead].mean() if j < n else 1.0 for j in range(n)])


def rankdata(x):
    u, inv, cnt = np.unique(np.asarray(x, float), return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2)[inv]


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def main(n_problems=1319, out=None):
    import blt_load
    import stripped_units as su
    from realblt_budget import GSM, MODEL
    from scale.rules import table_lookup
    from transformers import AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() and not os.environ.get("RED_CPU") else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev).eval()
    cfg = model.config
    # nov: paper 2's table fitted on stripped GSM8K train text (the bundle's novelty.npz is fitted on annotated text)
    tabs = {k: np.load(f) for k, f in (("nov", os.path.expanduser("~/.cache/segresearch-blt/novtab_gsm.npz")),
                                        ("novc", f"{BUNDLE}/novelty_corpus.npz"))}
    cols = {k: [] for k in ("s_small", "s_large", "ent_small", "nov", "novc", "lz_true", "lz_true4", "tok_new", "is_num", "kind")}
    used = 0
    for p in [json.loads(l) for l in open(GSM)][:n_problems]:
        text, s0 = su.gsm8k_text(p)
        b = text.encode()
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] > MAXT or ids.shape[1] != len(b) + 1:
            continue
        used += 1
        with torch.no_grad():
            _, _, plog = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                             max_patch_length=cfg.max_patch_length)
            llog = model(input_ids=ids, use_cache=False).logits
        tgt = ids[0, 1:] - 4                                          # byte j = token j+1
        lps = torch.log_softmax(plog[0, :-1, 4:260].float(), -1); lpl = torch.log_softmax(llog[0, :-1, 4:260].float(), -1)
        ss = (-lps.gather(1, tgt[:, None])[:, 0] / np.log(2)).cpu().numpy()
        sl = (-lpl.gather(1, tgt[:, None])[:, 0] / np.log(2)).cpu().numpy()
        es = (-(lps.exp() * lps).sum(-1) / np.log(2)).cpu().numpy()
        arr = np.frombuffer(b, np.uint8)
        kind = np.array([""] * len(b), object)
        for a, e, k in su.units(text, s0):
            kind[len(text[:a].encode())] = k                          # char -> byte offset (GSM8K has non-ASCII)
        cols["s_small"].append(ss); cols["s_large"].append(sl); cols["ent_small"].append(es)
        for k, t in tabs.items():
            cols[k].append(table_lookup(t["T2"], t["T1"], float(t["glob"]), arr))
        cols["lz_true"].append(lz_true(b)); cols["lz_true4"].append(lz_true(b, ahead=4)); cols["kind"].append(kind)
        cols["tok_new"].append(tok_new(b)); cols["is_num"].append(tok_isnum(b))
        if used % 100 == 0:
            print(f"  {used} problems", flush=True)
    C = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(COLS, **{k: (v.astype(str) if k == "kind" else v) for k, v in C.items()}, used=used)
    analyse(C, used, out)


def analyse(C, used, out=None):
    red = C["s_small"] - C["s_large"]
    kinds = ("computed", "copied", "final", "word")
    res = {"n_problems": used, "n_bytes": int(len(red)), "format": "stripped (realblt_budget.problem)",
           "by_kind": {}, "spearman": {}, "coverage_top10": {}, "plane": {}, "tok_new": {}}
    if "tok_new" in C:
        res["tok_new"]["all bytes"] = {"share_new": float(C["tok_new"].mean()), "share_on_numbers": float(C["is_num"].mean()),
                                       **{f"share_new_at_{k}": float(C["tok_new"][C["kind"] == k].mean()) for k in kinds}}
    for k in kinds + ("all bytes",):
        m = (C["kind"] == k) if k != "all bytes" else np.ones(len(red), bool)
        res["by_kind"][k] = {"n": int(m.sum()), **{c: float(np.mean(C[c][m])) for c in
                             ("s_small", "s_large", "ent_small", "nov", "novc", "lz_true", "lz_true4")}, "reducible": float(red[m].mean())}
    scores = {"ent_small": C["ent_small"], "nov": C["nov"], "novc": C["novc"], "s_small": C["s_small"],
              "reducible (oracle)": red, "lz_true (oracle)": C["lz_true"], "lz_true4 (oracle)": C["lz_true4"]}
    for name in ("ent_small", "nov", "novc", "s_small"):
        res["spearman"][name] = {"with_reducible": spearman(scores[name], red),
                                 "with_lz_true": spearman(scores[name], C["lz_true"]),
                                 "with_lz_true4": spearman(scores[name], C["lz_true4"])}
    res["spearman"]["lz_true"] = {"with_reducible": spearman(C["lz_true"], red)}
    lz_p = rankdata(C["lz_true"]) / len(red); sl_p = rankdata(C["s_large"]) / len(red)
    lz4_p = rankdata(C["lz_true4"]) / len(red)
    for k in kinds:
        m = C["kind"] == k
        res["plane"][k] = {"lz_true_pct": float(lz_p[m].mean()), "lz_true4_pct": float(lz4_p[m].mean()),
                           "s_large_pct": float(sl_p[m].mean())}
    rng = np.random.default_rng(0)
    for name, x in scores.items():
        order = np.lexsort((rng.random(len(x)), -x))                  # ties broken at random
        top = np.zeros(len(x), bool); top[order[: int(0.10 * len(x))]] = True
        res["coverage_top10"][name] = {k: float(top[C["kind"] == k].mean()) for k in kinds}
        res["plane"][f"selected by {name}"] = {"lz_true_pct": float(lz_p[top].mean()), "lz_true4_pct": float(lz4_p[top].mean()),
                                               "s_large_pct": float(sl_p[top].mean())}
        if "tok_new" in C:                                            # issue #18: share of selections on new tokens
            boot = [C["tok_new"][rng.choice(np.flatnonzero(top), top.sum())].mean() for _ in range(200)]
            res["tok_new"][name] = {"share_new": float(C["tok_new"][top].mean()),
                                    "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                    "share_on_numbers": float(C["is_num"][top].mean()),
                                    "share_new_among_numbers": float(C["tok_new"][top & C["is_num"]].mean()),
                                    "share_new_among_words": float(C["tok_new"][top & ~C["is_num"]].mean()),
                                    "s_large_pct": float(sl_p[top].mean())}
    out = out or os.path.join(HERE, "results", "reducible_stripped.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["axes"]:
        z = np.load(COLS, allow_pickle=False)
        analyse({k: (z[k].astype(object) if k == "kind" else z[k]) for k in z.files if k != "used"}, int(z["used"]),
                *sys.argv[2:3])
    else:
        main(*(int(x) for x in sys.argv[1:2]), *sys.argv[2:3])
