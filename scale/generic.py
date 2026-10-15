"""Does novelty patching beat entropy patching outside math? (local; Kaggle T4 x2)

The scaling runs use math text, so their measures (computed results, final answers) do not exist elsewhere. Here,
per domain, a 12M BLT-style model (scale/prep.py SIZES["12m"], unclipped like the pilot) is trained with
    entropy10   the small model's next-byte entropy, top 10% of corpus bytes      (BLT's rule)
    (also built: novelty_t10, novelty at entropy10's 10% TRAINING budget; entropy_tnov, entropy at novelty's
    training budget: the training-matched pairs, when matching eval rates leaves the training budgets unequal;
    SEGR_GENERIC_ENT_BUDGETS="14 17" adds entropy_t14, entropy_t17: entropy's accuracy-vs-compute curve)
    novelty     the self-supervised novelty table fitted on the corpus itself, its budget set so that its EVAL
                patch rate matches entropy10's (both rules then spend the same compute where it is measured)
and scored by bits per byte on held-out text: overall, and split into NEW bytes (no forward match of >= 4 bytes
in the previous 128) vs COPIED bytes (a diagnostic label for evaluation only; the rules never see it).

    domains: trace  synthetic execution traces (traces(); computed vs copied printed values; accuracy on both)
             code   Python files installed on the machine (stdlib + site-packages; deduplicated, sorted, ~600 MB),
                    5% of files held out by path hash
             prose  WikiText-103 (raw), train / validation splits (downloaded from Hugging Face)

    python -m scale.generic prep DOMAIN WORK          # corpus, entropy model, novelty table, thresholds, masks
    python -m scale.generic run DOMAIN WORK RULE SEED # train and score one model -> WORK/results/RULE_sSEED.json
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch

from scale.patchers import ENTROPY, score_stream, train_entropy_lm
from scale.rules import table_lookup
from scale.train import CTX, device, train

TOY = os.environ.get("SEGR_GENERIC_TOY") == "1"          # Mac dry run: small corpus, few steps
CODE_CAP = (20 if TOY else 600) << 20
VAL_CAP = (1 if TOY else 2) << 20
CH = 8 << 20
TRACE_CAP = (20 if TOY else 300) << 20


def build_corpus(domain, work):
    if os.path.exists(f"{work}/train.u8"):
        return
    os.makedirs(work, exist_ok=True)
    if domain == "code":
        roots = sorted({os.path.dirname(os.__file__), *[p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]})
        files = sorted({f for r in roots if os.path.isdir(r) for f in glob.glob(f"{r}/**/*.py", recursive=True)})
        seen, tr, va, size = set(), [], [], 0
        for f in files:
            try:
                t = open(f, "rb").read()
            except OSError:
                continue
            h = hashlib.sha1(t).hexdigest()
            if h in seen or len(t) < 200:
                continue
            seen.add(h)
            (va if int(h[:4], 16) % 20 == 0 else tr).append(t)
            size += len(t)
            if size > CODE_CAP:
                break
        train_b, val_b = b"\n\n".join(tr), b"\n\n".join(va)[:VAL_CAP]
    elif domain == "trace":
        train_b, val_b = traces(TRACE_CAP, seed=0), traces(VAL_CAP, seed=1)
    else:
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        train_b = "".join(d["train"]["text"]).encode()
        val_b = "".join(d["validation"]["text"]).encode()[:VAL_CAP]
    np.frombuffer(train_b, np.uint8).tofile(f"{work}/train.u8")
    np.frombuffer(val_b, np.uint8).tofile(f"{work}/val.u8")
    json.dump({"domain": domain, "train_bytes": len(train_b), "val_bytes": len(val_b),
               "train_sha256": hashlib.sha256(train_b).hexdigest()}, open(f"{work}/corpus.json", "w"))
    print(f"{domain}: {len(train_b) / 1e6:.0f} MB train, {len(val_b) / 1e6:.1f} MB val", flush=True)


def traces(n_bytes, seed):
    """Synthetic execution traces: short straight-line programs whose lines print their value after ' -> '.
        a = 17                 literal (no printed value)
        b = a + 5 -> 22        computed: a new value
        c = b -> 22            copied: the value of another variable, printed a few lines earlier
        print(d) -> 44         copied
    Programs of 6-14 lines over a handful of live variables, values 0-999, separated by a blank line."""
    rng = np.random.default_rng(seed)
    names = [chr(c) for c in range(97, 123)] + [a + b for a in "xyzpq" for b in "0123456789"]
    out, size = [], 0
    while size < n_bytes:
        env, lines = {}, []
        for _ in range(int(rng.integers(6, 15))):
            live = list(env)
            r = rng.random()
            if len(live) < 2 or r < 0.2:
                v = names[int(rng.integers(len(names)))]; env[v] = int(rng.integers(0, 100))
                lines.append(f"{v} = {env[v]}")
                continue
            u = live[int(rng.integers(len(live)))]
            if r < 0.4:
                lines.append(f"print({u}) -> {env[u]}")
                continue
            v = names[int(rng.integers(len(names)))]
            if r < 0.55:
                env[v] = env[u]; lines.append(f"{v} = {u} -> {env[v]}")
                continue
            for _ in range(5):
                op = "+-*"[int(rng.integers(3))]
                if rng.random() < 0.5:
                    w = live[int(rng.integers(len(live)))]; x, wt = env[w], w
                else:
                    x = int(rng.integers(1, 20)); wt = str(x)
                val = env[u] + x if op == "+" else env[u] - x if op == "-" else env[u] * x
                if 0 <= val < 1000:
                    env[v] = val; lines.append(f"{v} = {u} {op} {wt} -> {val}")
                    break
            else:
                env[v] = env[u]; lines.append(f"{v} = {u} -> {env[v]}")
        s = ("\n".join(lines) + "\n\n").encode()
        out.append(s); size += len(s)
    return b"".join(out)[:n_bytes]


def trace_targets(val):
    """Value spans [s, e) after ' -> ' in held-out traces: computed (the line has an operator) or copied (an
    assignment from one variable, or print). Same window rule as code_targets."""
    s_all = val.tobytes(); n = len(s_all)
    ok_pos = lambda s, e: (s - 1) // CTX == (e - 2) // CTX and (s - 1) % CTX >= CTX // 2 and e <= n - 1
    out = {"computed": [], "copied": []}
    for m in re.finditer(rb"(?m)^(.*) -> (\d+)$", s_all):
        s, e = m.span(2)
        if ok_pos(s, e):
            out["computed" if re.search(rb" [-+*] ", m.group(1)) else "copied"].append((s, e))
    return out


def forward_match(b, W=CTX, cap=16):
    """Length of the longest prefix of b[t:] that also starts in the previous W bytes (reuse_screen.match_signals)."""
    n = len(b); b = b.astype(np.int16); nov = np.zeros(n, np.int16)
    for d in range(1, W):
        eq = np.zeros(n, bool); eq[d:] = b[d:] == b[:-d]
        run = np.zeros(n + 1, np.int16)
        for _ in range(cap):
            run[:-1] = eq * (1 + run[1:])
        nov = np.maximum(nov, np.minimum(run[:-1], cap))
    return nov


def tab_fit(b, target):
    """Mean target per preceding 2-byte context (fallback 1 byte, then the global mean), as reuse_screen.tab_fit."""
    b = b.astype(np.int64); p1 = np.r_[0, b[:-1]]; p2 = np.r_[0, 0, b[:-2]]
    key = p2 * 256 + p1
    s2, c2 = np.bincount(key, target, 65536), np.bincount(key, None, 65536)
    s1, c1 = np.bincount(p1, target, 256), np.bincount(p1, None, 256)
    return (np.where(c2 >= 20, s2 / np.maximum(c2, 1), np.nan), np.where(c1 >= 20, s1 / np.maximum(c1, 1), np.nan),
            float(target.mean()))


def prep(domain, work):
    t0 = time.time()
    build_corpus(domain, work)
    b = np.memmap(f"{work}/train.u8", np.uint8, "r"); val = np.fromfile(f"{work}/val.u8", np.uint8)
    dev = device()
    rng = np.random.default_rng(0)
    sample = np.concatenate([np.asarray(b[s:s + 200_000]) for s in rng.integers(0, len(b) - 200_000, 4 if TOY else 20)])
    ent = train_entropy_lm(np.asarray(b[:200_000_000]), dict(ENTROPY, steps=200) if TOY else ENTROPY, dev=dev, log_every=0)
    torch.save(ent.state_dict(), f"{work}/entropy.pt")
    H = np.zeros(len(b), np.float32)
    for c0 in range(0, len(b), CH):                       # chunked scoring, same values as one pass
        c1 = min(len(b), c0 + CH); lo, hi = max(0, c0 - CTX), min(len(b), c1 + CTX)
        H[c0:c1] = score_stream(ent, np.asarray(b[lo:hi]), surprisal=False)[0][c0 - lo:c1 - lo]
    Hv = score_stream(ent, val, surprisal=False)[0]
    tab = tab_fit(sample, -forward_match(sample))
    np.savez(f"{work}/novelty.npz", T2=tab[0], T1=tab[1], glob=tab[2])
    Hs = H[rng.integers(0, len(b), 5_000_000)]
    thr_e = float(np.quantile(Hs, 0.9))
    Ns = table_lookup(*tab, sample); Nv = table_lookup(*tab, val)
    rate_e = float((Hv[CTX:] > thr_e).mean())
    # the novelty threshold whose EVAL rate is closest to entropy10's (the table's values come in steps)
    cands = np.unique(np.quantile(Ns, np.linspace(0.70, 0.97, 55)))
    thr_n = float(min(cands, key=lambda t: abs((Nv[CTX:] > t).mean() - rate_e)))
    masks = {"entropy10": H > thr_e}
    Nall = np.concatenate([table_lookup(*tab, np.asarray(b[max(0, c0 - 2):c0 + CH]))[min(c0, 2):]
                           for c0 in range(0, len(b), CH)])
    masks["novelty"] = Nall > thr_n
    # training-matched variants: novelty at entropy's 10% training budget, entropy at novelty's training budget
    thr_n10 = float(np.quantile(Ns, 0.9))
    thr_et = float(np.quantile(Hs, 1 - float(masks["novelty"].mean())))
    masks["novelty_t10"] = Nall > thr_n10
    masks["entropy_tnov"] = H > thr_et
    if os.environ.get("SEGR_GENERIC_NOVNET") == "1":
        # novelty_net: the same novelty target predicted from the last 128 bytes by a small causal network
        # (scale.learned_boundary) fitted on the sample; threshold set so its EVAL rate matches entropy10's
        from scale.learned_boundary import fit_net, predict_batched
        net = fit_net(sample, -forward_match(sample).astype(np.float32), steps=300 if TOY else 3000, dev=dev)
        Pv = predict_batched(net, val, dev)
        thr_nn = float(np.quantile(Pv[CTX:], 1 - rate_e))
        masks["novelty_net"] = np.concatenate([predict_batched(net, np.asarray(b[c0:c0 + CH]), dev) > thr_nn
                                               for c0 in range(0, len(b), CH)])
        torch.save(net.state_dict(), f"{work}/novelty_net.pt")
    vals = {"entropy10": Hv > thr_e, "novelty": Nv > thr_n, "novelty_t10": Nv > thr_n10, "entropy_tnov": Hv > thr_et}
    if "novelty_net" in masks:
        vals["novelty_net"] = Pv > thr_nn
    for r in os.environ.get("SEGR_GENERIC_ENT_BUDGETS", "").split():   # extra entropy budgets: entropy_t<R>
        t = float(np.quantile(Hs, 1 - int(r) / 100))
        masks[f"entropy_t{r}"] = H > t; vals[f"entropy_t{r}"] = Hv > t
    for name, m in masks.items():
        m.astype(np.uint8).tofile(f"{work}/mask_{name}.u8")
    for name, m in vals.items():
        m.astype(np.uint8).tofile(f"{work}/val_mask_{name}.u8")
    copy = forward_match(val) >= 4
    np.save(f"{work}/val_copy.npy", copy)
    info = {"thr_entropy": thr_e, "thr_novelty": thr_n, "thr_novelty_t10": thr_n10, "thr_entropy_tnov": thr_et,
            "train_rate": {k: float(m.mean()) for k, m in masks.items()},
            "val_rate": {k: float(v[CTX:].mean()) for k, v in vals.items()},
            "val_copy_share": float(copy.mean()), "seconds": round(time.time() - t0)}
    json.dump(info, open(f"{work}/rules.json", "w"), indent=1)
    print(json.dumps(info), flush=True)


TOKEN = re.compile(rb"[A-Za-z_][A-Za-z0-9_]+|\d[\d_]*(?:\.\d+)?")


def code_targets(val):
    """Token spans [s, e) of held-out code for exact-match accuracy, split by kind (code's analogue of math's
    computed results vs copied numbers; labels for evaluation only, the rules never see them):
        id_copy / id_new     identifiers (>= 2 bytes) that do / do not occur in the previous 128 bytes
        num_new              numeric literals not in the previous 128 bytes
        rhs_copy / rhs_new   the first token of a value: right after '=' (not '=='), 'return ' or '(' (spaces skipped)
    Only spans inside one evaluation window and in its second half (>= CTX // 2 bytes of context) count."""
    s_all = val.tobytes(); n = len(s_all)
    ok_pos = lambda s, e: (s - 1) // CTX == (e - 2) // CTX and (s - 1) % CTX >= CTX // 2 and e <= n - 1
    seen = lambda s, e: s_all[s:e] in s_all[max(0, s - CTX):s]
    out = {k: [] for k in ("id_copy", "id_new", "num_new", "rhs_copy", "rhs_new")}
    rhs_starts = set()
    for m in re.finditer(rb"(?<![=!<>])=(?!=)|return |\(", s_all):
        j = m.end()
        while j < n and s_all[j] == 32:
            j += 1
        rhs_starts.add(j)
    for m in TOKEN.finditer(s_all):
        s, e = m.span()
        if not ok_pos(s, e):
            continue
        cp = seen(s, e)
        if s in rhs_starts:
            out["rhs_copy" if cp else "rhs_new"].append((s, e))
        if s_all[s:s + 1].isdigit():
            if not cp:
                out["num_new"].append((s, e))
        else:
            out["id_copy" if cp else "id_new"].append((s, e))
    return out


def run(domain, work, rule, seed):
    from scale.prep import SIZES
    out = f"{work}/results/{rule}_s{seed}.json"
    if os.path.exists(out):
        return
    os.makedirs(f"{work}/results", exist_ok=True)
    b = np.memmap(f"{work}/train.u8", np.uint8, "r"); M = np.memmap(f"{work}/mask_{rule}.u8", np.uint8, "r")
    val = np.fromfile(f"{work}/val.u8", np.uint8); mv = np.fromfile(f"{work}/val_mask_{rule}.u8", np.uint8)
    copy = np.load(f"{work}/val_copy.npy")
    dev = device()
    cfg = dict(SIZES["12m"], seed=seed)
    if TOY:
        cfg.update(steps=30, warmup=5, micro=32)
    t0 = time.time()
    model = train(cfg, b, M, ckpt=f"{work}/results/{rule}_s{seed}.pt", dev=dev).eval()
    ev = val.astype(np.int64); n = (len(ev) - 1) // CTX
    X = np.stack([ev[k * CTX:(k + 1) * CTX] for k in range(n)]); Y = np.stack([ev[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    BD = np.stack([mv[k * CTX:k * CTX + CTX + 1].astype(np.int64) for k in range(n)]); BD[:, 0] = 1
    C = np.stack([copy[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    with torch.no_grad():
        lp = np.concatenate([torch.log_softmax(model(torch.from_numpy(X[i:i + 128]).to(dev), torch.from_numpy(BD[i:i + 128]).to(dev)).float(), -1).cpu().numpy().astype(np.float64)
                             for i in range(0, n, 128)])
    bt = -np.take_along_axis(lp, Y[..., None], -1)[..., 0] / np.log(2)
    keep = np.zeros_like(bt, bool); keep[:, CTX // 2:] = True
    res = {"domain": domain, "rule": rule, "seed": seed, "bpb": float(bt[keep].mean()),
           "bpb_new": float(bt[keep & ~C].mean()), "bpb_copy": float(bt[keep & C].mean()),
           "val_rate": float(mv.mean()), "train_seconds": round(time.time() - t0), "cfg": cfg,
           "device": torch.cuda.get_device_name() if dev.type == "cuda" else dev.type}
    if domain in ("code", "trace"):
        # exact match of whole tokens given the true prefix (greedy decoding of a token is right iff every byte's
        # argmax is right; the rules are causal, so the layout on the true prefix is the one greedy decoding sees)
        ok = np.zeros(len(val), bool)
        ok[1:n * CTX + 1] = (lp.argmax(-1) == Y).reshape(-1)
        cs = np.r_[0, np.cumsum(~ok)]
        for k, spans in (code_targets(val) if domain == "code" else trace_targets(val)).items():
            hit = [cs[e] - cs[s] == 0 for s, e in spans]
            res[f"acc_{k}"] = float(np.mean(hit)) if hit else float("nan"); res[f"n_{k}"] = len(hit)
    json.dump(res, open(out, "w"), indent=1)
    if os.path.exists(f"{work}/results/{rule}_s{seed}.pt"):
        os.remove(f"{work}/results/{rule}_s{seed}.pt")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "prep":
        prep(a[1], a[2])
    else:
        run(a[1], a[2], a[3], int(a[4]))
