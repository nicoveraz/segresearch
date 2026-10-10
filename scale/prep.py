"""Driver for the scaling pilot (#27). Three stages, all resumable:

    python -m scale.prep mac  BUNDLE           # Mac: eval files, contamination list, corpus, entropy model,
                                               #      dependence table -> a small bundle for the GPU
    python -m scale.prep pod  BUNDLE WORK      # GPU: rebuild the corpus (SHA-256 must match), score entropy,
                                               #      fit thresholds, build the masks of every arm
    python -m scale.prep arms BUNDLE WORK ARM...      # add arms (e.g. jump10 entropy20) to a built work dir
    python -m scale.prep bench BUNDLE WORK     # GPU: time an update at each size, project hours per run
    python -m scale.prep run  BUNDLE WORK SIZE ARM SEED    # train and evaluate one model

SIZES and ARMS below define the grid. SEGR_SCALE_TOY=1 shrinks everything (corpus, steps, eval) for a dry run.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

from scale import data
from scale.evaluate import accuracy, bits, end_to_end
from scale.model import EntropyLM
from scale.patchers import ENTROPY, REFERENCE, fit_dependence, score_stream, train_entropy_lm
from scale.rules import Rule, syntax_starts, table_lookup
from scale.train import CTX, device, make_model, train

TOY = os.environ.get("SEGR_SCALE_TOY") == "1"
ARMS = ("entropy10", "dep10", "syntax+entropy10")
# bytes seen per run: 1M as in the paper's runs (32,000 x 32 x 128), 12M 0.5B, 50M 1.5B; lr scaled ~1/D from 3e-3 at D=128
SIZES = {
    "1m":  dict(d=128, glayers=4, bs=32, steps=32000, lr=3e-3, warmup=100),
    "12m": dict(d=320, glayers=8, bs=128, steps=30500, lr=1.2e-3, warmup=1000, micro=128),
    "50m": dict(d=576, glayers=12, bs=256, steps=45800, lr=6.7e-4, warmup=1000, micro=64),
}
N_ACC, N_E2E = (40, 10) if TOY else (1000, 100)
THRESH_SAMPLE = 2_000_000 if TOY else 50_000_000      # positions used to fit the rule thresholds
CHUNK = 8 << 20                                        # bytes per chunk when scoring / masking the corpus


def _toy(cfg):
    return dict(cfg, steps=60, warmup=10) if TOY else cfg


MICRO = int(os.environ.get("SEGR_SCALE_MICRO", "0"))   # override the micro-batch (memory) without changing the update


# ----------------------------------------------------------------------------- Mac stage
def mac(bundle):
    os.makedirs(bundle, exist_ok=True)
    cache = os.path.expanduser("~/.cache/segresearch-scale")
    import mathexp                                    # only here: the eval split comes from the MLX-era math cache
    z = np.load(mathexp.NPZ)
    val, roles, mtrain = z["val_bytes"], z["val_roles"], z["train_bytes"]
    val.tofile(f"{bundle}/val.u8"); roles.astype(np.int8).tofile(f"{bundle}/val_roles.i8"); mtrain.tofile(f"{bundle}/math_train.u8")
    paths = data.download(f"{cache}/owm", data.SHARDS[:1] if TOY else data.SHARDS)
    drop_path = f"{bundle}/owm_drop.json"
    if not os.path.exists(drop_path):
        print("contamination check", flush=True)
        drop = data.contaminated(paths, data.eval_ngrams(val))
        data.save_json(drop, drop_path)
    drop = json.load(open(drop_path))
    corpus = f"{cache}/train{'_toy' if TOY else ''}.u8"
    meta_path = f"{bundle}/corpus_meta.json"
    if not os.path.exists(meta_path):
        print("building the corpus", flush=True)
        meta = data.build_corpus(paths, mtrain, drop, corpus)
        if TOY:                                       # toy: keep the first 40 MB
            b = data.load_bytes(corpus)[:40_000_000].copy(); b.tofile(corpus)
            meta.update(bytes=len(b), sha256=data.sha256(corpus), toy_truncated=True)
        data.save_json(meta, meta_path)
    meta = json.load(open(meta_path))
    print(f"corpus: {meta['bytes'] / 1e9:.2f} GB, sha256 {meta['sha256'][:16]}, {meta['owm_dropped']} documents dropped", flush=True)
    corp = data.load_bytes(corpus)
    dev = device()
    if not os.path.exists(f"{bundle}/entropy.pt"):
        print("training the entropy model", flush=True)
        ent = train_entropy_lm(corp, _toy(ENTROPY) if TOY else ENTROPY, dev=dev)
        torch.save(ent.state_dict(), f"{bundle}/entropy.pt")
    if not os.path.exists(f"{bundle}/dependence.npz"):
        # The paper's dependence table (deptrigger.py marginal mode, fitted on GSM8K/MATH training text with no
        # answer labels), so the dep arm is the paper's dep10 rule. A table fitted on this corpus instead
        # (SEGR_SCALE_DEP=fit) is dominated by OpenWebMath contexts and covers 0% of computed-result starts.
        if os.environ.get("SEGR_SCALE_DEP") == "fit" or TOY:
            print("fitting boundary dependence on the corpus", flush=True)
            tab = fit_dependence(corp, _toy(REFERENCE), n_windows=64 if TOY else 4096, dev=dev, ref_bytes=200_000_000)
            np.savez(f"{bundle}/dependence.npz", **tab)
        else:
            import shutil
            shutil.copy(os.path.join(mathexp.CACHE, "deptrigger.npz"), f"{bundle}/dependence.npz")
    if not os.path.exists(f"{bundle}/novelty.npz"):
        # The self-supervised novelty table of the 1M runs (reuse_screen.py fit, on GSM8K/MATH training text), so
        # the nov arm is mathexp's nov10, as the dep arm is the paper's dep10
        import shutil
        shutil.copy(os.path.join(mathexp.CACHE, "novtab.npz"), f"{bundle}/novelty.npz")
    print(f"bundle ready: {bundle}", flush=True)


def table_for(bundle, arm):
    """The lookup table an arm uses: the novelty table for nov arms, else the dependence table."""
    return dict(np.load(f"{bundle}/{'novelty' if arm.startswith('nov') else 'dependence'}.npz"))


# ----------------------------------------------------------------------------- GPU stage
def _entropy_model(bundle, dev):
    ent = EntropyLM(ENTROPY["d"], ENTROPY["layers"], ENTROPY["heads"], CTX)
    ent.load_state_dict(torch.load(f"{bundle}/entropy.pt", map_location="cpu"))
    return ent.to(dev).eval()


def _chunks(n):
    """[c0, c1) chunks with start on the 64-byte scoring grid, so chunked scoring equals whole-stream scoring."""
    for c0 in range(0, n, CHUNK):
        yield c0, min(c0 + CHUNK, n)


def _score_into(ent, b, H_out):
    """Entropy of every byte of b (memmap) into H_out (float32 memmap), chunk by chunk (same values as one pass)."""
    for c0, c1 in _chunks(len(b)):
        lo, hi = max(0, c0 - CTX), min(len(b), c1 + CTX)
        H, _ = score_stream(ent, np.asarray(b[lo:hi]))
        H_out[c0:c1] = H[c0 - lo:c1 - lo]


def pod(bundle, work):
    os.makedirs(work, exist_ok=True)
    meta = json.load(open(f"{bundle}/corpus_meta.json"))
    corpus = f"{work}/train.u8"
    if not (os.path.exists(corpus) and data.sha256(corpus) == meta["sha256"]):
        paths = data.download(f"{work}/owm", data.SHARDS[:len(meta["shards"])])
        print("building the corpus", flush=True)
        m2 = data.build_corpus(paths, np.fromfile(f"{bundle}/math_train.u8", np.uint8),
                               json.load(open(f"{bundle}/owm_drop.json")), corpus)
        if meta.get("toy_truncated"):
            b = data.load_bytes(corpus)[:meta["bytes"]].copy(); b.tofile(corpus); m2["sha256"] = data.sha256(corpus)
        if m2["sha256"] != meta["sha256"]:
            raise SystemExit(f"corpus SHA-256 {m2['sha256']} does not match the Mac's {meta['sha256']}")
    print(f"corpus verified: {meta['bytes'] / 1e9:.2f} GB", flush=True)
    build_arms(bundle, work, ARMS)


def build_arms(bundle, work, arms):
    """Fit each arm's threshold, write its train and eval masks, and merge it into rules.json (resumable; can add
    arms to a finished work directory: python -m scale.prep arms BUNDLE WORK ARM...)."""
    b = data.load_bytes(f"{work}/train.u8")
    val = np.fromfile(f"{bundle}/val.u8", np.uint8)
    dev = device()
    ent = _entropy_model(bundle, dev)
    H = np.lib.format.open_memmap(f"{work}/train_H.npy", "w+", np.float32, (len(b),)) \
        if not os.path.exists(f"{work}/train_H.done") else np.load(f"{work}/train_H.npy", mmap_mode="r")
    if not os.path.exists(f"{work}/train_H.done"):
        t = time.time(); _score_into(ent, b, H); H.flush(); open(f"{work}/train_H.done", "w").close()
        print(f"entropy scored in {time.time() - t:.0f}s", flush=True)
    Hv, _ = score_stream(ent, val)
    idx = np.random.default_rng(0).choice(len(b), min(THRESH_SAMPLE, len(b)), replace=False)
    idx.sort()
    rpath = f"{work}/rules.json"
    rules = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for arm in arms:
        r = _fit_rule(arm, b, H, idx, table_for(bundle, arm))          # thresholds from a fixed sample of train positions
        rules[arm] = r.state()
        mpath = f"{work}/mask_{arm}.u8"
        if not os.path.exists(mpath + ".done"):
            M = np.memmap(mpath, np.uint8, "w+", shape=(len(b),))
            for c0, c1 in _chunks(len(b)):
                lo = max(0, c0 - 8)
                M[c0:c1] = r.mask(np.asarray(b[lo:c1]), np.asarray(H[lo:c1]))[c0 - lo:]
            M.flush(); open(mpath + ".done", "w").close()
        M = np.memmap(mpath, np.uint8, "r")
        mv = r.mask(val, Hv).astype(np.uint8)
        mv.tofile(f"{work}/val_mask_{arm}.u8")
        print(f"  {arm:17s} threshold {float(r.thr):.4f}  train rate {M[idx].mean():.4f}  val rate {mv.mean():.4f}", flush=True)
        data.save_json(rules, rpath)


def _fit_rule(arm, b, H, idx, table):
    """Rule thresholds fitted on a fixed sample of train positions (idx, sorted). Syntax and dependence are computed
    chunk by chunk over the corpus (vectorized) and read at the sampled positions."""
    if arm.startswith("entropy"):
        return Rule(arm, np.zeros(1, np.uint8), np.asarray(H[idx]))
    if arm.startswith("jump"):                        # entropy rise H[t] - H[t-1] at the sampled positions
        r = Rule.__new__(Rule)
        r.name, r.kind = arm, "jump"
        J = np.asarray(H[idx]).astype(np.float64) - np.asarray(H[np.maximum(idx - 1, 0)])
        J[idx == 0] = 0.0
        r.thr = np.quantile(J.astype(np.float32), 1 - int(arm[4:]) / 100)
        return r
    kind, R = re.fullmatch(r"(entropy|dep|nov|syntax\+entropy)(\d+)", arm).groups()
    R = int(R) / 100
    r = Rule.__new__(Rule)
    r.name, r.kind = arm, kind
    tab = kind in ("dep", "nov")
    if tab:
        r.table = (np.asarray(table["T2"]), np.asarray(table["T1"]), float(table["glob"]))
    vals = np.zeros(len(idx), np.float64 if tab else bool)
    for c0, c1 in _chunks(len(b)):
        sel = slice(np.searchsorted(idx, c0), np.searchsorted(idx, c1))
        if sel.start == sel.stop:
            continue
        lo = max(0, c0 - 8)
        cb = np.asarray(b[lo:c1])
        v = table_lookup(*r.table, cb) if tab else syntax_starts(cb)
        vals[sel] = v[idx[sel] - lo]
    if tab:
        r.thr = np.quantile(vals.astype(np.float32), 1 - R)      # float32 values, as Rule fits them
    else:
        Hs = np.asarray(H[idx]); rest = ~vals
        r.thr = np.quantile(Hs[rest], 1 - (R - vals.mean()) / rest.mean())
    return r


# ----------------------------------------------------------------------------- throughput
def bench(bundle, work, updates=None):
    """Seconds per update and peak memory for each size (dep10 mask), and the projected hours per run."""
    updates = updates or (3 if TOY else 40)
    b = data.load_bytes(f"{work}/train.u8")
    M = np.memmap(f"{work}/mask_dep10.u8", np.uint8, "r")
    dev = device()
    out = {}
    for size, cfg in SIZES.items():
        cfg = dict(cfg, steps=updates + 2, seed=0)
        if MICRO:
            cfg["micro"] = MICRO
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        train(dict(cfg), b, M, dev=dev, log_every=0, losses=times)     # warm-up and timing below
        t = time.time()
        train(dict(cfg, steps=updates), b, M, dev=dev, log_every=0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sec = (time.time() - t) / updates
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
        out[size] = {"sec_per_update": sec, "hours_per_run": sec * SIZES[size]["steps"] / 3600, "peak_gib": mem}
        print(f"  {size}: {sec * 1000:.0f} ms/update -> {out[size]['hours_per_run']:.2f} h per run, peak {mem:.1f} GiB", flush=True)
    data.save_json(out, f"{work}/bench.json")
    return out


# ----------------------------------------------------------------------------- one run
def run(bundle, work, size, arm, seed):
    import registry
    out = f"{work}/runs/{size}_{arm}_s{seed}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/result.json"):
        print(f"done already: {out}"); return
    cfg = dict(_toy(SIZES[size]), seed=seed)
    if MICRO:
        cfg["micro"] = MICRO
    b = data.load_bytes(f"{work}/train.u8")
    M = np.memmap(f"{work}/mask_{arm}.u8", np.uint8, "r")
    dev = device()
    t0 = time.time()
    model = train(cfg, b, M, ckpt=f"{out}/ckpt.pt", dev=dev).eval()
    t_train = time.time() - t0
    val = np.fromfile(f"{bundle}/val.u8", np.uint8); roles = np.fromfile(f"{bundle}/val_roles.i8", np.int8)
    mv = np.fromfile(f"{work}/val_mask_{arm}.u8", np.uint8)
    rules = json.load(open(f"{work}/rules.json"))
    rule = Rule.from_state(rules[arm], table_for(bundle, arm))
    ent = _entropy_model(bundle, dev)
    with torch.no_grad():
        o = bits(model, val, mv, roles)
        o.update(accuracy(model, ent, rule, val, roles, seed, n_acc=N_ACC))
        o.update(end_to_end(model, ent, rule, val, roles, n=N_E2E, seed=seed))
    n_par = sum(p.numel() for p in model.parameters())
    o.update(size=size, arm=arm, seed=seed, params=n_par, train_seconds=round(t_train), cfg=cfg,
             device=torch.cuda.get_device_name() if dev.type == "cuda" else dev.type, toy=TOY)
    data.save_json(o, f"{out}/result.json")
    line = (f"RESULT math rule={arm} seed={seed} steps={cfg['steps']} D={cfg['d']} glayers={cfg['glayers']} "
            f"pool=xattn local=window rate={o['rate']:.3f} (patches {o['rate']:.3f}, scratchpads 0.000) bpb={o['bpb']:.4f} | "
            + " ".join(f"{k}: {o[k + '_bits']:.3f} bits, acc {o.get(k + '_acc', float('nan')):.3f} (n={o.get(k + '_n', 0)})"
                       for k in ("computed", "copy", "final", "boxed")) + f" | {t_train:.0f}s")
    if TOY:                                           # dry runs never reach the results registry
        print(line, flush=True)
    else:
        packed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMMIT")
        registry.emit("scale", line, experiment="scale", corpus="owm+math", size=size, params=n_par,
                      e2e_acc=o["e2e_acc"], e2e_n=o["e2e_n"], device=o["device"],
                      code_commit=open(packed).read().strip() if os.path.exists(packed) else None)
    if os.path.exists(f"{out}/ckpt.pt"):
        os.remove(f"{out}/ckpt.pt")                   # results are kept; checkpoints are not (disk)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "mac":
        mac(a[1])
    elif a[0] == "pod":
        pod(a[1], a[2])
    elif a[0] == "arms":
        build_arms(a[1], a[2], a[3:])
    elif a[0] == "bench":
        bench(a[1], a[2])
    elif a[0] == "run":
        run(a[1], a[2], a[3], a[4], int(a[5]))
    else:
        raise SystemExit(__doc__)
