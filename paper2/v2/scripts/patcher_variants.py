"""R16 (paper3/R16_PATCHER_TRAINING_RULES.md): does changing how paper 2's tiny patcher is trained strengthen the
computed-value signal in its state? Variants V0 (next byte, as paper 2), V1 (+ a "new number at the next byte" head),
V2 (+ heads for bytes 2-4 ahead), V3 (larger: d 64, 2 layers), V4 (V1 + V2); same corpus windows, schedule and seed.
Each variant's state (final normed state + entropy) at R10's candidates feeds R13's label-free probe, scored on the
GSM8K prose rewrites and the equation-style originals.

    .venv-blt/bin/python patcher_variants.py      -> results/patcher_variants.json   (PV_SMOKE=1: 200 updates, 40 / 20 problems)
"""
import json
import os
import re

import numpy as np

import _seg  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

import realblt_learned as rl
from realblt_autolabel import autolabel
from scale.model import EntropyLM, ln
from scale.patchers import ENTROPY
from scale.train import CTX, lr_at

SMOKE = os.environ.get("PV_SMOKE") == "1"
CACHE = os.path.expanduser("~/.cache/segresearch-scale")
VARIANTS = {"V0": dict(d=32, layers=1, aux=False, ahead=0), "V1": dict(d=32, layers=1, aux=True, ahead=0),
            "V2": dict(d=32, layers=1, aux=False, ahead=3), "V3": dict(d=64, layers=2, aux=False, ahead=0),
            "V4": dict(d=32, layers=1, aux=True, ahead=3)}
STEPS = 200 if SMOKE else ENTROPY["steps"]
NUM = re.compile(rb"\d[\d,]*(?:\.\d+)?")
NBOOT = rl.NBOOT
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def isal(c):
    return 48 <= c <= 57 or 65 <= c <= 90 or 97 <= c <= 122


class Patcher(nn.Module):
    def __init__(self, d, layers, aux, ahead):
        super().__init__()
        self.lm = EntropyLM(d, layers, ENTROPY["heads"], CTX)          # created first: same init as paper 2's patcher
        self.aux = nn.Linear(d, 1) if aux else None
        self.ahead = nn.ModuleList(nn.Linear(d, d) for _ in range(ahead))

    def forward(self, x):
        lm = self.lm; T = x.shape[1]
        h = lm.emb[x] + lm.pos[:T]
        ar = torch.arange(T, device=x.device)
        mask = (ar[:, None] >= ar[None])[None].expand(x.shape[0], T, T)
        for L in lm.layers:
            h = L(h, mask, lm.heads)
        s = ln(h, lm.lnf)
        return s, s @ lm.emb.T                                          # = EntropyLM.forward for the logits


def new_number_targets(w):
    """For a window w (CTX + 1 bytes): t[j] = 1 if a number starts at w[j + 1] after a non-alphanumeric byte and its
    value does not occur earlier in the window."""
    t = np.zeros(len(w) - 1, np.float32); seen = set()
    for m in NUM.finditer(w):
        s, v = m.start(), m.group(0).replace(b",", b"")
        if s >= 1 and not isal(w[s - 1]) and v not in seen:
            t[s - 1] = 1.0
        seen.add(v)
    return t


def train(cfg, stream, seed=0):
    torch.manual_seed(seed)
    model = Patcher(**cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    rng = np.random.default_rng(seed)                                   # same windows for every variant
    for k in range(STEPS):
        i = rng.integers(0, len(stream) - CTX - 1, ENTROPY["bs"])
        W = [bytes(stream[j:j + CTX + 1]) for j in i]
        x = torch.from_numpy(np.stack([np.frombuffer(w[:-1], np.uint8) for w in W]).astype(np.int64)).to(DEV)
        y = torch.from_numpy(np.stack([np.frombuffer(w[1:], np.uint8) for w in W]).astype(np.int64)).to(DEV)
        for g in opt.param_groups:
            g["lr"] = lr_at(k, ENTROPY["lr"], STEPS)
        s, logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if model.aux is not None:
            tgt = torch.from_numpy(np.stack([new_number_targets(w) for w in W])).to(DEV)
            loss = loss + F.binary_cross_entropy_with_logits(model.aux(s)[..., 0], tgt)
        if len(model.ahead):
            la = [F.cross_entropy((A(s[:, :-q]) @ model.lm.emb.T).reshape(-1, 256), y[:, q:].reshape(-1))
                  for q, A in enumerate(model.ahead, start=1)]
            loss = loss + sum(la) / len(la)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (k + 1) % 1000 == 0:
            print(f"    update {k + 1} loss {loss.item():.3f}", flush=True)
    return model.eval()


@torch.no_grad()
def read(model, b, batch=512):
    """State (final normed), entropy, surprisal and new-number-head logit predicting each byte of b, read with at
    most CTX bytes of context: the first window fills positions 1..CTX, later windows (every CTX/2) the second half."""
    L = len(b); d = model.lm.emb.shape[1]; half = CTX // 2
    S = np.zeros((L, d), np.float32); H = np.zeros(L, np.float32); U = np.zeros(L, np.float32); A = np.zeros(L, np.float32)
    bb = np.frombuffer(b, np.uint8).astype(np.int64)
    starts = [0] + list(range(half, max(half, L - half), half))
    for k0 in range(0, len(starts), batch):
        st = starts[k0:k0 + batch]
        X = np.zeros((len(st), CTX), np.int64); Y = np.zeros((len(st), CTX), np.int64)
        for r, s0 in enumerate(st):
            w = bb[s0:s0 + CTX + 1]; X[r, :min(CTX, len(w))] = w[:CTX]; Y[r, :len(w) - 1] = w[1:]
        s, logits = model(torch.from_numpy(X).to(DEV))
        lp = torch.log_softmax(logits.float(), -1)
        ent = (-(lp.exp() * lp).sum(-1) / np.log(2)).cpu().numpy()
        sur = (-lp.gather(-1, torch.from_numpy(Y).to(DEV)[..., None])[..., 0] / np.log(2)).cpu().numpy()
        aux = model.aux(s)[..., 0].float().cpu().numpy() if model.aux is not None else None
        s = s.float().cpu().numpy()
        for r, s0 in enumerate(st):
            j0 = 0 if s0 == 0 else half
            n = min(CTX, L - 1 - s0) - j0                              # state j predicts byte s0 + j + 1 < L
            if n <= 0:
                continue
            sl = slice(s0 + j0 + 1, s0 + j0 + 1 + n)
            S[sl] = s[r, j0:j0 + n]; H[sl] = ent[r, j0:j0 + n]; U[sl] = sur[r, j0:j0 + n]
            if aux is not None:
                A[sl] = aux[r, j0:j0 + n]
    return S, H, U, A


def main():
    corp = np.memmap(f"{CACHE}/train.u8", np.uint8, "r")
    recs = {sp: [json.loads(l) for l in open(os.path.join(rl.DATA, f"{sp}.jsonl"))] for sp in ("train", "test")}
    if SMOKE:
        recs = {"train": recs["train"][:40], "test": recs["test"][:20]}
    X = {(f, sp): [] for f in rl.FORMATS for sp in recs}
    for f in rl.FORMATS:
        for sp, rs in recs.items():
            for r in rs:
                x = rl.build(r, f); x["idx"] = r["idx"]; x["ya"] = autolabel(x)
                x["num"] = np.array([48 <= x["b"][i] <= 57 for i in x["cand"]], float)
                X[(f, sp)].append(x)
    held = bytes(corp[len(corp) - 2_000_000:])                         # corpus text the 16M-byte sample almost surely skipped
    val = open(f"{CACHE}/bundle/val.u8", "rb").read()
    models = {name: train(cfg, corp) for name, cfg in VARIANTS.items()}
    ref = Patcher(32, 1, False, 0).to(DEV); ref.lm.load_state_dict(torch.load(f"{CACHE}/bundle/entropy.pt", map_location="cpu"))
    models["V0_bundle"] = ref.eval()
    res = {"loss": {}, "auc": {}, "diff": {}, "verdicts": {}}
    for name, m in models.items():
        res["loss"][name] = {"heldout_corpus_bits": float(read(m, held)[2][1:].mean()), "val_bits": float(read(m, val)[2][1:].mean())}
        for (f, sp), xs in X.items():
            for x in xs:
                S, H, U, A = read(m, x["b"]); c = x["cand"]
                x.setdefault("feat", {})[name] = np.c_[S[c], H[c]].astype(np.float32)
                x.setdefault("sig", {})[name] = {"entropy": H[c], "head": A[c]}
        for f in rl.FORMATS:
            tr = X[(f, "train")]
            for x in tr:
                x["keep"] = (x["ya"] > 0) | (np.random.default_rng(x["idx"]).random(len(x["ya"])) < rl.NEG)
            F_ = np.concatenate([x["feat"][name][x["keep"]] for x in tr]); Y = np.concatenate([x["ya"][x["keep"]] for x in tr])
            pm = rl.fit_logit(F_, Y)
            for x in X[(f, "test")]:
                x.setdefault("score", {})[name] = rl.apply(pm, x["feat"][name])
        print(f"{name}: loss {res['loss'][name]}", flush=True)
    brng = np.random.default_rng(1)

    def auc_on(xs, name, meas, sig="probe"):
        y = np.concatenate([x["y"] for x in xs])
        s = np.concatenate([x["score"][name] if sig == "probe" else x["sig"][name][sig] for x in xs])
        if meas == "within_numbers":
            k = np.concatenate([x["num"] for x in xs]) > 0; y, s = y[k], s[k]
        return rl.auc(y, s)

    for f in rl.FORMATS:
        xs = X[(f, "test")]
        B = [[xs[i] for i in brng.integers(0, len(xs), len(xs))] for _ in range(NBOOT)]
        for meas in ("all", "within_numbers"):
            for name in models:
                for sig in ("probe", "entropy") + (("head",) if VARIANTS.get(name, {}).get("aux") else ()):
                    res["auc"][f"{f}|{meas}|{name}|{sig}"] = float(auc_on(xs, name, meas, sig))
            for a, b_ in [(k, "V0") for k in ("V1", "V2", "V3", "V4")] + [(k, "V3") for k in ("V1", "V2", "V4")]:
                bd = [auc_on(pk, a, meas) - auc_on(pk, b_, meas) for pk in B]
                res["diff"][f"{f}|{meas}|{a}-{b_}"] = {"value": float(auc_on(xs, a, meas) - auc_on(xs, b_, meas)),
                                                       "ci": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))]}
        for k, v in res["auc"].items():
            if k.startswith(f + "|"):
                print(f"AUC {k:40s} {v:.3f}", flush=True)
        for k, v in res["diff"].items():
            if k.startswith(f + "|"):
                print(f"DIFF {k:32s} {v['value']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]", flush=True)
    for u, meas in (("U1", "all"), ("U2", "within_numbers")):
        D = res["diff"]; above = lambda k: D[f"prose|{meas}|{k}"]["ci"][0] > 0
        over0 = [k for k in ("V1", "V2", "V4") if above(f"{k}-V0")]
        both = [k for k in over0 if above(f"{k}-V3")]
        res["verdicts"][u] = {"verdict": "HOLDS" if both else "FAILS" if not over0 else "MIXED", "above_V0": over0, "above_V0_and_V3": both}
    json.dump(res, open(os.path.join(_seg.HERE, "results", "patcher_variants_smoke.json" if SMOKE else "patcher_variants.json"), "w"), indent=1)
    for k, v in res["verdicts"].items():
        print("VERDICT", k, json.dumps(v))


if __name__ == "__main__":
    main()
