"""Evaluation for the scaling pilot (#27), matching mathexp._bits and mathexp._accuracy, without MLX.

  bits(model, ...)      bits per byte on non-overlapping CTX windows of the eval split (second half of each window
                        scored), overall and per answer type
  accuracy(model, ...)  exact match of greedy decoding given the true prefix, for computed results, final answers
                        and MATH \\boxed{} answers; patch boundaries decided online by the rule from the bytes seen
                        so far and the entropy model. All windows have the same length (CTX - 1), so every target of
                        a type decodes in one batch; each row is computed exactly as the sequential MLX loop.
  end_to_end(model,...) greedy generation of whole GSM8K solutions from the question (the model sees at most
                        CTX - 1 bytes), scored on the number after '#### '. A curiosity at this size; expect ~0.
"""
import numpy as np
import torch

from scale.train import CTX

ROLES = ["TEXT", "STRUCT", "HEX", "VAR", "VALUE", "OPERAND", "ANS_LOCAL", "ANS_LONG"]      # prepare.ROLES
RI = {r: i for i, r in enumerate(ROLES)}
ANSWER_ROLES = {"computed": "ANS_LOCAL", "copy": "VALUE", "final": "ANS_LONG", "boxed": "VAR"}  # mathexp.ANSWER_ROLES


def _dev(model):
    return next(model.parameters()).device


@torch.no_grad()
def bits(model, ev_bytes, ev_mask, ev_roles, batch=128):
    dev = _dev(model)
    ev, bev = ev_bytes.astype(np.int64), ev_mask.astype(np.int64)
    n = (len(ev) - 1) // CTX
    X = np.stack([ev[k * CTX:(k + 1) * CTX] for k in range(n)])
    Y = np.stack([ev[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    R = np.stack([ev_roles[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    BD = np.stack([bev[k * CTX:k * CTX + CTX + 1] for k in range(n)]); BD[:, 0] = 1
    lp = np.concatenate([torch.log_softmax(model(torch.from_numpy(X[b:b + batch]).to(dev),
                                                 torch.from_numpy(BD[b:b + batch]).to(dev)).double(), -1).cpu().numpy()
                         for b in range(0, n, batch)])
    bt = -np.take_along_axis(lp, Y[..., None], -1)[..., 0] / np.log(2)
    keep = np.zeros_like(bt, bool); keep[:, CTX // 2:] = True
    out = {"bpb": float(bt[keep].mean()), "rate": float((ev_mask > 0).mean())}
    for k, role in ANSWER_ROLES.items():
        m = keep & (R == RI[role])
        out[f"{k}_bits"] = float(bt[m].mean()) if m.any() else float("nan")
    return out


def targets(b, roles):
    """(answer type, start, end) of every answer span with a full window of context (mathexp._targets)."""
    out = []
    for k, role in ANSWER_ROLES.items():
        if k == "copy":
            continue
        m = roles == RI[role]
        starts = np.flatnonzero(m & ~np.r_[False, m[:-1]]); ends = np.flatnonzero(m & ~np.r_[m[1:], False]) + 1
        out += [(k, int(s), int(e)) for s, e in zip(starts, ends) if s > CTX]
    return out


@torch.no_grad()
def _layouts(rule, ent, W):
    """Patch flags (n, CTX) for windows W (n, CTX - 1): the rule on each window plus the flag for the next byte."""
    dev = next(ent.parameters()).device
    lp = torch.log_softmax(ent(torch.from_numpy(W.astype(np.int64)).to(dev)).double(), -1)
    h = (-(lp.exp() * lp).sum(-1) / np.log(2)).cpu().numpy()             # h[:, t]: entropy of the byte after t
    H = np.c_[np.zeros(len(W)), h[:, :-1]].astype(np.float32)
    bd = np.zeros((len(W), W.shape[1] + 1), np.int64)
    for r in range(len(W)):
        bd[r, :-1] = rule.mask(W[r], H[r])
        bd[r, -1] = rule.mask(np.r_[W[r], 0].astype(np.uint8), np.r_[H[r], np.float32(h[r, -1])])[-1]
    bd[:, 0] = 1
    return bd


@torch.no_grad()
def _greedy(model, ent, rule, W, steps, batch=256):
    """Greedy continuation of windows W (n, CTX - 1) for `steps` bytes, boundaries decided online."""
    dev = _dev(model)
    gen = np.zeros((len(W), steps), np.uint8)
    for b in range(0, len(W), batch):
        w = W[b:b + batch].copy()
        for s in range(steps):
            bd = _layouts(rule, ent, w)
            logits = model(torch.from_numpy(w.astype(np.int64)).to(dev), torch.from_numpy(bd).to(dev))
            nxt = logits[:, -1].float().argmax(-1).cpu().numpy().astype(np.uint8)
            gen[b:b + batch, s] = nxt
            w = np.c_[w[:, 1:], nxt]
    return gen


def accuracy(model, ent, rule, b, roles, seed, n_acc=1000):
    rng = np.random.default_rng(seed); res = {}
    by_type = {}
    for k, s, e in targets(b, roles):
        by_type.setdefault(k, []).append((s, e))
    for k, spans in by_type.items():
        pick = [spans[i] for i in rng.permutation(len(spans))[:n_acc]]
        W = np.stack([b[s - (CTX - 1):s] for s, _ in pick])
        gold = [b[s:e].tobytes() for s, e in pick]
        gen = _greedy(model, ent, rule, W, max(len(g) for g in gold))
        hits = sum(gen[i, :len(g)].tobytes() == g for i, g in enumerate(gold))
        res[f"{k}_acc"] = hits / max(len(pick), 1); res[f"{k}_n"] = len(pick)
    return res


def gsm8k_problems(b, roles):
    """(question bytes, gold final answer) for every GSM8K problem in the eval split."""
    out = []
    for seg in b.tobytes().split(b"\n\n"):
        if b"\n#### " in seg:
            q = seg.split(b"\n", 1)[0] + b"\n"
            out.append((q, seg.rsplit(b"#### ", 1)[1].strip()))
    return out


def end_to_end(model, ent, rule, b, roles, n=100, seed=0, max_bytes=320):
    probs = gsm8k_problems(b, roles)
    rng = np.random.default_rng(seed)
    pick = [probs[i] for i in rng.permutation(len(probs))[:n]]
    W = np.stack([np.frombuffer((b"\n" * CTX + q)[-(CTX - 1):], np.uint8) for q, _ in pick])
    gen = _greedy(model, ent, rule, W, max_bytes)
    hits = 0
    for (q, gold), g in zip(pick, gen):
        text = g.tobytes()
        if b"#### " in text:
            ans = text.split(b"#### ", 1)[1].split(b"\n", 1)[0].strip()
            hits += ans == gold
    return {"e2e_acc": hits / max(len(pick), 1), "e2e_n": len(pick)}
