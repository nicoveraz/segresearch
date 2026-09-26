"""FIXED. Do not modify.

The evaluation harness: a small BLT-style patch model whose ONLY varying input is
the boundary mask produced by boundary.py.

Architecture (BLT-lite):
  - a patch starts at every byte where the mask is True (the first byte of a window always starts one)
  - patch embedding = sum of (byte embedding + offset-in-patch embedding)
  - a global causal transformer runs once per patch  -> the mask decides where global compute goes
  - a local decoder predicts each byte, attending only to bytes of its OWN patch, plus the global
    state of the last COMPLETED patch. A byte that starts a patch is predicted with fresh global
    context; bytes inside a patch only see stale global context.
Every experiment uses the same model, data, steps and seed, so differences come from the mask.
"""
import ast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from prepare import BUDGET, CTX, MAIN_BS, MAIN_STEPS, RI, ROLES, V, _ln, adamw, log_softmax_np, make_step

D = 64
ALLOWED_IMPORTS = {"numpy", "math", "collections", "itertools", "functools", "heapq", "typing", "__future__"}
FORBIDDEN_NAMES = {"ord", "chr", "eval", "exec", "open", "compile", "__import__", "globals", "locals",
                   "getattr", "setattr", "vars", "input", "breakpoint"}


# ----------------------------------------------------------------------------- research hygiene
def check_boundary_source(path="boundary.py"):
    """boundary.py must be a generic rule: no string/bytes literals (so no hardcoded delimiters),
    no ord/chr, no file or module access beyond the numeric allow-list, no dunder tricks."""
    tree = ast.parse(open(path).read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                docstrings.add(id(node.body[0].value))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and id(node) not in docstrings:
            problems.append(f"line {node.lineno}: string/bytes literal {node.value!r}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of {m!r} not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: use of {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"line {node.lineno}: dunder attribute {node.attr!r} not allowed")
        elif isinstance(node, ast.JoinedStr):
            problems.append(f"line {node.lineno}: f-strings not allowed")
    if problems:
        raise SystemExit("boundary.py rejected:\n  " + "\n  ".join(problems))


def _top(s, thr):
    """Bytes strictly above thr, plus ties at thr (in order) until the budget is filled."""
    m = s > thr
    need = int(BUDGET * len(s)) - int(m.sum())
    if need > 0:
        m[np.flatnonzero(s == thr)[:need]] = True
    return m


def masks(module, sig_train, sig_eval):
    """Calls boundary.fit / boundary.score and enforces the compute budget.
    Float scores -> top BUDGET fraction (threshold fitted on train; clipped to budget on eval).
    Bool mask    -> must use at most BUDGET (+1% tolerance) on both splits."""
    state = module.fit(sig_train) if hasattr(module, "fit") else None
    s_tr = np.asarray(module.score(sig_train, state)); s_ev = np.asarray(module.score(sig_eval, state))
    for s, sig in ((s_tr, sig_train), (s_ev, sig_eval)):
        if s.shape != sig.bytes.shape:
            raise SystemExit(f"score() must return one value per byte: got {s.shape}, expected {sig.bytes.shape}")
        if s.dtype != bool and not np.all(np.isfinite(s)):
            raise SystemExit("score() returned NaN or inf")
    if s_tr.dtype == bool:
        m_tr, m_ev = s_tr, s_ev
    else:
        m_tr = _top(s_tr, np.quantile(s_tr, 1 - BUDGET))
        m_ev = _top(s_ev, np.quantile(s_tr, 1 - BUDGET))
        if m_ev.mean() > BUDGET:                      # distribution shift: clip to budget on eval
            m_ev = _top(s_ev, np.quantile(s_ev, 1 - BUDGET))
    for name, m in (("train", m_tr), ("eval", m_ev)):
        if m.mean() > BUDGET + 0.01:
            raise SystemExit(f"boundary rate on {name} is {m.mean():.3f}, above budget {BUDGET}")
    return m_tr, m_ev


# ----------------------------------------------------------------------------- BLT-lite model
def _init(key):
    ks = mx.random.split(key, 16); s = 0.02
    n = lambda k, sh: mx.random.normal(sh, key=k) * s
    blk = lambda k: {"ln1": mx.ones(D), "ln2": mx.ones(D), "qkv": n(k[0], (D, 3 * D)),
                     "o": n(k[1], (D, D)), "w1": n(k[2], (D, 4 * D)), "w2": n(k[3], (4 * D, D))}
    return {"emb": n(ks[0], (V, D)), "off": n(ks[1], (CTX, D)), "ppos": n(ks[2], (CTX, D)),
            "g": [blk(ks[3:7]), blk(ks[7:11])], "loc": blk(ks[11:15]),
            "gproj": n(ks[15], (D, D)), "lnf": mx.ones(D)}


def _block(L, h, mask, nh=4):
    B, T, _ = h.shape; hd = D // nh
    q, k, v = mx.split(_ln(h, L["ln1"]) @ L["qkv"], 3, axis=-1)
    sh = lambda t: t.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
    a = mx.where(mask[:, None], sh(q) @ sh(k).transpose(0, 1, 3, 2) / np.sqrt(hd), -1e9)
    h = h + (mx.softmax(a, axis=-1) @ sh(v)).transpose(0, 2, 1, 3).reshape(B, T, D) @ L["o"]
    return h + nn.gelu_approx(_ln(h, L["ln2"]) @ L["w1"]) @ L["w2"]


def _model(p, x, bd):
    """x: (B,T) bytes; bd: (B,T+1) patch-start flags for bytes x_0..x_T, bd[:,0] == 1.
    Position t predicts x_{t+1}."""
    B, T = x.shape
    ar = mx.arange(T)
    one_hot = lambda i: (i[:, :, None] == ar).astype(mx.float32)   # index -1 -> all-zero row
    pid = mx.cumsum(bd[:, :T], axis=1) - 1
    start = mx.where(bd[:, :T] == 1, ar[None], 0)
    off = ar[None] - mx.cummax(start, axis=1)
    e = p["emb"][x] + p["off"][off]
    pe = one_hot(pid).transpose(0, 2, 1) @ e + p["ppos"][None]
    G = pe
    causal = (ar[:, None] >= ar[None])[None]
    for L in p["g"]:
        G = _block(L, G, causal)
    gidx = mx.where(bd[:, 1:T + 1] == 1, pid, pid - 1)       # fresh context iff x_{t+1} starts a patch
    ctx = one_hot(gidx) @ G
    h = e + ctx @ p["gproj"]
    local = (pid[:, :, None] == pid[:, None, :]) & causal
    h = _block(p["loc"], h, local)
    return _ln(h, p["lnf"]) @ p["emb"].T


def train_eval(train_bytes, train_mask, eval_bytes, eval_mask, eval_roles, seed=0):
    tr = train_bytes.astype(np.int32); btr = train_mask.astype(np.int32)
    lossf = lambda p, x, y, bd: nn.losses.cross_entropy(_model(p, x, bd), y, reduction="mean")
    p, step = make_step(_init(mx.random.key(seed)), lossf, adamw(3e-3, MAIN_STEPS, 3e-4))

    rng = np.random.default_rng(seed)
    for _ in range(MAIN_STEPS):
        i = rng.integers(0, len(tr) - CTX - 1, MAIN_BS)
        x = np.stack([tr[j:j + CTX] for j in i]); y = np.stack([tr[j + 1:j + CTX + 1] for j in i])
        bd = np.stack([btr[j:j + CTX + 1] for j in i]); bd[:, 0] = 1
        step(x, y, bd)

    ev = eval_bytes.astype(np.int32); bev = eval_mask.astype(np.int32)
    n = (len(ev) - 1) // CTX
    X = np.stack([ev[k * CTX:(k + 1) * CTX] for k in range(n)])
    Y = np.stack([ev[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    Rl = np.stack([eval_roles[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    BD = np.stack([bev[k * CTX:k * CTX + CTX + 1] for k in range(n)]); BD[:, 0] = 1
    lp = np.concatenate([log_softmax_np(_model(p, mx.array(X[b:b + 128]), mx.array(BD[b:b + 128])))
                         for b in range(0, n, 128)])
    bits = -np.take_along_axis(lp, Y[..., None], -1)[..., 0] / np.log(2)
    acc = lp.argmax(-1) == Y
    keep = np.zeros_like(bits, bool); keep[:, CTX // 2:] = True   # score bytes with >= CTX/2 context
    out = {"bpb": float(bits[keep].mean()), "boundary_rate": float(eval_mask.mean())}
    ans = keep & np.isin(Rl, [RI["ANS_LOCAL"], RI["ANS_LONG"]])
    out["ans_bits"], out["ans_acc"] = float(bits[ans].mean()), float(acc[ans].mean())
    for r in ROLES:
        m = keep & (Rl == RI[r])
        out[f"{r}_bits"], out[f"{r}_acc"] = float(bits[m].mean()), float(acc[m].mean())
        out[f"{r}_start_rate"] = float(eval_mask[1:][eval_roles[1:] == RI[r]].mean())
    return out
