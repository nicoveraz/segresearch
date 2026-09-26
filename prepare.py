"""FIXED. Do not modify (agent and human alike, once experiments have started).

Synthetic data, scorer models, and cached per-byte signals.

Three corpus formats share the same underlying structure but differ in surface
syntax (delimiters, keywords). A and B are the development formats the agent
optimizes on; C is held out and only used by test_final.py.

Every byte carries a role label (used for evaluation only; never given to boundary.py):
  TEXT       templated clinical phrases                      predictable
  STRUCT     delimiters and keywords                         predictable
  HEX        random 8-char ids, never referenced again       irreducible, useless later
  VAR        variable names in assignments/queries           random choice, needed later
  VALUE      assigned values                                 irreducible, needed later
  OPERAND    operands of local sums                          irreducible, needed immediately
  ANS_LOCAL  answer of a sum whose operands are in the same record   (short-range)
  ANS_LONG   answer of a query over variables assigned 1-6 records earlier (long-range)

Answers are written with digits reversed (units first), which makes addition
learnable left to right.

Usage:  uv run prepare.py          # one-time; trains scorers and caches signals
        SEGR_SMOKE=1 uv run prepare.py   # tiny end-to-end smoke test
"""
import os
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

# ----------------------------------------------------------------------------- constants
SMOKE = os.environ.get("SEGR_SMOKE") == "1"
CACHE = os.path.expanduser(os.environ.get("SEGR_CACHE", "~/.cache/segresearch"))
if SMOKE:
    CACHE += "-smoke"

CTX = 128                 # bytes per training window
BUDGET = 0.25             # max fraction of bytes that may start a patch (= global model steps)
MAIN_STEPS = 60 if SMOKE else 2000
MAIN_BS = 32
N_RECORDS = {"train": 3000, "val": 400, "test": 400} if SMOKE else \
            {"train": 120_000, "val": 3000, "test": 3000}
SEEDS = {"train": 1, "val": 2, "test": 3}
PATCHER = dict(d=32, layers=1, heads=2, lr=3e-3,
               steps=150 if SMOKE else 4000,
               ckpts=(50, 100, 150) if SMOKE else (200, 1000, 4000))
REFERENCE = dict(d=64, layers=3, heads=4, lr=2e-3, steps=100 if SMOKE else 8000)
FORMATS_DEV = ("A", "B")
FORMAT_HIDDEN = "C"
ROLES = ["TEXT", "STRUCT", "HEX", "VAR", "VALUE", "OPERAND", "ANS_LOCAL", "ANS_LONG"]
RI = {r: i for i, r in enumerate(ROLES)}
V = 256

# ----------------------------------------------------------------------------- corpus
PHRASES = ["pt stable", "no chest pain", "bp normal", "afebrile", "alert and oriented",
           "denies nausea", "lungs clear", "abdomen soft"]
VARS = "wxyz"
HEXCH = "0123456789abcdef"
SEP = {"A": "\n", "B": ";", "C": "|"}
TEMPLATES = {
    "A": {"text": [("{phrase}", "TEXT")],
          "hex": [("id:", "STRUCT"), ("{hex}", "HEX")],
          "local": [("{a}", "OPERAND"), ("+", "STRUCT"), ("{b}", "OPERAND"), ("=", "STRUCT"), ("{ans}", "ANS_LOCAL")],
          "assign": [("set ", "STRUCT"), ("{v}", "VAR"), ("=", "STRUCT"), ("{n}", "VALUE")],
          "query": [("{v}", "VAR"), ("+", "STRUCT"), ("{w}", "VAR"), ("=", "STRUCT"), ("{ans}", "ANS_LONG")]},
    "B": {"text": [("{phrase}", "TEXT")],
          "hex": [("ref#", "STRUCT"), ("{hex}", "HEX")],
          "local": [("{a}", "OPERAND"), (" plus ", "STRUCT"), ("{b}", "OPERAND"), (" is ", "STRUCT"), ("{ans}", "ANS_LOCAL")],
          "assign": [("let ", "STRUCT"), ("{v}", "VAR"), (" be ", "STRUCT"), ("{n}", "VALUE")],
          "query": [("{v}", "VAR"), (" plus ", "STRUCT"), ("{w}", "VAR"), (" is ", "STRUCT"), ("{ans}", "ANS_LONG")]},
    "C": {"text": [("{phrase}", "TEXT")],
          "hex": [("uid ", "STRUCT"), ("{hex}", "HEX")],
          "local": [("sum ", "STRUCT"), ("{a}", "OPERAND"), (" ", "STRUCT"), ("{b}", "OPERAND"), (" = ", "STRUCT"), ("{ans}", "ANS_LOCAL")],
          "assign": [("{v}", "VAR"), (" <- ", "STRUCT"), ("{n}", "VALUE")],
          "query": [("sum ", "STRUCT"), ("{v}", "VAR"), (" ", "STRUCT"), ("{w}", "VAR"), (" = ", "STRUCT"), ("{ans}", "ANS_LONG")]},
}
KINDS = ["text", "hex", "local", "assign", "query"]
KIND_P = [0.25, 0.20, 0.15, 0.25, 0.15]
MAX_QUERY_LAG = 6   # a query only uses variables assigned within the last 6 records


def generate(fmt, n_records, seed):
    rng = np.random.default_rng(seed)
    tpl, sep = TEMPLATES[fmt], SEP[fmt]
    b, r, last = [], [], {}
    for i in range(n_records):
        kind = KINDS[rng.choice(len(KINDS), p=KIND_P)]
        recent = sorted(v for v, (_, j) in last.items() if i - j <= MAX_QUERY_LAG)
        if kind == "query" and len(recent) < 2:
            kind = "assign"
        k = {}
        if kind == "text":
            k["phrase"] = PHRASES[rng.integers(len(PHRASES))]
        elif kind == "hex":
            k["hex"] = "".join(rng.choice(list(HEXCH), 8))
        elif kind == "local":
            a, c = (int(x) for x in rng.integers(10, 100, 2))
            k.update(a=str(a), b=str(c), ans=str(a + c)[::-1])
        elif kind == "assign":
            v = VARS[rng.integers(len(VARS))]
            n = int(rng.integers(10, 100))
            last[v] = (n, i)
            k.update(v=v, n=str(n))
        else:
            v, w = rng.choice(recent, 2, replace=False)
            k.update(v=str(v), w=str(w), ans=str(last[v][0] + last[w][0])[::-1])
        for lit, role in tpl[kind]:
            s = k[lit[1:-1]] if lit.startswith("{") and lit.endswith("}") else lit
            b.extend(s.encode()); r.extend([RI[role]] * len(s))
        b.extend(sep.encode()); r.extend([RI["STRUCT"]] * len(sep))
    return np.array(b, np.uint8), np.array(r, np.int8)


# ----------------------------------------------------------------------------- MLX training utilities
class _Params(nn.Module):
    """Wraps a plain parameter tree so MLX's optimizer and compile can update it in place."""
    def __init__(self, tree):
        super().__init__()
        for k, v in tree.items():
            setattr(self, k, v)


def adamw(lr, steps, end_lr):
    """AdamW with linear warmup to lr, then cosine decay to end_lr (optax.warmup_cosine_decay_schedule)."""
    warm = min(100, steps // 4)
    sched = optim.join_schedules([optim.linear_schedule(0.0, lr, warm), optim.cosine_decay(lr, steps - warm, end_lr)],
                                 [warm])
    return optim.AdamW(sched, weight_decay=0.01, bias_correction=True)


def make_step(params, lossf, opt):
    """Compiled training step. lossf(p, *batch) -> scalar loss. Returns (model, step); step(*batch)
    updates model's parameters in place and returns the loss. model is dict-like: model["emb"], ..."""
    model = _Params(params)
    vg = nn.value_and_grad(model, lambda *b: lossf(model, *b))
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def _step(*batch):
        l, g = vg(*batch)
        opt.update(model, g)
        return l

    def step(*batch):
        l = _step(*(mx.array(b) for b in batch))
        mx.eval(l, state)
        return l
    return model, step


def log_softmax_np(lg):
    """float64 log-softmax of an MLX logits array, as numpy."""
    lg = np.array(lg.astype(mx.float32), np.float64)
    lg -= lg.max(-1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(-1, keepdims=True))


# ----------------------------------------------------------------------------- small causal LM
def _ln(x, g):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) * mx.rsqrt(v + 1e-5) * g


def init_lm(key, d, n_layers):
    ks = mx.random.split(key, 2 + 4 * n_layers); s = 0.02
    n = lambda k, sh: mx.random.normal(sh, key=k) * s
    p = {"emb": n(ks[0], (V, d)), "pos": n(ks[1], (CTX, d)), "lnf": mx.ones(d), "layers": []}
    for i in range(n_layers):
        k = ks[2 + 4 * i: 6 + 4 * i]
        p["layers"].append({"ln1": mx.ones(d), "ln2": mx.ones(d),
                            "qkv": n(k[0], (d, 3 * d)), "o": n(k[1], (d, d)),
                            "w1": n(k[2], (d, 4 * d)), "w2": n(k[3], (4 * d, d))})
    return p


def lm_forward(p, x, n_heads):
    T = x.shape[1]; d = p["emb"].shape[1]; hd = d // n_heads
    h = p["emb"][x] + p["pos"][:T]
    ar = mx.arange(T)
    mask = ar[:, None] >= ar[None]
    for L in p["layers"]:
        q, k, v = mx.split(_ln(h, L["ln1"]) @ L["qkv"], 3, axis=-1)
        sh = lambda t: t.reshape(t.shape[0], T, n_heads, hd).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        a = mx.where(mask, q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd), -1e9)
        h = h + (mx.softmax(a, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(h.shape) @ L["o"]
        h = h + nn.gelu_approx(_ln(h, L["ln2"]) @ L["w1"]) @ L["w2"]
    return _ln(h, p["lnf"]) @ p["emb"].T


def train_lm(stream, cfg, ckpts, seed):
    steps, nh = cfg["steps"], cfg["heads"]
    opt = adamw(cfg["lr"], steps, cfg["lr"] * 0.1)
    lossf = lambda p, x, y: nn.losses.cross_entropy(lm_forward(p, x, nh), y, reduction="mean")
    model, step = make_step(init_lm(mx.random.key(seed), cfg["d"], cfg["layers"]), lossf, opt)

    rng = np.random.default_rng(seed); out = {}
    s32 = stream.astype(np.int32)
    for s in range(1, steps + 1):
        i = rng.integers(0, len(s32) - CTX - 1, 32)
        l = step(np.stack([s32[j:j + CTX] for j in i]), np.stack([s32[j + 1:j + CTX + 1] for j in i]))
        if s in ckpts:
            out[s] = tree_map(lambda a: a, model.parameters())   # MLX arrays are immutable: a snapshot
            print(f"    step {s:6d} loss {l.item():.3f}", flush=True)
    return out


def score_stream(p, n_heads, stream):
    """Per-byte entropy and surprisal (bits) of predicting byte i from bytes < i.
    Every byte i >= CTX//2 + 1 is scored with at least CTX//2 bytes of context; earlier bytes are 0."""
    H = np.zeros(len(stream), np.float32); S = np.zeros(len(stream), np.float32)
    s32 = stream.astype(np.int32); half = CTX // 2
    starts = np.arange(0, len(s32) - CTX, half)
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]
        X = np.stack([s32[s:s + CTX] for s in st])
        lp = log_softmax_np(lm_forward(p, mx.array(X), n_heads))
        h = -(np.exp(lp) * lp).sum(-1) / np.log(2)
        for s, hr, lr in zip(st, h, lp):
            tgt = s32[s + half + 1: s + CTX + 1]
            H[s + half + 1: s + CTX + 1] = hr[half:]
            S[s + half + 1: s + CTX + 1] = -lr[half:][np.arange(len(tgt)), tgt] / np.log(2)
    return H, S


# ----------------------------------------------------------------------------- public API
@dataclass(frozen=True)
class Signals:
    """Everything boundary.py is allowed to see for one split of one format.

    bytes              uint8 array, the raw byte stream
    patcher_entropy    {train_step: float32 array}; entropy (bits) of the SMALL model's
                       prediction for byte i given bytes < i, at several training checkpoints
    patcher_surprisal  float32 array; -log2 p(actual byte i) under the final small model
    reference_entropy  float32 array; entropy under the STRONG reference model
    reference_surprisal float32 array; surprisal under the strong reference model
    The first CTX//2 + 1 entries of every signal are 0 (not enough context to score).
    """
    bytes: np.ndarray
    patcher_entropy: dict
    patcher_surprisal: np.ndarray
    reference_entropy: np.ndarray
    reference_surprisal: np.ndarray


def _path(fmt):
    return os.path.join(CACHE, f"format_{fmt}.npz")


def load(fmt, split):
    """Returns (Signals, roles). roles is for the harness only."""
    z = np.load(_path(fmt))
    steps = sorted(int(k.split("_")[-1]) for k in z.files if k.startswith(f"{split}_Hp_"))
    sig = Signals(bytes=z[f"{split}_bytes"],
                  patcher_entropy={s: z[f"{split}_Hp_{s}"] for s in steps},
                  patcher_surprisal=z[f"{split}_Sp"],
                  reference_entropy=z[f"{split}_Hr"],
                  reference_surprisal=z[f"{split}_Sr"])
    return sig, z[f"{split}_roles"]


def main():
    os.makedirs(CACHE, exist_ok=True)
    for fmt in FORMATS_DEV + (FORMAT_HIDDEN,):
        t0 = time.time()
        if os.path.exists(_path(fmt)):
            print(f"format {fmt}: cached at {_path(fmt)}"); continue
        data = {sp: generate(fmt, N_RECORDS[sp], SEEDS[sp]) for sp in N_RECORDS}
        print(f"format {fmt}: train {len(data['train'][0]):,} bytes. Example:\n"
              + data["val"][0][:160].tobytes().decode(), flush=True)
        print("  training patcher (small model)", flush=True)
        pat = train_lm(data["train"][0], PATCHER, set(PATCHER["ckpts"]), seed=0)
        print("  training reference (strong model)", flush=True)
        ref = train_lm(data["train"][0], REFERENCE, {REFERENCE["steps"]}, seed=1)[REFERENCE["steps"]]
        out = {}
        for sp, (bts, roles) in data.items():
            out[f"{sp}_bytes"], out[f"{sp}_roles"] = bts, roles
            for s, p in pat.items():
                H, S = score_stream(p, PATCHER["heads"], bts)
                out[f"{sp}_Hp_{s}"] = H
                if s == max(pat):
                    out[f"{sp}_Sp"] = S
            out[f"{sp}_Hr"], out[f"{sp}_Sr"] = score_stream(ref, REFERENCE["heads"], bts)
        np.savez_compressed(_path(fmt), **out)
        # sanity check: the reference must actually learn the answers, or "excess" signals are meaningless
        r, m = out["val_roles"], np.arange(len(out["val_roles"])) > CTX
        print("  val entropy by role (bits)   patcher-final / reference")
        for name in ROLES:
            sel = m & (r == RI[name])
            print(f"    {name:10s} {out['val_Hp_' + str(max(pat))][sel].mean():6.2f} / {out['val_Hr'][sel].mean():6.2f}")
        if out["val_Hr"][m & (r == RI["ANS_LONG"])].mean() > 0.5:
            print("  WARNING: reference has not learned ANS_LONG (> 0.5 bits). Increase REFERENCE['steps'].")
        print(f"  done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
