"""Specialized-math experiment: does patch placement change answer ACCURACY, not just loss?

Data: GSM8K (worked solutions with <<expr=R>>R calculator annotations and a final '#### N') plus the
MATH competition set (LaTeX solutions whose final answer sits in \\boxed{...}).

Answer spans (for evaluation only; rules never see them):
    GSM8K computed result R inside <<expr=R>>          role ANS_LOCAL
    GSM8K copy of R after >>                           role VALUE
    GSM8K final answer after ####                      role ANS_LONG
    MATH boxed final answer, the content of \\boxed{}   role VAR (reused as a free role name)

Patch rules, in two compute groups:
    25%:  entropy        BLT: top 25% of the small model's next-byte entropy
          jump           top 25% of rises in that entropy (label-free)
          syntax+jump    math-syntax starts plus the largest entropy rises, to 25%
    ~equal to words+syntax:
          words          a patch at every word start
          words+jump     word starts plus the largest non-word entropy rises (label-free)
          words+syntax   word starts plus math-syntax starts
          entropyR / jumpR / syntax+entropyR   tight budget (R% of bytes, e.g. entropy10): entropy, entropy rises, or math
                                syntax with the rest of the budget filled by entropy
          ndepR                 tight budget, label-free: a small network predicting the value of a patch start from
                                the last K bytes (neural_patcher.py), top R of bytes
          novR / entnovR / final+novR   tight budget, label-free, self-supervised (#31): expected novelty per
                                preceding 2-byte context (how often the next bytes do NOT repeat the last 128 bytes;
                                reuse_screen.py fit), alone, plus entropy (standardized), or with a patch forced after
                                the final-answer markers '#### ' and '\\boxed{' and the rest of the budget by novelty
          entdepR / depR        tight budget, label-free: entropy + boundary dependence (standardized), or dependence
                                alone, to R% (table fitted by deptrigger.py from the model's own losses)
          syntax / stride6+syntax   math syntax alone / plus a patch every 6 bytes (syntax without word alignment)
          words+syntax+digits   plus a fresh patch after each digit of a number that began right after '='
          words+syntax+ops      plus a patch after each operator (+ - * /) and after '<<' (operand-aligned)
          words+syntax+rand07/22  matched-compute controls: as many extra boundaries as +digits / +ops add,
                                at pseudo-random positions (a hash of the previous three bytes)
    Scratchpad Patching (run with SEGR_SCRATCH=1 SEGR_LOCAL=window): patches every 8 bytes plus 6% scratchpads
          sp:none / sp:dense5   no scratchpads (patches every 8 / every 5 bytes)
          sp:entropy            scratchpads where next-byte entropy is high (the paper's trigger)
          sp:jump / sp:syntax / sp:random   scratchpads at entropy rises / math syntax + rises / random positions
          sp16:learned          scratchpads where a learned predictor expects the largest loss drop (gaintrigger.py;
                                trained on the model's own loss, no answer labels)
          sp16:*                same with patches every 16 bytes (the paper's headline setting); sp16:dense8 =
                                patches every 8 bytes, no scratchpads (matched compute)
    Math syntax = the byte right after '=', '\\boxed{', '#### ' or '>>': a hand-written rule that a
    math-specialized model may legitimately use. No rule uses the answer labels.

Metrics: bits per byte on each answer type and overall, and exact-match ACCURACY: given the true text up
to an answer, does greedy decoding produce the exact answer? Patch boundaries during decoding are
decided online by the same rule, from the bytes and small-model entropies seen so far.

    uv run mathexp.py prepare                                     # download MATH, build the corpus, train the small model
    SEGR_POOL=xattn SEGR_LOCAL=window uv run mathexp.py run RULE SEED STEPS
    SEGR_D=128 SEGR_GLAYERS=4 ...                                  # a larger model
    SEGR_MATH_SMOKE=1 ...                                          # tiny corpus and model run, to check the pipeline
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import prepare
import realtext
import registry
from prepare import BUDGET, CTX, MAIN_BS, PATCHER, RI, adamw, log_softmax_np, make_step

SMOKE = os.environ.get("SEGR_MATH_SMOKE") == "1"
CACHE = os.path.expanduser(os.environ.get("SEGR_CACHE", "~/.cache/segresearch")) + ("-math-smoke" if SMOKE else "-math")
NPZ = os.path.join(CACHE, "math.npz")
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
N_VAL_GSM, N_VAL_MATH = (60, 60) if SMOKE else (660, 700)    # validation problems (from each test set)
N_ACC = int(os.environ.get("SEGR_MATH_NACC", 20 if SMOKE else 300))   # accuracy targets per answer type
ANSWER_ROLES = {"computed": "ANS_LOCAL", "copy": "VALUE", "final": "ANS_LONG", "boxed": "VAR"}
SYNTAX = [b"=", b"\\boxed{", b"#### ", b">>"]
FINAL = [b"#### ", b"\\boxed{"]


# ----------------------------------------------------------------------------- data
def _math_rows(config, split):
    path = os.path.join(CACHE, f"math_{config}_{split}.jsonl")
    if not os.path.exists(path):
        rows, off = [], 0
        while True:
            q = urllib.parse.urlencode({"dataset": "EleutherAI/hendrycks_math", "config": config,
                                        "split": split, "offset": off, "length": 100})
            for attempt in range(8):                                  # the API rate-limits (HTTP 429): back off
                try:
                    page = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q, timeout=60))
                    break
                except urllib.error.HTTPError as err:
                    if err.code != 429 or attempt == 7:
                        raise
                    time.sleep(10 * 2 ** attempt)
            rows += [r["row"] for r in page["rows"]]
            off += 100
            if off >= page["num_rows_total"]:
                break
            time.sleep(1.5)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return [json.loads(l) for l in open(path)]


def _encode_math(problems):
    """problem + solution; the content of the last \\boxed{...} in each solution is the answer (role VAR)."""
    b, r = bytearray(), []
    def add(s, role):
        e = s.encode(); b.extend(e); r.extend([RI[role]] * len(e))
    for p in problems:
        add(p["problem"] + "\n", "TEXT")
        sol = p["solution"]; k = sol.rfind("\\boxed{")
        if k < 0:
            add(sol, "TEXT")
        else:
            depth, j = 1, k + 7
            while j < len(sol) and depth:
                depth += {"{": 1, "}": -1}.get(sol[j], 0); j += 1
            add(sol[:k + 7], "TEXT"); add(sol[k + 7:j - 1], "VAR"); add(sol[j - 1:], "TEXT")
        add("\n\n", "STRUCT")
    return np.frombuffer(bytes(b), np.uint8).copy(), np.array(r, np.int8)


def _interleave(parts, seed):
    """Concatenate problem-level chunks from both datasets in a fixed random order (split on blank lines)."""
    chunks = []
    for bts, roles in parts:
        cut = np.flatnonzero((bts[:-1] == 10) & (bts[1:] == 10) & (roles[1:] == RI["STRUCT"])) + 2
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(bts)]):
            if e > s:
                chunks.append((bts[s:e], roles[s:e]))
    order = np.random.default_rng(seed).permutation(len(chunks))
    return np.concatenate([chunks[i][0] for i in order]), np.concatenate([chunks[i][1] for i in order])


def prepare_cache():
    os.makedirs(CACHE, exist_ok=True)
    gsm_tr, gsm_te = realtext._load("train"), realtext._load("test")
    math_tr = [r for c in MATH_CONFIGS for r in _math_rows(c, "train")]
    math_te = [r for c in MATH_CONFIGS for r in _math_rows(c, "test")]
    rng = np.random.default_rng(0)
    math_te = [math_te[i] for i in rng.permutation(len(math_te))[:N_VAL_MATH]]
    if SMOKE:
        gsm_tr, math_tr = gsm_tr[:600], math_tr[:600]
    data = {"train": _interleave([realtext._encode(gsm_tr), _encode_math(math_tr)], 1),
            "val": _interleave([realtext._encode(gsm_te[:N_VAL_GSM]), _encode_math(math_te)], 2)}
    for sp, (bts, roles) in data.items():
        print(f"{sp}: {len(bts):,} bytes; " + ", ".join(f"{k} {np.mean(roles == RI[v]):.3f}" for k, v in ANSWER_ROLES.items()))
    cfg = dict(PATCHER, steps=150) if SMOKE else PATCHER
    print("training the small entropy model", flush=True)
    pat = prepare.train_lm(data["train"][0], cfg, {cfg["steps"]}, seed=0)[cfg["steps"]]
    out = {f"pat_{'/'.join(map(str, k)) if isinstance(k, tuple) else k}": v for k, v in _flatten(pat)}
    for sp, (bts, roles) in data.items():
        H, _ = prepare.score_stream(pat, cfg["heads"], bts)
        out.update({f"{sp}_bytes": bts, f"{sp}_roles": roles, f"{sp}_H": H})
    np.savez_compressed(NPZ, **out)
    print("cached", NPZ)


def _flatten(tree, prefix=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, prefix + (k,))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flatten(v, prefix + (i,))
    else:
        yield prefix, np.asarray(tree)


def _patcher(z):
    """Rebuild the small entropy model from the cache (needed to decide boundaries online during decoding)."""
    p = {"layers": [{} for _ in range(PATCHER["layers"])]}
    for k in z.files:
        if k.startswith("pat_"):
            path = k[4:].split("/")
            if path[0] == "layers":
                p["layers"][int(path[1])][path[2]] = mx.array(z[k])
            else:
                p[path[0]] = mx.array(z[k])
    return p


# ----------------------------------------------------------------------------- rules
def _syntax_starts(b, markers=None):
    """True at the byte right after any math-syntax marker."""
    m = np.zeros(len(b), bool); s = b.tobytes()
    for tok in markers or SYNTAX:
        i = s.find(tok)
        while i >= 0:
            if i + len(tok) < len(b):
                m[i + len(tok)] = True
            i = s.find(tok, i + 1)
    return m


def _after_result_digits(b):
    """True at t when byte t-1 is a digit in a number that began right after '=' (a fresh patch per result digit)."""
    digit = (b >= 48) & (b <= 57)
    m = np.zeros(len(b), bool); run_after_eq = False
    for i in range(len(b) - 1):
        if digit[i]:
            if i == 0 or not digit[i - 1]:
                run_after_eq = i > 0 and b[i - 1] == 61                # '='
            m[i + 1] = run_after_eq
    return m


def _after_operators(b):
    """True at t when byte t-1 is an arithmetic operator or ends '<<' (operand-aligned patches)."""
    prev = np.r_[0, b[:-1]]
    return np.isin(prev, [43, 45, 42, 47]) | np.r_[False, False, (b[:-2] == 60) & (b[1:-1] == 60)]


def _hash_prev(b):
    """A pseudo-random value in [0, 1) at each t from bytes t-3..t-1: deterministic, causal, uninformative."""
    x = np.r_[0, 0, 0, b.astype(np.uint64)]
    h = (x[2:-1] * np.uint64(961) + x[1:-2] * np.uint64(31) + x[:-3]) * np.uint64(2654435761) % np.uint64(2 ** 32)
    return h.astype(np.float64) / 2 ** 32


def _rand_extra(b, q):
    return _hash_prev(b) < q


SP_RATE = 0.06        # scratchpads, as a share of all bytes


class Rule:
    """A patch rule with thresholds fitted on the training split, applicable to any byte window + entropies."""
    def __init__(self, name, z):
        self.name = name
        btr, Htr = z["train_bytes"], z["train_H"]
        Jtr = np.diff(Htr, prepend=Htr[0]); wtr = realtext._word_starts(btr); str_ = _syntax_starts(btr)
        target_words = (wtr | str_).mean()                           # the words+syntax budget, shared by the word group
        self.kind = None
        if name.startswith(("sp:", "sp16:")):
            # Scratchpad Patching: fixed patches every K bytes, plus SP_RATE scratchpads chosen by a trigger
            prefix, trig = name.split(":")
            self.K = {"dense5": 5, "dense8": 8}.get(trig, 16 if prefix == "sp16" else 8)
            name = "sp:" + trig
            free = np.arange(len(btr)) % self.K != 0
            if name == "sp:entropy":
                self.thr = np.quantile(Htr[free], 1 - SP_RATE / free.mean())
            elif name == "sp:jump":
                self.thr = np.quantile(Jtr[free], 1 - SP_RATE / free.mean())
            elif name == "sp:syntax":
                syn = str_ & free; rest = free & ~syn
                self.thr = np.quantile(Jtr[rest], 1 - max(SP_RATE - syn.mean(), 0) / rest.mean())
            elif name == "sp:learned":
                import gaintrigger
                self.w = np.load(os.path.join(CACHE, "gaintrigger.npz"))["w"]
                self.predict = lambda b, H: gaintrigger.features(b, H) @ self.w if len(b) else np.zeros(0)
                g = self.predict(btr[:2_000_000], Htr[:2_000_000]); fr = free[:2_000_000]
                self.thr = np.quantile(g[fr], 1 - SP_RATE / fr.mean())
            elif name == "sp:random":
                hv = _hash_prev(btr); qs = np.linspace(0, 0.2, 401)
                self.q = qs[np.argmin([abs((free & (hv < q)).mean() - SP_RATE) for q in qs])]
            elif name not in ("sp:none", "sp:dense5", "sp:dense8"):
                raise SystemExit(f"unknown rule {name}")
            self.name = name
            return
        m = re.fullmatch(r"(entropy|jump|syntax\+entropy|dep|entdep|ndep|nov|entnov|final\+nov)(\d+)", name)
        if m:
            # tight budget: R% of bytes start a patch (entropy10, dep15, syntax+entropy20, ...)
            self.kind, R = m.group(1), int(m.group(2)) / 100
            if self.kind in ("dep", "entdep"):
                import deptrigger
                t = np.load(os.path.join(CACHE, "deptrigger.npz"))
                self.dep = lambda b: deptrigger.table_lookup(t["T2"], t["T1"], float(t["glob"]), b) if len(b) else np.zeros(0, np.float32)
                Dtr = self.dep(btr)
                self.mu = (float(Htr.mean()), float(Htr.std()), float(Dtr.mean()), float(Dtr.std()))
                self.thr = np.quantile(self.score(btr, Htr) if self.kind == "entdep" else Dtr, 1 - R)
            elif self.kind in ("nov", "entnov", "final+nov"):            # self-supervised novelty table (#31)
                import deptrigger
                t = np.load(os.path.join(CACHE, "novtab.npz"))
                self.dep = lambda b: deptrigger.table_lookup(t["T2"], t["T1"], float(t["glob"]), b) if len(b) else np.zeros(0, np.float32)
                Ntr = self.dep(btr)
                self.mu = (float(Htr.mean()), float(Htr.std()), float(Ntr.mean()), float(Ntr.std()))
                if self.kind == "final+nov":
                    fin = _syntax_starts(btr, FINAL); rest = ~fin
                    self.thr = np.quantile(Ntr[rest], 1 - (R - fin.mean()) / rest.mean())
                else:
                    self.thr = np.quantile(self.score(btr, Htr) if self.kind == "entnov" else Ntr, 1 - R)
            elif self.kind == "ndep":                                    # learned patcher (neural_patcher.py)
                import neural_patcher
                net = neural_patcher.load(os.environ.get("SEGR_NPATCH"))
                self.ndep = lambda b: neural_patcher.predict(net, b)
                self.thr = np.quantile(self.ndep(btr), 1 - R)
            elif self.kind == "entropy":
                self.thr = np.quantile(Htr, 1 - R)
            elif self.kind == "jump":
                self.thr = np.quantile(Jtr, 1 - R)
            else:
                rest = ~str_; self.thr = np.quantile(Htr[rest], 1 - (R - str_.mean()) / rest.mean())
            return
        self.kind = None
        if name == "entropy":
            self.thr = np.quantile(Htr, 1 - BUDGET)
        elif name == "jump":
            self.thr = np.quantile(Jtr, 1 - BUDGET)
        elif name == "syntax+jump":
            extra = BUDGET - str_.mean()
            self.thr = np.quantile(Jtr[~str_], 1 - extra / (~str_).mean())
        elif name == "words+jump":
            extra = target_words - wtr.mean()
            self.thr = np.quantile(Jtr[~wtr], 1 - extra / (~wtr).mean())
        elif name in ("words+syntax+rand07", "words+syntax+rand22"):
            # matched-compute controls: extra boundaries at pseudo-random positions, as many as +digits / +ops add
            base = wtr | str_; extra = {"words+syntax+rand07": 0.0074, "words+syntax+rand22": 0.0228}[name]
            hv = _hash_prev(btr); qs = np.linspace(0, 0.2, 401)
            self.q = qs[np.argmin([abs((base | (hv < q)).mean() - base.mean() - extra) for q in qs])]
        elif name not in ("words", "syntax", "stride6+syntax", "words+syntax", "words+syntax+digits", "words+syntax+ops"):
            raise SystemExit(f"unknown rule {name}")

    def score(self, b, H):
        mh, sh, md, sd = self.mu
        return (H - mh) / sh + (self.dep(b) - md) / sd

    def mask(self, b, H):
        J = np.diff(H, prepend=H[0]) if len(H) else H
        if self.kind is not None:
            return {"entdep": lambda: self.score(b, H) > self.thr, "dep": lambda: self.dep(b) > self.thr,
                    "entnov": lambda: self.score(b, H) > self.thr, "nov": lambda: self.dep(b) > self.thr,
                    "final+nov": lambda: _syntax_starts(b, FINAL) | (self.dep(b) > self.thr),
                    "ndep": lambda: self.ndep(b) > self.thr,
                    "entropy": lambda: H > self.thr, "jump": lambda: J > self.thr,
                    "syntax+entropy": lambda: _syntax_starts(b) | (H > self.thr)}[self.kind]()
        if self.name.startswith("sp:"):
            stride = np.arange(len(b)) % self.K == 0
            trig = {"sp:entropy": lambda: H > self.thr, "sp:jump": lambda: J > self.thr,
                    "sp:syntax": lambda: _syntax_starts(b) | (J > self.thr),
                    "sp:random": lambda: _hash_prev(b) < self.q,
                    "sp:learned": lambda: self.predict(b, H) > self.thr}.get(self.name, lambda: np.zeros(len(b), bool))()
            flags = stride.astype(np.int8); flags[trig & ~stride] = 2      # 1 = patch start, 2 = scratchpad
            return flags
        if self.name == "entropy":
            return H > self.thr
        if self.name == "jump":
            return J > self.thr
        if self.name == "syntax+jump":
            return _syntax_starts(b) | (J > self.thr)
        words = realtext._word_starts(b) if len(b) else np.zeros(0, bool)
        if self.name == "words":
            return words
        if self.name == "words+syntax":
            return words | _syntax_starts(b)
        if self.name == "syntax":
            return _syntax_starts(b)
        if self.name == "stride6+syntax":                              # math syntax without word alignment
            return (np.arange(len(b)) % 6 == 0) | _syntax_starts(b)
        if self.name == "words+syntax+digits":
            return words | _syntax_starts(b) | _after_result_digits(b)
        if self.name == "words+syntax+ops":
            return words | _syntax_starts(b) | _after_operators(b)
        if self.name in ("words+syntax+rand07", "words+syntax+rand22"):
            return words | _syntax_starts(b) | _rand_extra(b, self.q)
        return words | (J > self.thr)                                  # words+jump


# ----------------------------------------------------------------------------- train / evaluate
def _train(tr_bytes, tr_mask, steps, seed):
    import harness
    tr, btr = tr_bytes.astype(np.int32), tr_mask.astype(np.int32)
    lossf = lambda p, x, y, bd: nn.losses.cross_entropy(harness._model(p, x, bd), y, reduction="mean")
    p, step = make_step(harness._init(mx.random.key(seed)), lossf, adamw(3e-3, steps, 3e-4))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = rng.integers(0, len(tr) - CTX - 1, MAIN_BS)
        bd = np.stack([btr[j:j + CTX + 1] for j in i]); bd[:, 0] = 1
        step(np.stack([tr[j:j + CTX] for j in i]), np.stack([tr[j + 1:j + CTX + 1] for j in i]), bd)
    return p


def _bits(p, ev_bytes, ev_mask, ev_roles):
    import harness
    ev, bev = ev_bytes.astype(np.int32), ev_mask.astype(np.int32)
    n = (len(ev) - 1) // CTX
    X = np.stack([ev[k * CTX:(k + 1) * CTX] for k in range(n)]); Y = np.stack([ev[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    R = np.stack([ev_roles[k * CTX + 1:(k + 1) * CTX + 1] for k in range(n)])
    BD = np.stack([bev[k * CTX:k * CTX + CTX + 1] for k in range(n)]); BD[:, 0] = 1
    lp = np.concatenate([log_softmax_np(harness._model(p, mx.array(X[b:b + 128]), mx.array(BD[b:b + 128])))
                         for b in range(0, n, 128)])
    bits = -np.take_along_axis(lp, Y[..., None], -1)[..., 0] / np.log(2)
    keep = np.zeros_like(bits, bool); keep[:, CTX // 2:] = True
    out = {"bpb": float(bits[keep].mean()), "rate": float((ev_mask > 0).mean()),
           "patch_rate": float((ev_mask == 1).mean()), "scratch_rate": float((ev_mask == 2).mean())}
    for k, role in ANSWER_ROLES.items():
        m = keep & (R == RI[role])
        out[f"{k}_bits"] = float(bits[m].mean()) if m.any() else float("nan")
    return out


def _entropy_window(pat, w):
    """Small-model entropy of each byte of window w given the bytes before it (0 for the first byte)."""
    lg = log_softmax_np(prepare.lm_forward(pat, mx.array(w[None].astype(np.int32)), PATCHER["heads"]))[0]
    h = -(np.exp(lg) * lg).sum(-1) / np.log(2)                        # h[t]: entropy of the byte after t
    return np.r_[0.0, h[:-1]].astype(np.float32), float(h[-1])


def _targets(b, roles):
    """(answer type, start, end) of every answer span; prefix = b[:start]."""
    out = []
    for k, role in ANSWER_ROLES.items():
        if k == "copy":
            continue
        m = roles == RI[role]
        starts = np.flatnonzero(m & ~np.r_[False, m[:-1]]); ends = np.flatnonzero(m & ~np.r_[m[1:], False]) + 1
        out += [(k, int(s), int(e)) for s, e in zip(starts, ends) if s > CTX]
    return out


def _accuracy(p, pat, rule, b, roles, seed):
    """Exact match of greedy decoding for each answer type, with boundaries decided online by the rule."""
    import harness
    rng = np.random.default_rng(seed); res = {}
    by_type = {}
    for k, s, e in _targets(b, roles):
        by_type.setdefault(k, []).append((s, e))
    for k, spans in by_type.items():
        pick = [spans[i] for i in rng.permutation(len(spans))[:N_ACC]]
        hits = 0
        for s, e in pick:
            gold = b[s:e].tobytes(); ctx = list(b[:s][-(CTX - 1):]); gen = []
            for _ in range(len(gold)):
                w = np.array(ctx, np.uint8)
                H, h_next = _entropy_window(pat, w)
                bd = np.r_[rule.mask(w, H), rule.mask(np.r_[w, 0].astype(np.uint8), np.r_[H, h_next])[-1]].astype(np.int32)
                bd[0] = 1
                logits = harness._model(p, mx.array(w[None].astype(np.int32)), mx.array(bd[None]))
                nxt = int(np.array(logits[0, -1]).argmax())
                gen.append(nxt); ctx = (ctx + [nxt])[-(CTX - 1):]
            hits += bytes(gen) == gold
        res[f"{k}_acc"] = hits / max(len(pick), 1); res[f"{k}_n"] = len(pick)
    return res


def run(name, seed, steps):
    import harness
    z = np.load(NPZ); rule = Rule(name, z); pat = _patcher(z)
    m_tr = rule.mask(z["train_bytes"], z["train_H"]); m_va = rule.mask(z["val_bytes"], z["val_H"])
    t0 = time.time()
    p = _train(z["train_bytes"], m_tr, steps, seed)
    o = _bits(p, z["val_bytes"], m_va, z["val_roles"])
    o.update(_accuracy(p, pat, rule, z["val_bytes"], z["val_roles"], seed))
    registry.emit("reasonexp" if "-reason" in CACHE else "mathexp", f"RESULT math rule={name} seed={seed} steps={steps} D={harness.D} glayers={harness.GLAYERS} "
          f"pool={harness.POOL} local={harness.LOCAL} rate={o['rate']:.3f} (patches {o['patch_rate']:.3f}, scratchpads {o['scratch_rate']:.3f}) bpb={o['bpb']:.4f} | "
          + " ".join(f"{k}: {o[k + '_bits']:.3f} bits, acc {o.get(k + '_acc', float('nan')):.3f} (n={o.get(k + '_n', 0)})"
                     for k in ANSWER_ROLES) + f" | {time.time() - t0:.0f}s", experiment=os.environ.get("SEGR_EXPERIMENT"), corpus="reason" if "-reason" in CACHE else "math", window=harness.WINDOW, scratch=harness.SCRATCH, nacc=N_ACC, smoke=SMOKE)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare_cache()
    elif sys.argv[1] == "rates":
        z = np.load(NPZ)
        for n in ("entropy", "jump", "syntax+jump", "words", "words+jump", "words+syntax", "words+syntax+digits", "words+syntax+ops", "words+syntax+rand07", "words+syntax+rand22", "entropy10", "jump10", "syntax+entropy10", "entdep10", "dep10",
                  "sp:none", "sp:dense5", "sp:entropy", "sp:jump", "sp:syntax", "sp:random",
                  "sp16:none", "sp16:dense8", "sp16:entropy", "sp16:jump", "sp16:syntax", "sp16:random", "sp16:learned"):
            r = Rule(n, z); m = r.mask(z["val_bytes"], z["val_H"]); a = realtext._answer_starts(z["val_roles"]) | \
                ((z["val_roles"] == RI["VAR"]) & ~np.r_[False, z["val_roles"][:-1] == RI["VAR"]])
            print(f"{n:13s} train {(r.mask(z['train_bytes'], z['train_H']) > 0).mean():.3f} val {(m > 0).mean():.3f} "
                  f"(scratchpads {(m == 2).mean():.3f}) | answer starts covered {(m[a] > 0).mean():.2f}, by a scratchpad {(m[a] == 2).mean():.2f}")
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
