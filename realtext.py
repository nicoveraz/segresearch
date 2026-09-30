"""Real-text check: does a patch start at the beginning of each answer beat BLT's entropy rule on real text?

Data: GSM8K (grade-school math word problems with worked solutions). Solutions carry calculator
annotations and a final answer, which give natural "answer" spans that depend on earlier text:

    Natalia sold 48/2 = <<48/2=24>>24 clips in May. ... #### 72
                          expr  R   copy                    final

Roles (reusing the harness's role names, for evaluation only):
    OPERAND    the expression inside <<...=R>>
    ANS_LOCAL  R, the computed result (arithmetic on the expression just before it)
    VALUE      the copy of R right after >>
    ANS_LONG   the final answer after #### (depends on the whole solution)
    STRUCT     << = >> ####
    TEXT       everything else

Masks compared at the same budget (<= 25% of bytes start a patch):
    entropy          BLT: top 25% of the small model's next-byte entropy
    entropy+ans      top (25% - a) of entropy, plus a patch start at the first byte of every answer span
    words            a patch start at the first byte of every word (after a space or newline)
    words+ans        word starts plus answer starts
    jump25           label-free: top 25% of rises in the small model's entropy (H2's signal)
    words+jump20/25  label-free: word starts plus the largest non-word entropy rises, to 20% / 25% of bytes
Answer starts come from the annotation syntax (a hand-written structural heuristic, not a learned rule):
the question is whether the effect exists on real text before building a detector for it.

    uv run realtext.py prepare                   # download GSM8K, train the small entropy model, cache signals
    SEGR_POOL=xattn SEGR_LOCAL=window uv run realtext.py run MASK SEED STEPS
    SEGR_REALDATA=code ...           # Python standard-library source instead of GSM8K; "answers" are
                                     # identifiers repeating a name seen <= 120 bytes earlier in the file
"""
import json
import os
import sys
import urllib.request

import numpy as np

import prepare
from prepare import BUDGET, PATCHER, RI

DATA = os.environ.get("SEGR_REALDATA", "gsm8k")      # gsm8k | code
assert DATA in ("gsm8k", "code")
CACHE = os.path.expanduser(os.environ.get("SEGR_CACHE", "~/.cache/segresearch")) + f"-{DATA}"
NPZ = os.path.join(CACHE, f"{DATA}.npz")
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{}.jsonl"
N_VAL = 660          # the first 660 GSM8K test problems are validation; the rest stay unused (held out)
CODE_TRAIN_BYTES, CODE_VAL_BYTES = 4_000_000, 350_000


def _load(split):
    path = os.path.join(CACHE, f"{split}.jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(split), path)
    return [json.loads(line) for line in open(path)]


def _encode(problems):
    """Byte stream + roles. Each problem is 'question\\nanswer\\n\\n'."""
    b, r = bytearray(), []
    def add(s, role):
        e = s.encode(); b.extend(e); r.extend([RI[role]] * len(e))
    for p in problems:
        add(p["question"] + "\n", "TEXT")
        text = p["answer"]
        i = 0
        while i < len(text):
            if text.startswith("<<", i) and ">>" in text[i:]:
                j = text.index(">>", i); inner = text[i + 2:j]
                expr, _, res = inner.rpartition("=")
                add("<<", "STRUCT"); add(expr, "OPERAND"); add("=", "STRUCT"); add(res, "ANS_LOCAL"); add(">>", "STRUCT")
                i = j + 2
                if text.startswith(res, i) and res:                  # the copy of R right after >>
                    add(res, "VALUE"); i += len(res)
            elif text.startswith("#### ", i):
                add("#### ", "STRUCT"); add(text[i + 5:], "ANS_LONG"); i = len(text)
            else:
                add(text[i], "TEXT"); i += 1
        add("\n\n", "STRUCT")
    return np.frombuffer(bytes(b), np.uint8).copy(), np.array(r, np.int8)


def _encode_code(paths, limit):
    """Python source as bytes. 'Answers' are identifiers that repeat a name seen earlier in the same file:
    ANS_LOCAL if the previous occurrence started <= 32 bytes before, ANS_LONG if 33-120 bytes before
    (within the model's context). Everything else is TEXT; files are separated by a blank line (STRUCT)."""
    import io, keyword, tokenize
    b, r = bytearray(), []
    for path in paths:
        src = open(path, "rb").read()
        if not src.isascii():
            continue
        text = src.decode()
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            continue
        line_start = np.r_[0, np.cumsum([len(l) for l in text.splitlines(keepends=True)])]
        roles = np.full(len(text), RI["TEXT"], np.int8)
        last = {}
        for t in toks:
            if t.type != tokenize.NAME or keyword.iskeyword(t.string):
                continue
            s = int(line_start[t.start[0] - 1] + t.start[1])
            prev = last.get(t.string)
            if prev is not None and s - prev <= 120:
                roles[s:s + len(t.string)] = RI["ANS_LOCAL" if s - prev <= 32 else "ANS_LONG"]
            last[t.string] = s
        b.extend(src + b"\n\n"); r.extend(roles.tolist() + [RI["STRUCT"]] * 2)
        if len(b) >= limit:
            break
    return np.frombuffer(bytes(b), np.uint8).copy(), np.array(r, np.int8)


def _code_data():
    import glob, hashlib, sysconfig
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(f for f in glob.glob(root + "/**/*.py", recursive=True)
                   if "/test" not in f and "site-packages" not in f and "/idlelib" not in f)
    is_val = lambda f: int(hashlib.md5(os.path.relpath(f, root).encode()).hexdigest(), 16) % 12 == 0   # ~8% of files
    return {"train": _encode_code([f for f in files if not is_val(f)], CODE_TRAIN_BYTES),
            "val": _encode_code([f for f in files if is_val(f)], CODE_VAL_BYTES)}


def prepare_cache():
    os.makedirs(CACHE, exist_ok=True)
    if DATA == "code":
        data = _code_data()
    else:
        train, test = _load("train"), _load("test")
        data = {"train": _encode(train), "val": _encode(test[:N_VAL])}
    for sp, (bts, roles) in data.items():
        print(f"{sp}: {len(bts):,} bytes; answer-span shares: "
              + ", ".join(f"{n} {np.mean(roles == RI[n]):.3f}" for n in ("OPERAND", "ANS_LOCAL", "VALUE", "ANS_LONG")))
    print("training the small entropy model", flush=True)
    pat = prepare.train_lm(data["train"][0], PATCHER, {PATCHER["steps"]}, seed=0)[PATCHER["steps"]]
    out = {}
    for sp, (bts, roles) in data.items():
        H, S = prepare.score_stream(pat, PATCHER["heads"], bts)
        out.update({f"{sp}_bytes": bts, f"{sp}_roles": roles, f"{sp}_H": H})
    np.savez_compressed(NPZ, **out)
    print("cached", NPZ)


def _answer_starts(roles):
    ans = np.isin(roles, [RI["ANS_LOCAL"], RI["VALUE"], RI["ANS_LONG"]])
    return ans & ~np.r_[False, ans[:-1]]


def _word_starts(b):
    return np.r_[True, (b[:-1] == 32) | (b[:-1] == 10)] & (b != 32) & (b != 10)


def _top(score, train_score, rate):
    """score above the train threshold for `rate`; if distribution shift pushes the share above `rate`,
    refit the threshold on this split (the harness does the same for boundary.py rules)."""
    m = score > np.quantile(train_score, 1 - rate)
    return m if m.mean() <= rate else score > np.quantile(score, 1 - rate)


def masks(name, z, split):
    b, roles, H = z[f"{split}_bytes"], z[f"{split}_roles"], z[f"{split}_H"]
    Htr = z["train_H"]
    if name == "entropy":
        return _top(H, Htr, BUDGET)
    starts = _answer_starts(roles)
    if name == "entropy+ans":
        room = min(BUDGET - _answer_starts(z["train_roles"]).mean(), BUDGET - starts.mean())
        return _top(H, Htr, room) | starts
    words = _word_starts(b)
    if name == "words":
        return words
    if name == "words+ans":
        return words | starts
    # label-free: rises in the small model's entropy (H2's signal), thresholds fitted on train
    J, Jtr = np.diff(H, prepend=H[0]), np.diff(Htr, prepend=Htr[0])
    if name == "jump25":
        return _top(J, Jtr, BUDGET)
    if name in ("words+jump20", "words+jump25"):
        extra = {"words+jump20": 0.015, "words+jump25": 0.065}[name]   # top non-word jumps, as a share of all bytes
        wtr = _word_starts(z["train_bytes"])
        return words | (J > np.quantile(Jtr[~wtr], 1 - extra / (~wtr).mean()))
    raise SystemExit(f"unknown mask {name}")


def run(name, seed, steps):
    import harness
    harness.MAIN_STEPS = steps
    z = np.load(NPZ)
    m_tr, m_va = masks(name, z, "train"), masks(name, z, "val")
    for sp, m in (("train", m_tr), ("val", m_va)):
        if m.mean() > BUDGET + 0.01:
            raise SystemExit(f"{name}: rate {m.mean():.3f} on {sp} is above the budget")
    o = harness.train_eval(z["train_bytes"], m_tr, z["val_bytes"], m_va, z["val_roles"], seed=seed)
    print(f"RESULT {DATA} mask={name} seed={seed} steps={steps} pool={harness.POOL} local={harness.LOCAL} | "
          f"ans {o['ans_bits']:.4f} (computed {o['ANS_LOCAL_bits']:.3f}, final {o['ANS_LONG_bits']:.3f}) | "
          f"copy {o['VALUE_bits']:.3f} | text {o['TEXT_bits']:.3f} | bpb {o['bpb']:.4f} | rate {o['boundary_rate']:.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare_cache()
    else:
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
