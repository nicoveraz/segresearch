"""Beyond arithmetic: the tight-budget test on Meta's BLT-1B with logic and state tracking.

The blind spot found on math: under a tight patch budget, entropy-level patching skips positions whose TYPE is
predictable but whose VALUE must be computed (results after '='). On code, repeated identifiers (a copy, not a
computation) showed only a small effect. This tests other places where a value must be worked out:

    logic     ProofWriter-style theories (facts and if-then rules about one person), a question, a worked proof
              ("Bob is red, so Bob is kind.") and an answer. Targets: each step's conclusion (found by matching a
              rule) and the True/False answer.
    direct    the same theories with no proof: the answer right after the question (chaining done implicitly).
    trace     short straight-line programs with a trace comment after each assignment ("# a is now 11"). Targets:
              traced values after an arithmetic line (computed) and after a constant line (copied), a control in
              the same text.

All problems are generated with a fixed seed and checked (values by executing the program, answers by forward
chaining). Layouts at equal patch counts, R = 15% and 10% of bytes:

    default      BLT's own entropy patches
    entropy@R    top R by BLT-1B's entropy
    dep@R        top R by boundary dependence, refit on this task's TRAIN problems (as in realblt_code.py)
    entdep@R     entropy + dependence (standardized)
    entdepM@R    entropy + the dependence table fit on GSM8K math (cross-domain transfer, no refit)
    oracle@R     boundaries forced at every target start, the rest by entropy (uses the targets)

Scored per target kind: bits per byte (teacher forced) and exact match.

    <env>/bin/python realblt_reason.py TASK [N_TRAIN] [N_TEST]        (TASK: logic | direct | trace)
"""
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

BUDGETS = (0.15, 0.10)
MATH_DEP = os.path.expanduser("~/.cache/segresearch-blt/dependence_table.npy")
NAMES = ["Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary", "Harry"]
ATTRS = ["big", "blue", "cold", "furry", "green", "kind", "nice", "quiet", "red", "rough", "round", "smart",
         "white", "young", "sad", "tall"]


def logic_problem(rng, direct):
    """One person, 2 starting facts, a hidden chain of 3-5 rules plus distractor rules, a query that is derivable
    (True) or not (False, closed world). Returns text and target spans [(kind, start, end)] in bytes."""
    name = rng.choice(NAMES)
    attrs = rng.sample(ATTRS, 12)
    facts, chain_len = attrs[:2], rng.randint(3, 5)
    chain = [rng.choice(facts)] + attrs[2:2 + chain_len]
    rules = [(chain[i], chain[i + 1]) for i in range(chain_len)]
    rest = attrs[2 + chain_len:]
    for _ in range(rng.randint(2, 4)):                                  # distractors: premise never derived
        a, b = rng.sample(rest, 2)
        rules.append((a, b))
    rng.shuffle(rules)
    known, steps = set(facts), []                                       # forward chaining, in rule order passes
    changed = True
    while changed:
        changed = False
        for a, b in rules:
            if a in known and b not in known:
                known.add(b); steps.append((a, b)); changed = True
    truth = rng.random() < 0.5
    query = rng.choice(sorted(known - set(facts))) if truth else rng.choice([x for x in attrs if x not in known])
    assert (query in known) == truth
    text = " ".join(f"{name} is {f}." for f in facts) + " "
    text += " ".join(f"If someone is {a} then they are {b}." for a, b in rules)
    text += f"\nQuestion: Is {name} {query}?\n"
    spans = []
    if not direct:
        text += "Proof:"
        for a, b in steps:
            text += f" {name} is {a}, so {name} is "
            spans.append(("step", len(text), len(text) + len(b)))
            text += b + "."
        text += "\n"
    text += "Answer: "
    ans = "True" if truth else "False"
    spans.append(("answer", len(text), len(text) + len(ans)))
    text += ans + "\n"
    return text, spans


def trace_problem(rng):
    """A straight-line program over a-e, 12-16 assignments, each followed by '# v is now N' (values kept in 0..999).
    Constant assignments make 'copy' targets, arithmetic ones 'computed' targets; checked by executing it."""
    env, lines, kinds = {}, [], []
    for _ in range(rng.randint(12, 16)):
        v = rng.choice("abcde")
        if len(env) < 2 or rng.random() < 0.3:
            stmt, kind = f"{v} = {rng.randint(0, 99)}", "copy"
        else:
            while True:
                x, y = rng.choice(sorted(env)), rng.choice(sorted(env) + [str(rng.randint(1, 9))])
                op = rng.choice("+-*")
                val = eval(f"{env[x]} {op} {env.get(y, y)}")
                if 0 <= val <= 999:
                    break
            stmt, kind = f"{v} = {x} {op} {y}", "computed"
        exec(stmt, {}, env)
        lines.append((stmt, v, kind))
    text, spans, check = "", [], {}
    for stmt, v, kind in lines:
        exec(stmt, {}, check)
        text += stmt + f"\n# {v} is now "
        val = str(check[v])
        spans.append((kind, len(text), len(text) + len(val)))
        text += val + "\n"
    return text, spans


def problems(task, n, seed):
    rng = random.Random(seed)
    if task == "trace":
        return [trace_problem(rng) for _ in range(n)]
    return [logic_problem(rng, task == "direct") for _ in range(n)]


def main(task, n_train, n_test):
    import torch
    from transformers import BltForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = BltForCausalLM.from_pretrained("itazap/blt-1b-hf", dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config

    def prep(b):
        ids = torch.tensor([[1] + [x + 4 for x in b]], device=dev)          # BOS + byte tokens (byte + 4)
        with torch.no_grad():
            ent, dl, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                             max_patch_length=cfg.max_patch_length)
        n = ids.shape[1]; e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]
        return ids, n, score, np.r_[0, np.cumsum(dl[0].cpu().numpy())[:-1]]

    def L(st, n):
        s = sorted(set(int(x) for x in st if 0 <= x < n))
        return torch.tensor([[b - a for a, b in zip(s, s[1:] + [n])]], device=dev)

    def logp(ids, st, n):
        with torch.no_grad():
            return torch.log_softmax(model(input_ids=ids, patch_lengths=L(st, n), use_cache=False).logits[0].float(), -1)

    t0 = time.time()
    print(f"[{task}] fitting boundary dependence on {n_train} train problems", flush=True)
    s2, s1 = defaultdict(list), defaultdict(list)
    for text, _ in problems(task, n_train, seed=1):
        b = text.encode()
        ids, n, score, default = prep(b)
        tight = np.r_[0, 1, np.argsort(-score)[:max(int(0.10 * n) - 2, 1)]]
        loss = lambda st: -logp(ids, st, n)[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
        gain = loss(tight) - loss(default)
        for i in range(2, len(b) - 4):
            d = float(gain[i:i + 4].mean())                                 # bytes i..i+3 (tokens i+1..i+4)
            s2[(b[i - 2], b[i - 1])].append(d); s1[b[i - 1]].append(d)
    own = {"T2": {k: float(np.mean(v)) for k, v in s2.items() if len(v) >= 20},
           "T1": {k: float(np.mean(v)) for k, v in s1.items() if len(v) >= 20},
           "glob": float(np.mean([x for v in s1.values() for x in v]))}
    top = sorted(own["T2"].items(), key=lambda kv: -kv[1])[:10]
    print(f"  {len(own['T2'])} contexts; top: " + ", ".join(f"{bytes(k)!r} {v:.2f}" for k, v in top), flush=True)
    math_tab = np.load(MATH_DEP, allow_pickle=True).item()
    lookup = lambda tab, b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"]))

    kinds = sorted({k for _, sp in problems(task, 50, seed=2) for k, _, _ in sp})
    layouts = ["default"] + [f"{k}@{int(r * 100)}" for r in BUDGETS
                             for k in ("entropy", "dep", "entdep", "entdepM", "oracle")]
    st_ = {l: {k: {"bits": [], "exact": [], "cov": []} for k in kinds} for l in layouts}
    for j, (text, spans) in enumerate(problems(task, n_test, seed=2)):
        b = text.encode()
        ids, n, score, default = prep(b)
        tstarts = [s + 1 for _, s, _ in spans]                              # token = byte + 1
        fin = np.isfinite(score); fin[:2] = False
        zs = lambda x: (x - x[fin].mean()) / x[fin].std()
        dep, depM = np.full(n, -np.inf), np.full(n, -np.inf)
        dep[2:] = [lookup(own, b, t - 1) for t in range(2, n)]
        depM[2:] = [lookup(math_tab, b, t - 1) for t in range(2, n)]
        lay = {"default": default}
        for r in BUDGETS:
            k = max(int(round(r * n)) - 2, 1); R = int(r * 100)
            lay[f"entropy@{R}"] = np.r_[0, 1, np.argsort(-score)[:k]]
            lay[f"dep@{R}"] = np.r_[0, 1, np.argsort(-dep)[:k]]
            lay[f"entdep@{R}"] = np.r_[0, 1, np.argsort(-np.where(fin, zs(score) + zs(dep), -np.inf))[:k]]
            lay[f"entdepM@{R}"] = np.r_[0, 1, np.argsort(-np.where(fin, zs(score) + zs(depM), -np.inf))[:k]]
            forced = [t for t in tstarts if t >= 2]; fs = set(forced)
            lay[f"oracle@{R}"] = np.r_[0, 1, forced, [t for t in np.argsort(-score) if t not in fs][:max(k - len(forced), 0)]]
        for name, st in lay.items():
            st = np.array(sorted(set(int(x) for x in st if 0 <= x < n)))
            lp = logp(ids, st, n); S = set(st.tolist())
            for (kind, s, e_), t in zip(spans, tstarts):
                tgt = ids[0, s + 1:e_ + 1]; pred = lp[s:e_]
                d = st_[name][kind]
                d["bits"].append(float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)))
                d["exact"].append(bool((pred.argmax(-1) == tgt).all())); d["cov"].append(t in S)
        if (j + 1) % 25 == 0:
            print(f"  {j + 1} test problems, {time.time() - t0:.0f}s", flush=True)
    print(f"\nBLT-1B, task {task}: {n_test} test problems; targets " +
          ", ".join(f"{k} {len(st_['default'][k]['exact'])}" for k in kinds))
    for name in layouts:
        print(f"RESULT {name:11s} " + " | ".join(
            f"{k}: covered {np.mean(s['cov']):5.1%}, {np.mean(s['bits']):.3f} bits, exact {np.mean(s['exact']):5.1%}"
            for k, s in st_[name].items()), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 300)
