"""Copied vs computed fields: does confidence-based "fast vs slow" routing fail on computed values? (local)

A structured-extraction benchmark in the spirit of fast decision models (System One-style: typed fields, one pass).
Generated invoices (fixed seed, gold answers computed by code) with three kinds of field:

    copied    vendor, invoice number, issue date, the unit price of a named item   (value appears verbatim)
    lookup    the most expensive item                                              (compare, then copy)
    computed  number of line items, total quantity, subtotal, total with tax, due date (issue date + terms)

Each field is answered twice by a language model: FAST (answer directly, few-shot, greedy) and SLOW (write the
working first, then "So the answer is ..."). For every fast answer we record confidence signals:

    seq_logp   log-probability of the answer tokens (higher = more confident)
    ent_mean   mean next-token entropy over the answer tokens
    first_p    probability of the first answer token
    typed_max  typed uncertainty (typed_screen.py, over token kinds: single digits, words, punctuation, other):
               P(likeliest kind) x within-kind entropy / log2(kind size), max over the answer tokens

Then the routing question: escalating a fraction q of fields from fast to slow, chosen by each signal (or by field
kind: computed first), what accuracy results? Predictions from the patching work: fast is fine on copied fields and
weak on computed ones; plain confidence misses part of the computed errors (confident errors), typed uncertainty
catches more, and the structural signal (field kind) catches what uncertainty cannot.

    .venv-blt/bin/python fields_bench.py [N_DOCS] [MODEL]          (default 20 docs, Qwen/Qwen2.5-1.5B)
    uv run --with mlx-lm python fields_bench.py 50 mlx-community/Qwen2.5-7B-Instruct-4bit   (MLX backend)
"""
import datetime as dt
import json
import math
import os
import random
import re
import sys
import time

import numpy as np

VENDORS = ["Andes Supply Co.", "Pacific Tools Ltd.", "Norte Medical SpA", "Lago Office Goods", "Cordillera Parts",
           "Austral Lab Systems", "Valle Hardware", "Costa Electric"]
ITEMS = ["copper wire", "valve kit", "safety gloves", "printer paper", "LED panel", "drill bits", "cable ties",
         "sensor module", "ink cartridge", "steel bolts", "glass vials", "power strip", "desk lamp", "pipe fitting"]
KIND = {"vendor": "copied", "invoice_number": "copied", "issue_date": "copied", "unit_price": "copied",
        "most_expensive": "lookup", "item_count": "computed", "total_quantity": "computed", "subtotal": "computed",
        "total": "computed", "due_date": "computed"}


def make_doc(rng):
    n = rng.randint(2, 6)
    items = rng.sample(ITEMS, n)
    rows = [(it, rng.randint(1, 9), round(rng.randint(150, 9900) / 100, 2)) for it in items]
    date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 300))
    terms = rng.choice([15, 30, 45, 60])
    tax = rng.choice([0, 5, 8, 10, 19])
    vendor, num = rng.choice(VENDORS), rng.randint(10000, 99999)
    text = (f"INVOICE #{num}\nVendor: {vendor}\nIssue date: {date.isoformat()}\nPayment terms: net {terms} days\n"
            f"Items:\n" + "".join(f"- {q} x {it} @ ${p:.2f}\n" for it, q, p in rows) + f"Tax: {tax}%\n")
    sub_c = sum(q * round(p * 100) for _, q, p in rows)               # exact cents
    sub = sub_c / 100
    tot_c = (sub_c * (100 + tax) + 50) // 100                          # round half up, as by hand
    pick = rng.choice(rows)
    fields = {
        "vendor": ("Who is the vendor?", vendor),
        "invoice_number": ("What is the invoice number?", str(num)),
        "issue_date": ("What is the issue date?", date.isoformat()),
        "unit_price": (f"What is the unit price of the {pick[0]}?", f"{pick[2]:.2f}"),
        "most_expensive": ("Which item has the highest unit price?", max(rows, key=lambda r: r[2])[0]),
        "item_count": ("How many different items are listed?", str(n)),
        "total_quantity": ("What is the total quantity of all items?", str(sum(q for _, q, _ in rows))),
        "subtotal": ("What is the subtotal before tax?", f"{sub:.2f}"),
        "total": ("What is the total including tax?", f"{tot_c / 100:.2f}"),
        "due_date": ("What is the due date?", (date + dt.timedelta(days=terms)).isoformat()),
    }
    return text, fields


def norm(s):
    s = s.strip().lower().replace("$", "").replace(",", "")
    s = s[:-1] if s.endswith(".") and not s.endswith(("ltd.", "co.", "spa.")) else s
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s


def correct(pred, gold):
    return pred is not None and norm(pred) == norm(gold)


EX_DOC = ("INVOICE #55120\nVendor: Sur Fasteners\nIssue date: 2026-02-03\nPayment terms: net 30 days\nItems:\n"
          "- 2 x hex nuts @ $4.10\n- 5 x washers @ $1.20\n- 1 x spring pins @ $3.75\nTax: 10%\n")
# one example of every question type, answered directly (FAST) or with the working first (SLOW)
EX_QA = [
    ("Who is the vendor?", "Sur Fasteners",
     "The vendor line says Sur Fasteners. So the answer is Sur Fasteners."),
    ("What is the invoice number?", "55120", "The header says #55120. So the answer is 55120."),
    ("What is the issue date?", "2026-02-03", "The issue date line says 2026-02-03. So the answer is 2026-02-03."),
    ("What is the unit price of the washers?", "1.20", "The washers line says @ $1.20. So the answer is 1.20."),
    ("Which item has the highest unit price?", "hex nuts",
     "Unit prices: hex nuts 4.10, washers 1.20, spring pins 3.75; the highest is 4.10. So the answer is hex nuts."),
    ("How many different items are listed?", "3", "The items are hex nuts, washers, spring pins: 3 lines. So the answer is 3."),
    ("What is the total quantity of all items?", "8", "Quantities 2 + 5 + 1 = 8. So the answer is 8."),
    ("What is the subtotal before tax?", "17.95",
     "2 x 4.10 = 8.20, 5 x 1.20 = 6.00, 1 x 3.75 = 3.75; 8.20 + 6.00 + 3.75 = 17.95. So the answer is 17.95."),
    ("What is the total including tax?", "19.75",
     "Subtotal 17.95, tax 10%: 17.95 x 1.10 = 19.745, rounded 19.75. So the answer is 19.75."),
    ("What is the due date?", "2026-03-05",
     "2026-02-03 plus 30 days: 25 days to 2026-02-28 (February has 28 days), then 5 more: 2026-03-05. "
     "So the answer is 2026-03-05."),
]


def prompt(doc, q, slow):
    out = EX_DOC + "".join(f"Question: {a}\nAnswer: {w if slow else f}\n" for a, f, w in EX_QA) + "\n"
    return out + doc + f"Question: {q}\nAnswer:"


def parse_slow(text, gold):
    """The answer from worked text: the last "answer is X"; else the last date or number (numeric/date fields)."""
    text = text.split("\nQuestion:")[0]
    hits = re.findall(r"answer is:?\s*\$?\**([^\n*]+?)\**\s*(?:\n|$)", text)
    if hits:
        ans = hits[-1].strip()
        if ans.endswith(".") and not re.search(r"\b(Ltd|Co|SpA)\.$", ans):
            ans = ans[:-1]
        return ans
    m = re.search(r"\bis\s+\**([^\n*]+?)\**\s*$", text.strip())        # "The vendor is X." (names)
    if m and not re.fullmatch(r"[\d.\-]+", gold):
        ans = m.group(1).strip()
        return ans[:-1] if ans.endswith(".") and not re.search(r"\b(Ltd|Co|SpA)\.$", ans) else ans
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", gold):
        d = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        return d[-1] if d else None
    if re.fullmatch(r"[\d.]+", gold):
        n = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
        return n[-1] if n else None
    return None


def token_kinds(tok):
    """Kind of every vocabulary token: 0 single digit, 1 word (letters), 2 punctuation/space, 3 other."""
    k = np.full(len(tok), 3, np.int64)
    for i in range(len(tok)):
        s = tok.decode([i])
        t = s.strip()
        if len(t) == 1 and t.isdigit():
            k[i] = 0
        elif t.isalpha():
            k[i] = 1
        elif t == "" or all(not c.isalnum() for c in t):
            k[i] = 2
    return k


def typed_tokens(lp, onehot, logsize):
    """typed_screen.typed over token kinds: lp (V,) natural-log probs -> T in [0, 1]."""
    p = np.exp(lp)
    Pc = p @ onehot
    Pcs = np.maximum(Pc, 1e-12)
    Hin = np.log2(Pcs) - ((p * lp / np.log(2)) @ onehot) / Pcs
    c = int(Pc.argmax())
    return float(Pc[c] * np.clip(Hin[c] / logsize[c], 0, 1))


def mlx_backend(model_id):
    """fast(doc, q) and slow(doc, q) on an MLX model (mlx_lm), same prompts and signals as the PyTorch path."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    model, tok = load(model_id)
    V = model.model.embed_tokens.weight.shape[0] if hasattr(model, "model") else len(tok)
    base = tok._tokenizer if hasattr(tok, "_tokenizer") else tok
    kinds = np.r_[token_kinds(base), np.full(max(0, V - len(base)), 3, np.int64)][:V]
    onehot = np.eye(4)[kinds]
    logsize = np.log2(np.maximum(onehot.sum(0), 2))
    print(f"{model_id} (MLX); vocab {V}; kind sizes {onehot.sum(0).astype(int).tolist()}", flush=True)
    greedy = lambda x: mx.argmax(x, axis=-1)

    def run(text, n, keep, stop):
        ids = mx.array(base.encode(text))
        gen, lps = [], []
        for (t, lp), _ in zip(generate_step(ids, model, max_tokens=n, sampler=greedy), range(n)):
            t = int(t); gen.append(t)
            if keep:
                lps.append(np.array(lp.astype(mx.float32)))
            if stop in base.decode(gen[-8:]):
                break
        return gen, lps

    def fast(doc, q):
        gen, lps = run(prompt(doc, q, False), 20, True, "\n")
        text = base.decode(gen).split("\n")[0].strip()
        k = max(1, len(base.encode(" " + text)))
        lp_tok = [float(l[t]) for l, t in zip(lps, gen)]
        ents = [float(-(np.exp(l) * l).sum() / np.log(2)) for l in lps]
        ts = [typed_tokens(l, onehot, logsize) for l in lps]
        return text, {"seq_logp": sum(lp_tok[:k]), "ent_mean": float(np.mean(ents[:k])), "first_p": math.exp(lp_tok[0]),
                      "typed_max": max(ts[:k])}

    def slow(doc, q, gold):
        gen, _ = run(prompt(doc, q, True), 256, False, "Question:")
        text = base.decode(gen).split("\nQuestion:")[0].strip()
        return parse_slow(text, gold), text

    return fast, slow


def main(n_docs, model_id):
    if model_id.startswith("mlx-community/"):
        fast, slow = mlx_backend(model_id)
        return evaluate(n_docs, model_id, fast, slow)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
    V = model.get_output_embeddings().weight.shape[0]
    kinds = np.r_[token_kinds(tok), np.full(max(0, V - len(tok)), 3, np.int64)][:V]
    onehot = np.eye(4)[kinds]
    logsize = np.log2(np.maximum(onehot.sum(0), 2))
    print(f"{model_id} on {dev}; vocab {V}; kind sizes {onehot.sum(0).astype(int).tolist()}", flush=True)

    def fast(doc, q):
        ids = tok(prompt(doc, q, False), return_tensors="pt").input_ids.to(dev)
        gen, lps, ents, ts = [], [], [], []
        with torch.no_grad():
            past = None; x = ids
            for _ in range(20):
                out = model(input_ids=x, past_key_values=past, use_cache=True)
                past = out.past_key_values
                lp = torch.log_softmax(out.logits[0, -1].float(), -1)
                nxt = int(lp.argmax())
                lpn = lp.cpu().numpy()
                lps.append(float(lpn[nxt])); ents.append(float(-(np.exp(lpn) * lpn).sum() / np.log(2)))
                ts.append(typed_tokens(lpn, onehot, logsize))
                gen.append(nxt)
                if "\n" in tok.decode(gen):
                    break
                x = torch.tensor([[nxt]], device=dev)
        text = tok.decode(gen).split("\n")[0].strip()
        k = max(1, len(tok(" " + text).input_ids))                       # signals over the answer tokens only
        return text, {"seq_logp": sum(lps[:k]), "ent_mean": float(np.mean(ents[:k])), "first_p": math.exp(lps[0]),
                      "typed_max": max(ts[:k])}

    def slow(doc, q, gold):
        ids = tok(prompt(doc, q, True), return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:]).split("\nQuestion:")[0].strip()
        return parse_slow(text, gold), text

    return evaluate(n_docs, model_id, fast, slow)


def evaluate(n_docs, model_id, fast, slow):
    rng = random.Random(7)
    rows, t0 = [], time.time()
    for i in range(n_docs):
        doc, fields = make_doc(rng)
        for name, (q, gold) in fields.items():
            fa, sig = fast(doc, q)
            sa, swork = slow(doc, q, gold)
            rows.append({"doc": i, "field": name, "kind": KIND[name], "gold": gold, "fast": fa, "slow": sa,
                         "fast_ok": correct(fa, gold), "slow_ok": correct(sa, gold), "slow_work": swork, **sig})
        print(f"  {i + 1}/{n_docs} docs, {time.time() - t0:.0f}s", flush=True)
    report(rows)
    out = f"results/fields_bench_{model_id.split('/')[-1]}_{n_docs}.json"
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"-> {out}")


def report(rows):
    print("\naccuracy by field kind (fast / slow), and how confident fast is when it is wrong vs right")
    for kind in ("copied", "lookup", "computed"):
        R = [r for r in rows if r["kind"] == kind]
        f = np.mean([r["fast_ok"] for r in R]); s = np.mean([r["slow_ok"] for r in R])
        wrong = [r for r in R if not r["fast_ok"]]; right = [r for r in R if r["fast_ok"]]
        conf_w = np.mean([math.exp(r["seq_logp"]) for r in wrong]) if wrong else float("nan")
        conf_r = np.mean([math.exp(r["seq_logp"]) for r in right]) if right else float("nan")
        hi_w = np.mean([math.exp(r["seq_logp"]) > 0.8 for r in wrong]) if wrong else float("nan")
        print(f"  {kind:9s} n={len(R):4d}  fast {f:5.1%}  slow {s:5.1%}  | fast answer prob: right {conf_r:.2f}, "
              f"wrong {conf_w:.2f}; wrong answers with prob > 0.8: {hi_w:5.1%}")
    sigs = {"low confidence (seq_logp)": lambda r: -r["seq_logp"], "high entropy (ent_mean)": lambda r: r["ent_mean"],
            "low first-token prob": lambda r: -r["first_p"], "typed uncertainty (typed_max)": lambda r: r["typed_max"],
            "field kind (computed first)": lambda r: (r["kind"] == "computed") + 0.5 * (r["kind"] == "lookup") - 1e-3 * r["seq_logp"],
            "kind + typed": lambda r: (r["kind"] == "computed") + r["typed_max"],
            "oracle (fast wrong first)": lambda r: (not r["fast_ok"]) + 0.0}
    print("\naccuracy when escalating a fraction q of fields to slow, chosen by each signal")
    qs = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    print(f"  {'signal':32s}" + "".join(f"{q:>7.0%}" for q in qs))
    rr = np.random.default_rng(0)
    for name, f in sigs.items():
        sc = np.array([f(r) for r in rows]) + 1e-9 * rr.random(len(rows))
        order = np.argsort(-sc)
        accs = []
        for q in qs:
            esc = set(order[:int(round(q * len(rows)))].tolist())
            accs.append(np.mean([(r["slow_ok"] if i in esc else r["fast_ok"]) for i, r in enumerate(rows)]))
        print(f"  {name:32s}" + "".join(f"{a:7.1%}" for a in accs))


def rescore(path):
    """Re-parse saved slow answers with the current parser and print the report again (no model run)."""
    rows = json.load(open(path))
    for r in rows:
        r["slow"] = parse_slow(r["slow_work"], r["gold"]); r["slow_ok"] = correct(r["slow"], r["gold"])
    json.dump(rows, open(path, "w"), indent=1)
    report(rows)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "rescore":
        rescore(a[1]); sys.exit()
    main(int(a[0]) if a else 20, a[1] if len(a) > 1 else "Qwen/Qwen2.5-1.5B")
