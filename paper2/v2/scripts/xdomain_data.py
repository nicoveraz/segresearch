"""Documents for the cross-domain test of where a patch start pays (paper3/XDOMAIN_RULES.md, issue #30).

Six domains, each split into FIT documents (gain measured; used to fit thresholds and gain tables), TEST documents
(gain measured; all verdicts) and a label-free POOL of raw text for the novelty table (disjoint from TEST):

    math      GSM8K, stripped format (no <<>> annotations; "The final answer is N"); fit/pool from train, test from test
    algebra   MATH, Algebra subject: "Problem: ...\\nSolution: ..." (LaTeX kept); fit/pool from train, test from test
    code      Python standard-library source, 800-byte chunks (realblt_code.chunks: files split by path hash)
    prose     English Wikipedia (wikimedia/wikipedia 20231101.en, streamed), paragraphs packed into documents;
              fit/pool from the first articles, test from later ones
    clinical  MedQA-USMLE (4 options) vignettes with options and the answer line; fit/pool and test disjoint
    traces    synthetic execution traces (scale/generic.py traces), different seeds per split

ASCII only (non-ASCII is transliterated or the document is skipped); 300-950 bytes per document (BLT-1B scripts cap
at 1,000 tokens). Writes ~/.cache/segresearch-xdomain/{domain}_{fit,test}.jsonl and {domain}_pool.txt.

    ~/anaconda3/bin/python xdomain_data.py
"""
import glob
import hashlib
import json
import os
import re
import sys
import sysconfig
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "segresearch"))
OUT = os.path.expanduser("~/.cache/segresearch-xdomain")
N_FIT, N_TEST, POOL_BYTES = 50, 60, 200_000
LO, HI = 300, 950


def ascii_(s):
    s = s.replace("–", "-").replace("—", " - ").replace("’", "'").replace("“", '"').replace("”", '"')
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s


def ok(s):
    return s.isascii() and LO <= len(s.encode()) <= HI


def pack(paras, lo=LO, hi=HI):
    """Pack consecutive paragraphs into documents of lo..hi bytes (a paragraph longer than hi is cut at a sentence)."""
    docs, cur = [], ""
    for p in paras:
        if len(p) > hi:
            cut = p[:hi].rfind(". ")
            p = p[:cut + 1] if cut > lo // 2 else p[:hi]
        if cur and len(cur) + 1 + len(p) > hi:
            if len(cur) >= lo:
                docs.append(cur)
            cur = ""
        cur = (cur + "\n" + p) if cur else p
    if len(cur) >= lo:
        docs.append(cur)
    return docs


def math_docs():
    def fmt(p):
        sol, ans = p["answer"].rsplit("#### ", 1)
        sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        return p["question"] + "\n" + sol + "\nThe final answer is " + ans.strip()
    tr = [fmt(json.loads(l)) for l in open(os.path.expanduser("~/.cache/segresearch-gsm8k/train.jsonl"))]
    te = [fmt(json.loads(l)) for l in open(os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl"))]
    return tr, te, tr


def algebra_docs():
    def load(split):
        rows = [json.loads(l) for l in open(os.path.expanduser(f"~/.cache/segresearch-math/math_algebra_{split}.jsonl"))]
        return ["Problem: " + r["problem"].strip() + "\nSolution: " + r["solution"].strip() for r in rows]
    tr = load("train")
    return tr, load("test"), tr


def code_docs():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(f for f in glob.glob(root + "/**/*.py", recursive=True)
                   if "/test" not in f and "site-packages" not in f and "/idlelib" not in f)

    def chunks(val):                                  # realblt_code.chunks, same split and chunking
        out = []
        for f in files:
            if (int(hashlib.md5(os.path.relpath(f, root).encode()).hexdigest(), 16) % 12 == 0) != val:
                continue
            src = open(f, "rb").read()
            if not src.isascii():
                continue
            cur = b""
            for line in src.split(b"\n"):
                if len(cur) + len(line) + 1 > 800 and len(cur) > 200:
                    out.append(cur); cur = b""
                cur += line + b"\n"
        rng = np.random.default_rng(0)
        return [out[i].decode() for i in rng.permutation(len(out))]
    tr = chunks(False)
    return tr, chunks(True), tr


def prose_docs():
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    fit, test = [], []
    for i, art in enumerate(ds):
        paras = [re.sub(r" {2,}", " ", re.sub(r"\(\s*[,;]?\s*\)", "", ascii_(p.strip())))     # dump artifacts:
                 for p in art["text"].split("\n") if len(p.strip()) > 80]                    # removed templates
        paras = [p for p in paras if p.isascii() and not p.startswith(("Category:", "See also"))]
        docs = pack(paras)
        (fit if i < 400 else test).extend(docs[:3])     # at most 3 documents per article
        if i >= 400 and len(test) >= 3 * N_TEST:
            break
    return fit, test, fit


def clinical_docs():
    from datasets import load_dataset
    d = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    docs = [ascii_(r["question"].strip() + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(r["options"].items()))
                   + f"\nAnswer: {r['answer_idx']}. {r['answer']}") for r in d]
    return docs[:1000], docs[1000:], docs[:1000]


def traces_docs():
    from scale.generic import traces
    def docs(seed):
        progs = [p for p in traces(300_000, seed).decode().split("\n\n") if p.strip()]
        return pack(progs, lo=LO, hi=800)
    return docs(1), docs(2), docs(3)


DOMAINS = {"math": math_docs, "algebra": algebra_docs, "code": code_docs, "prose": prose_docs,
           "clinical": clinical_docs, "traces": traces_docs}


def main(only=None):
    os.makedirs(OUT, exist_ok=True)
    for name, fn in DOMAINS.items():
        if only and name not in only:
            continue
        fit_src, test_src, pool_src = fn()
        fit = [s for s in fit_src if ok(s)][:N_FIT]
        test = [s for s in test_src if ok(s)][:N_TEST]
        pool, size = [], 0
        for s in pool_src:
            if s.isascii() and s not in test:
                pool.append(s); size += len(s) + 2
                if size >= POOL_BYTES:
                    break
        for split, docs in (("fit", fit), ("test", test)):
            with open(os.path.join(OUT, f"{name}_{split}.jsonl"), "w") as f:
                for s in docs:
                    f.write(json.dumps({"text": s}) + "\n")
        open(os.path.join(OUT, f"{name}_pool.txt"), "w").write("\n\n".join(pool))
        lens = [len(s) for s in test]
        print(f"{name:9s} fit {len(fit)} test {len(test)} (bytes {min(lens)}-{max(lens)}, mean {np.mean(lens):.0f})"
              f" pool {size} bytes", flush=True)
        assert len(fit) == N_FIT and len(test) == N_TEST, name


if __name__ == "__main__":
    main(sys.argv[1:] or None)
    sys.stdout.flush()
    os._exit(0)                  # the streamed Wikipedia reader leaves an Arrow thread pool that hangs at exit
