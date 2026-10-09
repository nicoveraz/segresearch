"""Training corpus for the scaling pilot (#27): an OpenWebMath slice plus GSM8K/MATH training solutions.

Deterministic, so the Mac and a rented GPU build byte-identical files (checked by SHA-256):
  - OpenWebMath: the first four parquet shards (fixed names and SHA-256 below), documents in shard/row order;
  - contamination: documents sharing any 13-word sequence (lowercased alphanumeric words) with a GSM8K or MATH
    test problem in the eval split are dropped. The Mac computes the dropped (shard, row) list once; the pod
    applies it;
  - GSM8K/MATH training solutions (the train split of the math cache) repeated to ~10% of the corpus bytes;
  - all documents shuffled with a fixed seed and joined with blank lines.

The eval split stays the existing one (660 GSM8K and 700 MATH test problems with answer roles), so results are
comparable with the paper's trained models.
"""
import hashlib
import json
import os
import re
import urllib.request

import numpy as np

OWM_URL = "https://huggingface.co/datasets/open-web-math/open-web-math/resolve/main/"
SHARDS = [
    ("data/train-00000-of-00114-5a023365406cb9c4.parquet", 236268485, "18d7ef70a6ef61f09c020950626d7e6e61d1300f3f85beddbd3b3c6801071a41"),
    ("data/train-00001-of-00114-e32fc2813a15f61c.parquet", 238479393, "d51cde7eeceb64d1814df384a0a05340422e40543db095200de38da41044dda1"),
    ("data/train-00002-of-00114-1429d96b99aec578.parquet", 238135043, "4be3d8421bbbdd687d49810e8a024d5ec6eaf6febd7b21c8308c92741356331f"),
    ("data/train-00003-of-00114-e7fc257ef044bc03.parquet", 241967659, "b0298ffe842ee4fadb52a6d193dc755568f1c5f200822ca22473c5b281b13c56"),
]
NGRAM = 13
WORD = re.compile(r"[a-z0-9]+")


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def download(dest, shards=SHARDS):
    """Download the OpenWebMath shards into dest and verify size and SHA-256. Returns the local paths."""
    os.makedirs(dest, exist_ok=True)
    paths = []
    for name, size, digest in shards:
        path = os.path.join(dest, os.path.basename(name))
        if not (os.path.exists(path) and os.path.getsize(path) == size and sha256(path) == digest):
            print(f"  downloading {name} ({size / 1e6:.0f} MB)", flush=True)
            urllib.request.urlretrieve(OWM_URL + name, path + ".part")
            os.replace(path + ".part", path)
            if os.path.getsize(path) != size or sha256(path) != digest:
                raise RuntimeError(f"{path}: size or SHA-256 mismatch")
        paths.append(path)
    return paths


def owm_docs(paths):
    """(shard index, row index, text) for every OpenWebMath document, in order."""
    import pyarrow.parquet as pq
    for s, path in enumerate(paths):
        pf = pq.ParquetFile(path)
        row = 0
        for batch in pf.iter_batches(columns=["text"], batch_size=4096):
            for text in batch.column(0).to_pylist():
                yield s, row, text
                row += 1


def _ngrams(text):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def eval_ngrams(val_bytes):
    """13-word sequences of every eval problem (GSM8K and MATH test problems, split on blank lines)."""
    out = set()
    for seg in val_bytes.tobytes().decode("utf-8", "replace").split("\n\n"):
        out |= _ngrams(seg)
    return out


def contaminated(paths, grams, log_every=200_000):
    """[(shard, row)] of OpenWebMath documents sharing a 13-word sequence with the eval problems."""
    drop = []
    for k, (s, r, text) in enumerate(owm_docs(paths)):
        w = WORD.findall(text.lower())
        if any(" ".join(w[i:i + NGRAM]) in grams for i in range(len(w) - NGRAM + 1)):
            drop.append((s, r))
        if log_every and (k + 1) % log_every == 0:
            print(f"    {k + 1} documents checked, {len(drop)} dropped", flush=True)
    return drop


def build_corpus(paths, math_train, drop, out_path, math_share=0.10, seed=0):
    """Write the training corpus (raw uint8 bytes) and return its metadata (size, SHA-256, composition)."""
    drop = {tuple(x) for x in drop}
    docs = [text.encode("utf-8") for s, r, text in owm_docs(paths) if (s, r) not in drop]
    owm_bytes = sum(len(d) + 2 for d in docs)
    math_docs = [d for d in math_train.tobytes().split(b"\n\n") if d]
    math_once = sum(len(d) + 2 for d in math_docs)
    reps = max(1, round(math_share / (1 - math_share) * owm_bytes / math_once))
    docs += math_docs * reps
    order = np.random.default_rng(seed).permutation(len(docs))
    with open(out_path + ".part", "wb") as f:
        for i in order:
            f.write(docs[i]); f.write(b"\n\n")
    os.replace(out_path + ".part", out_path)
    meta = {"bytes": os.path.getsize(out_path), "sha256": sha256(out_path), "owm_docs": len(docs) - len(math_docs) * reps,
            "owm_dropped": len(drop), "owm_bytes": owm_bytes, "math_repeats": reps, "math_bytes": math_once * reps,
            "shards": [s[0] for s in SHARDS[:len(paths)]], "seed": seed, "math_share": math_share}
    return meta


def load_bytes(path):
    """The corpus as a read-only memmap of uint8."""
    return np.memmap(path, dtype=np.uint8, mode="r")


def save_json(obj, path):
    """Atomic: readers (runs starting their evaluation) never see a half-written file."""
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)
