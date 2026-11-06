"""Write the texts of the two copy tests (O6b, O6c) as JSON lines, from the stored patcher outputs (no model; new in
v2). Record of how data/copy_latch_texts/ was made: the patcher outputs it reads (results/blt_copy_latch_fresh_patcher.npz,
16.8 MB, and results/blt_copy_latch_cutoff_patcher.npz, 3.5 MB, git-ignored in the author's private working repository)
are not shipped. Each record keeps the stored metadata (group, GSM8K train index, ID, ID kind, distance, byte count,
sha256, unit spans) and the text; nothing is recomputed. scripts/check_copy_texts.py checks the result against the
text hashes in the shipped block files.

    python paper2/v2/scripts/extract_copy_texts.py [FRESH_NPZ] [CUTOFF_NPZ]
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # paper2/v2
OUT = os.path.join(HERE, "data", "copy_latch_texts")
KEEP = ("group", "idx", "id", "kind", "distance", "n_bytes", "sha256", "units")


def texts(path):
    z = np.load(path)
    meta, b = json.loads(str(z["meta"])), z["b"]
    off = 0
    for m in meta:                                   # as realblt_copy_latch_fresh.load: text = b[off:off + n - 1]
        t = b[off:off + m["n"] - 1].tobytes()
        off += m["n"] - 1
        assert len(t) == m["n_bytes"] and hashlib.sha256(t).hexdigest() == m["sha256"], m["idx"]
        yield {k: m[k] for k in KEEP if k in m} | {"text": t.decode("utf-8")}
    assert off == len(b)


def main():
    a = sys.argv[1:] + [None, None]
    src = {"o6b_fresh.jsonl": a[0] or os.path.join(HERE, "results", "blt_copy_latch_fresh_patcher.npz"),
           "o6c_cutoff.jsonl": a[1] or os.path.join(HERE, "results", "blt_copy_latch_cutoff_patcher.npz")}
    os.makedirs(OUT, exist_ok=True)
    for name, path in src.items():
        rows = list(texts(path))
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} texts from {os.path.basename(path)} "
              f"(sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]}...)")


if __name__ == "__main__":
    main()
