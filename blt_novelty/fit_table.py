"""Fit a novelty lookup table for BLT's lookup-table patcher.

    python -m blt_novelty.fit_table OUT.pt --rate 0.10 FILE [FILE ...]

FILEs: plain text, or .jsonl with a "text" field per line. Up to --max-bytes (default 4 MB, as in the paper) are
sampled in order from the files. Writes a torch file with the boolean 260 x 260 table ("lookup_table"), the dense
scores, the threshold and the patch rate measured on the fitting text.
"""
import argparse
import json

import numpy as np

from blt_novelty.novelty import blt_lookup_table, fit_scores, scores_for_bytes, threshold_for_rate


def read_bytes(paths, max_bytes):
    out, n = [], 0
    for p in paths:
        if p.endswith(".jsonl"):
            chunks = (json.loads(l)["text"].encode() for l in open(p, encoding="utf8") if l.strip())
        else:
            chunks = [open(p, "rb").read()]
        for c in chunks:
            out.append(c); n += len(c)
            if n >= max_bytes:
                return np.frombuffer(b"\n\n".join(out)[:max_bytes], np.uint8)
    return np.frombuffer(b"\n\n".join(out), np.uint8)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("files", nargs="+")
    ap.add_argument("--rate", type=float, default=0.10, help="target fraction of bytes that start a patch")
    ap.add_argument("--max-bytes", type=int, default=4 << 20)
    a = ap.parse_args()
    b = read_bytes(a.files, a.max_bytes)
    S = fit_scores(b)
    thr = threshold_for_rate(S, b, a.rate)
    rate = float((scores_for_bytes(S, b) > thr).mean())
    torch.save({"lookup_table": blt_lookup_table(S, thr), "scores": torch.from_numpy(S), "threshold": thr,
                "rate_on_fit_text": rate, "fit_bytes": int(len(b)), "source": "arXiv:2610.22100"}, a.out)
    print(f"fitted on {len(b):,} bytes; threshold {thr:.4f}; patch rate on the fitting text {rate:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
