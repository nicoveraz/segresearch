"""Check data/copy_latch_texts/ against the block files of the two copy tests (CPU, stdlib only; new in v2).

Every text must be listed in its block file (results/blt_copy_latch_fresh_blocks.json for O6b,
results/blt_copy_latch_cutoff_blocks.json for O6c) under the same group and GSM8K train index, with the same byte
count and sha256, and every listed text must be present; where a text carries a ticket ID, the ID must sit at its
stored spans (units: [start, end) byte offsets into the text).

    python paper2/v2/scripts/check_copy_texts.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # paper2/v2
R, D = os.path.join(HERE, "results"), os.path.join(HERE, "data", "copy_latch_texts")


def listed(blocks_file):
    o = json.load(open(os.path.join(R, blocks_file)))
    if "blocks" in o:                                  # O6b: blocks[F][fit|test]
        return {(f"{F}.{sp}", e["idx"]): e for F, v in o["blocks"].items() for sp in ("fit", "test") for e in v.get(sp, [])}
    return {(sp, e["idx"]): e for sp in ("fit", "test") for e in o["block"][sp]}     # O6c: block[fit|test]


def main():
    bad = 0
    for name, blocks in (("o6b_fresh.jsonl", "blt_copy_latch_fresh_blocks.json"),
                         ("o6c_cutoff.jsonl", "blt_copy_latch_cutoff_blocks.json")):
        want = listed(blocks)
        rows = [json.loads(l) for l in open(os.path.join(D, name), encoding="utf-8")]
        seen = set()
        for r in rows:
            k = (r["group"], r["idx"])
            t = r["text"].encode("utf-8")
            e = want.get(k)
            ok = (e is not None and k not in seen and len(t) == e["n_bytes"] == r["n_bytes"]
                  and hashlib.sha256(t).hexdigest() == e["sha256"] == r["sha256"]
                  and all(e.get(f) == r.get(f) for f in ("id", "kind", "distance")))
            if ok and r.get("id"):
                ok = all(t[s:e2].decode("utf-8") == r["id"] for u in ("id_new", "id_copy") for s, e2 in r["units"][u])
            bad += not ok
            seen.add(k)
        missing = set(want) - seen
        bad += len(missing)
        groups = sorted({g for g, _ in seen})
        print(f"{name}: {len(rows)} texts, {len(groups)} groups ({', '.join(groups)}); "
              f"missing from the data {len(missing)}; failures so far {bad}")
    print("OK" if not bad else "FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
