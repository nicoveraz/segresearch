"""Boundary enrichment (SOMBRERO, arXiv 2601.22805) of our patch rules on the math evaluation set (local).

SOMBRERO measures how strongly patch starts concentrate on hard-to-predict bytes:

    B = mean surprisal at patch starts / mean surprisal over all bytes      (B > 1: starts on hard bytes)

with a rate-matched circular-shift null (Z_B). It then trains boundaries to raise B. Here B is computed for the
rules of the 50M runs (exact thresholds from the pods' rules.json; surprisal from the bundle's small byte LM, the same
model entropy patching uses), next to each rule's coverage of computed-result and final-answer starts and its 50M
accuracy, to ask whether high enrichment goes with computing results correctly.

    .venv-scale/bin/python -m scale.enrichment         # -> results/enrichment.json
"""
import glob
import json
import os

import numpy as np
import torch

from scale.evaluate import RI
from scale.model import EntropyLM
from scale.patchers import ENTROPY, score_stream
from scale.prep import table_for
from scale.rules import Rule
from scale.train import CTX, device

B = os.path.expanduser("~/.cache/segresearch-scale/bundle")
RULES = {**json.load(open(os.path.expanduser("~/Downloads/segresearch_pod_50m2/results_all/rules.json"))),
         **json.load(open(os.path.expanduser("~/Downloads/segresearch_pod_novc50/results_all/rules.json")))}
ROOT = os.path.join(os.path.dirname(__file__), "..")


def acc50(rule):
    fs = sorted(glob.glob(f"{ROOT}/results/scale/runs/50m_{rule}_s*_clip/result.json"))
    R = [json.load(open(f)) for f in fs]
    return ({k: float(np.mean([r[k] for r in R])) for k in ("computed_acc", "final_acc", "bpb")} | {"seeds": len(R)}) if R else None


def main(n_null=200):
    dev = device()
    val = np.fromfile(f"{B}/val.u8", np.uint8); roles = np.fromfile(f"{B}/val_roles.i8", np.int8)
    ent = EntropyLM(ENTROPY["d"], ENTROPY["layers"], ENTROPY["heads"], CTX).to(dev)
    ent.load_state_dict(torch.load(f"{B}/entropy.pt", map_location=dev)); ent.eval()
    with torch.no_grad():
        H, S = score_stream(ent, val, dev=dev)
    keep = np.zeros(len(val), bool); keep[CTX:len(val) - CTX] = True          # positions both arrays cover
    starts = {k: (roles == RI[r]) & np.r_[True, roles[:-1] != RI[r]] for k, r in (("computed", "ANS_LOCAL"), ("final", "ANS_LONG"))}
    rng = np.random.default_rng(0)
    out = {}
    print(f"{len(val) / 1e6:.1f} MB eval; mean surprisal {S[keep].mean():.2f} bits")
    print(f"{'rule':18s} {'rate':>6s} {'B':>6s} {'Z_B':>7s} {'cov computed':>13s} {'cov final':>10s} {'50M computed':>13s} {'50M final':>10s}")
    for name, st in RULES.items():
        if name.startswith(("typed", "final+typed")):
            continue                                                             # needs typed values; not in this screen
        rule = Rule.from_state(st, table_for(B, name) if name.startswith(("dep", "nov")) else None)
        m = rule.mask(val, H) & keep
        Bv = S[m].mean() / S[keep].mean()
        sk = S[keep]; mk = m[keep]
        null = [sk[np.roll(mk, int(rng.integers(1, len(sk) - 1)))].mean() / sk.mean() for _ in range(n_null)]
        z = (Bv - np.mean(null)) / np.std(null)
        cov = {k: float((m & s & keep).sum() / max((s & keep).sum(), 1)) for k, s in starts.items()}
        a = acc50(name)
        out[name] = {"rate": float(mk.mean()), "B": float(Bv), "Z_B": float(z), "coverage": cov, "acc50m_clip": a}
        print(f"{name:18s} {mk.mean():6.1%} {Bv:6.2f} {z:7.1f} {cov['computed']:13.0%} {cov['final']:10.0%} "
              + (f"{100 * a['computed_acc']:12.1f}% {100 * a['final_acc']:9.1f}%  ({a['seeds']} seeds)" if a else "           –          –"))
    # surprisal of the target bytes themselves: are computed results 'hard' for the patcher's model?
    for k, s in starts.items():
        out[f"surprisal_at_{k}_starts"] = float(S[s & keep].mean())
    print(f"mean surprisal at computed-result starts {out['surprisal_at_computed_starts']:.2f} bits, "
          f"final-answer starts {out['surprisal_at_final_starts']:.2f}, all bytes {S[keep].mean():.2f}")
    json.dump(out, open(f"{ROOT}/results/enrichment.json", "w"), indent=1)


if __name__ == "__main__":
    main()
