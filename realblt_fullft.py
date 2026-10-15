"""Study B (prereg/2026-10-13_scale150_bltft.md): Meta's BLT-1B fully fine-tuned at a tight patch budget (local).

realblt_finetune.py adapted BLT-1B with LoRA. Here every weight is trained except the entropy patcher (frozen: it
defines the layouts), under one layout rule at budget R, on GSM8K training solutions written as realblt_budget.py
writes them ("question\\nsolution\\nThe final answer is N"). Same data order, steps and optimizer for every rule.
Rules (blt_layouts.py, train-fitted thresholds, causal):

    entropy      BLT-1B's entropy                                          (control: BLT's rule at the budget)
    nov          the self-supervised novelty table (realblt_combo.py)
    nov+final    a boundary after "The final answer is ", the rest by novelty

Then the model is scored under its own rule: teacher-forced exact match on in-line results and final answers (as
in the paper) on N_TEST GSM8K test problems, and END-TO-END GSM8K on N_E2E problems (greedy, zero-shot in the
training format, patch boundaries decided online from the bytes generated so far; no KV cache).

    <env>/bin/python realblt_fullft.py RULE [R] [UPDATES] [N_TEST] [N_E2E] [SEED]   e.g. nov+final 0.10 1800 300 500 0
    SEGR_FT_TOY=1 ...   Mac dry run: trains only one global layer, a few updates, tiny evaluation

Training: AdamW lr 1e-5 (warmup 100, linear decay to 10%), 8 problems per update (accumulated), gradient clipping
1.0, fp32 master weights with bf16 autocast on CUDA. Needs ~25 GB of GPU memory plus activations.
"""
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

import blt_layouts
import blt_load
import registry
from realblt_budget import GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem
from realblt_combo import MARKER, novelty_table

TOY = os.environ.get("SEGR_FT_TOY") == "1"
N_FIT = 300
ACC = 2 if TOY else 8                          # problems per update
RULES = {"entropy": "entropy", "nov": "nov", "nov+final": "forced[final]:nov"}
MAX_NEW = 60 if TOY else 320
OUT = os.path.expanduser("~/.cache/segresearch-blt/fullft")


def number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s or "")
    return m[0].replace(",", "").rstrip(".") if m else None


def main(rule, R, updates, n_test, n_e2e, seed):
    from transformers import AutoTokenizer
    assert blt_layouts.MODE == "train"
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = blt_load.load(MODEL, dev, dtype=torch.bfloat16 if (TOY or dev != "cuda") else torch.float32)
    cfg = model.config
    patcher = model.model.patcher
    nov = novelty_table()
    kind = RULES[rule]

    def marker_starts(b):
        out, i = [], b.find(MARKER)
        while i >= 0:
            out.append(i + len(MARKER) + 1); i = b.find(MARKER, i + 1)
        return out

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, _, _ = patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        s = np.full(n, -np.inf); s[2:] = e[1:n - 1]
        nv = np.full(n, -np.inf); nv[2:] = nov(np.frombuffer(b, np.uint8))[1:n - 1]   # token t = byte t-1
        return {"entropy": s, "nov": nv}

    def layout(b, ids):
        return blt_layouts.layout(kind, R, signals(b, ids), fit, {"final": marker_starts(b)})

    train = [json.loads(l) for l in open(GSM_TRAIN)]
    fit = blt_layouts.Fit((R,))
    for p in train[:N_FIT if not TOY else 20]:
        pr = problem(p, tok, dev)
        if pr is not None:
            fit.add(signals(pr[0], pr[1]), {"final": marker_starts(pr[0])})
    fit.done()
    print(f"{rule}@{R:.2f}: train patch rate {fit.rate(kind, R):.3f} on {len(fit.sigs)} problems; device {dev}", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [p for n_, p in model.named_parameters() if not n_.startswith("model.patcher")]
    if TOY:                                       # one global layer only, so a 16 GB Mac can run the mechanics
        trainable = list(model.model.global_transformer.layers[-1].parameters())
    for p in trainable:
        p.requires_grad_(True)
    print(f"trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.0)
    warm = min(100, max(1, updates // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.1, 1 - s / updates))
    pool = train[N_FIT:]
    rng = random.Random(seed); torch.manual_seed(seed)
    model.train(); t0 = time.time(); losses = []
    amp = dev == "cuda"
    for step in range(updates):
        tot = 0.0
        for _ in range(ACC):
            pr = None
            while pr is None:
                pr = problem(rng.choice(pool), tok, dev)
            b, ids, spans, _ = pr
            st = layout(b, ids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(input_ids=ids, patch_lengths=lengths_from_starts(st, ids.shape[1]).to(dev), use_cache=False)
            loss = torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), ids[0, 1:]) / ACC
            loss.backward(); tot += loss.item()
            del out, loss
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot)
        if dev == "mps":
            torch.mps.empty_cache()
        if (step + 1) % max(1, updates // 20) == 0:
            print(f"  update {step + 1}/{updates} loss {np.mean(losses[-max(1, updates // 20):]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    if os.environ.get("SEGR_FT_SAVE") == "1":
        os.makedirs(OUT, exist_ok=True)
        torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f"{OUT}/{rule}{int(R * 100)}_s{seed}.pt")

    # teacher-forced scoring under the model's own rule (as realblt_budget.py)
    tf = {"res_exact": [], "fin_exact": [], "rate": []}
    for p in [json.loads(l) for l in open(GSM)][:n_test]:
        pr = problem(p, tok, dev)
        if pr is None:
            continue
        b, ids, spans, _ = pr
        st = np.array(sorted({int(x) for x in layout(b, ids) if 0 <= x < ids.shape[1]}))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, ids.shape[1]).to(dev),
                                         use_cache=False).logits[0].float(), -1)
        tf["rate"].append(len(st) / ids.shape[1])
        for kd, s, e_ in spans:
            tf[f"{kd}_exact"].append(bool((lp[s - 1:e_ - 1].argmax(-1) == ids[0, s:e_]).all()))

    # end to end: greedy generation, boundaries decided online
    def next_byte(b):
        ids = torch.tensor([[1] + [x + 4 for x in b + b"\0"]], device=dev)
        n = ids.shape[1] - 1
        st = np.array(sorted({int(x) for x in layout(b + b"\0", ids) if 0 <= x < n + 1}))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg = model(input_ids=ids, patch_lengths=lengths_from_starts(st, n + 1).to(dev), use_cache=False).logits[0, n - 1]
        return int(lg[4:260].float().argmax()), len(st) / (n + 1)

    e2e = {"correct": [], "rate": [], "pred": []}
    t1 = time.time()
    for j, p in enumerate([json.loads(l) for l in open(GSM)][:n_e2e]):
        gold = number(p["answer"].rsplit("#### ", 1)[1])
        ctx = f"{p['question']}\n".encode(); gen = b""; rates = []
        for _ in range(MAX_NEW):
            nb, rate = next_byte(ctx + gen)
            gen += bytes([nb]); rates.append(rate)
            if re.search(rb"The final answer is [^\n]*\n", gen) or gen.endswith(b"\n\n"):
                break
        m = re.search(rb"The final answer is ([^\n]*)", gen)
        pred = number(m.group(1).decode(errors="replace")) if m else None
        e2e["correct"].append(pred is not None and gold is not None and float(pred) == float(gold))
        e2e["rate"].append(float(np.mean(rates))); e2e["pred"].append(pred)
        if (j + 1) % 25 == 0:
            print(f"  e2e {j + 1}/{n_e2e}: {np.mean(e2e['correct']):.1%} ({time.time() - t1:.0f}s)", flush=True)

    name = f"{rule}@{int(round(R * 100))}"
    line = (f"RESULT fullft {name:12s} seed={seed} updates={updates} | teacher-forced: results exact "
            f"{np.mean(tf['res_exact']):5.1%}, final exact {np.mean(tf['fin_exact']):5.1%}, rate {np.mean(tf['rate']):.3f} | "
            f"end-to-end GSM8K {np.mean(e2e['correct']):5.1%} (n={len(e2e['correct'])}, rate {np.mean(e2e['rate']):.3f}) | "
            f"final train loss {np.mean(losses[-50:]):.4f}")
    if TOY:
        print(line, flush=True)
    else:
        registry.emit("realblt_fullft", line, experiment="blt_fullft", rule=rule, budget=R, updates=updates, seed=seed,
                      window=blt_load.WINDOW)
        registry.save_items(f"blt_fullft_{rule}{int(R * 100)}_s{seed}{blt_load.SUFFIX}", {"teacher_forced": tf, "e2e": e2e},
                            experiment="blt_fullft", rule=rule, budget=R, updates=updates, seed=seed)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]) if len(a) > 1 else 0.10, int(a[2]) if len(a) > 2 else 1800,
         int(a[3]) if len(a) > 3 else 300, int(a[4]) if len(a) > 4 else 500, int(a[5]) if len(a) > 5 else 0)
