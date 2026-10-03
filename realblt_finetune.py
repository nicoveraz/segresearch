"""Adapt Meta's BLT-1B to a tight patch budget with a light fine-tune (issue #11), then score it.

BLT-1B was trained with patches on ~28% of bytes; the other BLT-1B scripts only change the layout at inference.
Here low-rank adapters (LoRA, r=16) on the global transformer and the decoder's cross-attention (the parts that make
and read patch states) are trained on GSM8K TRAIN solutions under one tight layout rule, so the model can adapt to
that budget. The weights themselves stay frozen. Layout rules at budget R (blt_layouts.py, train-fitted thresholds):

    entropy      BLT-1B's entropy                         (BLT's rule)
    entropy+dep  entropy + boundary dependence            (label-free)
    forced       a boundary after every '= ', rest entropy (hand-written results rule)

Training never sees the test split. After training, the script scores the adapted model on GSM8K TEST exactly as
realblt_budget.py does (in-line results after '= ' and final answers, exact match given the true prefix), under
its own training layout and under the other two at the same R.

    <env>/bin/python realblt_finetune.py RULE [R] [STEPS] [N_TEST] [SEED]   e.g. entropy 0.10 1500 300 0
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BltForCausalLM

import blt_layouts
import registry
from realblt_budget import DEP, GSM, GSM_TRAIN, MODEL, lengths_from_starts, problem

N_FIT = 300                                   # train problems used to fit budget thresholds (not trained on)
OUT = os.path.expanduser("~/.cache/segresearch-blt/adapters")
RULES = {"entropy": "entropy", "entdep": "entropy+dep", "results": "forced"}


def main(rule, R, steps, n_test, seed=0):
    assert blt_layouts.MODE == "train"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = BltForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    cfg = model.config
    patcher = model.model.patcher                 # kept before the LoRA wrapper changes model.model
    tab = np.load(DEP, allow_pickle=True).item()
    dep_at = lambda b, i: tab["T2"].get((b[i - 2], b[i - 1]), tab["T1"].get(b[i - 1], tab["glob"])) if i >= 2 else -np.inf

    def signals(b, ids):
        n = ids.shape[1]
        with torch.no_grad():
            ent, default_len, _ = patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold,
                                                      max_patch_length=cfg.max_patch_length)
        e = ent[0].float().cpu().numpy()
        score = np.full(n, -np.inf); score[2:] = e[1:n - 1]
        dep = np.full(n, -np.inf); dep[2:] = [dep_at(b, t - 1) for t in range(2, n)]
        return {"entropy": score, "dep": dep}, np.r_[0, np.cumsum(default_len[0].cpu().numpy())[:-1]]

    train = [json.loads(l) for l in open(GSM_TRAIN)]
    fit = blt_layouts.Fit((R,))
    for p in train[:N_FIT]:
        pr = problem(p, tok, dev)
        if pr is not None:
            fit.add(signals(pr[0], pr[1])[0], pr[3])
    fit.done()
    kind = RULES[rule]
    print(f"thresholds fitted on {len(fit.sigs)} train problems; {rule}@{R:.2f} train rate {fit.rate(kind, R):.3f}", flush=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=r".*(global_transformer\.layers\.\d+\.(self_attn|mlp)|local_decoder\.cross_attn_layers\.\d+)\.(" + "|".join(targets) + ")")
    torch.manual_seed(seed)                       # LoRA initialization
    model = get_peft_model(model, lcfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA: {n_tr / 1e6:.1f}M trainable parameters", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.1, 1 - s / steps))
    pool = [p for p in train[N_FIT:]]
    rng = random.Random(seed)                     # training data order
    model.train(); t0 = time.time(); losses = []
    for step in range(steps):
        pr = None
        while pr is None:
            pr = problem(rng.choice(pool), tok, dev)
        b, ids, spans, res_starts = pr
        n = ids.shape[1]
        sig, _ = signals(b, ids)
        st = blt_layouts.layout(kind, R, sig, fit, res_starts if kind == "forced" else ())
        out = model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev), use_cache=False)
        loss = torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), ids[0, 1:])
        loss.backward(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        del out, loss
        if dev == "mps":
            torch.mps.empty_cache()                                       # varying lengths fragment the MPS cache
        if (step + 1) % max(1, steps // 15) == 0:
            print(f"  step {step + 1}/{steps} loss {np.mean(losses[-max(1, steps // 15):]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    tag = f"{rule}{int(round(R * 100))}_s{seed}"
    model.save_pretrained(os.path.join(OUT, tag))

    # score on GSM8K test under each rule at the same budget (as realblt_budget.py)
    model.eval()
    layouts = [f"{k}@{int(round(R * 100))}" for k in RULES]
    stats = {l: {"res_exact": [], "fin_exact": [], "res_bits": [], "fin_bits": [], "rate": []} for l in layouts}
    res_pid, used = [], 0
    for p in [json.loads(l) for l in open(GSM)][:n_test]:
        pr = problem(p, tok, dev)
        if pr is None:
            continue
        b, ids, spans, res_starts = pr
        n = ids.shape[1]; used += 1; res_pid += [used - 1] * len(res_starts)
        sig, _ = signals(b, ids)
        for name in layouts:
            k = RULES[name.split("@")[0]]
            st = blt_layouts.layout(k, R, sig, fit, res_starts if k == "forced" else ())
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids, patch_lengths=lengths_from_starts(st, n).to(dev), use_cache=False).logits[0].float(), -1)
            stats[name]["rate"].append(len(st) / n)
            for kd, s, e_ in spans:
                tgt = ids[0, s:e_]; pred = lp[s - 1:e_ - 1]
                stats[name][f"{kd}_bits"].append(float(-pred.gather(1, tgt[:, None]).mean() / np.log(2)))
                stats[name][f"{kd}_exact"].append(bool((pred.argmax(-1) == tgt).all()))
    print(f"\nBLT-1B + LoRA trained under {rule}@{R:.2f} ({steps} steps), {used} GSM8K test problems")
    for name in layouts:
        s = stats[name]
        registry.emit("realblt_finetune", f"RESULT {name:11s} patch rate {np.mean(s['rate']):.3f} | "
                      f"in-line results: {np.mean(s['res_bits']):.3f} bits, exact {np.mean(s['res_exact']):5.1%} | "
                      f"final: {np.mean(s['fin_bits']):.3f} bits, exact {np.mean(s['fin_exact']):5.1%}",
                      experiment="blt_finetune", trained_rule=rule, budget=R, steps=steps, seed=seed, thresh=blt_layouts.MODE)
    registry.save_items(f"blt_finetune_{tag}", {"layouts": stats, "res_pid": res_pid}, experiment="blt_finetune",
                        trained_rule=rule, budget=R, steps=steps, seed=seed)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]) if len(a) > 1 else 0.10, int(a[2]) if len(a) > 2 else 1500, int(a[3]) if len(a) > 3 else 300,
         int(a[4]) if len(a) > 4 else 0)
