"""Label-free 'difficulty ahead' signal for BLT-1B. Fit on GSM8K TRAIN: for each position, the main model's mean
loss on the next 4 bytes (default patches); average it per preceding 2-byte context (backoff to 1 byte).
At inference the score at t is a lookup on bytes t-2, t-1: causal, no labels. Screen coverage on TEST."""
import json, re, os, numpy as np, torch
from collections import defaultdict
from transformers import AutoTokenizer, BltForCausalLM
tok = AutoTokenizer.from_pretrained("itazap/blt-1b-hf")
model = BltForCausalLM.from_pretrained("itazap/blt-1b-hf", dtype=torch.bfloat16).to("mps").eval(); cfg = model.config
def load(split, n):
    out = []
    for p in [json.loads(l) for l in open(os.path.expanduser(f"~/.cache/segresearch-gsm8k/{split}.jsonl"))][:n]:
        sol, ans = p["answer"].rsplit("#### ", 1); sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        out.append(p["question"] + "\n" + sol + "\nThe final answer is " + ans.strip())
    return out
def run(text):
    ids = tok(text, return_tensors="pt").input_ids.to("mps")
    with torch.no_grad():
        ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold, max_patch_length=cfg.max_patch_length)
        lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), -1)
    loss = -lp[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()     # loss[j] = loss on token j+1
    return ids.shape[1], ent[0].float().cpu().numpy(), loss
s2, s1 = defaultdict(list), defaultdict(list)
for text in load("train", 500):
    b = text.encode(); n, e, loss = run(text)
    if n != len(b) + 1 or n > 1000: continue
    for i in range(2, len(b) - 4):
        t = i + 1; d = float(loss[t - 1:t + 3].mean())                   # loss on tokens t..t+3 (bytes i..i+3)
        s2[(b[i - 2], b[i - 1])].append(d); s1[b[i - 1]].append(d)
T2 = {k: np.mean(v) for k, v in s2.items() if len(v) >= 20}; T1 = {k: np.mean(v) for k, v in s1.items() if len(v) >= 20}
glob = np.mean([x for v in s1.values() for x in v])
lookup = lambda b, i: T2.get((b[i - 2], b[i - 1]), T1.get(b[i - 1], glob))
print("fitted contexts:", len(T2), "| difficulty after '= ':", round(T2.get((61, 32), float('nan')), 2), "after 'e ':", round(T2.get((101, 32), float('nan')), 2), "global", round(glob, 2))
rows = []
for text in load("test", 300):
    b = text.encode(); n, e, _ = run(text)
    if n != len(b) + 1 or n > 1000: continue
    res = {m.start() + 2 for m in re.finditer(rb"= \$?\d", b)}
    for i in range(2, len(b)):
        lab = "result" if i in res else ("word" if b[i - 1] == 32 and b[i] != 32 else "other")
        rows.append((lab, e[i], lookup(b, i)))                            # entropy deciding token t=i+1 sits at t-1=i
lab = np.array([r[0] for r in rows]); H = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows])
z = lambda x: (x - x.mean()) / x.std()
for name, s in {"entropy (BLT)": H, "difficulty ahead": D, "entropy + difficulty": z(H) + z(D)}.items():
    print(f"{name:22s} | " + " | ".join(f"{r:.0%}: result {np.mean(s[lab == 'result'] > np.quantile(s, 1 - r)):5.1%} word {np.mean(s[lab == 'word'] > np.quantile(s, 1 - r)):5.1%}" for r in (0.15, 0.10, 0.06)))
np.save(os.path.expanduser("~/.cache/segresearch-blt/difficulty_table.npy"), {"T2": T2, "T1": T1, "glob": glob}, allow_pickle=True)
