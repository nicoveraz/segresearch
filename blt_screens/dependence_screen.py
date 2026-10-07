"""Label-free 'boundary dependence' signal for BLT-1B. On GSM8K TRAIN: loss of the main model on the next 4 bytes
under a tight layout (top 10% by entropy) minus under its default layout (~30%), averaged per preceding 2-byte
context (backoff to 1). How much those bytes need a boundary. Causal lookup at inference, no labels."""
import json, re, os, numpy as np, torch
from collections import defaultdict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import blt_load
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("itazap/blt-1b-hf")
model = blt_load.load("itazap/blt-1b-hf", "mps").eval(); cfg = model.config
def load(split, n):
    out = []
    for p in [json.loads(l) for l in open(os.path.expanduser(f"~/.cache/segresearch-gsm8k/{split}.jsonl"))][:n]:
        sol, ans = p["answer"].rsplit("#### ", 1); sol = re.sub(r"<<[^>]*>>", "", sol).strip()
        out.append(p["question"] + "\n" + sol + "\nThe final answer is " + ans.strip())
    return out
def L(st, n):
    s = sorted(set(int(x) for x in st if 0 <= x < n)); return torch.tensor([[b - a for a, b in zip(s, s[1:] + [n])]], device="mps")
def losses(ids, pl):
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=ids, patch_lengths=pl, use_cache=False).logits[0].float(), -1)
    return -lp[:-1].gather(1, ids[0, 1:, None])[:, 0].cpu().numpy()
s2, s1 = defaultdict(list), defaultdict(list)
for text in load("train", 400):
    b = text.encode(); ids = tok(text, return_tensors="pt").input_ids.to("mps"); n = ids.shape[1]
    if n != len(b) + 1 or n > 1000: continue
    with torch.no_grad():
        ent, dl, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold, max_patch_length=cfg.max_patch_length)
    e = ent[0].float().cpu().numpy(); score = np.full(n, -np.inf); score[2:] = e[1:n - 1]
    tight = np.r_[0, 1, np.argsort(-score)[:max(int(0.10 * n) - 2, 1)]]
    default = np.r_[0, np.cumsum(dl[0].cpu().numpy())[:-1]]
    gain = losses(ids, L(tight, n)) - losses(ids, L(default, n))
    for i in range(2, len(b) - 4):
        t = i + 1; d = float(gain[t - 1:t + 3].mean())
        s2[(b[i - 2], b[i - 1])].append(d); s1[b[i - 1]].append(d)
T2 = {k: float(np.mean(v)) for k, v in s2.items() if len(v) >= 20}; T1 = {k: float(np.mean(v)) for k, v in s1.items() if len(v) >= 20}
glob = float(np.mean([x for v in s1.values() for x in v]))
print("contexts:", len(T2), "| dependence after '= ':", round(T2.get((61, 32), float("nan")), 3), "after 'e ':", round(T2.get((101, 32), float("nan")), 3), "global", round(glob, 3))
top = sorted(T2.items(), key=lambda kv: -kv[1])[:10]
print("top contexts:", ", ".join(f"{bytes(k)!r} {v:.2f}" for k, v in top))
os.makedirs(os.path.expanduser("~/.cache/segresearch-blt"), exist_ok=True); np.save(os.path.expanduser(f"~/.cache/segresearch-blt/dependence_table{blt_load.SUFFIX}.npy"), {"T2": T2, "T1": T1, "glob": glob}, allow_pickle=True)
lookup = lambda b, i: T2.get((b[i - 2], b[i - 1]), T1.get(b[i - 1], glob))
rows = []
for text in load("test", 300):
    b = text.encode(); ids = tok(text, return_tensors="pt").input_ids.to("mps"); n = ids.shape[1]
    if n != len(b) + 1 or n > 1000: continue
    with torch.no_grad():
        ent, _, _ = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold, max_patch_length=cfg.max_patch_length)
    e = ent[0].float().cpu().numpy(); res = {m.start() + 2 for m in re.finditer(rb"= \$?\d", b)}
    for i in range(2, len(b)):
        lab = "result" if i in res else ("word" if b[i - 1] == 32 and b[i] != 32 else "other")
        rows.append((lab, e[i], lookup(b, i)))
lab = np.array([r[0] for r in rows]); H = np.array([r[1] for r in rows]); D = np.array([r[2] for r in rows])
z = lambda x: (x - x.mean()) / x.std()
for name, s in {"entropy (BLT)": H, "dependence": D, "entropy + dependence": z(H) + z(D)}.items():
    print(f"{name:22s} | " + " | ".join(f"{r:.0%}: result {np.mean(s[lab == 'result'] > np.quantile(s, 1 - r)):5.1%} word {np.mean(s[lab == 'word'] > np.quantile(s, 1 - r)):5.1%}" for r in (0.15, 0.10, 0.06)))
