"""Screen label-free signals on BLT-1B's own entropy model: which ranks GSM8K in-line results after '= ' highest?
Byte kinds: digit, lowercase, uppercase, whitespace, other ASCII, non-ASCII. For the distribution predicting token t:
H = total entropy, Hk = entropy over kinds, Hv = H - Hk (uncertainty within the kind)."""
import json, re, os, numpy as np, torch
from transformers import AutoTokenizer, BltForCausalLM
tok = AutoTokenizer.from_pretrained("itazap/blt-1b-hf")
model = BltForCausalLM.from_pretrained("itazap/blt-1b-hf", dtype=torch.bfloat16).to("mps").eval(); cfg = model.config
kind = np.full(260, 5)
for byte in range(256):
    c = bytes([byte])
    kind[byte + 4] = 0 if c.isdigit() else 1 if c.islower() else 2 if c.isupper() else 3 if c in b" \n\t\r" else 4 if byte < 128 else 5
K = torch.tensor(kind, device="mps")
probs = [json.loads(l) for l in open(os.path.expanduser("~/.cache/segresearch-gsm8k/test.jsonl"))][:300]
rows = []
for p in probs:
    sol, ans = p["answer"].rsplit("#### ", 1); sol = re.sub(r"<<[^>]*>>", "", sol).strip()
    text = p["question"] + "\n" + sol + "\nThe final answer is " + ans.strip(); b = text.encode()
    ids = tok(text, return_tensors="pt").input_ids.to("mps")
    if ids.shape[1] > 1000 or ids.shape[1] != len(b) + 1: continue
    with torch.no_grad():
        _, _, logits = model.model.patcher(ids, patch_size=cfg.patch_size, threshold=cfg.patching_threshold, max_patch_length=cfg.max_patch_length)
    lp = torch.log_softmax(logits[0].float(), -1); pr = lp.exp()
    H = -(pr * lp).sum(-1)
    pk = torch.zeros(pr.shape[0], 6, device="mps").index_add_(1, K, pr)
    Hk = -(pk * torch.log(pk.clamp_min(1e-12))).sum(-1)
    H, Hk = H.cpu().numpy(), Hk.cpu().numpy()
    res = {m.start() + 2 for m in re.finditer(rb"= \$?\d", b)}
    for i in range(2, len(b)):
        t = i + 1; d = t - 1                       # distribution predicting token t sits at position t-1
        lab = "result" if i in res else ("word" if b[i - 1] == 32 and b[i] != 32 else "other")
        rows.append((lab, H[d], Hk[d], H[d] - Hk[d]))
lab = np.array([r[0] for r in rows]); H = np.array([r[1] for r in rows]); Hk = np.array([r[2] for r in rows]); Hv = np.array([r[3] for r in rows])
signals = {"entropy H (BLT)": H, "within-kind Hv": Hv, "Hv - Hk": Hv - Hk, "Hv * (Hk < 0.5)": Hv * (Hk < 0.5)}
print("medians (nats): " + ", ".join(f"{l}: H {np.median(H[lab == l]):.2f} Hk {np.median(Hk[lab == l]):.2f} Hv {np.median(Hv[lab == l]):.2f}" for l in ("result", "word", "other")))
for name, s in signals.items():
    out = []
    for r in (0.15, 0.10, 0.06):
        thr = np.quantile(s, 1 - r); out.append(f"{r:.0%}: result {np.mean(s[lab == 'result'] > thr):5.1%} word {np.mean(s[lab == 'word'] > thr):5.1%}")
    print(f"{name:18s} | " + " | ".join(out))
