# Scaling pilot (#27): does the gap hold at 1M, 12M and 50M parameters?

Small BLT-style models trained from scratch at a 10% patch budget under three rules (BLT's entropy, label-free
boundary dependence, the hand-written results rule), at three sizes, on an OpenWebMath slice plus GSM8K/MATH
training solutions. Evaluation is the paper's: GSM8K/MATH test problems, exact match given the true prefix with
boundaries decided online, plus an end-to-end GSM8K check.

The code is a PyTorch port of the MLX harness, so the same code runs on the Mac (MPS) and on a rented GPU (CUDA).

| file | role |
|---|---|
| `model.py` | BLT-lite and the entropy model (global model runs over real patches only) |
| `train.py` | training loop: AdamW + warmup-cosine, bf16 on CUDA, gradient accumulation, checkpoint/resume |
| `rules.py`, `patchers.py` | patch rules; entropy model training and scoring; boundary-dependence fit |
| `evaluate.py` | bits per byte, online-boundary exact match, end-to-end GSM8K |
| `data.py` | OpenWebMath download (fixed shards, SHA-256), contamination filter, deterministic corpus |
| `prep.py` | driver: `mac`, `pod`, `bench`, `run` |
| `cloud/pack.sh`, `cloud/run.sh` | archive for the GPU; one unattended GPU session |
| `check_port.py`, `check_rules.py`, `check_patchers.py`, `check_eval.py` | Mac-only checks against the MLX code |

## Checks against the MLX code (run on the Mac)

```bash
.venv-scale/bin/python -m scale.check_port      # logits (2.6e-6), causality, training equivalence
.venv-scale/bin/python -m scale.check_rules     # masks identical to mathexp.Rule
.venv-scale/bin/python -m scale.check_patchers  # entropy scoring matches prepare.score_stream
.venv-scale/bin/python -m scale.check_eval      # bits and exact-match counts match mathexp
```

`.venv-scale` holds torch, numpy, pyarrow and mlx (`uv venv .venv-scale && uv pip install -p .venv-scale torch numpy pyarrow mlx`).

## Running it

1. **Mac** (free): build the bundle (corpus check, entropy model, dependence table), then pack it.
   ```bash
   .venv-scale/bin/python -m scale.prep mac ~/.cache/segresearch-scale/bundle
   bash scale/cloud/pack.sh ~/.cache/segresearch-scale/bundle scale_pilot.tgz
   ```
2. **RunPod**: rent one community RTX 4090 pod with a PyTorch template and enough container disk (40 GB). No
   network volume (it is billed while stopped). Copy `scale_pilot.tgz` to the pod (`runpodctl send` on the Mac,
   `runpodctl receive <code>` on the pod, or `scp` through the pod's SSH).
3. **On the pod**:
   ```bash
   tar xzf scale_pilot.tgz && cd pilot && nohup bash scale/cloud/run.sh > run.out 2>&1 &
   ```
   It builds the corpus (checked against the Mac's SHA-256), the masks, times one update per size, then runs the
   1M, 12M and 50M stages. It skips the 50M stage if a run is projected above `MAX_HOURS_50M` (default 3 h).
   Finished when `/workspace/work/DONE` exists; results are in `/workspace/work/results.tgz`.
4. Copy `results.tgz` back to the Mac, then **terminate** the pod (not just stop it).

Toy dry run of the whole pipeline on the Mac: `SEGR_SCALE_TOY=1` with the same commands (separate bundle/work dirs).
