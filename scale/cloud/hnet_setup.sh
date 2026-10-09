#!/bin/bash
# H-Net blind-spot check (#21): environment and checkpoints, on a CUDA pod. H-Net needs flash_attn, mamba_ssm and
# causal_conv1d (CUDA-only kernels), so this runs on the pod, next to the training runs:
#
#     bash scale/cloud/hnet_setup.sh && /workspace/hnetenv/bin/python scale/cloud/hnet_check.py
#
# Installs into its own venv (system site packages visible, torch pinned to the pod's version) so the training
# runs' environment is never changed. Prebuilt wheels are fetched where they exist; set MAX_JOBS low if a kernel
# has to compile, so it does not starve the training runs of CPU.
set -euo pipefail
W=${WORK:-/workspace}
ENV=$W/hnetenv
[ -d "$ENV" ] || python -m venv --system-site-packages "$ENV"
P=$ENV/bin/pip
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch==$TORCH" > "$W/hnet_constraints.txt"
export MAX_JOBS=${MAX_JOBS:-4}
$P install -q -c "$W/hnet_constraints.txt" einops optree regex omegaconf packaging ninja
$P install -q -c "$W/hnet_constraints.txt" --no-build-isolation causal-conv1d mamba-ssm flash-attn
[ -d "$W/hnet" ] || git clone -q https://github.com/goombalab/hnet "$W/hnet"
(cd "$W/hnet" && git rev-parse HEAD > "$W/hnet_commit.txt")
$P install -q --no-deps -e "$W/hnet"
$ENV/bin/python -c "import torch, flash_attn, mamba_ssm, hnet; print('ok', torch.__version__, flash_attn.__version__, mamba_ssm.__version__)"
mkdir -p "$W/hnet_ckpt"
for m in ${HNET_MODELS:-hnet_2stage_XL hnet_1stage_XL}; do
  [ -f "$W/hnet_ckpt/$m.pt" ] || curl -sSL -o "$W/hnet_ckpt/$m.pt.tmp" "https://huggingface.co/cartesia-ai/$m/resolve/main/$m.pt" \
    && mv -f "$W/hnet_ckpt/$m.pt.tmp" "$W/hnet_ckpt/$m.pt" 2>/dev/null || true
  ls -la "$W/hnet_ckpt/$m.pt"
done
