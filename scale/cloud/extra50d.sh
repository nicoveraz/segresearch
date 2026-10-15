#!/bin/bash
# 50M baselines entropy10 and dep10, seeds 1 and 2, clipped like the 50M nov10 runs (local follow-up). On a fresh pod,
# from the unpacked bundle directory:
#
#     nohup bash scale/cloud/extra50d.sh > extra_d.out 2>&1 &
#
# Rebuilds the corpus (SHA-256 checked), builds only these two masks, runs the four at once, packs
# $WORK/results_all.tgz and writes $WORK/DONE_ALL (what the Mac's download-and-terminate loop waits for). Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${BUNDLE:-$PWD/bundle}
W=${WORK:-/workspace/work}
mkdir -p "$W/logs"
exec > >(tee -a "$W/session.log") 2>&1
echo "== setup $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.__version__, torch.cuda.get_device_name())"
pip install -q numpy pyarrow
echo "== corpus and the entropy10 and dep10 masks $(date)"
python -c "from scale import prep; prep.ARMS = ('entropy10', 'dep10'); prep.pod('$B', '$W')"
echo "== runs $(date)"
export SEGR_SCALE_CLIP=1.0 SEGR_SCALE_TAG=_clip
printf "%s\n" "50m entropy10 1" "50m dep10 1" "50m entropy10 2" "50m dep10 2" |
  xargs -P 4 -L 1 bash -c 'python -m scale.prep run "'"$B"'" "'"$W"'" $0 $1 $2 > "'"$W"'/logs/$0_$1_s$2_clip.log" 2>&1 && echo "  done $0 $1 s$2" || echo "  FAILED $0 $1 s$2"'
echo "== packing $(date)"
cp results/registry/scale.jsonl "$W/registry_scale.jsonl" 2>/dev/null || true
tar czf "$W/results_all.tgz.tmp" --exclude='*.pt' --exclude='*.tmp' -C "$W" runs logs rules.json session.log \
    $([ -f "$W/registry_scale.jsonl" ] && echo registry_scale.jsonl) && mv "$W/results_all.tgz.tmp" "$W/results_all.tgz"
find "$W/runs" -name result.json | wc -l | xargs echo "  results:"
touch "$W/DONE_ALL"
echo "== DONE $(date)"
