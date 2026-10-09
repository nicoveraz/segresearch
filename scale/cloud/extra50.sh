#!/bin/bash
# Extra 50M seeds for the scaling pilot (#27): the single-seed arms of the paper (BLT's jump rule, entropy at 20%,
# and the hand-written rule's third seed). On a fresh pod, from the unpacked bundle directory:
#
#     nohup bash scale/cloud/extra50.sh > extra.out 2>&1 &
#
# Rebuilds the corpus (SHA-256 checked) and masks, adds the jump10 and entropy20 arms, runs the five runs three at a
# time, and packs $WORK/results_extra.tgz, then writes $WORK/DONE_ALL (what the Mac's download-and-terminate loop
# waits for). Resumable: rerunning skips finished work.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${BUNDLE:-$PWD/bundle}
W=${WORK:-/workspace/work}
PAR=${PAR:-3}
mkdir -p "$W/logs"
exec > >(tee -a "$W/session.log") 2>&1
echo "== setup $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.__version__, torch.cuda.get_device_name())"
pip install -q numpy pyarrow
echo "== corpus and masks $(date)"
python -m scale.prep pod "$B" "$W"
python -m scale.prep arms "$B" "$W" jump10 entropy20
echo "== runs $(date)"
printf '%s\n' "50m jump10 1" "50m entropy20 1" "50m syntax+entropy10 2" "50m jump10 2" "50m entropy20 2" |
  xargs -P "$PAR" -L 1 bash -c 'python -m scale.prep run "'"$B"'" "'"$W"'" $0 $1 $2 > "'"$W"'/logs/$0_$1_s$2.log" 2>&1 && echo "  done $0 $1 s$2" || echo "  FAILED $0 $1 s$2"'
echo "== packing $(date)"
cp results/registry/scale.jsonl "$W/registry_scale.jsonl" 2>/dev/null || true
tar czf "$W/results_all.tgz.tmp" --exclude='*.pt' --exclude='*.tmp' -C "$W" runs logs rules.json session.log \
    $([ -f "$W/registry_scale.jsonl" ] && echo registry_scale.jsonl) && mv "$W/results_all.tgz.tmp" "$W/results_all.tgz"
find "$W/runs" -name result.json | wc -l | xargs echo "  results:"
touch "$W/DONE_ALL"
echo "== DONE $(date)"
