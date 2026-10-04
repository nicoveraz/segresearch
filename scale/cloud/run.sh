#!/bin/bash
# One unattended GPU session for the scaling pilot (#27). On the pod, from the unpacked bundle directory:
#
#     bash scale/cloud/run.sh            # everything: setup, corpus + masks, throughput check, the grid
#     STAGES="1m 12m" bash scale/cloud/run.sh     # only some stages
#     DRYRUN=1 WORK=/tmp/w STAGES=1m bash scale/cloud/run.sh   # Mac dry run on a toy bundle
#
# Writes everything under $WORK (default /workspace/work) and finishes with $WORK/DONE and
# $WORK/results.tgz (result.json and log of every run, the registry lines, bench.json). Every step is resumable:
# rerunning skips finished work. Nothing here needs credentials.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${BUNDLE:-$PWD/bundle}
W=${WORK:-/workspace/work}
STAGES=${STAGES:-"1m 12m 50m"}
MAX50=${MAX_HOURS_50M:-3.0}          # stop before the 50M stage if one run is projected to take longer
PAR12=${PAR12:-3}
PAR50=${PAR50:-2}
mkdir -p "$W/logs"
exec > >(tee -a "$W/session.log") 2>&1

echo "== setup $(date)"
if [ "${DRYRUN:-0}" = 1 ]; then          # Mac dry run: toy sizes, no CUDA check, no installs
  export SEGR_SCALE_TOY=1
else
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.__version__, torch.cuda.get_device_name())"
  pip install -q numpy pyarrow
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo "== corpus and masks $(date)"
python -m scale.prep pod "$B" "$W"

echo "== throughput $(date)"
[ -f "$W/bench.json" ] || python -m scale.prep bench "$B" "$W"

jobs() {   # jobs PAR "size arm seed" ...
  local par=$1; shift
  printf '%s\n' "$@" | xargs -P "$par" -L 1 bash -c \
    'python -m scale.prep run "'"$B"'" "'"$W"'" $0 $1 $2 > "'"$W"'/logs/$0_$1_s$2.log" 2>&1 && echo "  done $0 $1 s$2" || echo "  FAILED $0 $1 s$2"'
}

for stage in $STAGES; do
  echo "== stage $stage $(date)"
  case $stage in
    1m)  jobs 2 "1m dep10 0" "1m entropy10 0" ;;
    12m) jobs "$PAR12" "12m entropy10 0" "12m dep10 0" "12m syntax+entropy10 0" \
                       "12m entropy10 1" "12m dep10 1" "12m syntax+entropy10 1" ;;
    50m) h=$(python -c "import json; print(json.load(open('$W/bench.json'))['50m']['hours_per_run'])")
         if python -c "import sys; sys.exit(0 if $h <= $MAX50 else 1)"; then
           jobs "$PAR50" "50m entropy10 0" "50m dep10 0" "50m entropy10 1" "50m dep10 1" "50m syntax+entropy10 0"
         else
           echo "  SKIPPED: a 50M run is projected at $h h (> $MAX50 h); decide before spending more"
         fi ;;
  esac
done

echo "== packing results $(date)"
cp results/registry/scale.jsonl "$W/registry_scale.jsonl" 2>/dev/null || true
tar czf "$W/results.tgz" --exclude='*.pt' --exclude='*.tmp' -C "$W" runs logs bench.json rules.json session.log \
    $([ -f "$W/registry_scale.jsonl" ] && echo registry_scale.jsonl)
find "$W/runs" -name result.json | wc -l | xargs echo "  results:"
touch "$W/DONE"
echo "== DONE $(date)   ->  $W/results.tgz"
