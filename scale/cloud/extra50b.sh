#!/bin/bash
# 50M runs of the new rules with gradient clipping (norm 1.0), after typed14 + cue collapsed without it (local follow-up).
# All 9 new runs clipped, plus clipped check runs of the pilot's entropy10 and dep10 (seed 0): if they match the
# unclipped pilot, clipping is neutral and the comparison with the pilot's rules holds. Runs as runs/<name>_clip.
# Reuses the corpus and masks already built in $WORK (by extra50.sh). From the unpacked bundle directory:
#
#     nohup bash scale/cloud/extra50b.sh > extra_b.out 2>&1 &
#
# Finishes with $WORK/results_all.tgz and $WORK/DONE_ALL, as extra50.sh. Resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${BUNDLE:-$PWD/bundle}
W=${WORK:-/workspace/work}
PAR=${PAR:-3}
export SEGR_SCALE_CLIP=1.0 SEGR_SCALE_TAG=_clip
exec > >(tee -a "$W/session.log") 2>&1
echo "== clipped runs $(date)"
printf '%s\n' "50m nov10 0" "50m typed14 0" "50m final+typed14 0" "50m entropy10 0" "50m dep10 0" "50m nov10 1" \
    "50m typed14 1" "50m final+typed14 1" "50m nov10 2" "50m typed14 2" "50m final+typed14 2" |
  xargs -P "$PAR" -L 1 bash -c 'python -m scale.prep run "'"$B"'" "'"$W"'" $0 $1 $2 > "'"$W"'/logs/$0_$1_s$2_clip.log" 2>&1 && echo "  done $0 $1 s$2 clip" || echo "  FAILED $0 $1 s$2 clip"'
echo "== packing $(date)"
cp results/registry/scale.jsonl "$W/registry_scale.jsonl" 2>/dev/null || true
tar czf "$W/results_all.tgz.tmp" --exclude='*.pt' --exclude='*.tmp' -C "$W" runs logs rules.json session.log \
    $([ -f "$W/registry_scale.jsonl" ] && echo registry_scale.jsonl) && mv "$W/results_all.tgz.tmp" "$W/results_all.tgz"
find "$W/runs" -name result.json | wc -l | xargs echo "  results:"
touch "$W/DONE_ALL"
echo "== DONE $(date)"
