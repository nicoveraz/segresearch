#!/bin/bash
# 50M runs of the new rules (local follow-up to the scaling pilot #27): novelty nov10, typed uncertainty typed14, and
# typed14 with the final-answer cue (final+typed14), 3 seeds each. typed14, not typed10: fitted on this corpus (90% OpenWebMath), typed uncertainty
# at 10% starts patches on only 6.7% of the GSM8K/MATH eval text; 14% gives 10.6% there, matching the other arms'
# eval rates (9.7-12.1%). It trains with more patch starts (14%); the eval budget is the matched one. On a fresh pod, from the unpacked bundle directory:
#
#     nohup bash scale/cloud/extra50.sh > extra.out 2>&1 &
#     DRYRUN=1 BUNDLE=... WORK=... bash scale/cloud/extra50.sh      # Mac dry run on a toy bundle (no CUDA, no installs)
#
# Rebuilds the corpus (SHA-256 checked) and masks, adds the nov10, typed14 and final+typed14 arms, runs the nine runs three at a
# time, and packs $WORK/results_all.tgz, then writes $WORK/DONE_ALL (what the Mac's download-and-terminate loop
# waits for). Resumable: rerunning skips finished work.
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${BUNDLE:-$PWD/bundle}
W=${WORK:-/workspace/work}
PAR=${PAR:-3}
mkdir -p "$W/logs"
exec > >(tee -a "$W/session.log") 2>&1
echo "== setup $(date)"
if [ "${DRYRUN:-0}" = 1 ]; then
  export SEGR_SCALE_TOY=1
else
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.__version__, torch.cuda.get_device_name())"
  pip install -q numpy pyarrow
fi
echo "== corpus and masks $(date)"
python -m scale.prep pod "$B" "$W"
python -m scale.prep arms "$B" "$W" nov10 typed14 final+typed14
echo "== runs $(date)"
# one seed of each rule per wave of three. typed14 vs final+typed14 isolates the answer cue: the pod's entropy model
# predicts final answers, so typed uncertainty alone covers 0.9% of them (final+typed14: 100%), as on BLT-1B.
# Left out for budget: the jump rule's seeds 1-2, syntax+entropy10 seed 2, entropy20 seeds 1-2.
printf '%s\n' "50m nov10 0" "50m typed14 0" "50m final+typed14 0" "50m nov10 1" "50m typed14 1" "50m final+typed14 1" \
    "50m nov10 2" "50m typed14 2" "50m final+typed14 2" |
  xargs -P "$PAR" -L 1 bash -c 'python -m scale.prep run "'"$B"'" "'"$W"'" $0 $1 $2 > "'"$W"'/logs/$0_$1_s$2.log" 2>&1 && echo "  done $0 $1 s$2" || echo "  FAILED $0 $1 s$2"'
echo "== packing $(date)"
cp results/registry/scale.jsonl "$W/registry_scale.jsonl" 2>/dev/null || true
tar czf "$W/results_all.tgz.tmp" --exclude='*.pt' --exclude='*.tmp' -C "$W" runs logs rules.json session.log \
    $([ -f "$W/registry_scale.jsonl" ] && echo registry_scale.jsonl) && mv "$W/results_all.tgz.tmp" "$W/results_all.tgz"
find "$W/runs" -name result.json | wc -l | xargs echo "  results:"
touch "$W/DONE_ALL"
echo "== DONE $(date)"
