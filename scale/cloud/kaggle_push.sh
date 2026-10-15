#!/bin/bash
# Run the scaling runs of the new rules on Kaggle (GPU T4 x2) from the Mac's terminal (local follow-up to #27).
#
#     bash scale/cloud/kaggle_push.sh upload                     # pack code + bundle, create/update the private dataset
#     bash scale/cloud/kaggle_push.sh run "1m nov10 0" "1m typed14 0" ...   # push a session that runs these
#     bash scale/cloud/kaggle_push.sh status                     # running / complete / error
#     bash scale/cloud/kaggle_push.sh logs [N]                   # last N lines of the running session's log
#     bash scale/cloud/kaggle_push.sh pull                       # download the session's results
#
# Needs the Kaggle CLI with the user's own token in ~/.kaggle/access_token (not handled here) and KAGGLE_USER set to
# the Kaggle username. Everything is private: the dataset
# and the kernel. Each session rebuilds the corpus and masks (~1 h on T4) and then runs two at a time until ~11 h.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"                  # uv tool install puts the kaggle CLI here
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
U=${KAGGLE_USER:-$(kaggle config view 2>/dev/null | awk -F': ' '/username/ {print $2}')}
[ -n "$U" ] || { echo "Kaggle CLI not configured (kaggle config view shows no username)"; exit 1; }
DS="$U/segresearch-scale-bundle"; K="$U/segresearch-scale-t4"
STAGE=${KAGGLE_STAGE:-$HOME/.cache/segresearch-scale/kaggle}
ARMS=${ARMS:-"nov10 typed14 final+typed14"}
mkdir -p "$STAGE/dataset" "$STAGE/kernel" "$STAGE/out"

case "${1:-}" in
upload)
  bash "$ROOT/scale/cloud/pack.sh" "$HOME/.cache/segresearch-scale/bundle" "$STAGE/dataset/pilot.tgz"
  cat > "$STAGE/dataset/dataset-metadata.json" <<EOF
{"title": "segresearch-scale-bundle", "id": "$DS", "licenses": [{"name": "other"}]}
EOF
  if kaggle datasets status "$DS" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE/dataset" -m "code $(cd "$ROOT" && git rev-parse --short HEAD)"
  else
    kaggle datasets create -p "$STAGE/dataset"        # private by default
  fi ;;
run)
  shift
  [ $# -gt 0 ] || { echo "give the runs, e.g. \"1m nov10 0\""; exit 1; }
  runs=$(printf '%s\n' "$@" | awk '{printf "(\"%s\", \"%s\", %s), ", $1, $2, $3}')
  arms=$(printf '"%s", ' $ARMS)
  sed -e "s|^RUNS = \[\].*|RUNS = [${runs}]|" -e "s|^ARMS = \[\].*|ARMS = [${arms}]|" \
      "$ROOT/scale/cloud/kaggle_job.py" > "$STAGE/kernel/kaggle_job.py"
  cat > "$STAGE/kernel/kernel-metadata.json" <<EOF
{"id": "$K", "title": "segresearch-scale-t4", "code_file": "kaggle_job.py", "language": "python",
 "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
 "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["$DS"], "competition_sources": [], "kernel_sources": [],
 "model_sources": []}
EOF
  grep -n "^RUNS\|^ARMS" "$STAGE/kernel/kaggle_job.py"
  kaggle kernels push -p "$STAGE/kernel" ;;
generic)                                               # novelty vs entropy outside math (scale/generic.py)
  doms=$(printf '"%s", ' ${GEN_DOMAINS:-code prose}); runs=$(printf '%s\n' ${GEN_RUNS:-entropy10:0 novelty:0 entropy10:1 novelty:1} | awk -F: '{printf "(\"%s\", %s), ", $1, $2}')
  sed -e "s|^DOMAINS = \[.*|DOMAINS = [${doms}]|" -e "s|^RUNS = \[.*|RUNS = [${runs}]|" \
      -e "s|^ENT_BUDGETS = .*|ENT_BUDGETS = \"${GEN_ENT_BUDGETS:-}\"|" -e "s|^NOVNET = .*|NOVNET = \"${GEN_NOVNET:-}\"|" \
      "$ROOT/scale/cloud/kaggle_generic_job.py" > "$STAGE/kernel/kaggle_job.py"
  grep -n "^DOMAINS\|^RUNS" "$STAGE/kernel/kaggle_job.py"
  printf '%s\n' "{\"id\": \"$K\", \"title\": \"segresearch-scale-t4\", \"code_file\": \"kaggle_job.py\"," \
    "\"language\": \"python\", \"kernel_type\": \"script\", \"is_private\": \"true\", \"enable_gpu\": \"true\"," \
    "\"enable_internet\": \"true\", \"machine_shape\": \"NvidiaTeslaT4\", \"dataset_sources\": [\"$DS\"]," \
    "\"competition_sources\": [], \"kernel_sources\": [], \"model_sources\": []}" > "$STAGE/kernel/kernel-metadata.json"
  kaggle kernels push -p "$STAGE/kernel" ;;
cross)                                                 # train with novelty, patch with other rules (kaggle_cross_job.py)
  cp "$ROOT/scale/cloud/kaggle_cross_job.py" "$STAGE/kernel/kaggle_job.py"
  printf '%s\n' "{\"id\": \"$K\", \"title\": \"segresearch-scale-t4\", \"code_file\": \"kaggle_job.py\"," \
    "\"language\": \"python\", \"kernel_type\": \"script\", \"is_private\": \"true\", \"enable_gpu\": \"true\"," \
    "\"enable_internet\": \"true\", \"machine_shape\": \"NvidiaTeslaT4\", \"dataset_sources\": [\"$DS\"]," \
    "\"competition_sources\": [], \"kernel_sources\": [], \"model_sources\": []}" > "$STAGE/kernel/kernel-metadata.json"
  kaggle kernels push -p "$STAGE/kernel" ;;
status)
  kaggle kernels status "$K" ;;
logs)
  kaggle kernels logs "$K" 2>&1 | grep -o '"data":"[^"]*"' | sed 's/"data":"//; s/"$//; s/\\n/\n/g' | grep -v "^$" | tail -${2:-30} ;;
pull)
  d="$STAGE/out/$(date -u +%Y%m%d_%H%M)"; mkdir -p "$d"
  kaggle kernels output "$K" -p "$d"
  echo "-> $d"; ls "$d" "$d/results" 2>/dev/null | head -30 ;;
*)
  sed -n 2,11p "$0"; exit 1 ;;
esac
