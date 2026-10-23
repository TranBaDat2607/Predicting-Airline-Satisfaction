#!/usr/bin/env bash
# Push Kaggle GPU kernels in order, each as soon as kgkit's preflight (quota, 2-session limit, inputs) accepts it.
#   bash src/gpu_queue.sh [--not-before 2026-10-24T00:03:00Z] dir:hours [dir:hours ...]
# Logs to artifacts/gpu_queue.log. A refused push is retried every 5 minutes.
set -u
KG=${KG:-$HOME/.claude/plugins/cache/claude-code-kaggle-grandmaster/kaggle-grandmaster/1.6.2}
LOG=artifacts/gpu_queue.log
mkdir -p artifacts
if [ "${1:-}" = "--not-before" ]; then                 # e.g. --not-before 2026-10-24T00:03:00Z
  target=$(date -u -d "$2" +%s)
  until [ "$(date -u +%s)" -ge "$target" ]; do sleep 30; done
  shift 2
fi
for item in "$@"; do
  dir=${item%%:*}; hours=${item##*:}
  while true; do
    out=$(PYTHONPATH=$KG python -m kgkit gpu push "$dir" --hours "$hours" 2>&1 | grep -v Warning)
    echo "$(date -u +%FT%TZ) $dir: $out" >> "$LOG"
    echo "$out" | grep -q "successfully pushed" && break
    sleep 300
  done
done
echo "$(date -u +%FT%TZ) queue done" >> "$LOG"
