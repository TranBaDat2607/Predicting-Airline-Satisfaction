#!/usr/bin/env bash
# Validate -> submit -> wait for the public score -> log it on the ledger.
#   bash src/submit.sh subs/0005_x.csv "message"
set -euo pipefail
f=$1; msg=${2:-$(basename "$f" .csv)}
P=C:/Users/Admin/anaconda3/envs/airline-satisfaction/python.exe
K=C:/Users/Admin/anaconda3/envs/airline-satisfaction/Scripts/kaggle.exe
export PYTHONIOENCODING=utf-8 PYTHONPATH="$(pwd)/src"
"$P" -m kgkit validate "$f" data/sample_submission.csv
"$K" competitions submit playground-series-s6e10 -f "$f" -m "$msg"
name=$(basename "$f")
for _ in $(seq 1 60); do
  row=$("$K" competitions submissions playground-series-s6e10 --csv 2>/dev/null | grep -m1 ",$name," || true)
  case "$row" in *COMPLETE*|*ERROR*) break ;; esac
  sleep 10
done
echo "$row"
score=$(echo "$row" | awk -F, '{print $(NF-1)}')
id=$(basename "$f" .csv)
[ -n "$score" ] && "$P" -m kgkit ledger lb "$id" "$score"
