#!/usr/bin/env bash
# Validate -> submit -> wait for the public score -> log it on the ledger.
#   bash src/submit.sh subs/<exp_id>.csv "message"      (<exp_id> = a ledger id or name; created lb-only if missing)
set -euo pipefail
f=$1; msg=${2:-$(basename "$f" .csv)}
P=${PY:-/home/nammovuivui/anaconda3/envs/airline-satisfaction/bin/python}
K=${KAGGLE:-kaggle}
export PYTHONIOENCODING=utf-8 PYTHONPATH="$(pwd)/src"
"$P" -m kgkit validate "$f" data/sample_submission.csv
for try in 1 2 3; do "$K" competitions submit playground-series-s6e10 -f "$f" -m "$msg" && break; sleep 10; done
name=$(basename "$f")
row=""
for _ in $(seq 1 90); do
  row=$("$K" competitions submissions playground-series-s6e10 --csv 2>/dev/null | grep -m1 ",$name," || true)
  case "$row" in *COMPLETE*|*ERROR*) break ;; esac
  sleep 10
done
echo "$row"
score=$(echo "$row" | awk -F, '{print $(NF-1)}')
id=$(basename "$f" .csv)
[ -n "$score" ] && "$P" -m kgkit ledger lb "$id" "$score"
