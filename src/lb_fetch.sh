#!/usr/bin/env bash
# Download submission.csv of every scored public notebook listed in data/tilt/kernels.csv
# into data/tilt/k/<owner>__<slug>/ (retries; 6 in parallel; already-downloaded ones are skipped).
#   bash src/lb_fetch.sh
set -u
cd "$(dirname "$0")/.."
OUT=data/tilt/k
mkdir -p "$OUT"
fetch() {
  ref="$1"; d="$OUT/${ref/\//__}"
  [ -s "$d/submission.csv" ] && return 0
  mkdir -p "$d"
  for i in 1 2 3 4; do
    timeout 300 kaggle kernels output "$ref" -p "$d" --file-pattern '^submission\.csv$' -q >/dev/null 2>&1
    [ -s "$d/submission.csv" ] && { echo "ok $ref"; return 0; }
    sleep $((i * 3))
  done
  echo "fail $ref"
}
export -f fetch
export OUT
python3 - <<'EOF' | xargs -P 6 -I{} bash -c 'fetch "$@"' _ {}
import csv
for r in csv.DictReader(open("data/tilt/kernels.csv")):
    s = r.get("best_public_score") or ""
    if s and float(s) > 0:
        print(r["ref"])
EOF
