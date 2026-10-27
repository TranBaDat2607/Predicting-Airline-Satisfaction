#!/usr/bin/env bash
# Second pass for notebooks whose output has no submission.csv: if exactly one output CSV has "submission"
# in its name, download it as data/tilt/k/<owner>__<slug>/submission.csv (ambiguous outputs are skipped:
# a wrong file would be a wrong data point for the tilt).
#   bash src/lb_fetch2.sh
set -u
cd "$(dirname "$0")/.."
OUT=data/tilt/k
fetch2() {
  ref="$1"; d="$OUT/${ref/\//__}"
  [ -s "$d/submission.csv" ] && return 0
  names=$(timeout 120 kaggle kernels files "$ref" -v --page-size 200 2>/dev/null | tail -n +2 | cut -d, -f1 | grep -i 'submission.*\.csv$')
  n=$(printf '%s\n' "$names" | grep -c . )
  if [ "$n" != "1" ]; then echo "skip $ref ($n candidates)"; return 0; fi
  mkdir -p "$d"
  for i in 1 2 3; do
    timeout 300 kaggle kernels output "$ref" -p "$d" --file-pattern "^$(printf '%s' "$names" | sed 's/[.[\*^$()+?{|]/\\&/g')\$" -q >/dev/null 2>&1
    f="$d/$names"
    if [ -s "$f" ]; then [ "$f" != "$d/submission.csv" ] && mv "$f" "$d/submission.csv"; echo "ok2 $ref ($names)"; return 0; fi
    sleep $((i * 3))
  done
  echo "fail2 $ref"
}
export -f fetch2
export OUT
grep "^fail" data/tilt/fetch.log | cut -d' ' -f2 | xargs -P 2 -I{} bash -c 'fetch2 "$@"' _ {}
