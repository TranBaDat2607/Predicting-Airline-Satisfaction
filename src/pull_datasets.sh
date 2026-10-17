#!/usr/bin/env bash
# Download Kaggle datasets into data/ext/<slug>/: whole-dataset download first, per-file fallback on failure
# (some public OOF libraries return 404 on the whole-dataset endpoint).
K=${KAGGLE:-C:/Users/Admin/anaconda3/envs/airline-satisfaction/Scripts/kaggle.exe}
export PYTHONIOENCODING=utf-8
for ds in "$@"; do
  d=data/ext/$(basename "$ds"); mkdir -p "$d"
  if [ -n "$(ls -A "$d" 2>/dev/null)" ]; then echo "$ds: already present"; continue; fi
  if ! "$K" datasets download "$ds" -p "$d" --unzip -q >/dev/null 2>&1 || [ -z "$(ls -A "$d")" ]; then
    "$K" datasets files "$ds" --page-size 200 --csv 2>/dev/null | tail -n +2 | cut -d, -f1 | while read -r f; do
      "$K" datasets download "$ds" -f "$f" -p "$d" -q >/dev/null 2>&1 || echo "FAIL $ds/$f"
      [ -f "$d/$(basename "$f").zip" ] && (cd "$d" && unzip -oq "$(basename "$f").zip" && rm "$(basename "$f").zip")
    done
  fi
  echo "$ds: $(ls "$d" | wc -l) files, $(du -sh "$d" | cut -f1)"
done
