"""Score-named files inside public notebook outputs (e.g. goodpjw2008's submission_part4_096213.csv): list every
competition notebook's output files (latest version), keep CSVs whose name carries a public score, download them to
data/tilt/named/<owner>__<slug>/. src/lb_collect.py adds them (source "named") and src/tilt_eb.py's Student-t
weights catch names that are not public scores (CV scores, claims).

    python src/lb_scan_outputs.py --workers 3       # -> data/tilt/named/, data/tilt/named_files.csv
"""
import argparse
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()
OUT = Path("data/tilt/named")
# a public score in a file name; "cv" right before the number marks a CV score and is skipped
PAT = re.compile(r"(?i)(?<!cv)(?<!cv[_-])(?:lb[_-]?|public[_-]?|pb[_-]?|_|^)0[._-]?(9[56]\d{3})(?=[_.\-a-z]|$)")


def score_of(name):
    stem = Path(name).stem
    if re.search(r"(?i)(^|[^a-z])(oof|cv|train)([^a-z]|$)|cv[_-]?0", stem):
        return None
    m = PAT.search(stem)
    return float("0." + m.group(1)) if m else None


def run(cmd, timeout=600):
    for i in range(6):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ""
        out = r.stdout + r.stderr
        if "rate limit" in out.lower() or "429" in out:
            time.sleep(20 + 20 * i)
            continue
        return out
    return ""


def work(ref):
    d = OUT / ref.replace("/", "__")
    if (d / ".listed").exists():
        names = (d / ".listed").read_text().split("\n")
    else:
        out = run(["kaggle", "kernels", "files", ref, "--page-size", "200", "-v"])
        names = [ln.split(",")[0] for ln in out.splitlines()[1:] if ln.strip().endswith(tuple("0123456789")) or "," in ln]
        names = [n for n in names if n.lower().endswith(".csv")]
        d.mkdir(parents=True, exist_ok=True)
        (d / ".listed").write_text("\n".join(names))
    hits = [(n, score_of(n)) for n in names if score_of(n)]
    got = []
    for n, s in hits:
        if not (d / Path(n).name).exists():
            run(["kaggle", "kernels", "output", ref, "-p", str(d), "--file-pattern", "^" + re.escape(n) + "$", "-q"])
        if (d / Path(n).name).exists():
            got.append({"ref": ref, "file": str(d / Path(n).name), "score": s})
    return got


refs = pd.read_csv("data/tilt/kernels.csv").ref.tolist()
rows = []
with ThreadPoolExecutor(a.workers) as ex:
    for got in ex.map(work, refs):
        rows += got
        for g in got:
            print(f"{g['ref']}: {Path(g['file']).name} -> {g['score']:.5f}", flush=True)
pd.DataFrame(rows).to_csv("data/tilt/named_files.csv", index=False)
print(f"{len(rows)} score-named output files from {len(set(r['ref'] for r in rows))} notebooks")
