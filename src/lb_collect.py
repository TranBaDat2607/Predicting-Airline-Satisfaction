"""Collect every test-prediction file with a known public score into one rank matrix.

Sources: public notebook outputs (data/tilt/k, scores from the search API in data/tilt/kernels.csv),
score-named files in public datasets (0.96xxx*.csv), the LB-surface atlas rank vectors, the public files
used by src/lb_surface.py, and our own submissions (ledger lb_public).
Exact duplicates (same ranking) are merged; a ranking seen with two different scores is dropped (the
search API reports a notebook's best version, which need not be the output we downloaded).

    python src/lb_collect.py      # -> data/tilt/ranks.npy (float32, files x test rows), data/tilt/files.csv
"""
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

IDS = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(IDS)
OUT = Path("data/tilt")


def ranks(v):
    return ((rankdata(v) - 0.5) / len(v)).astype(np.float32)


def read_sub(path):
    d = pd.read_csv(path)
    if "id" not in d.columns or len(d) != N:
        return None
    col = "satisfaction" if "satisfaction" in d.columns else d.columns[-1]
    v = pd.to_numeric(d.set_index("id")[col], errors="coerce").reindex(IDS).to_numpy(float)
    if not np.isfinite(v).all() or np.unique(v[:5000]).size < 50:
        return None
    return v


items = []  # (name, source, score, values)

kern = pd.read_csv(OUT / "kernels.csv")
for r in kern.itertuples():
    p = OUT / "k" / r.ref.replace("/", "__") / "submission.csv"
    if p.exists() and r.best_public_score and r.best_public_score > 0:
        items.append((r.ref, "notebook", float(r.best_public_score), p))

for p in sorted(glob.glob("data/ext/*/0.96*") + glob.glob("data/ext/*/0.95*")):
    m = re.match(r"(0\.9\d{4})", Path(p).name)
    if m:
        items.append((p, "dataset", float(m.group(1)), Path(p)))
for p in glob.glob("data/ext/goodpjw-pub/*_0962*.csv"):
    m = re.search(r"_(0962\d\d)_", Path(p).name)
    items.append((p, "dataset", float("0." + m.group(1)[1:]) if m else None, Path(p)))
# rehosted public files whose name carries the PUBLIC score (a "cv-0-9xxxx" number is a CV score and is ignored)
LB_NAME = [r"lb-0-(9\d{4})", r"0-(9\d{4})-public-score", r"_0(9[56]\d{3})(?:_|\.)"]
for lib in ("s6e10-top18-artifacts", "ps6e10-best-public-submissions"):
    for p in sorted(glob.glob(f"data/ext/{lib}/*")):
        for pat in LB_NAME:
            m = re.search(pat, Path(p).name)
            if m:
                items.append((p, "dataset", float("0." + m.group(1)), Path(p)))
                break
KNOWN = {"data/ext/dataset-submission-for-lb-score-0-96192/submission_alpha_1p40.csv": 0.96192}  # dataset title
for p, s in KNOWN.items():
    if Path(p).exists():
        items.append((p, "dataset", s, Path(p)))

atlas = Path("data/ext/s6e10-lb-surface-atlas")
for npz, csv, key, col in [("own_scored_ranks.npz", "own_scores.csv", "key", "public_lb"),
                           ("probe_ranks.npz", "probe_scores.csv", "key", "public_lb"),
                           ("round6_probe_ranks.npz", "round6_probe_scores.csv", "key", "public_lb")]:
    z = np.load(atlas / npz)
    sc = pd.read_csv(atlas / csv).set_index(key)[col]
    for k in z.files:
        if k in sc.index and pd.notna(sc[k]):
            items.append((f"atlas:{k}", "atlas", float(sc[k]), z[k]))
for f in ["rabari_v41_submission.csv", "rabari_v43_submission.csv", "rabari_v54_submission.csv"]:
    pass  # scores unknown except v43 (in lb_surface.PUBLIC)

import sys
sys.path.insert(0, "src")
try:
    from lb_surface import PUBLIC
except Exception:
    PUBLIC = {}
for p, s in PUBLIC.items():
    if Path(p).exists():
        items.append((p, "lb", float(s), Path(p)))

for line in Path(".kaggle-gm/ledger.jsonl").read_text().splitlines():
    r = json.loads(line)
    lb, sub = r.get("lb_public"), r.get("submission")
    if lb and sub and Path(sub).exists():
        items.append((sub, "own", float(lb), Path(sub)))

print("candidate files:", len(items))
rows, R = [], []
seen = {}
for name, src, score, obj in items:
    if score is None:
        continue
    v = obj if isinstance(obj, np.ndarray) else read_sub(obj)
    if v is None or len(v) != N:
        continue
    r = ranks(v)
    h = hashlib.md5(np.round(r * 1e5).astype(np.int32).tobytes()).hexdigest()
    if h in seen:
        j = seen[h]
        if abs(rows[j]["score"] - score) > 1e-9:
            rows[j]["conflict"] = True
        rows[j]["dups"] += 1
        continue
    seen[h] = len(rows)
    rows.append({"name": name, "source": src, "score": score, "dups": 0, "conflict": False})
    R.append(r)

df = pd.DataFrame(rows)
R = np.stack(R)
keep = ~df.conflict.to_numpy()
df, R = df[keep].reset_index(drop=True), R[keep]
np.save(OUT / "ranks.npy", R)
df.to_csv(OUT / "files.csv", index=False)
print(f"unique files {len(df)} (dropped {int((~keep).sum())} with conflicting scores)")
print(df.groupby("source").score.describe())
