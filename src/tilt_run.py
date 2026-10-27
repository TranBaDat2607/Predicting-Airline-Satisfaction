"""Fit the public-label tilt (src/tilt.py) on every collected scored file and write tilted candidates.

Base = our best public file. Its prior probabilities come from its test ranks mapped through the calibrated
OOF stack (0020): the r-th quantile of the stack's OOF probabilities, i.e. "a file of the stack's quality
assigns this probability at this rank". Files far from the base (rank corr below --min-corr) or scoring below
--min-score are left out: the linearisation is only accurate near the base.

    python src/tilt_run.py --base subs/0044_lbb-settf8-10.csv --base-score 0.96214 --write 0.5 1.0
Writes subs/<id>_tilt-eta<eta>.csv candidates, tagged [public-LB fitted] - never a final pick.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from tilt import Tilt

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="subs/0044_lbb-settf8-10.csv")
ap.add_argument("--base-score", type=float, default=0.96214)
ap.add_argument("--min-score", type=float, default=0.955)
ap.add_argument("--min-corr", type=float, default=0.90)
ap.add_argument("--noise", type=float, default=1.0)
ap.add_argument("--no-nuisance", action="store_true")
ap.add_argument("--exclude", nargs="*", default=[], help="substrings of file names to leave out")
ap.add_argument("--write", type=float, nargs="*", default=[])
ap.add_argument("--prefix", default="")
a = ap.parse_args()

ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
y = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
oof = oof if oof.max() <= 1 else expit(oof)
qs = np.sort(oof)

b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / len(b)
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)

R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
keep = (files.score >= a.min_score).to_numpy().copy()
for s in a.exclude:
    keep &= ~files.name.str.contains(s, regex=False).to_numpy()
corr = np.array([np.corrcoef(R[i], rb)[0, 1] if keep[i] else 0 for i in range(len(files))])
keep &= corr >= a.min_corr
keep &= np.abs(corr - 1) > 1e-7  # the base itself (or an exact copy)
sel = np.flatnonzero(keep)
print(f"files {len(files)}, used {len(sel)} (score >= {a.min_score}, rank corr >= {a.min_corr})")
print(files.iloc[sel].groupby("source").score.agg(["count", "min", "max"]))

t = Tilt(p, f=0.2, noise_units=a.noise, nuisance=not a.no_nuisance)
t.fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], a.base_score)
scale = 1e5 / (t.n1 * t.n0)
print(f"LOO R2 {t.loo_r2:.3f}; surprise sd {t.y.std() * scale:.2f}u; LOO residual sd {t.loo.std() * scale:.2f}u")
out = files.iloc[sel][["name", "source", "score"]].copy()
out["corr"] = corr[sel]
out["surprise_u"] = t.y * scale
out["loo_resid_u"] = t.loo * scale
out.to_csv("data/tilt/fit_report.csv", index=False)
print(out.reindex(out.surprise_u.abs().sort_values(ascending=False).index).head(12).to_string())

for eta in a.write:
    s = t.score(eta)
    rs = (rankdata(s) - 0.5) / len(s)
    print(f"eta {eta}: rank corr with base {np.corrcoef(rs, rb)[0, 1]:.6f}")
    path = f"subs/{a.prefix}tilt-eta{eta:g}.csv"
    pd.DataFrame({"id": ids, "satisfaction": s}).to_csv(path, index=False, float_format="%.8f")
    print("wrote", path)
