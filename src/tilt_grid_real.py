"""Noise settings of the exponential tilt on the real files: each (k0, k1, nu) is fitted without today's own files; reported are its own tilt's
prediction and how well it predicts the held-out own files. In simulation (reports/tilt/exp_grid_kaggle_*.log) the best settings are the ones with
small optimism. Writes each candidate's ranks to data/tilt/grid/<name>.npy.

    python src/tilt_grid_real.py --holdout 0072 0073 0074 0075 0076 0077 --grid 0:0:2 0:12:2 0:0:1 0:12:1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_exp import TiltExp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", nargs="*", default=[])
ap.add_argument("--grid", nargs="+", required=True, help="k0:k1:nu; k1 = 0 means REML")
ap.add_argument("--anchor", default="0073")
ap.add_argument("--exact-sources", nargs="+", default=["own"])
a = ap.parse_args()
ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv("subs/0044_lbb-settf8-10.csv").set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / N
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
keep = (files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7)
hold = np.zeros(len(files), bool)
for s in a.holdout:
    hold |= files.name.str.contains(f"subs/{s}", regex=False).to_numpy()
sel, hsel = np.flatnonzero(keep & ~hold), np.flatnonzero(keep & hold)
print(f"used {len(sel)}, held out {len(hsel)}", flush=True)
t = TiltExp(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
exact = files.source.isin(a.exact_sources).to_numpy()[sel]
out = Path("data/tilt/grid")
out.mkdir(exist_ok=True)
hr = {files.name[i].split("/")[-1][:4]: np.asarray(R[i], np.float64) for i in hsel}
hs = {files.name[i].split("/")[-1][:4]: files.score[i] for i in hsel}
anc = np.flatnonzero(files.name.str.contains(f"subs/{a.anchor}", regex=False).to_numpy())
anchor = np.asarray(R[anc[0]], np.float64) if len(anc) else None
for g in a.grid:
    k0, k1, nu = (float(v) for v in g.split(":"))
    fixed = None if k1 == 0 else (max(k0, 1e-9), k1)
    t.fit_noise(nu=nu, fixed=fixed, exact=exact, verbose=False)
    t.fit_exp(nu=nu, em=3, exact=exact, verbose=False)
    s = t.score_exp(1.0)
    rk = (rankdata(s) - 0.5) / N
    np.save(out / f"k{k0:g}_{k1:g}_nu{nu:g}.npy", rk.astype(np.float32))
    errs = " ".join(f"{k}:{(t.predict_exp(r) - hs[k]) * 1e5:+.1f}" for k, r in hr.items())
    print(f"{g:10s} k0 {t.k0:.3g} k1 {t.k1:.3g} w<.5 {(t.w_exp < .5).sum():3d} own pred {t.predict_exp(s):.5f} corr base {np.corrcoef(rk, rb)[0, 1]:.5f}"
          + (f" diff to anchor {(rk - anchor).std():.4f}" if anchor is not None else "") + f" | held-out pred - actual (u): {errs}", flush=True)
