"""Correlated model-error kernel (TiltEB.kernel_power): REML and held-out own tilts for each power m, Gaussian fit.

    python src/tilt_kernel_check.py --holdout 0068 0070 0071 0072 --powers 0 1 2 4 8
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--holdout", nargs="*", default=["0068", "0070", "0071", "0072"])
ap.add_argument("--powers", type=float, nargs="+", default=[0, 1, 2, 4, 8])
ap.add_argument("--nu", type=float, default=2.0)
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
    hold |= files.name.str.contains(s, regex=False).to_numpy()
sel, hsel = np.flatnonzero(keep & ~hold), np.flatnonzero(keep & hold)
print(f"used {len(sel)}, held out {len(hsel)}", flush=True)
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
exact = files.source.to_numpy()[sel] == "own"
scale = 1e5 / (t.n1 * t.n0)
for m in a.powers:
    t.kernel_power = m
    t.fit_noise(nu=a.nu, exact=exact, verbose=False)
    s1 = t.score(1.0)
    rows = [f"{files.name[i].split('/')[-1][:4]} {t.predict(np.asarray(R[i], np.float64)):.5f}/{files.score[i]:.5f}" for i in hsel]
    print(f"m {m:g}: k0 {t.k0:.3g} k1 {t.k1:.3g} REML {t.reml(t.noise_var):.2f} w<.5 {(t.w < .5).sum()} LOO R2 {t.loo_r2:.3f} own tilt pred {t.predict(s1):.5f} | held-out pred/actual: " + "  ".join(rows), flush=True)
