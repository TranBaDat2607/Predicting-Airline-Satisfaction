"""Empirical-Bayes noise level for the tilt: refit the same files at several noise levels and compare the
leave-one-out prediction of our own scored tilt submissions (exact public scores) with what they scored.
The K x K system does not depend on the noise, so one build serves the whole scan.

    python src/tilt_noise_scan.py --noise 1 2 4 8 16 32
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt import Tilt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="subs/0044_lbb-settf8-10.csv")
ap.add_argument("--base-score", type=float, default=0.96214)
ap.add_argument("--min-score", type=float, default=0.955)
ap.add_argument("--min-corr", type=float, default=0.90)
ap.add_argument("--noise", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32])
ap.add_argument("--probe", default="tilt", help="substring naming our own probe files to report")
a = ap.parse_args()

ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / len(b)
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
keep = (files.score >= a.min_score).to_numpy() & (corr >= a.min_corr) & (np.abs(corr - 1) > 1e-7)
sel = np.flatnonzero(keep)
t = Tilt(p, noise_units=1.0).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel],
                                 a.base_score)
own = np.flatnonzero(files.name.to_numpy()[sel].astype(str).__contains__ if False else
                     np.array([a.probe in str(n) for n in files.name.to_numpy()[sel]]))
scale = 1e5 / (t.n1 * t.n0)
sd1 = t.sd
for nu in a.noise:
    t.sd = sd1 * nu
    t.solve()
    pred = (t.y - t.loo) * scale
    obs = t.y * scale
    msg = "  ".join(f"{files.name.iloc[sel[i]].split('/')[-1][:22]}: obs {obs[i]:+.1f}u pred {pred[i]:+.1f}u" for i in own)
    print(f"noise x{nu:<5g} LOO R2 {t.loo_r2:+.3f}  resid sd {t.loo.std() * scale:6.2f}u  | {msg}", flush=True)
