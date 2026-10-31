"""Which noise model predicts our own scored tilts best? Fit the tilt without the held-out files (default: our
three scored tilt submissions), then predict their public scores under several noise models: global x1..x32
(tilt.py), REML (n0, n1) and REML + Student-t weights. Also prints each model's predicted public at a few etas.

    python src/tilt_eb_check.py --holdout tilt-v
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
ap.add_argument("--base", default="subs/0044_lbb-settf8-10.csv")
ap.add_argument("--base-score", type=float, default=0.96214)
ap.add_argument("--min-score", type=float, default=0.955)
ap.add_argument("--min-corr", type=float, default=0.90)
ap.add_argument("--holdout", nargs="*", default=["tilt-v"])
ap.add_argument("--etas", type=float, nargs="*", default=[0.6, 1.0, 1.5, 2.0])
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
hold = np.zeros(len(files), bool)
for s in a.holdout:
    hold |= files.name.str.contains(s, regex=False).to_numpy()
sel, hsel = np.flatnonzero(keep & ~hold), np.flatnonzero(keep & hold)
print(f"fit on {len(sel)} files; held out: {[files.name[i] for i in hsel]}", flush=True)
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], a.base_score)
exact = files.source.to_numpy()[sel] == "own"
H = [np.asarray(R[i], dtype=np.float64) for i in hsel]
models = [("global x1", dict(fixed=(1, 1e-9))), ("global x4", dict(fixed=(16, 1e-9))),
          ("global x16", dict(fixed=(256, 1e-9))), ("global x24", dict(fixed=(576, 1e-9))),
          ("global x32", dict(fixed=(1024, 1e-9))), ("REML", dict()), ("REML t4", dict(nu=4)),
          ("REML t2", dict(nu=2))]
for name, kw in models:
    t.fit_noise(exact=exact, verbose=False, **kw)
    pred = "  ".join(f"{files.name[i].split('/')[-1][:18]} {t.predict(h):.5f} (obs {files.score[i]:.5f})"
                     for i, h in zip(hsel, H))
    et = "  ".join(f"eta{e:g} {t.predict(t.score(e)):.5f}" for e in a.etas)
    print(f"{name:11s} REML {t.reml(t.noise_var):9.2f} k0 {t.k0:8.3g} k1 {t.k1:8.3g} w<.5 {(t.w < 0.5).sum():3d} "
          f"LOO R2 {t.loo_r2:.3f} | {pred} | {et}", flush=True)
