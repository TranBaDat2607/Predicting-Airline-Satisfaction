"""Build the tilt system once and compare noise models by leave-own-tilt-out prediction: refit without one of our
scored tilts (exact public scores), predict it, and report the error; plus each model's predicted public per eta.

    python src/tilt_explore.py --models t2 t1 t4 reml --holdout 0062 0058
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
ap.add_argument("--models", nargs="+", default=["t2", "t1", "t4", "reml"])
ap.add_argument("--holdout", nargs="*", default=["0062", "0058"])
ap.add_argument("--etas", type=float, nargs="*", default=[0.8, 1.0, 1.25, 1.5])
ap.add_argument("--no-nuisance", action="store_true")
a = ap.parse_args()
SPEC = {"x24": dict(fixed=(576, 1e-9)), "reml": dict(), "t4": dict(nu=4), "t2": dict(nu=2), "t1": dict(nu=1),
        "t05": dict(nu=0.5), "t2k5": dict(nu=2, fixed=(1e-6, 5)), "t2k20": dict(nu=2, fixed=(1e-6, 20)),
        "t2k40": dict(nu=2, fixed=(1e-6, 40)), "t2k80": dict(nu=2, fixed=(1e-6, 80)), "t2k160": dict(nu=2, fixed=(1e-6, 160)), "t2k320": dict(nu=2, fixed=(1e-6, 320))}

ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / len(b)
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
names = files.name.to_numpy()[sel]
print(f"{len(sel)} files", flush=True)
T = TiltEB(p, nuisance=not a.no_nuisance).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], a.base_score)
exact = files.source.to_numpy()[sel] == "own"
for name in a.models:
    errs = []
    for h in a.holdout:
        j = [i for i, n in enumerate(names) if h in str(n)]
        if not j:
            continue
        keep = np.setdiff1d(np.arange(len(sel)), j)
        t = T.subset(keep)
        t.fit_noise(exact=exact[keep], verbose=False, **SPEC[name])
        s = np.asarray(R[sel[j[0]]], dtype=np.float64)
        pr = t.predict(s)
        errs.append(f"{str(names[j[0]]).split('/')[-1][:20]} pred {pr:.5f} obs {files.score.to_numpy()[sel[j[0]]]:.5f}")
    T.fit_noise(exact=exact, verbose=False, **SPEC[name])
    et = "  ".join(f"eta{e:g} {T.predict(T.score(e)):.5f}" for e in a.etas)
    print(f"{name:5s} k0 {T.k0:7.3g} k1 {T.k1:7.3g} w<.2 {(T.w < 0.2).sum():3d} | " + " | ".join(errs) + f" || {et}",
          flush=True)
