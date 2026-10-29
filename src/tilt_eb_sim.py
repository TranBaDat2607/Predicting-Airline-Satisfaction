"""Does the empirical-Bayes noise model (src/tilt_eb.py) recover the clean-data tilt when some file scores are wrong?

Pseudo-test as src/tilt_sim.py (train rows + OOF pool, 20 % "public"). A share --corrupt of the files gets a
wrong score: the score of another version of the notebook, modelled as the true score + N(0, --corrupt-sd u).
Each noise model is fitted on the same files; reported: public / private AUC of the tilt per eta and the
posterior-predicted public (calibration).

    python src/tilt_eb_sim.py --k 370 --corrupt 0.25 --corrupt-sd 15
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=370)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corrupt", type=float, default=0.25)
ap.add_argument("--corrupt-sd", type=float, default=15.0, help="sd of a wrong score, in 1e-5 units")
ap.add_argument("--etas", type=float, nargs="+", default=[0.6, 1, 1.5, 2])
ap.add_argument("--models", nargs="+", default=["x1", "x24", "reml", "t4", "t2"])
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
y_all = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
pool = np.load("artifacts/pool/oof.npy", mmap_mode="r")
base_all = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
n_test = 299_844
T = np.sort(rng.choice(len(y_all), n_test, replace=False))
y = y_all[T]
pub = np.zeros(n_test, bool)
pub[rng.choice(n_test, int(round(0.2 * n_test)), replace=False)] = True
P = np.asarray(pool[T], dtype=np.float32)
m = P.shape[1]


def auc5(s):
    return round(roc_auc_score(y[pub], s[pub]), 5)


files = []
for j in rng.choice(m, min(a.k // 2, m), replace=False):
    files.append(P[:, j].astype(np.float64))
while len(files) < a.k:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    files.append(P[:, rng.choice(m, len(w), replace=False)] @ w)
del P
scores = np.array([auc5(f) for f in files])
bad = rng.random(a.k) < a.corrupt
scores_obs = scores + bad * np.round(rng.normal(0, a.corrupt_sd, a.k)) * 1e-5
p = np.clip(base_all[T] if base_all.max() <= 1 else expit(base_all[T]), 1e-6, 1 - 1e-6)
b5 = auc5(p)
print(f"base public {b5:.5f} private {roc_auc_score(y[~pub], p[~pub]):.6f}; corrupted files {bad.sum()} / {a.k}", flush=True)
t = TiltEB(p).fit(files, scores_obs, b5)
del files
spec = {"x1": dict(fixed=(1, 1e-9)), "x24": dict(fixed=(576, 1e-9)), "x32": dict(fixed=(1024, 1e-9)), "reml": dict(), "t4": dict(nu=4), "t2": dict(nu=2), "t1": dict(nu=1)}
for name in a.models:
    t.fit_noise(verbose=False, **spec[name])
    wbad = np.median(t.w[bad]) if bad.any() else np.nan
    line = [f"{name:5s} k0 {t.k0:7.3g} k1 {t.k1:7.3g} median w bad {wbad:.2f} good {np.median(t.w[~bad]):.2f} |"]
    for eta in a.etas:
        s = t.score(eta)
        line.append(f"eta{eta:g} pub {roc_auc_score(y[pub], s[pub]):.5f} (pred {t.predict(s):.5f}) "
                    f"priv {roc_auc_score(y[~pub], s[~pub]):.5f}")
    print("  ".join(line), flush=True)
