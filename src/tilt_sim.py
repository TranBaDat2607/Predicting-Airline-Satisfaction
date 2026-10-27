"""Simulate a public-label tilt on train before spending real submissions on one.

A pseudo-test of 299,844 train rows is split 20 % "public" / 80 % "private". K "scored files" are built from
the OOF pool (single members and random logit blends); each gets its public AUC rounded to 5 decimals, exactly
what a scored public file gives us. The tilt (src/tilt.py) estimates which rows are public positives /
negatives from those scores alone and re-ranks the base. Reported: public and private AUC of the tilted base.

    python src/tilt_sim.py --k 300 --seed 0
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from tilt import Tilt

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--frac-public", type=float, default=0.2)
ap.add_argument("--singles", type=float, default=0.5, help="share of single members among the files")
ap.add_argument("--noise", type=float, default=1.0, help="noise sd in units of one rounding step (1e-5 AUC)")
ap.add_argument("--temp", type=float, default=1.0, help="miscalibrate the prior: logit(p) * temp")
ap.add_argument("--no-nuisance", action="store_true")
ap.add_argument("--etas", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
y_all = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
pool = np.load("artifacts/pool/oof.npy", mmap_mode="r")  # (n_train, m) logits
base_all = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
n_test = 299_844
T = np.sort(rng.choice(len(y_all), n_test, replace=False))
y = y_all[T]
pub = np.zeros(n_test, bool)
pub[rng.choice(n_test, int(round(a.frac_public * n_test)), replace=False)] = True
P = np.asarray(pool[T], dtype=np.float32)
m = P.shape[1]
print(f"pseudo-test {n_test}, public {pub.sum()}, pool members {m}")

def auc5(s):
    return round(roc_auc_score(y[pub], s[pub]), 5)

files = []
n_single = int(a.k * a.singles)
for j in rng.choice(m, min(n_single, m), replace=False):
    files.append(P[:, j].astype(np.float64))
while len(files) < a.k:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    idx = rng.choice(m, len(w), replace=False)
    files.append(P[:, idx] @ w)
scores = np.array([auc5(f) for f in files])
base = base_all[T]
p = np.clip(base if base.max() <= 1 else expit(base), 1e-6, 1 - 1e-6)
print(f"base public {auc5(p):.5f} private {roc_auc_score(y[~pub], p[~pub]):.6f}; files public {scores.min():.5f}..{scores.max():.5f}")

from scipy.special import logit as _lg
p = expit(_lg(p) * a.temp)
t = Tilt(p, f=a.frac_public, noise_units=a.noise, nuisance=not a.no_nuisance)
t.fit(files, scores, auc5(p))
print(f"LOO R2 of the public-score surprises: {t.loo_r2:.3f}  (surprise sd {1e5 * t.y.std() / (t.n1 * t.n0):.2f}u)")
for eta in a.etas:
    s = t.score(eta)
    print(f"eta {eta:5.2f}  public {roc_auc_score(y[pub], s[pub]):.6f}  private {roc_auc_score(y[~pub], s[~pub]):.6f}")
