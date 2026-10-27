"""Simulate the submission loop on top of the tilt: K0 scored files, then T rounds of
"submit the current tilt (+ small random perturbation), observe its rounded public AUC, add it, refit".
Reports the best public AUC reached and the private AUC of that file (pseudo-test on train, as tilt_sim.py).

    python src/tilt_active_sim.py --k0 300 --t 40
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from tilt import Tilt

ap = argparse.ArgumentParser()
ap.add_argument("--k0", type=int, default=300)
ap.add_argument("--t", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--etas", type=float, nargs="+", default=[1.0, 1.5])
ap.add_argument("--jitter", type=float, default=0.15, help="sd of the random logit perturbation of each probe")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
y_all = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
pool = np.load("artifacts/pool/oof.npy", mmap_mode="r")
base_all = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
n_test = 299_844
T = np.sort(rng.choice(len(y_all), n_test, replace=False))
y = y_all[T]
pub = np.zeros(n_test, bool)
pub[rng.choice(n_test, int(0.2 * n_test), replace=False)] = True
P = np.asarray(pool[T], dtype=np.float32)
m = P.shape[1]


def auc5(s):
    return round(roc_auc_score(y[pub], s[pub]), 5)


files = [P[:, j].astype(np.float64) for j in rng.choice(m, min(a.k0 // 2, m), replace=False)]
while len(files) < a.k0:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    files.append(P[:, rng.choice(m, len(w), replace=False)] @ w)
p = np.clip(base_all[T], 1e-6, 1 - 1e-6)
b5 = auc5(p)
t = Tilt(p).fit(files, [auc5(f) for f in files], b5)
best = (b5, roc_auc_score(y[~pub], p[~pub]))
print(f"base public {b5:.5f}; K0 = {a.k0}")
for r in range(a.t):
    eta = a.etas[r % len(a.etas)]
    s = logit(np.clip(t.score(eta), 1e-9, 1 - 1e-9)) + a.jitter * rng.standard_normal(n_test) * (r > 0)
    sc = auc5(s)
    if sc > best[0]:
        best = (sc, roc_auc_score(y[~pub], s[~pub]))
    t.add([s], [sc])
    if r % 5 == 0 or r == a.t - 1:
        print(f"probe {r + 1:3d} eta {eta}: public {sc:.5f}  best {best[0]:.5f} (private {best[1]:.6f})  LOO R2 {t.loo_r2:.3f}", flush=True)
