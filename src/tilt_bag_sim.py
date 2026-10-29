"""Does bagging the robust tilt over file subsets realise more public AUC than the full posterior? (The full
posterior's own tilt has scored 10-18u under its prediction on the real LB.) Pseudo-test as src/tilt_eb_sim.py:
K public-like files, a share with wrong scores (one-sided: the listed score of a notebook is its BEST version's,
so a wrong pair is too high), robust Student-t fit. Compared on the same files: the full posterior tilt, and the
mean posterior shift over B random subsets (share --frac of the files), at a few etas.

    python src/tilt_bag_sim.py --k 400 --bags 8 --frac 0.7
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
ap.add_argument("--k", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corrupt", type=float, default=0.25)
ap.add_argument("--corrupt-sd", type=float, default=15.0)
ap.add_argument("--one-sided", action="store_true", help="wrong scores are always too high")
ap.add_argument("--bags", type=int, default=8)
ap.add_argument("--frac", type=float, default=0.7)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--train", default="data/train.csv")
ap.add_argument("--pool", default="artifacts/pool/oof.npy")
ap.add_argument("--base", default="artifacts/0020_stack-lr-pool319-kaggle/oof.npy")
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
y_all = pd.read_csv(a.train, usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
pool = np.load(a.pool, mmap_mode="r")
base_all = np.load(a.base).astype(float)
n_test = 299_844
T = np.sort(rng.choice(len(y_all), n_test, replace=False))
y = y_all[T]
pub = np.zeros(n_test, bool)
pub[rng.choice(n_test, int(round(0.2 * n_test)), replace=False)] = True
P = np.asarray(pool[T], dtype=np.float32)
m = P.shape[1]


def auc5(s):
    return round(roc_auc_score(y[pub], s[pub]), 5)


files = [P[:, j].astype(np.float64) for j in rng.choice(m, min(a.k // 2, m), replace=False)]
while len(files) < a.k:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    files.append(P[:, rng.choice(m, len(w), replace=False)] @ w)
del P
scores = np.array([auc5(f) for f in files])
bad = rng.random(a.k) < a.corrupt
err = rng.normal(0, a.corrupt_sd, a.k)
err = np.abs(err) if a.one_sided else err
scores = scores + bad * np.round(err) * 1e-5
p = np.clip(base_all[T] if base_all.max() <= 1 else expit(base_all[T]), 1e-6, 1 - 1e-6)
t = TiltEB(p).fit(files, scores, auc5(p))
del files
t.fit_noise(nu=a.nu, verbose=False)
da_full, db_full = t.da.copy(), t.db.copy()
res = {}
for eta in (0.6, 0.8, 1.0):
    s = t.score(eta)
    res[f"full eta{eta}"] = (roc_auc_score(y[pub], s[pub]), t.predict(s))
DA, DB = np.zeros_like(da_full), np.zeros_like(db_full)
for bgi in range(a.bags):
    idx = np.sort(rng.choice(a.k, int(a.frac * a.k), replace=False))
    ts = t.subset(idx)
    ts.fit_noise(nu=a.nu, verbose=False)
    DA += ts.da / a.bags
    DB += ts.db / a.bags
t.da, t.db = DA, DB
for eta in (0.6, 0.8, 1.0, 1.2):
    s = t.score(eta)
    res[f"bag eta{eta}"] = (roc_auc_score(y[pub], s[pub]), None)
t.da, t.db = da_full, db_full
print(f"base {auc5(p):.5f}; K {a.k}; corrupt {bad.sum()} ({'one-sided' if a.one_sided else 'two-sided'}); "
      f"bags {a.bags} x {a.frac:g}")
for k, (pubv, pr) in res.items():
    print(f"{k:14s} public {pubv:.5f}" + (f"  (predicted {pr:.5f}, optimism {(pr - pubv) * 1e5:+.1f}u)" if pr else ""))
