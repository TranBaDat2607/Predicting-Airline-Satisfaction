"""Does the exponential-tilt posterior (src/tilt_exp.py) realise more public AUC than the Gaussian one (src/tilt_eb.py)
on the same files? Pseudo-test as src/tilt_bag_sim.py: K public-like files, a share with one-sided wrong scores (the listed
notebook score is the BEST version's), robust Student-t fit. Reported: public AUC and optimism per variant.

    python src/tilt_exp_sim.py --k 400 --corrupt 0.3 --relin 2
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from tilt_exp import TiltExp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corrupt", type=float, default=0.3)
ap.add_argument("--corrupt-sd", type=float, default=15.0)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--em", type=int, default=2)
ap.add_argument("--relin", type=int, default=2, help="relinearisation rounds of the bilinear model")
ap.add_argument("--etas", type=float, nargs="+", default=[0.7, 0.85, 1.0, 1.15, 1.3])
ap.add_argument("--grid", nargs="*", default=[], help="noise settings k0:k1:nu to compare at eta 1 instead of the full report")
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


def auc(s):
    return roc_auc_score(y[pub], s[pub])


def auc5(s):
    return round(auc(s), 5)


files = [P[:, j].astype(np.float32) for j in rng.choice(m, min(a.k // 2, m), replace=False)]
while len(files) < a.k:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    files.append((P[:, rng.choice(m, len(w), replace=False)] @ w).astype(np.float32))
del P
scores = np.array([auc5(f) for f in files])
bad = rng.random(a.k) < a.corrupt
scores = scores + bad * np.round(np.abs(rng.normal(0, a.corrupt_sd, a.k))) * 1e-5
p = np.clip(base_all[T] if base_all.max() <= 1 else expit(base_all[T]), 1e-6, 1 - 1e-6)
b5 = auc5(p)
print(f"base {b5:.5f}; K {a.k}; corrupted {bad.sum()} (one-sided)", flush=True)
t0 = time.time()
t = TiltExp(p).fit([np.asarray(f, np.float64) for f in files], scores, b5)
print(f"built system {time.time() - t0:.0f}s", flush=True)
if a.grid:
    bad_w = lambda: (t.w_exp[bad].mean(), t.w_exp[~bad].mean())  # noqa: E731
    for g in a.grid:
        k0, k1, nu = (float(v) for v in g.split(":"))
        t.fit_noise(nu=nu, fixed=(max(k0, 1e-9), max(k1, 1e-9)), verbose=False)
        t.fit_exp(nu=nu, em=a.em, verbose=False)
        s = t.score_exp(1.0)
        pv, pr = auc(s), t.predict_exp(s)
        print(f"k0 {k0:g} k1 {k1:g} nu {nu:g}: public {pv:.5f} gain {(pv - b5) * 1e5:+7.1f}u predicted {pr:.5f} (optimism {(pr - pv) * 1e5:+.1f}u); weights corrupted {bad_w()[0]:.2f} clean {bad_w()[1]:.2f}; {time.time() - t0:.0f}s", flush=True)
    sys.exit(0)
t.fit_noise(nu=a.nu, verbose=True)
print(f"Gaussian solved {time.time() - t0:.0f}s", flush=True)


def show(name, s, pred=None):
    pv = auc(s)
    print(f"{name:26s} public {pv:.5f}  gain {(pv - b5) * 1e5:+7.1f}u" + (f"  predicted {pred:.5f} (optimism {(pred - pv) * 1e5:+.1f}u)" if pred else ""),
          flush=True)


for eta in a.etas:
    s = t.score(eta)
    show(f"gaussian eta{eta:g}", s, t.predict(s))
for eta in a.etas:
    show(f"one-step exp eta{eta:g}", t.score_exp_one_step(eta))
t.fit_exp(nu=a.nu, em=a.em)
print(f"exp fit {time.time() - t0:.0f}s", flush=True)
for eta in a.etas:
    s = t.score_exp(eta)
    show(f"exp eta{eta:g}", s, t.predict_exp(s))
# how many of the wrong files does each fit catch
print(f"weights: gaussian corrupted {t.w[bad].mean():.2f} clean {t.w[~bad].mean():.2f}; exp corrupted {t.w_exp[bad].mean():.2f} clean {t.w_exp[~bad].mean():.2f}", flush=True)
for r in range(a.relin):
    t.relinearize(files, scores)
    t.noise_var = t.nv_exp
    t._newton(t.nv_exp, t.lam_exp, t.beta_exp, iters=5)
    print(f"relinearised {r + 1}: {time.time() - t0:.0f}s, E[n1] {t.qp.sum():.0f}, E[n0] {t.qn.sum():.0f}", flush=True)
    for eta in a.etas:
        s = t.score_exp(eta)
        show(f"exp relin{r + 1} eta{eta:g}", s, t.predict_exp(s))
