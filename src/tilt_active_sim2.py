"""Which probe strategy gains most per own submission? Pseudo-test as src/tilt_eb_sim.py (K public files, a share
with wrong scores, robust Student-t fit), then T rounds of: build a probe, observe its exact rounded public AUC, add
it (exact, weight >= 1), refit. Reported per round: the probe's public AUC and the best public AUC so far.

Strategies
  greedy : the tilt at the eta the posterior predicts best
  eta2   : alternate eta 0.6 / 1.4 (learns the scale along the tilt)
  half   : the greedy tilt on a random half of the rows (the other half keeps the prior) every other round
  noise  : greedy tilt + logit jitter (sd --jitter) every other round

    python src/tilt_active_sim2.py --strategy greedy --t 15
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=370)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corrupt", type=float, default=0.25)
ap.add_argument("--corrupt-sd", type=float, default=15.0)
ap.add_argument("--t", type=int, default=15)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--strategy", default="greedy")
ap.add_argument("--jitter", type=float, default=0.3)
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


files = [P[:, j].astype(np.float64) for j in rng.choice(m, min(a.k // 2, m), replace=False)]
while len(files) < a.k:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    files.append(P[:, rng.choice(m, len(w), replace=False)] @ w)
del P
scores = np.array([auc5(f) for f in files])
bad = rng.random(a.k) < a.corrupt
scores = scores + bad * np.round(rng.normal(0, a.corrupt_sd, a.k)) * 1e-5
p = np.clip(base_all[T] if base_all.max() <= 1 else expit(base_all[T]), 1e-6, 1 - 1e-6)
b5 = auc5(p)
t = TiltEB(p).fit(files, scores, b5)
del files
exact = np.zeros(a.k, bool)
t.fit_noise(nu=a.nu, exact=exact, verbose=False)
ETAS = [0.6, 0.8, 1.0, 1.2, 1.5]


def greedy():
    pr = [t.predict(t.score(e)) for e in ETAS]
    e = ETAS[int(np.argmax(pr))]
    return t.score(e), e, max(pr)


s0, e0, pr0 = greedy()
print(f"[{a.strategy}] base {b5:.5f}; tilt before probes: eta {e0} public {roc_auc_score(y[pub], s0[pub]):.5f} "
      f"(pred {pr0:.5f}) private {roc_auc_score(y[~pub], s0[~pub]):.5f}", flush=True)
best = 0.0
for r in range(a.t):
    s, e, pr = greedy()
    if a.strategy == "eta2":
        e = 0.6 if r % 2 == 0 else 1.4
        s = t.score(e)
    elif a.strategy == "half" and r % 2 == 1:
        half = rng.random(n_test) < 0.5
        s = np.where(half, s, p)
    elif a.strategy == "noise" and r % 2 == 1:
        s = expit(logit(np.clip(s, 1e-9, 1 - 1e-9)) + a.jitter * rng.standard_normal(n_test))
    sc = auc5(s)
    best = max(best, sc)
    t.add([s], [sc])
    exact = np.append(exact, True)
    t.fit_noise(nu=a.nu, exact=exact, verbose=False)
    g, eg, prg = greedy()
    print(f"round {r + 1:2d}: probe eta {e} public {sc:.5f} | best {best:.5f} | next greedy public "
          f"{roc_auc_score(y[pub], g[pub]):.5f} (pred {prg:.5f})", flush=True)
