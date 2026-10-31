"""Which NEW-direction probes are worth an own submission? Pseudo-test as src/tilt_eb_sim.py: K public-like files
(a share with wrong scores) from part of the OOF pool, robust Student-t fit. The other pool members are candidate
probes (their pseudo-test predictions = our unscored models). Each round one candidate is "submitted" (exact
rounded public AUC), added, the model refit, and the greedy tilt re-scored.

  random : a random unused candidate
  maxvar : the candidate with the largest posterior predictive sd (least explained by the files so far)
  greedy : the greedy tilt itself (control: no new direction)

    python src/tilt_probe_sim.py --strategy maxvar --t 12
Paths: --train, --pool, --base (defaults are this repo's; a Kaggle kernel passes its input paths).
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
ap.add_argument("--k", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--corrupt", type=float, default=0.25)
ap.add_argument("--corrupt-sd", type=float, default=15.0)
ap.add_argument("--t", type=int, default=12)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--strategy", default="maxvar")
ap.add_argument("--n-cand", type=int, default=80)
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
perm = rng.permutation(m)
cand_idx, file_idx = perm[: a.n_cand], perm[a.n_cand:]


def auc5(s):
    return round(roc_auc_score(y[pub], s[pub]), 5)


files = [P[:, j].astype(np.float64) for j in file_idx[: a.k // 2]]
while len(files) < a.k:
    w = rng.dirichlet(np.ones(rng.integers(2, 30)))
    files.append(P[:, rng.choice(file_idx, len(w), replace=False)] @ w)
cands = {int(j): P[:, j].astype(np.float64) for j in cand_idx}
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
ETAS = [0.8, 1.0, 1.2]


def greedy():
    pr = [t.predict(t.score(e)) for e in ETAS]
    return t.score(ETAS[int(np.argmax(pr))]), max(pr)


g, pg = greedy()
print(f"[{a.strategy}] base {b5:.5f}; K {a.k}; tilt public {roc_auc_score(y[pub], g[pub]):.5f} (pred {pg:.5f})", flush=True)
for r in range(a.t):
    if a.strategy == "greedy":
        key, s = None, g
    elif a.strategy == "random":
        key = list(cands)[rng.integers(len(cands))]
        s = cands[key]
    else:
        q = np.linalg.inv(t._cov(t.noise_var))
        sds = {j: t.predict_dist(v, q)[1] for j, v in cands.items()}
        key = max(sds, key=sds.get)
        s = cands[key]
    if key is not None:
        cands.pop(key)
    sc = auc5(s)
    t.add([s], [sc])
    exact = np.append(exact, True)
    t.fit_noise(nu=a.nu, exact=exact, verbose=False)
    g, pg = greedy()
    print(f"round {r + 1:2d}: probe {key} public {sc:.5f} | tilt public {roc_auc_score(y[pub], g[pub]):.5f} "
          f"(pred {pg:.5f}) private {roc_auc_score(y[~pub], g[~pub]):.5f}", flush=True)
