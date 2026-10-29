"""Is Kaggle's public/private split of the test rows structured? The tilt posterior's shift of each row's public
membership, d_pub = E[a + b | scores] - f, should be unrelated to the row id under a random split. Correlations
with id order, id parity / residues and contiguous id blocks are compared with their null sd (1/sqrt(n)).

    python src/split_check.py --noise 24
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
ap.add_argument("--noise", type=float, default=24)
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
sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
t = Tilt(p, noise_units=a.noise).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel],
                                     a.base_score)
dpub = t.da + t.db
dpos = t.da / np.maximum(t.abar, 1e-9) - t.db / np.maximum(t.bbar, 1e-9)
n = len(ids)
print(f"test ids {ids.min()}..{ids.max()} (n {n}, contiguous {ids.max() - ids.min() + 1 == n}); null sd {1 / np.sqrt(n):.4f}")
z = (dpub - dpub.mean()) / dpub.std()
order = np.argsort(ids)
print(f"corr(d_pub, id) {np.corrcoef(z, ids)[0, 1]:+.4f}   corr(d_pub, row index) {np.corrcoef(z, np.arange(n))[0, 1]:+.4f}")
for k in (2, 3, 4, 5, 10):
    g = ids % k
    means = [z[g == r].mean() for r in range(k)]
    print(f"id mod {k}: mean z per residue " + " ".join(f"{m:+.4f}" for m in means) + f"  (null sd {np.sqrt(k / n):.4f})")
for nb in (5, 10, 50):
    blk = np.minimum((rankdata(ids) - 1) * nb // n, nb - 1).astype(int)
    means = np.array([z[blk == j].mean() for j in range(nb)])
    print(f"{nb} id blocks: sd of block means {means.std():.4f} (null {np.sqrt(nb / n):.4f}), max |mean| {np.abs(means).max():.4f}")
# the same for the label direction (control: should also be flat)
zp = (dpos - dpos.mean()) / dpos.std()
print(f"control: corr(d_pos, id) {np.corrcoef(zp, ids)[0, 1]:+.4f}")
