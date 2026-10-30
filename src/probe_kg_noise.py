"""Knowledge gradient of random-nudge probes around the current best tilt (src/tilt_eb.TiltEB.lookahead). Theory: for a row-level
direction the gain of one measurement is 1/2 sum(v^2 V)/sum(v^2 H), and V_i/H_i = public rows per unit log-odds at row i, so
nudges should sit where the log-odds density peaks. Compared: nudge sizes and row masks.

    python src/probe_kg_noise.py --tilt subs/0072_tilt-v13-exp-eta1.csv
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tilt", default="subs/0072_tilt-v13-exp-eta1.csv")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv("subs/0044_lbb-settf8-10.csv").set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / N
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
t.fit_noise(nu=2.0, exact=files.source.to_numpy()[sel] == "own")
q = np.linalg.inv(t._cov(t.noise_var))
cur = pd.read_csv(a.tilt).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rk = (rankdata(cur) - 0.5) / N
scale = np.sort(logit(p))
z = scale[np.rint(rk * N + 0.5).astype(int) - 1]
hist, edges = np.histogram(z, bins=40)
print("rows per unit log-odds:", np.round(hist / np.diff(edges) / 1000).astype(int).tolist(), "edges", edges[[0, -1]].round(2), flush=True)
s1 = t.score(1.0)
p0 = t.predict(s1)
print(f"Gaussian posterior tilt predicted {p0:.5f}; current file predicted {t.predict(rk):.5f}", flush=True)
nodes, wts = np.polynomial.hermite_e.hermegauss(5)
wts = wts / wts.sum()
rng = np.random.default_rng(a.seed)
lo = z < np.quantile(z, 0.45)
hi = z > np.quantile(z, 0.60)
masks = {"low45": lo, "high40": hi, "all": np.ones(N, bool), "low20": z < np.quantile(z, 0.2), "mid": (z > -2.5) & (z < 2.5)}
for mname, m in masks.items():
    for tt in (0.03, 0.06, 0.12, 0.25):
        n = rng.standard_normal(N) * tt * m
        r2 = (rankdata(z + n) - 0.5) / N
        mu, sd = t.predict_dist(r2, q)
        sp, sn = t.last_parts
        preds = t.lookahead(r2, mu + sd * nodes, q=q)
        print(f"{mname:7s} t {tt:4.2f} rank-diff sd {(r2 - rk).std():.5f} mu {mu:.5f} sd {sd * 1e5:5.2f}u (post {sp * 1e5:5.2f}u noise {sn * 1e5:4.2f}u) KG {(np.dot(wts, preds) - p0) * 1e5:+6.2f}u", flush=True)
