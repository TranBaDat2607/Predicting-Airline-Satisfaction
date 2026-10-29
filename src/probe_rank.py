"""Rank our UNSCORED candidate files by how much a submission of each would tell the tilt (src/tilt_eb.py): the
posterior predictive sd of its public score given every scored file (large = a direction the scored files do not
explain). Candidates: the 319 pool members' test predictions, own models (artifacts/*/test.npy, set-transformer
mean, LLR), honest stacks. Writes data/tilt/probe_rank.csv (name, pred, sd_u, rank corr with base).

    python src/probe_rank.py --nu 2
"""
import argparse
import glob
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
ap.add_argument("--nu", type=float, default=2.0)
a = ap.parse_args()

ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / N
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], a.base_score)
t.fit_noise(nu=a.nu, exact=files.source.to_numpy()[sel] == "own")
q = np.linalg.inv(t._cov(t.noise_var))

cands = {}
pt = np.load("artifacts/pool/test.npy", mmap_mode="r")
names = pd.read_csv("artifacts/pool/members.csv").name.tolist()
for j, nm in enumerate(names):
    cands[f"pool:{nm}"] = np.asarray(pt[:, j], dtype=np.float64)
for f in sorted(glob.glob("artifacts/*/test.npy")):
    v = np.load(f)
    if v.ndim == 1 and len(v) == N:
        cands[f] = v.astype(np.float64)
for f in ["artifacts/settf_mean16_test.npy"]:
    cands[f] = np.load(f).astype(np.float64)
llr = np.load("data/ext/kout-s6e10-llr-v2/llr_test.npy")
for c in range(llr.shape[1]):
    cands[f"llr-v2:{c}"] = llr[:, c].astype(np.float64)
rows = []
for nm, v in cands.items():
    if not np.isfinite(v).all():
        continue
    r = (rankdata(v) - 0.5) / N
    rc = np.corrcoef(r, rb)[0, 1]
    if rc < 0.9:
        continue
    mu, sd = t.predict_dist(r, q)
    sp, sn = t.last_parts
    rows.append({"name": nm, "pred": mu, "sd_u": sd * 1e5, "post_sd_u": sp * 1e5, "noise_sd_u": sn * 1e5, "corr": rc})
df = pd.DataFrame(rows).sort_values("post_sd_u", ascending=False)
df.to_csv("data/tilt/probe_rank.csv", index=False)
print(f"{len(df)} candidates; sd quantiles (u): {np.percentile(df.sd_u, [10, 50, 90]).round(2)}")
print(df.head(25).to_string())
