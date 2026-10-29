"""Knowledge-gradient ranking of candidate probes (unscored own/pool files) for the public-label tilt.

For each candidate: its predictive distribution N(mu, sd) of the public score under the robust tilt posterior,
then at Gauss-Hermite outcomes y the re-fitted tilt's predicted public score (TiltEB.lookahead). KG = expected
predicted public of the re-fitted tilt minus that of the current tilt. Candidates from data/tilt/probe_rank.csv
(top --top by posterior sd) and their source arrays.

    python src/probe_kg.py --top 60
"""
import argparse
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
ap.add_argument("--top", type=int, default=60)
ap.add_argument("--eta", type=float, default=1.0)
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
p0 = t.predict(t.score(a.eta))
print(f"current tilt eta {a.eta}: predicted public {p0:.5f}", flush=True)

pt = np.load("artifacts/pool/test.npy", mmap_mode="r")
pool_names = pd.read_csv("artifacts/pool/members.csv").name.tolist()


def load(name):
    if name.startswith("pool:"):
        return np.asarray(pt[:, pool_names.index(name[5:])], dtype=np.float64)
    if name.startswith("llr-v2:"):
        return np.load("data/ext/kout-s6e10-llr-v2/llr_test.npy")[:, int(name.split(":")[1])].astype(np.float64)
    return np.load(name).astype(np.float64)


nodes, wts = np.polynomial.hermite_e.hermegauss(5)
wts = wts / wts.sum()
pr = pd.read_csv("data/tilt/probe_rank.csv").head(a.top)
rows = []
for r in pr.itertuples():
    v = load(r.name)
    rk = (rankdata(v) - 0.5) / N
    mu, sd = t.predict_dist(rk, q)
    preds = t.lookahead(rk, mu + sd * nodes, eta=a.eta, q=q)
    kg = float(np.dot(wts, preds) - p0)
    rows.append({"name": r.name, "mu": mu, "sd_u": sd * 1e5, "kg_u": kg * 1e5,
                 "best_case_u": (max(preds) - p0) * 1e5, "worst_case_u": (min(preds) - p0) * 1e5})
    print(f"{r.name[:70]:70s} mu {mu:.5f} sd {sd * 1e5:5.1f}u  KG {kg * 1e5:+6.2f}u", flush=True)
df = pd.DataFrame(rows).sort_values("kg_u", ascending=False)
df.to_csv("data/tilt/probe_kg.csv", index=False)
print(df.head(15).to_string())
