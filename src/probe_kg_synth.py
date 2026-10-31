"""Knowledge gradient of SYNTHETIC probes for the public-label tilt (TiltEB.lookahead), to compare with model files
(src/probe_kg.py): exaggerated tilts (eta 1.5, 2, 3), the tilt applied only to the M rows it moves most (others keep
the prior), and the tilt minus those rows.

    python src/probe_kg_synth.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

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
t.fit_noise(nu=2.0, fixed=(1e-6, 80), exact=files.source.to_numpy()[sel] == "own")
q = np.linalg.inv(t._cov(t.noise_var))
s1 = t.score(1.0)
p0 = t.predict(s1)
print(f"current tilt: predicted {p0:.5f}", flush=True)
move = np.abs(rankdata(s1) - rankdata(p)) / N
order = np.argsort(-move)
probes = {f"eta{e:g}": t.score(e) for e in (0.7, 1.4)}
for m in (5000, 20000, 60000):
    top = np.zeros(N, bool)
    top[order[:m]] = True
    for e in (0.5, 0.75, 1.5):
        probes[f"top{m}-x{e:g}"] = np.where(top, t.score(e), s1)
    rest = ~top
    probes[f"rest{m}-x1.5"] = np.where(rest, t.score(1.5), s1)
nodes, wts = np.polynomial.hermite_e.hermegauss(5)
wts = wts / wts.sum()
for name, s in probes.items():
    rk = (rankdata(s) - 0.5) / N
    mu, sd = t.predict_dist(rk, q)
    sp, sn = t.last_parts
    preds = t.lookahead(rk, mu + sd * nodes, q=q)
    print(f"{name:16s} mu {mu:.5f} sd {sd * 1e5:5.1f}u (post {sp * 1e5:5.1f}u) KG {(np.dot(wts, preds) - p0) * 1e5:+6.2f}u",
          flush=True)
