"""Do shared model errors within a family of nudge probes predict held-out nudges better? Fit without one probe, predict its
score difference to the anchor, with and without a noise block over the rest of the family (Gaussian fit)."""
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
names = files.name.to_numpy()[sel].astype(str)
fam = ["0073", "0074", "0075", "0076"]
idx = {k: int(np.flatnonzero(np.char.find(names, f"subs/{k}") >= 0)[0]) for k in fam}
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
exact = files.source.to_numpy()[sel] == "own"
for h in fam[1:]:
    keep = np.setdiff1d(np.arange(len(sel)), [idx[h]])
    ts = t.subset(keep)
    ex = exact[keep]
    pos = {k: int(np.flatnonzero(keep == idx[k])[0]) for k in fam if k != h}
    rh = np.asarray(R[sel[idx[h]]], np.float64)
    ra = np.asarray(R[sel[idx["0073"]]], np.float64)
    act = files.score.to_numpy()[sel[idx[h]]] - files.score.to_numpy()[sel[idx["0073"]]]
    for blocks in (None, [np.array(list(pos.values()))]):
        ts.noise_blocks = blocks
        ts.fit_noise(nu=2, exact=ex, verbose=False)
        print(f"hold out {h} blocks {blocks is not None}: k1 {ts.k1:.2f} REML {ts.reml(ts.noise_var):.2f} pred diff to 0073 {(ts.predict(rh) - ts.predict(ra)) * 1e5:+.2f}u actual {act * 1e5:+.2f}u; pred own tilt {ts.predict(ts.score(1.0)):.5f}", flush=True)
