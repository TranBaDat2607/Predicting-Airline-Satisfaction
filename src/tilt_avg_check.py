"""Predicted public score (robust tilt posterior, all scored files incl. our tilts) of rank averages of our scored
tilt files, next to the posterior's own new tilt. Averages of measured files are interpolations, which the model
predicts better than its own posterior-mean tilt (that one has been 13-18u optimistic).

    python src/tilt_avg_check.py --fixed 1e-6 80
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fixed", type=float, nargs=2, default=None)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--subs", nargs="+", default=["0062", "0063", "0064", "0065", "0066"])
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
t.fit_noise(nu=a.nu, fixed=a.fixed, exact=files.source.to_numpy()[sel] == "own")
own = {}
for s in a.subs:
    i = [j for j, n in enumerate(files.name) if f"/{s}_" in str(n)]
    own[s] = (np.asarray(R[i[0]], dtype=np.float64), files.score[i[0]])
for s, (r, sc) in own.items():
    print(f"{s}: observed {sc:.5f} predicted (in-sample) {t.predict(r):.5f}")
print(f"posterior tilt eta 1: predicted {t.predict(t.score(1.0)):.5f}")
for k in (2, 3, len(own)):
    for combo in itertools.combinations(own, k):
        r = np.mean([own[c][0] for c in combo], 0)
        print(f"avg {'+'.join(combo)}: predicted {t.predict(r):.5f}")
