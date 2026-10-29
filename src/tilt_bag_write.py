"""Bagged public-label tilt: average the robust posterior shift (src/tilt_eb.py) over B random subsets of the scored
files (our own exact files are in every bag), write the tilt as a submission ([public-LB fitted], never a final
pick). See src/tilt_bag_sim.py for when bagging realises more public AUC than the full posterior.

    python src/tilt_bag_write.py --bags 8 --frac 0.75 --eta 1.0 --name tilt-bag
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402
from kgkit.experiment import Ledger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bags", type=int, default=8)
ap.add_argument("--frac", type=float, default=0.75)
ap.add_argument("--eta", type=float, default=1.0)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default="tilt-bag")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
base = "subs/0044_lbb-settf8-10.csv"
b = pd.read_csv(base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / N
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
exact = files.source.to_numpy()[sel] == "own"
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
t.fit_noise(nu=a.nu, exact=exact)
full = t.predict(t.score(a.eta))
DA, DB = np.zeros(N), np.zeros(N)
pub = np.flatnonzero(~exact)
for k in range(a.bags):
    idx = np.sort(np.concatenate([np.flatnonzero(exact), rng.choice(pub, int(a.frac * len(pub)), replace=False)]))
    ts = t.subset(idx)
    ts.fit_noise(nu=a.nu, exact=exact[idx], verbose=False)
    DA += ts.da / a.bags
    DB += ts.db / a.bags
    print(f"bag {k + 1}: own-tilt predicted by the bag {ts.predict(ts.score(a.eta)):.5f}", flush=True)
t.da, t.db = DA, DB
s = t.score(a.eta)
pred = t.predict(s)
rc = np.corrcoef((rankdata(s) - 0.5) / N, rb)[0, 1]
print(f"full-posterior tilt predicted {full:.5f}; bagged tilt predicted (by the full model) {pred:.5f}; rank corr {rc:.5f}")
rec = Ledger().log(f"{a.name}-eta{a.eta:g}", None, model="tilt", tags=["tilt", "public-lb-fitted"],
                   notes=(f"[public-LB fitted] bagged tilt: mean posterior shift over {a.bags} bags of {a.frac:g} of "
                          f"the public files (own files in every bag), {len(sel)} files, eta {a.eta:g}, predicted "
                          f"{pred:.5f}; never a final pick"),
                   params={"base": base, "eta": a.eta, "bags": a.bags, "frac": a.frac, "nu": a.nu, "seed": a.seed},
                   pool=False)
path = f"subs/{rec['id']}.csv"
pd.DataFrame({"id": ids, "satisfaction": s}).to_csv(path, index=False, float_format="%.9f")
Ledger().update(rec["id"], submission=path)
print(path)
