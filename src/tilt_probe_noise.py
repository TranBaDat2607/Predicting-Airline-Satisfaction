"""One random-nudge probe around a tilt: Gaussian noise of size t (log-odds) on the rows where the log-odds density peaks,
which maximises the knowledge gradient of a measurement (src/probe_kg_noise.py: about +0.8u per submission). Each probe is a
new direction for the next tilt fit, and sometimes a new best itself. Ledger-logged [public-LB fitted]; never a final pick.

    python src/tilt_probe_noise.py --tilt subs/0073_x.csv --seed 1 --t 0.08 --mask low45 --name probe-n1
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from kgkit.experiment import Ledger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tilt", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--t", type=float, default=0.08)
ap.add_argument("--mask", default="low45", choices=["low45", "high40", "all"])
ap.add_argument("--name", required=True)
a = ap.parse_args()
ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv("subs/0044_lbb-settf8-10.csv").set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / N
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
cur = pd.read_csv(a.tilt).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rk = (rankdata(cur) - 0.5) / N
z = np.sort(logit(p))[np.rint(rk * N + 0.5).astype(int) - 1]
mask = {"low45": z < np.quantile(z, 0.45), "high40": z > np.quantile(z, 0.60), "all": np.ones(N, bool)}[a.mask]
n = np.random.default_rng(a.seed).standard_normal(N) * a.t * mask
r2 = (rankdata(z + n) - 0.5) / N
rec = Ledger().log(a.name, None, model="tilt", tags=["tilt", "public-lb-fitted", "probe"], pool=False,
                   notes=(f"[public-LB fitted] nudge probe of {a.tilt}: N(0, {a.t:g}) log-odds on {a.mask}, seed {a.seed} "
                          f"(rank-diff sd {(r2 - rk).std():.5f}); never a final pick"),
                   params={"tilt": a.tilt, "seed": a.seed, "t": a.t, "mask": a.mask})
path = f"subs/{rec['id']}.csv"
pd.DataFrame({"id": ids, "satisfaction": r2}).to_csv(path, index=False, float_format="%.9f")
Ledger().update(rec["id"], submission=path)
print(path, f"rank-diff sd {(r2 - rk).std():.5f}")
