"""Write one synthetic tilt probe as a submission (ledger-logged, [public-LB fitted], never a final pick).

  topM-xE : the tilt at eta 1, with the M rows it moves most taken to eta E (src/probe_kg_synth.py ranks these)
  etaE    : the plain tilt at eta E

    python src/tilt_probe_write.py --probe top10000-x2 --name probe-top10k-x2
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402
from kgkit.experiment import Ledger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--probe", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--nu", type=float, default=2.0)
a = ap.parse_args()

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
t = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
t.fit_noise(nu=a.nu, exact=files.source.to_numpy()[sel] == "own")
s1 = t.score(1.0)
m = re.fullmatch(r"top(\d+)-x([\d.]+)", a.probe)
if m:
    k, e = int(m.group(1)), float(m.group(2))
    move = np.abs(rankdata(s1) - rankdata(p))
    top = np.zeros(N, bool)
    top[np.argsort(-move)[:k]] = True
    s = np.where(top, t.score(e), s1)
else:
    s = t.score(float(a.probe.removeprefix("eta")))
pred = t.predict(s)
q = np.linalg.inv(t._cov(t.noise_var))
mu, sd = t.predict_dist((rankdata(s) - 0.5) / N, q)
rc = np.corrcoef((rankdata(s) - 0.5) / N, rb)[0, 1]
print(f"{a.probe}: predicted public {pred:.5f} (sd {sd * 1e5:.1f}u), rank corr with base {rc:.5f}, {len(sel)} files")
notes = (f"[public-LB fitted] tilt probe {a.probe} (Student-t nu {a.nu:g}, {len(sel)} scored files; predicted public "
         f"{pred:.5f}, sd {sd * 1e5:.1f}u, KG-chosen); never a final pick")
rec = Ledger().log(a.name, None, model="tilt", notes=notes, tags=["tilt", "public-lb-fitted", "probe"],
                   params={"base": base, "probe": a.probe, "n_files": int(len(sel)), "nu": a.nu}, pool=False)
path = f"subs/{rec['id']}.csv"
pd.DataFrame({"id": ids, "satisfaction": s}).to_csv(path, index=False, float_format="%.9f")
Ledger().update(rec["id"], submission=path)
print(path)
