"""Does one Gauss-Newton re-linearisation (TiltGN) remove the optimism of the tilt's own predictions? Fit the
robust tilt, re-linearise around its posterior mean, refit; then leave-own-tilt-out predictions for both models
and each model's predicted public for its own new tilt.

    python src/tilt_gn_check.py --holdout 0068 0067 0066 [--write 1.0 --name tilt-v10-gn]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB, TiltGN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--fixed", type=float, nargs=2, default=None)
ap.add_argument("--eta1", type=float, default=1.0, help="linearisation point: first fit's tilt at this eta")
ap.add_argument("--holdout", nargs="*", default=["0068", "0067", "0066"])
ap.add_argument("--write", type=float, nargs="*", default=[])
ap.add_argument("--name", default="tilt-gn")
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
names = files.name.to_numpy()[sel].astype(str)
exact = files.source.to_numpy()[sel] == "own"
F = lambda: (np.asarray(R[i], dtype=np.float64) for i in sel)  # noqa: E731
sc = files.score.to_numpy()[sel]
t1 = TiltEB(p).fit(list(F()), sc, 0.96214)
t1.fit_noise(nu=a.nu, fixed=a.fixed, exact=exact, verbose=True)
ah, bh = t1.abar + a.eta1 * t1.da, t1.bbar + a.eta1 * t1.db
t2 = TiltGN(p, ah, bh, a.eta1 * t1.da, a.eta1 * t1.db).fit(list(F()), sc, 0.96214)
t2.fit_noise(nu=a.nu, fixed=a.fixed, exact=exact, verbose=True)
for name, t in (("linear", t1), ("gauss-newton", t2)):
    errs = []
    for hname in a.holdout:
        j = [i for i, n in enumerate(names) if f"/{hname}_" in n]
        if not j:
            continue
        keep = np.setdiff1d(np.arange(len(sel)), j)
        ts = t.subset(keep)
        ts.fit_noise(nu=a.nu, fixed=a.fixed, exact=exact[keep], verbose=False)
        pr = ts.predict(np.asarray(R[sel[j[0]]], dtype=np.float64))
        errs.append(f"{hname} pred {pr:.5f} obs {sc[j[0]]:.5f} ({(pr - sc[j[0]]) * 1e5:+.1f}u)")
    print(f"{name:13s} | " + " | ".join(errs) + f" || own tilt eta1 predicted {t.predict(t.score(1.0)):.5f}", flush=True)
s_gn = t2.score(1.0)
print(f"rank corr GN tilt vs linear tilt {np.corrcoef(rankdata(s_gn), rankdata(t1.score(1.0)))[0, 1]:.5f}")
if a.write:
    from kgkit.experiment import Ledger  # noqa: E402
    for eta in a.write:
        s = t2.score(eta)
        rec = Ledger().log(f"{a.name}-eta{eta:g}", None, model="tilt", tags=["tilt", "public-lb-fitted"],
                           notes=(f"[public-LB fitted] Gauss-Newton tilt (re-linearised at the first fit's eta "
                                  f"{a.eta1:g} tilt), {len(sel)} files, eta {eta:g}, predicted {t2.predict(s):.5f}; "
                                  "never a final pick"),
                           params={"base": base, "eta": eta, "n_files": int(len(sel)), "nu": a.nu, "gn": True},
                           pool=False)
        path = f"subs/{rec['id']}.csv"
        pd.DataFrame({"id": ids, "satisfaction": s}).to_csv(path, index=False, float_format="%.9f")
        Ledger().update(rec["id"], submission=path)
        print(path)
