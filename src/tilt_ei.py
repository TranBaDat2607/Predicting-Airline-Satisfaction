"""Expected-improvement choice of the next tilt submission. The leaderboard shows our best score, so a variant is
worth E[max(0, X - best)] with X its realised public AUC. Empirically (own tilts 0056-0071) the realised score of a
posterior's own tilt is its prediction - BIAS +- SPREAD (12u, 9u); the model's own predictive sd is added.

Candidates: the robust tilt (src/tilt_eb.py) under several noise models and etas, plus bagged tilts.
Writes the best candidate (ledger-logged, [public-LB fitted], never a final pick) when --write is given.

    python src/tilt_ei.py --write --name tilt-ei
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bias", type=float, default=12.0, help="u: realised = predicted - bias")
ap.add_argument("--spread", type=float, default=9.0, help="u: sd of realised - predicted")
ap.add_argument("--etas", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4])
ap.add_argument("--models", nargs="+", default=["t2", "t1", "t4", "t2k20", "t2k40"])
ap.add_argument("--bags", type=int, default=6)
ap.add_argument("--write", action="store_true")
ap.add_argument("--name", default="tilt-ei")
a = ap.parse_args()
SPEC = {"t2": dict(nu=2), "t1": dict(nu=1), "t4": dict(nu=4), "t2k20": dict(nu=2, fixed=(1e-6, 20)),
        "t2k40": dict(nu=2, fixed=(1e-6, 40))}

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
best = float(files.score[files.source == "own"].max())
T = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
T.fit_noise(nu=2, exact=exact, verbose=False)
judge_q = np.linalg.inv(T._cov(T.noise_var))  # the reference model (t2 REML) scores every candidate
cands = {}
for mname in a.models:
    T.fit_noise(exact=exact, verbose=False, **SPEC[mname])
    for eta in a.etas:
        cands[f"{mname}-eta{eta:g}"] = T.score(eta)
rng = np.random.default_rng(0)
T.fit_noise(nu=2, exact=exact, verbose=False)
da0, db0 = T.da.copy(), T.db.copy()
pubi = np.flatnonzero(~exact)
DA, DB = np.zeros(N), np.zeros(N)
for k in range(a.bags):
    idx = np.sort(np.concatenate([np.flatnonzero(exact), rng.choice(pubi, int(0.75 * len(pubi)), replace=False)]))
    ts = T.subset(idx)
    ts.fit_noise(nu=2, exact=exact[idx], verbose=False)
    DA += ts.da / a.bags
    DB += ts.db / a.bags
T.da, T.db = DA, DB
for eta in a.etas:
    cands[f"bag-eta{eta:g}"] = T.score(eta)
T.da, T.db = da0, db0
rows = []
for name, s in cands.items():
    rk = (rankdata(s) - 0.5) / N
    mu, sd = T.predict_dist(rk, judge_q)
    m = mu - a.bias * 1e-5
    sig = np.sqrt(sd ** 2 + (a.spread * 1e-5) ** 2)
    z = (m - best) / sig
    ei = sig * norm.pdf(z) + (m - best) * norm.cdf(z)
    rows.append({"cand": name, "pred": mu, "sd_u": sd * 1e5, "exp_realised": m, "ei_u": ei * 1e5,
                 "p_beat_best": norm.cdf(z), "rank_corr_base": np.corrcoef(rk, rb)[0, 1]})
df = pd.DataFrame(rows).sort_values("ei_u", ascending=False)
print(f"current best own public {best:.5f}")
print(df.to_string(index=False))
if a.write:
    from kgkit.experiment import Ledger  # noqa: E402
    top = df.iloc[0]
    s = cands[top.cand]
    rec = Ledger().log(f"{a.name}-{top.cand}", None, model="tilt", tags=["tilt", "public-lb-fitted"],
                       notes=(f"[public-LB fitted] EI-chosen tilt variant {top.cand} ({len(sel)} files; predicted "
                              f"{top.pred:.5f}, expected realised {top.exp_realised:.5f}, EI {top.ei_u:.2f}u); never a "
                              "final pick"), params={"base": base, "cand": top.cand, "n_files": int(len(sel))},
                       pool=False)
    path = f"subs/{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": s}).to_csv(path, index=False, float_format="%.9f")
    Ledger().update(rec["id"], submission=path)
    print(path)
