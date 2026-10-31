"""Tilt v4: tilt_run.py with the empirical-Bayes noise model of src/tilt_eb.py (per-file noise that grows with the
file's size, REML-fitted; Student-t weights for files whose score the others cannot explain).

    python src/tilt_run2.py --nu 4 --scan 0.6 1 1.5 2              # fit + predicted public per eta
    python src/tilt_run2.py --nu 4 --write 1.5 --name tilt-v4      # also write + log the candidate
Writes subs/<id>.csv tagged [public-LB fitted] - never a final pick.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB, onesided_em  # noqa: E402
from tilt_exp import TiltExp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="subs/0044_lbb-settf8-10.csv")
ap.add_argument("--base-score", type=float, default=0.96214)
ap.add_argument("--min-score", type=float, default=0.955)
ap.add_argument("--min-corr", type=float, default=0.90)
ap.add_argument("--nu", type=float, default=4.0, help="Student-t dof for the per-file weights (0 = Gaussian)")
ap.add_argument("--fixed", type=float, nargs=2, default=None, help="fixed (n0, n1) instead of REML")
ap.add_argument("--exclude", nargs="*", default=[], help="substrings of file names to leave out")
ap.add_argument("--exclude-file", default=None, help="file with one file name per line to leave out")
ap.add_argument("--holdout", nargs="*", default=[], help="substrings: fit without these files, then predict them")
ap.add_argument("--scan", type=float, nargs="*", default=[])
ap.add_argument("--write", type=float, nargs="*", default=[])
ap.add_argument("--name", default="tilt-v4")
ap.add_argument("--save", default="data/tilt/fit_eb.npz")
ap.add_argument("--onesided", action="store_true", help="one-sided score-error EM on the suspect files")
ap.add_argument("--suspect-file", default="data/tilt/suspect_files.txt")
ap.add_argument("--em-iters", type=int, default=8)
ap.add_argument("--exact-sources", nargs="+", default=["own"], help="sources whose scores are exact: their weight is never below 1")
ap.add_argument("--exp", action="store_true", help="exponential-tilt posterior (src/tilt_exp.py) for --scan and --write")
a = ap.parse_args()

ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / len(b)
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)

R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
keep = (files.score >= a.min_score).to_numpy().copy()
for s in a.exclude:
    keep &= ~files.name.str.contains(s, regex=False).to_numpy()
if a.exclude_file:
    keep &= ~files.name.isin(set(open(a.exclude_file).read().split())).to_numpy()
corr = np.array([np.corrcoef(R[i], rb)[0, 1] if keep[i] else 0 for i in range(len(files))])
keep &= corr >= a.min_corr
keep &= np.abs(corr - 1) > 1e-7
hold = np.zeros(len(files), bool)
for s in a.holdout:
    hold |= files.name.str.contains(s, regex=False).to_numpy()
sel = np.flatnonzero(keep & ~hold)
hsel = np.flatnonzero(keep & hold)
print(f"files {len(files)}, used {len(sel)}, held out {len(hsel)}")

t = (TiltExp if a.exp else TiltEB)(p, f=0.2).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], a.base_score)
exact = files.source.isin(a.exact_sources).to_numpy()[sel]
t.fit_noise(nu=a.nu, fixed=a.fixed, exact=exact)
if a.onesided:
    sus = files.name.isin(set(open(a.suspect_file).read().split())).to_numpy()[sel]
    print(f"one-sided EM on {int(sus.sum())} suspect files; plain tilt predicted {t.predict(t.score(1.0)):.5f}")
    kw = {"fixed": a.fixed} if a.fixed else {}
    onesided_em(t, nu=a.nu, exact=exact, suspect=sus, iters=a.em_iters, **kw)
if a.exp:
    print(f"gaussian tilt at eta 1 predicted {t.predict(t.score(1.0)):.5f}", flush=True)
    t.fit_exp(nu=a.nu, em=3, exact=exact)
    t.noise_var, t.w = t.nv_exp, t.w_exp
    t.lam = t.lam_exp
    q_ = t.q_inv
    t.loo = t.lam_exp / np.diag(q_)
    t.y = t.yraw - t._x() @ t.beta_exp
    t.loo_r2 = 1 - float(t.loo @ t.loo) / float(t.y @ t.y)
scale = 1e5 / (t.n1 * t.n0)
out = files.iloc[sel][["name", "source", "score"]].copy()
out["corr"] = corr[sel]
out["surprise_u"] = t.y * scale
out["loo_resid_u"] = t.loo * scale
out["noise_sd_u"] = np.sqrt(t.noise_var) * scale
out["w"] = t.w
out.to_csv("data/tilt/fit_eb_report.csv", index=False)
print(f"LOO R2 {t.loo_r2:.3f}; surprise sd {t.y.std() * scale:.2f}u; LOO resid sd {t.loo.std() * scale:.2f}u")
print(out.groupby("source").agg(n=("w", "size"), w_mean=("w", "mean"), w_lt05=("w", lambda v: (v < 0.5).sum()),
                                 noise_med_u=("noise_sd_u", "median"), resid_sd=("loo_resid_u", "std")).to_string())
print("lowest weights:")
print(out.sort_values("w").head(10).to_string())
print("own files (surprise vs LOO residual):")
print(out[out.source == "own"].tail(8).to_string())

def score_of(eta):
    return t.score_exp(eta) if a.exp else t.score(eta)


def pred_of(s):
    return t.predict_exp(s) if a.exp else t.predict(s)


for i in hsel:
    s = np.asarray(R[i], dtype=np.float64)
    print(f"held-out {files.name[i]}: predicted {pred_of(s):.5f} actual {files.score[i]:.5f}")
for eta in a.scan:
    print(f"eta {eta:5.2f}: predicted public {pred_of(score_of(eta)):.5f}", flush=True)
np.savez(a.save, da=t.da, db=t.db, abar=t.abar, bbar=t.bbar, p=p, sel=sel, noise_var=t.noise_var, w=t.w,
         lam=t.lam_exp if a.exp else t.lam)

from kgkit.experiment import Ledger  # noqa: E402

for eta in a.write:
    s = score_of(eta)
    rs = (rankdata(s) - 0.5) / len(s)
    rc = np.corrcoef(rs, rb)[0, 1]
    notes = (f"[public-LB fitted] tilt {'v13 exponential posterior' if a.exp else 'v4'} (EB noise k0 {t.k0:.3g} k1 {t.k1:.3g}, "
             f"Student-t nu {a.nu:g}) of {a.base} from {len(sel)} scored files, eta {eta:g} (rank corr {rc:.5f} with base, "
             f"predicted public {pred_of(s):.5f}); never a final pick")
    rec = Ledger().log(f"{a.name}-eta{eta:g}", None, model="tilt", notes=notes, tags=["tilt", "public-lb-fitted"],
                       params={"base": a.base, "eta": eta, "n_files": int(len(sel)), "nu": a.nu, "k0": float(t.k0),
                               "k1": float(t.k1), "min_corr": a.min_corr, "min_score": a.min_score, "exp": bool(a.exp)},
                       pool=False)
    path = f"subs/{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": rs if a.exp else s}).to_csv(path, index=False, float_format="%.9f")
    Ledger().update(rec["id"], submission=path)
    print(f"eta {eta}: rank corr with base {rc:.6f} -> {path}")
