"""Why did a tilt realise far under its prediction? Refit on file subsets (leave out probes, the newest mined pairs, ...), then score
each candidate under the posterior of ALL files including the failed tilt's own score.

    python src/tilt_diag.py --failed 0077 --group probes=0074,0075,0076 --group newmined=x_kava1,x_hittmg
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_exp import TiltExp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--failed", default="0077")
ap.add_argument("--anchor", default="0073")
ap.add_argument("--group", action="append", default=[])
ap.add_argument("--nu", type=float, default=2.0)
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
names = files.name.to_numpy()[sel].astype(str)
t = TiltExp(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
exact = files.source.to_numpy()[sel] == "own"
print(f"{len(sel)} files", flush=True)
groups = {}
for g in a.group:
    k, v = g.split("=")
    groups[k] = np.array([any(s in n for s in v.split(",")) for n in names])
    print(f"group {k}: {groups[k].sum()} files", flush=True)
failed = np.array([f"subs/{a.failed}" in n for n in names])
anchor_r = np.asarray(R[sel[np.flatnonzero([f"subs/{a.anchor}" in n for n in names])[0]]], np.float64)
failed_r = np.asarray(R[sel[np.flatnonzero(failed)[0]]], np.float64)
print(f"rank-diff sd failed vs anchor {(failed_r - anchor_r).std():.5f}", flush=True)
cands = {}
combos = {"all-no-failed": []}
for k in groups:
    combos[f"no-{k}"] = [k]
if len(groups) > 1:
    combos["no-" + "+".join(groups)] = list(groups)
for name, gs in combos.items():
    drop = failed.copy()
    for g in gs:
        drop |= groups[g]
    keep = np.flatnonzero(~drop)
    ts = t.subset(keep)
    ts.fit_noise(nu=a.nu, exact=exact[keep], verbose=False)
    ts.fit_exp(nu=a.nu, em=3, exact=exact[keep], verbose=False)
    s = ts.score_exp(1.0)
    rk = (rankdata(s) - 0.5) / N
    cands[name] = rk
    print(f"{name:24s} files {len(keep)} k1 {ts.k1:.2f} own pred {ts.predict_exp(s):.5f} rank-diff sd to anchor {(rk - anchor_r).std():.5f} to failed {(rk - failed_r).std():.5f} corr base {np.corrcoef(rk, rb)[0, 1]:.5f}", flush=True)
t.fit_noise(nu=a.nu, exact=exact, verbose=False)
t.fit_exp(nu=a.nu, em=3, exact=exact, verbose=False)
s = t.score_exp(1.0)
rk = (rankdata(s) - 0.5) / N
cands["ALL"] = rk
print(f"ALL files k1 {t.k1:.2f} own pred {t.predict_exp(s):.5f} rank-diff sd to anchor {(rk - anchor_r).std():.5f} corr base {np.corrcoef(rk, rb)[0, 1]:.5f}", flush=True)
print("under the posterior of ALL files:", flush=True)
for name, r in cands.items():
    print(f"  {name:24s} predicted {t.predict_exp(r):.5f}", flush=True)
print(f"  anchor predicted {t.predict_exp(anchor_r):.5f}; failed predicted {t.predict_exp(failed_r):.5f}", flush=True)
np.savez("data/tilt/diag_cands.npz", **cands)
