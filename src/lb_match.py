"""Pick, for every multi-version scored notebook, the version output that matches its listed (best-version) score.

Robust tilt fit (src/tilt_eb.py, Student-t) on the current files; for each notebook the posterior WITHOUT that
notebook predicts the public score of every candidate output (latest = data/tilt/k, older = data/tilt/kv/...).
The candidate with the smallest |z| = |listed - predicted| / predictive sd wins if it beats the latest output by
a likelihood ratio > --lr. Winners are copied to data/tilt/k_fixed/<owner>__<slug>/submission.csv, which
src/lb_collect.py prefers over the latest output.

    python src/lb_match.py --nu 2
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, "src")
from tilt_eb import TiltEB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="subs/0044_lbb-settf8-10.csv")
ap.add_argument("--base-score", type=float, default=0.96214)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=20.0, help="likelihood ratio needed to replace the latest output")
ap.add_argument("--zmax", type=float, default=3.0, help="the chosen version must itself fit (|z| below this)")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
qs = np.sort(oof if oof.max() <= 1 else expit(oof))
b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
rb = (rankdata(b) - 0.5) / len(b)
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
files = pd.read_csv("data/tilt/files.csv")
corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
names = files.name.to_numpy()[sel].astype(str)
T = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], a.base_score)
exact = files.source.to_numpy()[sel] == "own"
T.fit_noise(nu=a.nu, exact=exact)


def read(path):
    d = pd.read_csv(path)
    if "id" not in d.columns or len(d) != N:
        return None
    col = "satisfaction" if "satisfaction" in d.columns else d.columns[-1]
    v = pd.to_numeric(d.set_index("id")[col], errors="coerce").reindex(ids).to_numpy(float)
    if not np.isfinite(v).all() or np.unique(v[:5000]).size < 50:
        return None
    return (rankdata(v) - 0.5) / N


kern = pd.read_csv("data/tilt/kernels.csv").set_index("ref").best_public_score
rows = []
for d in sorted(Path("data/tilt/kv").iterdir()):
    ref = d.name.replace("__", "/", 1)
    if ref not in kern.index or not kern[ref] > 0:
        continue
    score = float(kern[ref])
    cands = [("latest", Path("data/tilt/k") / d.name / "submission.csv")]
    cands += [(f"{v.name}/{f.name}", f) for v in sorted(d.glob("v*"), key=lambda x: -int(x.name[1:]))
              for f in sorted(v.glob("*.csv"))]
    cands = [(c, f) for c, f in cands if f.exists()]
    has_latest = any(c == "latest" for c, _ in cands)
    if len(cands) < (2 if has_latest else 1):
        continue
    j = np.flatnonzero(names == ref)
    t = T.subset(np.setdiff1d(np.arange(len(sel)), j))
    t.noise_var = np.delete(T.noise_var, j)
    t.w = np.delete(T.w, j)
    t.solve_nv(t.noise_var)
    q = np.linalg.inv(t._cov(t.noise_var))
    res = []
    seen = set()
    for c, f in cands:
        r = read(f)
        if r is None:
            continue
        key = hash(np.round(r * 1e5).astype(np.int32).tobytes())
        if key in seen:
            continue
        seen.add(key)
        mu, sd = t.predict_dist(r, q)
        res.append((c, f, mu, sd, (score - mu) / sd))
    if not res:
        continue
    lat = [x for x in res if x[0] == "latest"]
    best = min(res, key=lambda x: abs(x[4]))
    zl = lat[0][4] if lat else np.inf
    lr = np.exp(min(0.5 * (zl ** 2 - best[4] ** 2), 700)) if np.isfinite(zl) else np.inf
    # no usable latest output (src/lb_fetch_missing.py): take the best-fitting candidate if it fits
    swap = best[0] != "latest" and abs(best[4]) < a.zmax and (lr > a.lr or not has_latest)
    rows.append({"ref": ref, "score": score, "n_cand": len(res), "z_latest": zl, "best": best[0], "z_best": best[4],
                 "pred_best": best[2], "sd_best_u": best[3] * 1e5, "lr": lr, "swap": swap})
    print(f"{ref}: {len(res)} distinct outputs, z latest {zl:+.2f}, best {best[0]} z {best[4]:+.2f} "
          f"(pred {best[2]:.5f} sd {best[3] * 1e5:.1f}u vs listed {score:.5f}){'  -> SWAP' if swap else ''}", flush=True)
    if swap and not a.dry:
        out = Path("data/tilt/k_fixed") / d.name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(best[1], out / "submission.csv")
        (out / "source.txt").write_text(f"{best[0]} z {best[4]:+.2f} (latest z {zl:+.2f})\n")
df = pd.DataFrame(rows)
df.to_csv("data/tilt/match_report.csv", index=False)
if len(df):
    print(f"{len(df)} notebooks with older versions; swaps {int(df.swap.sum())}; |z latest| > 3: {(df.z_latest.abs() > 3).sum()}")
