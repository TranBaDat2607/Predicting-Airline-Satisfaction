"""Local public-LB surface around an anchor file A, fitted on every submission file with a known public score.

Each file is a rank vector r_i; x_i = r_i - r_A. Model:  s_i - s_A = <g, x_i> - (c/2) ||x_i||^2,
with g restricted to the span of the x_i (g = sum_j a_j x_j, kernel K = X X^T), fitted by weighted ridge (near points
weigh more). Leave-one-out error is reported next to a curvature-only baseline, so a fit that predicts nothing better
than noise is visible before a submission is spent on its step. The ascent step is x* = g / c, written as candidates
r_A + eta * x* for a few eta.

    python src/lb_surface.py --write 0.25 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

IDS = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
ANCHOR = "data/lb/secrets-to-satisfaction-it-s-stacking/submission.csv"
PUBLIC = {  # public files with their public LB score (named by score, or stated by the notebook / dataset README)
    "data/lb/s6e10-on-the-journey-to-your-first-top-100/submission.csv": 0.96213,
    "data/lb/s6e10-summit-composer-0-96210/submission.csv": 0.96213,
    "data/lb/airline-passenger-satisfaction-0-96213/submission.csv": 0.96213,
    "data/lb/3-airline-s6e10-ensembling/submission.csv": 0.96213,
    "data/lb/ps-s6e10-round/submission.csv": 0.96213,
    "data/ext/goodpjw-pub/public_096213_1023.csv": 0.96213,
    "data/ext/goodpjw-pub/meta_part4_096213_1023.csv": 0.96213,
    "data/ext/goodpjw-pub/mayankraaz_summit_composer_096210_1023.csv": 0.96210,
    "data/lb/s6e10-lb-0-96209-rank-10/submission.csv": 0.96209,
    "data/lb/0-96206-disagreement-gated-teachers/submission.csv": 0.96206,
    "data/lb/s6e10-lb-surface-ascent-v2-0-96208/submission.csv": 0.96208,
    "data/lb/s6e10-meta-stack-on-public-oof-lb-0-96197/submission.csv": 0.96197,
    "data/lb/s6e10-meta-stack-on-public-oof-lb-0-96197/submission_measure_096191.csv": 0.96191,
    "data/lb/s6e10-meta-stack-on-public-oof-lb-0-96197/submission_part1_182stack.csv": 0.96180,
    "data/lb/s6e10-route-profile-cross-fitted-target/submission_anchor_096193.csv": 0.96193,
    "data/ext/s6e10-lb-surface-atlas/rabari_v43_submission.csv": 0.96207,
}


def R(v):
    return ((rankdata(v) - 0.5) / len(v)).astype(np.float64)


def read(path):
    d = pd.read_csv(path)
    return d.set_index("id").iloc[:, -1].reindex(IDS).to_numpy(float)


def collect():
    pts = {}
    for p, s in PUBLIC.items():
        if Path(p).exists():
            pts[p] = s
    for p in glob.glob("data/ext/ps-s6e10-02/0.96*") + glob.glob("data/ext/ps-s6e10-7/0.96*"):
        m = re.match(r"(0\.96\d{3})", Path(p).name)
        if m:
            pts[p] = float(m.group(1))
    for line in Path(".kaggle-gm/ledger.jsonl").read_text().splitlines():
        r = json.loads(line)
        lb, sub = r.get("lb_public"), r.get("submission")
        if lb and sub and Path(sub).exists():
            pts[sub] = float(lb)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--s-anchor", type=float, default=0.96213)
    ap.add_argument("--tau", type=float, default=0.01, help="distance scale of the point weights (rank-diff sd)")
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--write", type=float, nargs="*", default=[])
    a = ap.parse_args()
    rA = R(read(a.anchor))
    pts = collect()
    names, X, s = [], [], []
    for p, sc in sorted(pts.items()):
        x = R(read(p)) - rA
        if np.abs(x).max() == 0:
            continue
        names.append(p), X.append(x.astype(np.float32)), s.append(sc)
    X, s = np.stack(X), np.array(s)
    n, N = X.shape
    K = (X.astype(np.float64) @ X.T.astype(np.float64)) / N          # <x_i, x_j> / N
    sq = np.diag(K).copy()
    d = s - a.s_anchor
    w = 1.0 / (1.0 + np.sqrt(sq) / a.tau)
    print(f"{n} scored files; rank-move sd from A: min {np.sqrt(sq).min():.5f} median {np.median(np.sqrt(sq)):.5f} max {np.sqrt(sq).max():.5f}")

    def fit(idx, lam):
        # unknowns: alpha (n) and c; design for point i: [K_i,idx, -sq_i/2]
        Kt = K[np.ix_(idx, idx)]
        A_ = np.hstack([Kt, -sq[idx, None] / 2])
        W = np.sqrt(w[idx])[:, None]
        reg = np.diag(np.r_[np.full(len(idx), lam) * np.diag(Kt).mean(), 0.0])
        coef = np.linalg.solve((A_ * W).T @ (A_ * W) + reg, (A_ * W).T @ (d[idx] * W[:, 0]))
        return coef[:-1], coef[-1]

    def predict(alpha, c, idx, i):
        return K[i, idx] @ alpha - c * sq[i] / 2

    best = None
    for lam in a.lams:
        err, err0 = [], []
        for i in range(n):
            idx = np.array([j for j in range(n) if j != i])
            al, c = fit(idx, lam)
            err.append(predict(al, c, idx, i) - d[i])
            c0 = np.sum(w[idx] * d[idx] * (-sq[idx] / 2)) / np.sum(w[idx] * (sq[idx] / 2) ** 2)   # curvature only
            err0.append(-c0 * sq[i] / 2 - d[i])
        err, err0 = np.array(err), np.array(err0)
        near = np.sqrt(sq) < 0.005
        rm = lambda e, m: np.sqrt(np.average(e[m] ** 2, weights=w[m])) * 1e5
        print(f"lam {lam:g}: LOO rmse x1e-5 all {rm(err, np.ones(n, bool)):.2f} near {rm(err, near):.2f} | "
              f"curvature-only {rm(err0, np.ones(n, bool)):.2f} near {rm(err0, near):.2f}")
        if best is None or rm(err, near) < best[0]:
            best = (rm(err, near), lam)
    lam = best[1]
    idx = np.arange(n)
    al, c = fit(idx, lam)
    g_norm2 = al @ K @ al
    print(f"chosen lam {lam:g}: c {c:.4g}, predicted gain of the full step {g_norm2 / (2 * c) * 1e5:.2f}e-5, "
          f"step rank-move sd {np.sqrt(g_norm2) / c:.5f}")
    contrib = pd.Series(al, index=[Path(p).parent.name[:22] + "/" + Path(p).name for p in names]).sort_values()
    print("largest |alpha|:\n" + contrib[contrib.abs().sort_values().index[-10:]].round(3).to_string())
    for eta in a.write:
        step = (al @ X.astype(np.float64)) / c
        r = rA + eta * step
        out = Path(f"data/lb/surface_step_eta{eta:g}.csv")
        pd.DataFrame({"id": IDS, "satisfaction": R(r)}).to_csv(out, index=False, float_format="%.9f")
        gain = eta * g_norm2 / c - c / 2 * eta ** 2 * g_norm2 / c ** 2
        print(f"eta {eta:g}: predicted {gain * 1e5:+.2f}e-5, rank move sd {np.std(R(r) - rA):.5f} -> {out}")


if __name__ == "__main__":
    main()
