"""OOF proxy of a public LB file: explain its test ranks by the honest pool members' test ranks (non-negative least squares), apply the same
weights to the members' OOF logits, and use the resulting proxy OOF to pre-screen LB directions on labelled rows.

The public files (nina2025 0.96210, the 0.96213 crowd file) have no OOF, so a new direction D can only be judged on the
LB. If the file is mostly a linear mix of public members, (1 - a) * R(proxy) + a * R(D_oof) on OOF says whether D adds
information the file lacks before a submission is spent on it.

    python src/lb_proxy.py --target data/ext/ps-s6e10-02/0.96210.csv --cand own:artifacts/<exp>/oof.npy:artifacts/<exp>/test.npy

Measured limitation (2026-10-23): for nina's 0.96210 file the NNLS fit reaches rank corr 0.9916 (19 members, proxy OOF
0.96200) and predicts +3.9e-5 for 10 % more honest stack, but the LB measured -1e-5 for that move (ledger 0017): the
LB-tuned part of the public chain is exactly what the members cannot explain, so this screens nothing on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def R(v):
    return (rankdata(v) - 0.5) / len(v)


def logit(p):
    p = np.asarray(p, float)
    if p.min() >= 0 and p.max() <= 1:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/ext/kout-s6e10-honest-stack/pool")
    ap.add_argument("--target", required=True, help="public submission csv (id, satisfaction)")
    ap.add_argument("--cand", nargs="*", default=[], help="name:oof.npy:test.npy (probabilities or logits)")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    a = ap.parse_args()
    pool = Path(a.pool)
    O, T = np.load(pool / "oof.npy"), np.load(pool / "test.npy")
    names = pd.read_csv(pool / "members.csv")["name"].tolist()
    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    tgt = pd.read_csv(a.target).set_index("id").iloc[:, 0].reindex(ids).to_numpy(float)
    # rank space on both sides: distribution free, so weights fitted on test transfer to the OOF rows
    rank_cols = lambda M: np.column_stack([R(M[:, j]) for j in range(M.shape[1])]).astype(np.float32)
    Rt, Ro = rank_cols(T), rank_cols(O)
    rt = R(tgt)
    from scipy.optimize import nnls
    sub = np.random.default_rng(0).choice(len(rt), min(len(rt), 120_000), replace=False)
    w, _ = nnls(Rt[sub].astype(np.float64), rt[sub])
    fit = Rt @ w
    r2 = 1 - ((rt - fit) ** 2).sum() / ((rt - rt.mean()) ** 2).sum()
    print(f"target {a.target}: NNLS rank fit R^2 {r2:.6f}, rank corr {np.corrcoef(R(fit), rt)[0, 1]:.6f}, "
          f"{(w > 0).sum()} members with weight, sum {w.sum():.3f}")
    proxy = Ro @ w
    base = roc_auc_score(y, proxy)
    print(f"proxy OOF AUC {base:.6f} | folds " + " ".join(f"{roc_auc_score(y[folds == k], proxy[folds == k]):.5f}" for k in range(5)))
    class _M: pass
    m = _M(); m.coef_ = w
    top = pd.Series(m.coef_, index=names).sort_values(ascending=False).head(12)
    print("largest proxy weights:\n" + top.round(4).to_string())
    rp = R(proxy)
    for spec in a.cand:
        name, fo, ft = spec.split(":")
        co = logit(np.load(fo))
        rc = R(co)
        line = [f"{name:28s} alone {roc_auc_score(y, co):.6f}"]
        for al in a.alpha:
            b = (1 - al) * rp + al * rc
            line.append(f"a={al:g}: {roc_auc_score(y, b) - roc_auc_score(y, rp):+.2e}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
