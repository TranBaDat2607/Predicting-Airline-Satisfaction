"""Local mini-pool of tabular-foundation-model members (TabPFN / TabICL / other TFMs) with OOF + test logits.

Used for the LB-blend layer: a low-noise "TFM direction" (the public 0.96213 file gained +3e-5 on the
public LB by adding 10 % of one TabPFN output). Reads the libraries pulled into data/ext/.

    python src/tfm_pool.py            # prints member AUCs, the plain logit mean and a nested LR stack
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = Path("artifacts/tfm_pool")


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def by_id(path, ids, col=None):
    d = pd.read_csv(path)
    c = col or ("pred" if "pred" in d.columns else [x for x in d.columns if x not in ("id", "fold", "target", "n_seeds")][-1])
    return d.set_index("id")[c].reindex(ids).to_numpy()


def members(tr_ids, te_ids):
    m = {}
    d = "data/ext/s6e10-tfm-oof-and-test-predictions"
    for f in sorted(glob.glob(f"{d}/oof_*.npy")):
        n = Path(f).stem[4:]
        if n in ("xgboost", "realmlp"):
            continue
        m[f"cd_{n}"] = (np.load(f), np.load(f.replace("oof_", "test_")))
    for n in ("tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt"):
        m[f"ab_{n}"] = (by_id(f"data/ext/oof-dataset/{n}_oof.csv", tr_ids), by_id(f"data/ext/oof-dataset/{n}_test.csv", te_ids))
    for f in sorted(glob.glob("data/ext/s6e10-tabpfn-member/oof_*.csv")):
        n = Path(f).stem[4:]
        m[f"gp_{n}"] = (by_id(f, tr_ids), by_id(f.replace("oof_", "test_"), te_ids))
    m["gp_tabpfn_ne4"] = (np.load("data/ext/goodpjw-pub/oof_tabpfn_ne4.npy"), np.load("data/ext/goodpjw-pub/test_tabpfn_ne4.npy"))
    for k in ("s6e10-exp01-tabpfn35-full-context", "s6e10-tabpfn-route-og", "s6e10-tabpfn-seeds"):
        d = f"data/ext/kout-{k}"
        m[f"k_{k[6:]}"] = (by_id(f"{d}/oof_exp01_tabpfn35.csv", tr_ids, "pred"), by_id(f"{d}/test_exp01_tabpfn35.csv", te_ids))
    for f in sorted(glob.glob("data/ext/kout-s6e10-tabpfn35-aux-catfd/oof_*.csv")):  # ours
        m["own_tabpfn_auxcatfd"] = (by_id(f, tr_ids, "pred"), by_id(f.replace("oof_", "test_"), te_ids))
    return {k: (logit(o), logit(t)) for k, (o, t) in m.items() if np.isfinite(o).all() and np.isfinite(t).all()}


def main():
    tr = pd.read_csv("data/train.csv", usecols=["id", "satisfaction"])
    te_ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    y = tr.satisfaction.astype(int).to_numpy()
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    m = members(tr.id.to_numpy(), te_ids)
    names = list(m)
    O = np.column_stack([m[n][0] for n in names])
    T = np.column_stack([m[n][1] for n in names])
    print(pd.Series({n: roc_auc_score(y, O[:, i]) for i, n in enumerate(names)}).sort_values(ascending=False).round(6).to_string())
    pfn = [i for i, n in enumerate(names) if "tabpfn" in n]
    print(f"\nTabPFN logit mean ({len(pfn)}): {roc_auc_score(y, O[:, pfn].mean(1)):.6f}")
    mu, sd = O.mean(0), O.std(0)
    Z, ZT = (O - mu) / sd, (T - mu) / sd
    s = np.zeros(len(y))
    for f in range(5):
        lr = LogisticRegression(C=0.03, max_iter=2000).fit(Z[folds != f], y[folds != f])
        s[folds == f] = lr.decision_function(Z[folds == f])
    print(f"TFM LR stack ({len(names)}) nested CV {roc_auc_score(y, s):.6f}")
    lr = LogisticRegression(C=0.03, max_iter=2000).fit(Z, y)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "oof_tfm_stack.npy", s)
    np.save(OUT / "test_tfm_stack.npy", lr.decision_function(ZT))
    np.save(OUT / "oof_pfn_mean.npy", O[:, pfn].mean(1))
    np.save(OUT / "test_pfn_mean.npy", T[:, pfn].mean(1))
    print(pd.Series(lr.coef_.ravel(), index=names).sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
