"""Non-linear level-2: shallow, heavily regularised LightGBM on member logits (CPU), nested on the frozen folds.
Early stopping uses a 10% split of each meta-training part, never the scored fold.
Then picks the LR/LGB blend weight on the nested OOFs (one scalar) and logs both.

    python src/stack_lgb.py --lr 0004 --name stack_lgb246 --log
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stack import load

P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=6, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", required=True, help="ledger id of the LR stack to blend with")
    ap.add_argument("--members")
    ap.add_argument("--name", default="stack_lgb")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    from kgkit.experiment import Ledger

    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    A, B, names = load(args.members)
    t0 = time.time()
    oof, its, scores = np.zeros(len(y)), [], []
    for f in range(5):
        a, b = np.where(folds != f)[0], np.where(folds == f)[0]
        ai = np.random.default_rng(f).permutation(a)
        cut = int(len(ai) * 0.9)
        m = lgb.train(P, lgb.Dataset(A[ai[:cut]], y[ai[:cut]]), 4000,
                      valid_sets=[lgb.Dataset(A[ai[cut:]], y[ai[cut:]])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        its.append(m.best_iteration)
        oof[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
        scores.append(roc_auc_score(y[b], oof[b]))
        print(f"fold {f}: {scores[-1]:.6f} ({m.best_iteration} it, {time.time() - t0:.0f}s)", flush=True)
    cv = roc_auc_score(y, oof)
    print(f"LGB stacker nested CV {cv:.6f}", flush=True)

    lr = Ledger().get(args.lr)
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    lr_oof, lr_test = lg(np.load(lr["oof_path"])), lg(np.load(lr["test_path"]))
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        z = (1 - w) * lr_oof + w * oof
        print(f"LR*{1 - w:.1f} + LGB*{w:.1f}: {roc_auc_score(y, z):.6f} | folds "
              + " ".join(f"{roc_auc_score(y[folds == f], z[folds == f]):.5f}" for f in range(5)), flush=True)
    if not args.log:
        return
    gb_test = lgb.train(P, lgb.Dataset(A, y), int(np.mean(its) * 1.1)).predict(B, raw_score=True)
    sig = lambda z: 1 / (1 + np.exp(-z))
    rec = Ledger().log(args.name, cv, fold_scores=scores, params={**P, "rounds": int(np.mean(its) * 1.1)},
                       features=names, model="stack_lgb", oof=sig(oof), test_pred=sig(gb_test), task="binary",
                       pool=False)
    print(f"logged {rec['id']}")


if __name__ == "__main__":
    main()
