"""TabICL member (in-context learner, GPU) on the frozen folds: raw columns + sachith7's aux expected values.

The 8 GB card cannot hold the full 560k-row context, so each fold uses a stratified subsample of its
training rows as context (--ctx) and predicts validation/test rows in batches.

    python src/train_tabicl.py --ctx 50000 --smoke
    python src/train_tabicl.py --ctx 150000 --name tabicl_aux_c150k
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

AUX = Path("data/ext/s6e10-stack-oof-predictions/aux_ev.parquet")


def features(aux: bool):
    tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    cols = [c for c in te.columns if c != "id"]
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(-1)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    if aux:
        X = pd.concat([X, pd.read_parquet(AUX).reset_index(drop=True)], axis=1)
    X = X.to_numpy(np.float32)
    return X[: len(tr)], X[len(tr):], tr["satisfaction"].astype(int).to_numpy()


def predict(m, X, bs):
    return np.concatenate([m.predict_proba(X[i:i + bs])[:, 1] for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--n-estimators", type=int, default=1)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--notes", default="")
    ap.add_argument("--smoke", action="store_true", help="fold 0, 20k validation rows, no test, not logged")
    args = ap.parse_args()
    from tabicl import TabICLClassifier

    X, Xt, y = features(not args.no_aux)
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    rng = np.random.default_rng(42)
    oof, test, scores = np.zeros(len(y)), np.zeros(len(Xt)), []
    t0 = time.time()
    for f in range(1 if args.smoke else 5):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        ctx = np.concatenate([rng.choice(tr[y[tr] == k], int(args.ctx * (y[tr] == k).mean()), replace=False)
                              for k in (0, 1)])
        m = TabICLClassifier(device="cuda", n_estimators=args.n_estimators, random_state=42)
        m.fit(X[ctx], y[ctx])
        if args.smoke:
            va = va[:20_000]
        oof[va] = predict(m, X[va], args.batch)
        scores.append(roc_auc_score(y[va], oof[va]))
        print(f"fold {f}: {scores[-1]:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if not args.smoke:
            test += predict(m, Xt, args.batch) / 5
        del m
        gc.collect()
        torch.cuda.empty_cache()
    if args.smoke:
        print(f"smoke OK, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB (not logged)")
        return
    cv = roc_auc_score(y, oof)
    print(f"CV {cv:.6f} | folds " + " ".join(f"{s:.5f}" for s in scores), flush=True)
    from kgkit.experiment import Ledger

    name = args.name or f"tabicl_c{args.ctx // 1000}k"
    rec = Ledger().log(name, cv, fold_scores=scores, params={"ctx": args.ctx, "n_estimators": args.n_estimators,
                       "aux": not args.no_aux}, model="tabicl", notes=args.notes, oof=oof, test_pred=test,
                       task="binary", runtime_s=round(time.time() - t0, 1))
    print(f"logged {rec['id']}")


if __name__ == "__main__":
    main()
