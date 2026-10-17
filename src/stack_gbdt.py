"""Level-2 GBDT stacker (XGBoost on GPU) over artifacts/pool member logits, optionally with raw features
and the LR-stack logit as a base margin.

Nested CV on the frozen folds. Rounds: early-stopped on an inner 10% split of the training folds (the
scoring fold is never used for stopping); the test model is refit on all rows with the mean best rounds.

    python src/stack_gbdt.py --raw --name stack_xgb_raw --log
    python src/stack_gbdt.py --base-margin <ledger id of an LR stack> ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

POOL = Path("artifacts/pool")


def raw_features():
    tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    cols = [c for c in te.columns if c != "id"]
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X.iloc[: len(tr)].to_numpy(np.float32), X.iloc[len(tr):].to_numpy(np.float32), cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="add the raw columns to the member logits")
    ap.add_argument("--members", help="file with member names to keep (default: all)")
    ap.add_argument("--base-margin", help="ledger id whose OOF/test logits are the starting margin (residual stack)")
    ap.add_argument("--params", type=json.loads, default={})
    ap.add_argument("--name", default="stack_xgb")
    ap.add_argument("--notes", default="")
    ap.add_argument("--folds-only", type=int, nargs="+", help="screen on these outer folds only")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    oof, test = np.load(POOL / "oof.npy"), np.load(POOL / "test.npy")
    names = pd.read_csv(POOL / "members.csv")["name"].tolist()
    if args.members:
        keep = [l.strip() for l in Path(args.members).read_text().splitlines() if l.strip()]
        idx = [names.index(k) for k in keep if k in names]
        oof, test, names = oof[:, idx], test[:, idx], [names[i] for i in idx]
    if args.raw:
        r_tr, r_te, r_cols = raw_features()
        oof, test, names = np.hstack([oof, r_tr]), np.hstack([test, r_te]), names + r_cols
    bm_oof = bm_test = None
    if args.base_margin:
        rec = next(json.loads(l) for l in Path(".kaggle-gm/ledger.jsonl").read_text().splitlines()
                   if json.loads(l)["id"] == args.base_margin)
        lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
        bm_oof, bm_test = lg(np.load(rec["oof_path"])), lg(np.load(rec["test_path"]))
    params = dict(objective="binary:logistic", eval_metric="auc", device="cuda", tree_method="hist",
                  learning_rate=0.02, max_depth=3, min_child_weight=50, subsample=0.8, colsample_bynode=0.5,
                  reg_lambda=10.0, max_bin=256)
    params.update(args.params)
    print(f"{oof.shape[1]} features, params {params}", flush=True)

    rng = np.random.default_rng(0)
    s_oof = np.full(len(y), np.nan)
    scores, rounds = [], []
    t0 = time.time()
    for f in args.folds_only or range(5):
        tr_all, va = np.where(folds != f)[0], np.where(folds == f)[0]
        inner = rng.random(len(tr_all)) < 0.1
        tr, es = tr_all[~inner], tr_all[inner]
        mk = lambda i: xgb.DMatrix(oof[i], label=y[i], base_margin=None if bm_oof is None else bm_oof[i])
        d_tr, d_es, d_va = mk(tr), mk(es), mk(va)
        b = xgb.train(params, d_tr, 20000, evals=[(d_es, "es")], early_stopping_rounds=300, verbose_eval=False)
        rounds.append(b.best_iteration + 1)
        s_oof[va] = b.predict(d_va, iteration_range=(0, b.best_iteration + 1), output_margin=True)
        scores.append(roc_auc_score(y[va], s_oof[va]))
        print(f"fold {f}: {scores[-1]:.6f} ({rounds[-1]} rounds, {time.time() - t0:.0f}s)", flush=True)
    mask = ~np.isnan(s_oof)
    cv = roc_auc_score(y[mask], s_oof[mask])
    print(f"nested CV {cv:.6f} | folds " + " ".join(f"{s:.5f}" for s in scores), flush=True)
    if not args.log or args.folds_only:
        return
    n_round = int(np.mean(rounds) * 1.1)
    b = xgb.train(params, xgb.DMatrix(oof, label=y, base_margin=bm_oof), n_round)
    s_test = b.predict(xgb.DMatrix(test, base_margin=bm_test), output_margin=True)
    from kgkit.experiment import Ledger

    sig = lambda z: 1 / (1 + np.exp(-z))
    rec = Ledger().log(args.name, cv, fold_scores=scores, params={**params, "rounds": n_round, "raw": args.raw,
                       "base_margin": args.base_margin}, features=names, model="stack_xgb", notes=args.notes,
                       oof=sig(s_oof), test_pred=sig(s_test), task="binary", pool=False)
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"]
    out = Path("subs") / f"{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": sig(s_test)}).to_csv(out, index=False)
    Ledger().update(rec["id"], submission=str(out).replace("\\", "/"))
    print(f"logged {rec['id']} -> {out}")


if __name__ == "__main__":
    main()
