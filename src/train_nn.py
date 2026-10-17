"""Neural members via pytabkit (GPU): RealMLP / TabM on the frozen folds, logged to the ledger.

Features: raw columns (categoricals as category dtype; Flight Distance also as a categorical "route" id),
optionally + sachith7's label-free auxiliary expected values (aux_ev.parquet, each rating predicted from the
other 20 columns, cross-predicted over train+test).

    python src/train_nn.py --model tabm --aux --smoke
    python src/train_nn.py --model tabm --aux --name tabm_aux
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

AUX = Path("data/ext/s6e10-stack-oof-predictions/aux_ev.parquet")


def features(aux: bool, route: bool):
    tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    cols = [c for c in te.columns if c != "id"]
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(X["Departure Delay in Minutes"])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category")
    if route:
        fd = X["Flight Distance"]
        vc = fd.map(fd.value_counts())
        X["route"] = fd.where(vc >= 20, -1).astype(str).astype("category")
    if aux:
        a = pd.read_parquet(AUX)
        assert len(a) == len(X)
        X = pd.concat([X, a.reset_index(drop=True)], axis=1)
    y = tr["satisfaction"].astype(int).to_numpy()
    return X.iloc[: len(tr)].reset_index(drop=True), X.iloc[len(tr):].reset_index(drop=True), y


def make(kind, seed, smoke, params):
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier

    common = dict(device="cuda", random_state=seed, verbosity=0, val_metric_name="1-auc_ovr")
    if kind == "tabm":
        p = dict(n_epochs=3 if smoke else 30, batch_size=2048, patience=6, tabm_k=32)
        p.update(params)
        return TabM_D_Classifier(**common, **p)
    p = dict(n_epochs=3 if smoke else 64, batch_size=1024, use_ls=False)
    p.update(params)
    return RealMLP_TD_Classifier(**common, **p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tabm", "realmlp"], default="tabm")
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--no-route", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--params", type=json.loads, default={})
    ap.add_argument("--name")
    ap.add_argument("--notes", default="")
    ap.add_argument("--smoke", action="store_true", help="fold 0 on 50k rows, 3 epochs, not logged")
    args = ap.parse_args()

    X, Xt, y = features(args.aux, not args.no_route)
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    print(f"{X.shape[1]} features, {len(X)} rows", flush=True)
    oof, test, scores = np.zeros(len(y)), np.zeros(len(Xt)), []
    t0 = time.time()
    for f in range(1 if args.smoke else 5):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        if args.smoke:
            tr = tr[:50_000]
        m = make(args.model, args.seed, args.smoke, args.params)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        test += m.predict_proba(Xt)[:, 1] / 5
        scores.append(roc_auc_score(y[va], oof[va]))
        print(f"fold {f}: {scores[-1]:.6f} ({time.time() - t0:.0f}s)", flush=True)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    if args.smoke:
        print("smoke OK (not logged)")
        return
    cv = roc_auc_score(y, oof)
    print(f"CV {cv:.6f} | folds " + " ".join(f"{s:.5f}" for s in scores), flush=True)
    from kgkit.experiment import Ledger

    name = args.name or f"{args.model}{'_aux' if args.aux else ''}"
    rec = Ledger().log(name, cv, fold_scores=scores, params={**args.params, "aux": args.aux,
                       "route": not args.no_route, "seed": args.seed}, features=list(X.columns), model=args.model,
                       notes=args.notes, oof=oof, test_pred=test, task="binary", seeds=[args.seed],
                       runtime_s=round(time.time() - t0, 1))
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"]
    out = Path("subs") / f"{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": test}).to_csv(out, index=False)
    Ledger().update(rec["id"], submission=str(out).replace("\\", "/"))
    print(f"logged {rec['id']} -> {out}")


if __name__ == "__main__":
    main()
