"""Family-wise L2 stack: a separate penalty per model family, implemented as a per-family column scale s_f
on the standardised logits (L2 on w with columns scaled by s_f == penalty C * s_f^2 on that family).

The scales are tuned by coordinate search on ONE tuning fold (meta fit on the other four, scored on it),
then the chosen scales are scored with the full nested CV — the honest number to compare.

    python src/stack_family.py --tune-fold 0 --C 0.01 --name stack_lr_family --log
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from stack import POOL, load, standardise

GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def scaled(X, fams, scales):
    return X * np.array([scales[f] for f in fams], dtype=X.dtype)


def fold_score(X, y, folds, f, C, model=None):
    tr, va = folds != f, folds == f
    m = model or LogisticRegression(C=C, max_iter=2000, tol=1e-6, warm_start=True)
    m.fit(X[tr], y[tr])
    return roc_auc_score(y[va], m.decision_function(X[va])), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--tune-fold", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--scales", type=json.loads, help="skip tuning, use these {family: scale}")
    ap.add_argument("--members")
    ap.add_argument("--name", default="stack_lr_family")
    ap.add_argument("--notes", default="")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    oof, test, names = load(args.members)
    fam_of = pd.read_csv(POOL / "members.csv").set_index("name")["family"]
    fams = [fam_of[n] for n in names]
    X, Xt = standardise(oof.astype(np.float64), test.astype(np.float64))
    scales = {f: 1.0 for f in sorted(set(fams))}
    if args.scales:
        scales.update(args.scales)
    else:
        t0 = time.time()
        best, model = fold_score(scaled(X, fams, scales), y, folds, args.tune_fold, args.C)
        print(f"start fold{args.tune_fold} {best:.6f} | {pd.Series(fams).value_counts().to_dict()}", flush=True)
        for r in range(args.rounds):
            for fam in sorted(set(fams), key=lambda f: -fams.count(f)):
                cur = scales[fam]
                for g in GRID:
                    if g == cur:
                        continue
                    trial = {**scales, fam: g}
                    s, _ = fold_score(scaled(X, fams, trial), y, folds, args.tune_fold, args.C, model)
                    if s > best + 1e-7:
                        best, scales = s, trial
                print(f"round {r} {fam:8s} -> {scales[fam]:g}  fold{args.tune_fold} {best:.6f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print("scales:", json.dumps(scales), flush=True)

    Xs, Xts = scaled(X, fams, scales), scaled(Xt, fams, scales)
    s_oof, scores = np.zeros(len(y)), []
    for f in range(5):
        tr, va = folds != f, folds == f
        m = LogisticRegression(C=args.C, max_iter=2000, tol=1e-6).fit(Xs[tr], y[tr])
        s_oof[va] = m.decision_function(Xs[va])
        scores.append(roc_auc_score(y[va], s_oof[va]))
    cv = roc_auc_score(y, s_oof)
    print(f"nested CV {cv:.6f} | folds " + " ".join(f"{s:.5f}" for s in scores), flush=True)
    if not args.log:
        return
    m = LogisticRegression(C=args.C, max_iter=2000, tol=1e-6).fit(Xs, y)
    s_test = m.decision_function(Xts)
    from kgkit.experiment import Ledger

    sig = lambda z: 1 / (1 + np.exp(-z))
    rec = Ledger().log(args.name, cv, fold_scores=scores, params={"C": args.C, "scales": scales,
                       "tune_fold": args.tune_fold, "n_members": len(names)}, features=names, model="stack_lr",
                       notes=args.notes, oof=sig(s_oof), test_pred=sig(s_test), task="binary", pool=False)
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"]
    out = Path("subs") / f"{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": sig(s_test)}).to_csv(out, index=False)
    Ledger().update(rec["id"], submission=str(out).replace("\\", "/"))
    print(f"logged {rec['id']} -> {out}")


if __name__ == "__main__":
    main()
