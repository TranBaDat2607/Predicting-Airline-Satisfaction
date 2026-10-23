"""Level-2 stack over artifacts/pool: L2 logistic regression on standardised member logits.

Honest score: nested CV on the frozen folds (meta-model fit on 4 folds of the OOF matrix, scored on the 5th).
Test prediction: meta-model refit on all OOF rows, applied to the member test logits.

    python src/stack.py --C 0.05 --name stack_lr --log
    python src/stack.py --members artifacts/pool/selected.txt ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

POOL = Path("artifacts/pool")


def load(members_file: str | None):
    oof, test = np.load(POOL / "oof.npy"), np.load(POOL / "test.npy")
    names = pd.read_csv(POOL / "members.csv")["name"].tolist()
    if members_file:
        keep = [l.strip() for l in Path(members_file).read_text().splitlines() if l.strip()]
        idx = [names.index(k) for k in keep if k in names]
        oof, test, names = oof[:, idx], test[:, idx], [names[i] for i in idx]
    return oof, test, names


def standardise(oof, test):
    mu, sd = oof.mean(0), oof.std(0) + 1e-9
    return (oof - mu) / sd, (test - mu) / sd


def fit_lr(X, y, C):
    m = LogisticRegression(C=C, max_iter=2000, tol=1e-6)
    m.fit(X, y)
    return m


def nested(X, y, folds, C):
    oof = np.zeros(len(y))
    scores = []
    for f in range(folds.max() + 1):
        tr, va = folds != f, folds == f
        m = fit_lr(X[tr], y[tr], C)
        oof[va] = m.decision_function(X[va])
        scores.append(roc_auc_score(y[va], oof[va]))
    return oof, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, nargs="+", default=[0.05])
    ap.add_argument("--members")
    ap.add_argument("--name", default="stack_lr")
    ap.add_argument("--notes", default="")
    ap.add_argument("--log", action="store_true", help="log the best C to the ledger + write subs/")
    ap.add_argument("--save", help="write the best C's nested OOF + full-fit test logits and weights to this dir")
    ap.add_argument("--raw", action="store_true", help="fit on raw member logits (no standardisation), as trust-the-cv does")
    args = ap.parse_args()

    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    oof, test, names = load(args.members)
    X, Xt = (oof.astype(np.float64), test.astype(np.float64)) if args.raw else \
        standardise(oof.astype(np.float64), test.astype(np.float64))
    print(f"{len(names)} members", flush=True)
    best = None
    for C in args.C:
        t0 = time.time()
        s_oof, scores = nested(X, y, folds, C)
        cv = roc_auc_score(y, s_oof)
        print(f"C={C:g}: nested CV {cv:.6f} (fold mean {np.mean(scores):.6f}) | folds " + " ".join(f"{s:.5f}" for s in scores)
              + f" | {time.time() - t0:.0f}s", flush=True)
        if best is None or cv > best[0]:
            best = (cv, C, s_oof, scores)
    if not (args.log or args.save):
        return
    cv, C, s_oof, scores = best
    m = fit_lr(X, y, C)
    s_test = m.decision_function(Xt)
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "oof_stack.npy", s_oof)
        np.save(out / "test_stack.npy", s_test)
        pd.Series(m.coef_.ravel(), index=names, name="coef").to_csv(out / "weights.csv")
        (out / "summary.txt").write_text(f"C={C} nested CV {cv:.6f} folds {scores} members {len(names)}\n")
        print(f"saved stack C={C} to {out}")
    if not args.log:
        return
    sys.path.insert(0, str(Path(__file__).parent))
    from kgkit.experiment import Ledger

    p_oof, p_test = 1 / (1 + np.exp(-s_oof)), 1 / (1 + np.exp(-s_test))
    rec = Ledger().log(args.name, cv, fold_scores=scores, params={"C": C, "n_members": len(names)},
                       features=names, model="stack_lr", notes=args.notes, oof=p_oof, test_pred=p_test,
                       task="binary", pool=False)
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"]
    out = Path("subs") / f"{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": p_test}).to_csv(out, index=False)
    Ledger().update(rec["id"], submission=str(out).replace("\\", "/"))
    top = pd.Series(m.coef_.ravel(), index=names).sort_values(key=np.abs, ascending=False)
    print("top weights:\n" + top.head(15).to_string())
    print(f"logged {rec['id']} -> {out} (C={C})")


if __name__ == "__main__":
    main()
