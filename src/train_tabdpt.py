"""TabDPT member (retrieval-based in-context learner, GPU) on the frozen folds.

Each predicted row gets its own context: its `--ctx` nearest training-fold neighbours (faiss retrieval),
a mechanism unlike TabPFN / TabICL's shared context, hence a diverse stack member.
Features: raw columns (+ sachith7's aux expected values unless --no-aux).

    python src/train_tabdpt.py --smoke
    python src/train_tabdpt.py --ctx 2048 --name tabdpt_aux_r2048
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from train_tabicl import features


def predict(m, X, ctx, bs, seed):
    out = []
    for i in range(0, len(X), bs):
        p = m.predict_proba(X[i:i + bs], context_size=ctx, seed=seed)
        out.append(p[:, 1])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--notes", default="")
    ap.add_argument("--smoke", action="store_true", help="fold 0, 20k validation rows, no test, not logged")
    args = ap.parse_args()
    from tabdpt import TabDPTClassifier

    X, Xt, y = features(not args.no_aux)
    folds = pd.read_csv("data/folds.csv")["fold"].to_numpy()
    oof, test, scores = np.zeros(len(y)), np.zeros(len(Xt)), []
    t0 = time.time()
    for f in range(1 if args.smoke else 5):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = TabDPTClassifier(device="cuda", context_reduction="retrieval", compile=False, verbose=False)
        m.fit(X[tr], y[tr])
        if args.smoke:
            va = va[:20_000]
        oof[va] = predict(m, X[va], args.ctx, args.batch, 42)
        scores.append(roc_auc_score(y[va], oof[va]))
        print(f"fold {f}: {scores[-1]:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if not args.smoke:
            test += predict(m, Xt, args.ctx, args.batch, 42) / 5
        del m
        gc.collect()
        torch.cuda.empty_cache()
    if args.smoke:
        print(f"smoke OK, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB (not logged)")
        return
    cv = roc_auc_score(y, oof)
    print(f"CV {cv:.6f} | folds " + " ".join(f"{s:.5f}" for s in scores), flush=True)
    from kgkit.experiment import Ledger

    rec = Ledger().log(args.name or f"tabdpt_r{args.ctx}", cv, fold_scores=scores,
                       params={"ctx": args.ctx, "aux": not args.no_aux, "retrieval": True}, model="tabdpt",
                       notes=args.notes, oof=oof, test_pred=test, task="binary", runtime_s=round(time.time() - t0, 1))
    print(f"logged {rec['id']}")


if __name__ == "__main__":
    main()
