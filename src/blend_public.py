"""LB-side blend: our honest stack (a ledger experiment) + public submission files that have no OOF.

The public files cannot be cross-validated, so their weights are an LB bet; the ledger record keeps the
stack's nested CV as `cv` and is tagged `lb-blend` so it is never mistaken for a CV-validated result.

    python src/blend_public.py --base 0003 --add data/ext/ps-s6e10-7/0.96183.csv:0.3 --name blend_s233_n183
    (weights apply to logits of the probabilities; --rank averages normalised ranks instead)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="ledger id of the stack")
    ap.add_argument("--add", nargs="+", required=True, help="path:weight of public submission files")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    from kgkit.experiment import Ledger

    led = Ledger()
    base = led.get(args.base)
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"]
    tf = (lambda p: rankdata(p) / len(p)) if args.rank else logit
    parts = [(tf(np.load(base["test_path"])), None)]
    w_add = 0.0
    for spec in args.add:
        path, w = spec.rsplit(":", 1)
        p = pd.read_csv(path).set_index("id")["satisfaction"].reindex(ids).to_numpy()
        assert np.isfinite(p).all(), path
        parts.append((tf(p), float(w)))
        w_add += float(w)
    parts[0] = (parts[0][0], 1.0 - w_add)
    z = sum(w * v for v, w in parts)
    pred = z if args.rank else 1 / (1 + np.exp(-z))
    rho = np.corrcoef(rankdata(pred), rankdata(parts[0][0]))[0, 1]
    rec = led.log(args.name, base["cv"], params={"base": base["id"], "add": args.add, "rank": args.rank},
                  model="lb_blend", notes=args.notes, test_pred=pred, tags=["lb-blend"], parent=base["id"],
                  pool=False)
    out = Path("subs") / f"{rec['id']}.csv"
    pd.DataFrame({"id": ids, "satisfaction": pred}).to_csv(out, index=False)
    led.update(rec["id"], submission=str(out).replace("\\", "/"))
    print(f"rank corr with base {rho:.6f}; logged {rec['id']} -> {out}")


if __name__ == "__main__":
    main()
