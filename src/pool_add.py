"""Append members to a downloaded pool (Kaggle stack kernel output) without rebuilding it from the 35 sources.

    python src/pool_add.py --base data/ext/kout-s6e10-honest-stack-v2/pool --out artifacts/pool_v3 \
        --add own:tabpfn_auxcatfd:data/ext/kout-s6e10-tabpfn35-aux-catfd/oof_tabpfn35_auxcatfd.csv:data/ext/kout-s6e10-tabpfn35-aux-catfd/test_tabpfn35_auxcatfd.csv
    ln -sfn ../artifacts/pool_v3 artifacts/pool && python src/stack.py --C 0.03

A member is NAME:OOF:TEST; .npy arrays are in train/test row order, .csv files are read by id (column "pred" or the last).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from pool import family, read_csv_pred, to_logit  # noqa: E402


def read(path: str, split: str) -> np.ndarray:
    return np.load(path).astype(np.float64).ravel() if path.endswith(".npy") else read_csv_pred(path, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add", nargs="+", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    O, T = np.load(base / "oof.npy"), np.load(base / "test.npy")
    m = pd.read_csv(base / "members.csv")
    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    cols_o, cols_t, rows = [O], [T], []
    for spec in a.add:
        name, fo, ft = spec.rsplit(":", 2)
        o, t = to_logit(read(fo, "train")), to_logit(read(ft, "test"))
        assert np.isfinite(o).all() and np.isfinite(t).all(), name
        auc = roc_auc_score(y, o)
        print(f"{name}: OOF AUC {auc:.6f}")
        cols_o.append(o.astype(np.float32)[:, None])
        cols_t.append(t.astype(np.float32)[:, None])
        rows.append(dict(name=name, family=family(name), auc=auc))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", np.concatenate(cols_o, 1))
    np.save(out / "test.npy", np.concatenate(cols_t, 1))
    pd.concat([m, pd.DataFrame(rows)], ignore_index=True).to_csv(out / "members.csv", index=False)
    print(f"wrote {out}: {len(m) + len(rows)} members")


if __name__ == "__main__":
    main()
