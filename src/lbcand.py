"""LB-layer candidates: weighted sums of public submission files and our own test predictions, logged as
lb-only ledger records (tag lb-blend) and written to subs/<id>.csv.

    python src/lbcand.py <name> "<notes>" --part N=data/ext/ps-s6e10-02/0.96210.csv:0.80 \
        --part T=artifacts/tfm_pool/test_pfn_mean.npy:0.15:sigmoid --part S=data/lb/trust-the-cv/submission.csv:0.05

A part is NAME=PATH:WEIGHT[:sigmoid|rank]: .csv files are read by id (column 2), .npy files are test-order arrays;
'sigmoid' maps logits to probabilities, 'rank' maps to normalised ranks. Parts are summed with their weights as given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from kgkit.experiment import Ledger  # noqa: E402

IDS = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()


def read(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        v = np.load(path).astype(float)
    else:
        d = pd.read_csv(path)
        v = d.set_index("id").iloc[:, 0].reindex(IDS).to_numpy(dtype=float)
    assert v.shape == IDS.shape and np.isfinite(v).all(), path
    return v


def build(parts):
    out = np.zeros(len(IDS))
    for spec in parts:
        name, rest = spec.split("=", 1)
        bits = rest.split(":")
        path, w = bits[0], float(bits[1])
        v = read(path)
        if len(bits) > 2 and bits[2] == "sigmoid":
            v = 1 / (1 + np.exp(-v))
        elif len(bits) > 2 and bits[2] == "rank":
            v = (rankdata(v) - 0.5) / len(v)
        out += w * v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("notes")
    ap.add_argument("--part", action="append", required=True)
    ap.add_argument("--ref", default="data/lb/secrets-to-satisfaction-it-s-stacking/submission.csv",
                    help="file to report the rank move against (public-LB noise rule: sd ~ 0.003 * sd(rank diff))")
    a = ap.parse_args()
    pred = build(a.part)
    r = (rankdata(pred) - 0.5) / len(pred)
    ref = read(a.ref)
    d = r - (rankdata(ref) - 0.5) / len(ref)
    rec = Ledger().log(a.name, None, model="lb_blend", notes=a.notes, params={"parts": a.part}, tags=["lb-blend"],
                       pool=False)
    out = Path("subs") / f"{rec['id']}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": IDS, "satisfaction": np.clip(pred, 1e-9, 1 - 1e-9)}).to_csv(out, index=False, float_format="%.10f")
    Ledger().update(rec["id"], submission=str(out))
    print(f"{rec['id']} -> {out} | rank move vs ref sd {d.std():.5f} (public sd ~{0.003 * d.std():.1e})")


if __name__ == "__main__":
    main()
