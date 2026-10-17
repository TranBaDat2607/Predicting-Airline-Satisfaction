"""Build the stacking pool: every public OOF/test pair (data/ext/*) plus our own ledger experiments,
converted to logits, audited (row count, AUC sanity, test range), and cached as float32 matrices.

    python src/pool.py                 # writes artifacts/pool/{oof,test}.npy + members.csv
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXT = Path("data/ext")
OUT = Path("artifacts/pool")
N_TRAIN, N_TEST = 699_635, 299_844
# stack-level OOFs: their training rows encode other folds' labels, never use them as members
BANNED = ("stack245", "stack", "blend", "meta")


FAMILIES = [("tabpfn", "tabpfn|pfn"), ("tabicl", "tabicl"), ("realmlp", "realmlp|rmlp"), ("tabm", "tabm"),
            ("deepfm", "deepfm"), ("cat", "cat"), ("lgb", "lgb|lgbm"), ("xgb", "xgb")]


def family(name: str, model: str = "") -> str:
    s = (model + " " + name.split(":", 1)[1]).lower()
    for fam, pat in FAMILIES:
        if re.search(pat, s):
            return fam
    return "tfm" if name.startswith("tfm:") else "other"


def to_logit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.min() >= 0 and a.max() <= 1:
        a = np.clip(a, 1e-7, 1 - 1e-7)
        return np.log(a / (1 - a))
    return a


def public_pairs():
    for lib in sorted(p for p in EXT.iterdir() if p.is_dir()):
        for f in sorted(glob.glob(str(lib / "oof_*.npy"))):
            name = os.path.basename(f)[4:-4]
            t = lib / f"test_{name}.npy"
            if not t.exists():
                continue
            yield f"{lib.name.split('-')[1] if '-' in lib.name else lib.name}:{name}", Path(f), t


def ledger_pairs():
    led = Path(".kaggle-gm/ledger.jsonl")
    if not led.exists():
        return
    for line in led.read_text().splitlines():
        r = json.loads(line)
        oof, test = r.get("oof_path"), r.get("test_path")
        if r.get("pool") is False or not oof or not test or not Path(oof).exists():
            continue
        yield f"own:{r['id']}:{r.get('model', '')}", Path(oof), Path(test)


def main():
    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    cols_oof, cols_test, rows, seen = [], [], [], {}
    for name, fo, ft in list(public_pairs()) + list(ledger_pairs() or []):
        short = name.split(":", 1)[1]
        if any(b in short.lower() for b in BANNED):
            rows.append(dict(name=name, status="banned (stack-level)"))
            continue
        o, t = np.load(fo, allow_pickle=False), np.load(ft, allow_pickle=False)
        if o.ndim > 1:
            o = o[:, -1]
        if t.ndim > 1:
            t = t[:, -1]
        if len(o) != N_TRAIN or len(t) != N_TEST:
            rows.append(dict(name=name, status=f"bad shape {o.shape} {t.shape}"))
            continue
        lo, lt = to_logit(o), to_logit(t)
        if not (np.isfinite(lo).all() and np.isfinite(lt).all()):
            rows.append(dict(name=name, status="non-finite"))
            continue
        auc = roc_auc_score(y, lo)
        if auc < 0.9:
            rows.append(dict(name=name, status=f"low AUC {auc:.4f} (misaligned?)", auc=auc))
            continue
        key = np.round(lo[:2000], 5).tobytes()  # exact duplicates across libraries
        if key in seen:
            rows.append(dict(name=name, status=f"duplicate of {seen[key]}", auc=auc))
            continue
        seen[key] = name
        cols_oof.append(lo.astype(np.float32))
        cols_test.append(lt.astype(np.float32))
        rows.append(dict(name=name, status="ok", auc=auc, oof_mean=float(lo.mean()), test_mean=float(lt.mean())))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "oof.npy", np.stack(cols_oof, 1))
    np.save(OUT / "test.npy", np.stack(cols_test, 1))
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "members_all.csv", index=False)
    m["family"] = [family(n.rsplit(":", 1)[0] if n.startswith("own:") else n,
                          n.rsplit(":", 1)[1] if n.startswith("own:") else "") for n in m.name]
    m[m.status == "ok"][["name", "family", "auc"]].to_csv(OUT / "members.csv", index=False)
    print(m.status.str.split(" ").str[0].value_counts().to_string())
    print(f"pool: {len(cols_oof)} members, best OOF AUC {m.auc.max():.5f}")
    print(m[m.status != "ok"].to_string())


if __name__ == "__main__":
    main()
