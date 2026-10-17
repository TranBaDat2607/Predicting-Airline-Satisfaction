"""Build the stacking pool: every public OOF/test pair (data/ext/*) plus our own ledger experiments,
converted to logits, audited (row count, AUC sanity, exact duplicates), and cached as float32 matrices.

Recognised layouts inside each data/ext/<lib>/ (the public libraries use all of them):
  oof_<n>.npy + test_<n>.npy | <n>_oof.npy + <n>_test.npy | oof_<n>.csv + test_<n>.csv (id + value)
  <n>_oof.csv + <n>_test.csv | <k>_*oof.csv + <k>_submission.csv | <n>.npz with oof/test (or train/test) keys

    python src/pool.py                 # writes artifacts/pool/{oof,test}.npy + members.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXT = Path("data/ext")
OUT = Path("artifacts/pool")
N_TRAIN, N_TEST = 699_635, 299_844
# stack-level OOFs: their training rows encode other folds' labels, never use them as members
BANNED = ("stack", "blend", "meta", "nj_01")  # nj_01 = najiama 01_blend inside the 124 set
# libraries already contained in another library (cdeotte's 124 set) or holding features / submissions only
SKIP_LIBS = ("s6e10-oof-library", "s6e10-oof-test-predictions", "ps-s6e10-02", "ps-s6e10-7",
             "kout-s6e10-passenger-context-foundation-ensemble", "s6e10-top18-artifacts")
LIB_TAG = {"s6e10-124-public-oof-and-test-predictions": "124", "s6e10-tfm-oof-and-test-predictions": "tfm",
           "s6e10-goodpjw2008-extra-members-oof": "goodpjw", "s6e10-golem-oof-library": "golem",
           "oof-dataset": "abdullah", "ps-s6e10-airline-satisfaction-artifacts": "sadam",
           "s6e10-busyaprime-five-members-oof": "bp", "s6e10-oof": "najiama", "s6e10-stack-oof-predictions": "sachith",
           "s6e10-ablation-members-oof": "kagan", "s6e10-tabpfn-member": "gptabpfn",
           "s6e10-route-tfm-nn-oof": "koumei", "kout-s6e10-source-transfer-foundation-model-oof": "kozy"}

FAMILIES = [("tabpfn", "tabpfn|pfn"), ("tabicl", "tabicl"), ("tabdpt", "tabdpt"), ("realmlp", "realmlp|rmlp"), ("tabm", "tabm"),
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


NON_PRED = ("id", "fold", "Unnamed: 0", "target", "satisfaction", "n_seeds", "y")
WIDE = [("oof_predictions.parquet", "test_predictions.parquet"), ("oof_members.csv", "test_members.csv")]
_IDS = {}


def ids(split):
    if split not in _IDS:
        _IDS[split] = pd.read_csv(f"data/{split}.csv", usecols=["id"])["id"].to_numpy()
    return _IDS[split]


def read_csv_pred(path, split):
    d = pd.read_csv(path)
    cols = [c for c in d.columns if c not in NON_PRED]
    val = "pred" if "pred" in cols else cols[-1] if cols else "satisfaction"
    if "id" in d.columns:
        return d.set_index("id")[val].reindex(ids(split)).to_numpy()
    return d[val].to_numpy()


def read_wide(path, split):
    d = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if "id" in d.columns:
        d = d.set_index("id").reindex(ids(split))
    return {c: d[c].to_numpy() for c in d.columns if c not in NON_PRED}


def lib_pairs(lib: Path):
    """Yield (member name, oof loader, test loader) for one library directory (and its subdirectories)."""
    for sub in sorted(p for p in lib.iterdir() if p.is_dir() and p.name not in ("catboost_info", "units")):
        for n, lo, lt in lib_pairs(sub):
            yield f"{sub.name}/{n}", lo, lt
    files = {p.name: p for p in lib.iterdir() if p.is_file()}
    for fo, ft in WIDE:
        if fo in files and ft in files:
            cols = read_wide(files[fo], "train").keys()
            for c in cols:
                yield c, (lambda f=files[fo], c=c: read_wide(f, "train")[c]),                     (lambda f=files[ft], c=c: read_wide(f, "test")[c])
    if "oof_preds.csv" in files and "submission.csv" in files:
        yield "oof_preds", (lambda p=files["oof_preds.csv"]: read_csv_pred(p, "train")),             (lambda p=files["submission.csv"]: read_csv_pred(p, "test"))
    for fn, p in sorted(files.items()):
        m = re.fullmatch(r"oof_(.+)\.(npy|csv)", fn) or re.fullmatch(r"(.+)_oof\.(npy|csv)", fn)
        if m:
            n, ext = m.group(1), m.group(2)
            t = files.get(f"test_{n}.{ext}") or files.get(f"{n}_test.{ext}")
            if t is not None:
                if ext == "npy":
                    yield n, (lambda p=p: np.load(p)), (lambda t=t: np.load(t))
                else:
                    yield n, (lambda p=p: read_csv_pred(p, "train")), (lambda t=t: read_csv_pred(t, "test"))
                continue
        m = re.fullmatch(r"(\d+)_(.+)_oof\.csv", fn)  # najiama: 07_XGBoost_oof.csv + 07_submission.csv
        if m and f"{m.group(1)}_submission.csv" in files:
            t = files[f"{m.group(1)}_submission.csv"]
            yield f"{m.group(1)}_{m.group(2)}", (lambda p=p: read_csv_pred(p, "train")), \
                (lambda t=t: read_csv_pred(t, "test"))
            continue
        if fn.endswith(".npz"):
            z = np.load(p)
            ko = "oof" if "oof" in z.files else "train" if "train" in z.files else None
            if ko and "test" in z.files and z[ko].ndim == 1:
                yield fn[:-4], (lambda p=p, ko=ko: np.load(p)[ko]), (lambda p=p: np.load(p)["test"])
            for k in z.files:  # amanatar: oof_<x> / test_<x> keys in one archive
                if k.startswith("oof_") and f"test_{k[4:]}" in z.files:
                    yield f"{fn[:-4]}/{k[4:]}", (lambda p=p, k=k: np.load(p)[k]),                         (lambda p=p, k=k: np.load(p)["test_" + k[4:]])


def public_pairs():
    for lib in sorted(p for p in EXT.iterdir() if p.is_dir()):
        if lib.name in SKIP_LIBS:
            continue
        tag = LIB_TAG.get(lib.name, lib.name)
        for n, lo, lt in lib_pairs(lib):
            yield f"{tag}:{n}", lo, lt


def ledger_pairs():
    led = Path(".kaggle-gm/ledger.jsonl")
    if not led.exists():
        return
    for line in led.read_text().splitlines():
        r = json.loads(line)
        oof, test = r.get("oof_path"), r.get("test_path")
        if r.get("pool") is False or not oof or not test or not Path(oof).exists():
            continue
        yield f"own:{r['id']}:{r.get('model', '')}", (lambda o=oof: np.load(o)), (lambda t=test: np.load(t))


def main():
    y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
    cols_oof, cols_test, rows, seen = [], [], [], {}
    for name, lo, lt in list(public_pairs()) + list(ledger_pairs() or []):
        short = name.split(":", 1)[1]
        if any(b in short.lower() for b in BANNED):
            rows.append(dict(name=name, status="banned (stack-level)"))
            continue
        try:
            o, t = np.asarray(lo(), dtype=np.float64), np.asarray(lt(), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(name=name, status=f"unreadable {type(e).__name__}"))
            continue
        if o.ndim > 1:
            o = o[:, -1]
        if t.ndim > 1:
            t = t[:, -1]
        if len(o) != N_TRAIN or len(t) != N_TEST:
            rows.append(dict(name=name, status=f"bad shape {o.shape} {t.shape}"))
            continue
        if not (np.isfinite(o).all() and np.isfinite(t).all()):
            rows.append(dict(name=name, status="non-finite / missing ids"))
            continue
        lo_, lt_ = to_logit(o), to_logit(t)
        auc = roc_auc_score(y, lo_)
        if auc < 0.9:
            rows.append(dict(name=name, status=f"low AUC {auc:.4f} (misaligned?)", auc=auc))
            continue
        key = np.round(lo_[:2000], 4).tobytes()  # exact duplicates across libraries
        if key in seen:
            rows.append(dict(name=name, status=f"duplicate of {seen[key]}", auc=auc))
            continue
        seen[key] = name
        cols_oof.append(lo_.astype(np.float32))
        cols_test.append(lt_.astype(np.float32))
        rows.append(dict(name=name, status="ok", auc=auc, oof_mean=float(lo_.mean()), test_mean=float(lt_.mean())))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "oof.npy", np.stack(cols_oof, 1))
    np.save(OUT / "test.npy", np.stack(cols_test, 1))
    m = pd.DataFrame(rows)
    m["family"] = [family(n.rsplit(":", 1)[0] if n.startswith("own:") else n,
                          n.rsplit(":", 1)[1] if n.startswith("own:") else "") for n in m.name]
    m.to_csv(OUT / "members_all.csv", index=False)
    m[m.status == "ok"][["name", "family", "auc"]].to_csv(OUT / "members.csv", index=False)
    print(m.status.str.split(" ").str[0].value_counts().to_string())
    print(f"pool: {len(cols_oof)} members, best OOF AUC {m.auc.max():.5f}")
    print(m[m.status != "ok"][["name", "status"]].to_string())


if __name__ == "__main__":
    main()
