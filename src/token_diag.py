"""Residual diagnostic: do GPT-2 token groups of the numeric columns carry signal the stack misses?

For each key, the stack's OOF residual (y - p) is target-encoded out of fold (same 5 folds), then
added to the stack logit with one fitted coefficient. A gain here means the key has signal that the
0.96225 stack (0020) does not already use.
"""
import argparse

import numpy as np
import pandas as pd
import tiktoken
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--oof", default="artifacts/0020_stack-lr-pool319-kaggle/oof.npy")
ap.add_argument("--smooth", type=float, default=50.0)
args = ap.parse_args()

tr = pd.read_csv("data/train.csv")
y = tr.satisfaction.astype(int).to_numpy()
fold = pd.read_csv("data/folds.csv")["fold"].to_numpy()
p = np.clip(np.load(args.oof).astype(float), 1e-6, 1 - 1e-6)
z = logit(p)
r = y - p
base = roc_auc_score(y, z)
print(f"stack OOF AUC {base:.6f}")

enc = tiktoken.get_encoding("gpt2")
_cache = {}


def toks(v, space):
    k = (v, space)
    if k not in _cache:
        s = (" " if space else "") + str(v)
        _cache[k] = [enc.decode([t]) for t in enc.encode(s)]
    return _cache[k]


def tok_cols(col, space):
    vals = tr[col].fillna(-1).astype(int)
    u = pd.Series(vals.unique())
    first = u.map(lambda v: toks(v, space)[0])
    last = u.map(lambda v: toks(v, space)[-1])
    ntok = u.map(lambda v: len(toks(v, space)))
    m = lambda s: vals.map(dict(zip(u, s)))
    return m(first), m(last), m(ntok)


def resid_te(key):
    key = pd.Series(pd.factorize(key)[0])
    out = np.zeros(len(key))
    for f in range(5):
        trm = fold != f
        g = pd.DataFrame({"k": key[trm], "r": r[trm]}).groupby("k").r.agg(["sum", "count"])
        te = g["sum"] / (g["count"] + args.smooth)
        out[~trm] = key[~trm].map(te).fillna(0).to_numpy()
    return out


def gain(feat):
    best = (0.0, base)
    for b in [0.5, 1, 2, 3, 4, 6, 8]:
        a = roc_auc_score(y, z + b * feat)
        if a > best[1]:
            best = (b, a)
    return best


fd = tr["Flight Distance"].astype(int)
keys = {"fd": fd}
for sp in (True, False):
    f1, fl, nt = tok_cols("Flight Distance", sp)
    tag = "sp" if sp else "ns"
    keys[f"fd_first_{tag}"] = f1
    keys[f"fd_last_{tag}"] = fl
    keys[f"fd_ntok_{tag}"] = nt
    for c in ["Class", "Type of Travel", "Customer Type"]:
        keys[f"fd_first_{tag}x{c[:5]}"] = f1.astype(str) + "|" + tr[c].astype(str)
for c in ["Class", "Type of Travel", "Customer Type", "Gender"]:
    keys[f"fd_x{c[:5]}"] = fd.astype(str) + "|" + tr[c].astype(str)
    keys[f"age_x{c[:5]}"] = tr["Age"].astype(str) + "|" + tr[c].astype(str)
keys["fd_mod10"] = fd % 10
keys["fd_mod100"] = fd % 100
keys["fd_div10"] = fd // 10
keys["fd_div100"] = fd // 100
keys["age"] = tr["Age"]
keys["dep"] = tr["Departure Delay in Minutes"]
keys["arr"] = tr["Arrival Delay in Minutes"].fillna(-1)
keys["dep_arr"] = tr["Departure Delay in Minutes"].astype(str) + "|" + tr["Arrival Delay in Minutes"].astype(str)

rows = []
for name, k in keys.items():
    feat = resid_te(k)
    b, a = gain(feat)
    rows.append((name, pd.Series(k).nunique(), b, a - base))
    print(f"{name:28s} nuniq {pd.Series(k).nunique():7d}  beta {b:4.1f}  gain {1e5 * (a - base):+7.2f}u", flush=True)
