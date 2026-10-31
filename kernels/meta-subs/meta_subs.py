"""Kaggle CPU: public scores of this competition's notebook versions from Meta Kaggle (official daily export).

Teams of a running competition are not exported, so submissions are reached through their source notebook version:
KernelVersionCompetitionSources -> Submissions.SourceKernelVersionId. Only PUBLIC score columns are extracted; private
score columns are counted (non-null), never written or printed: a private score of a running competition must not be used.

Output: versions_of_comp.csv (every notebook version attached to the competition), subs_public.csv, kernels.csv, users.csv.
"""
import glob
import os

import pandas as pd

CID = 125224  # playground-series-s6e10


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return hits[0]


head = pd.read_csv(find("KernelVersionCompetitionSources.csv"), nrows=5)
print("competition source columns", list(head.columns))
rows = []
for c in pd.read_csv(find("KernelVersionCompetitionSources.csv"), chunksize=5_000_000, low_memory=False):
    rows.append(c[c.SourceCompetitionId == CID])
src = pd.concat(rows)
vids = set(src.KernelVersionId.astype("int64"))
print("notebook versions attached to the competition", len(vids))

keep = ["Id", "ScriptId", "AuthorUserId", "CreationDate", "VersionNumber", "Title", "EvaluationDate", "RunningTimeInMilliseconds"]
rows = []
for c in pd.read_csv(find("KernelVersions.csv"), chunksize=2_000_000, usecols=keep, low_memory=False):
    rows.append(c[c.Id.isin(vids)])
kv = pd.concat(rows)
kv.to_csv("versions_of_comp.csv", index=False)
print("kernel versions found", len(kv), "notebooks", kv.ScriptId.nunique())

head = pd.read_csv(find("Submissions.csv"), nrows=5)
pub_cols = [c for c in head.columns if "private" not in c.lower()]
priv_cols = [c for c in head.columns if "private" in c.lower()]
rows, n_priv = [], {c: 0 for c in priv_cols}
for c in pd.read_csv(find("Submissions.csv"), chunksize=2_000_000, low_memory=False):
    m = c[c.SourceKernelVersionId.isin(vids)]
    for p in priv_cols:
        n_priv[p] += int(m[p].notna().sum())
    rows.append(m[pub_cols])
subs = pd.concat(rows)
print(f"submissions linked to these versions {len(subs)}; private non-null counts (values never read): {n_priv}")
subs.to_csv("subs_public.csv", index=False)
print(subs.describe().T.to_string())
print(subs.tail(5).to_string())

sids = set(kv.ScriptId.dropna().astype("int64"))
keep = ["Id", "AuthorUserId", "CurrentKernelVersionId", "CurrentUrlSlug", "MadePublicDate"]
rows = []
for c in pd.read_csv(find("Kernels.csv"), chunksize=2_000_000, usecols=keep, low_memory=False):
    rows.append(c[c.Id.isin(sids)])
ks = pd.concat(rows)
ks.to_csv("kernels.csv", index=False)
print("kernels found", len(ks))
uids = set(ks.AuthorUserId.dropna().astype("int64"))
rows = []
for c in pd.read_csv(find("Users.csv"), chunksize=2_000_000, usecols=["Id", "UserName"], low_memory=False):
    rows.append(c[c.Id.isin(uids)])
pd.concat(rows).to_csv("users.csv", index=False)
print(os.listdir("."))
