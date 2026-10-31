"""Scored multi-version notebooks whose listed score no downloaded output explains (src/lb_match.py: no candidate within |z| < 1, or the best beyond 1.5):
download every CSV (except OOF/train/fold files) of ALL their committed versions into data/tilt/kv/, so the matcher can find the scored one.

    python src/lb_fetch_unmatched.py --workers 3
"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()
KV = Path("data/tilt/kv")
m = pd.read_csv("data/tilt/match_report.csv")
refs = set(m[(m.n_fit == 0) | (m.z_best.abs() > 1.5)].ref)
v = pd.read_csv("data/tilt/meta_versions.csv")
v = v[v.VersionNumber.notna() & (v.best > 0)]
# scored notebooks with no candidate at all in the report (no output in any fetched version) count too
have = set(m.ref)
refs |= {r for r in v.ref.unique() if r not in have and not (Path("data/tilt/k") / r.replace("/", "__") / "submission.csv").exists()}
pairs = [(r, int(x)) for r in sorted(refs) for x in v[v.ref == r].VersionNumber]
print(len(refs), "notebooks,", len(pairs), "versions", flush=True)


def fetch(pair):
    ref, ver = pair
    d = KV / ref.replace("/", "__") / f"v{ver}"
    if (d / ".done_all").exists():
        return "cached"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        try:
            r = subprocess.run(["kaggle", "kernels", "output", f"{ref}/{ver}", "-p", str(d), "--file-pattern", r"(?i)^(?!.*(oof|train|fold|feature|importance)).*\.csv$", "-q"],
                               capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            (d / ".done_all").write_text("timeout")
            return "timeout"
        msg = (r.stdout + r.stderr).lower()
        if "rate limit" in msg or "429" in msg:
            time.sleep(20 + 20 * i)
            continue
        (d / ".done_all").write_text(msg[-300:])
        (d / ".done").write_text(msg[-300:])
        return "ok" if list(d.glob("*.csv")) else "none"
    return "fail"


with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(fetch, pairs))
print(pd.Series(res).value_counts().to_string())
