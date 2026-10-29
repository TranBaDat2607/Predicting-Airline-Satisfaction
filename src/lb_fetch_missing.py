"""Scored public notebooks with no usable output (no submission.csv in the latest version, other file names, failed
latest run): download every CSV of their newest --versions versions into data/tilt/kv/<owner>__<slug>/v<k>/.
src/lb_match.py then picks the candidate whose predicted public score matches the listed one.

    python src/lb_fetch_missing.py --versions 4 --workers 2      # refs from data/tilt/missing_files.csv
"""
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ap = argparse.ArgumentParser()
ap.add_argument("--versions", type=int, default=4)
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
OUT = Path("data/tilt/kv")
refs = pd.read_csv("data/tilt/missing_files.csv").ref.tolist()
api = KaggleApi()
api.authenticate()


def version_of(ref):
    owner, slug = ref.split("/")
    for i in range(6):
        try:
            with api.build_kaggle_client() as kc:
                q = ApiGetKernelRequest()
                q.user_name, q.kernel_slug = owner, slug
                return int(kc.kernels.kernels_api_client.get_kernel(q).metadata.current_version_number)
        except Exception:
            time.sleep(5 + 10 * i)
    return None


def fetch(ref, v):
    d = OUT / ref.replace("/", "__") / f"v{v}"
    if (d / ".done_all").exists():
        return len(list(d.glob("*.csv")))
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        try:
            r = subprocess.run(["kaggle", "kernels", "output", f"{ref}/{v}", "-p", str(d), "--file-pattern",
                                r"(?i)^[^/]*\.csv$", "-q"], capture_output=True, text=True, timeout=2400)
        except subprocess.TimeoutExpired:  # huge outputs (dozens of probe CSVs): keep what arrived
            (d / ".done_all").write_text("timeout")
            return len(list(d.glob("*.csv")))
        msg = (r.stdout + r.stderr).lower()
        if "rate limit" in msg or "429" in msg:
            time.sleep(20 + 20 * i)
            continue
        (d / ".done_all").write_text(msg[-300:])
        (d / ".done").write_text(msg[-300:])
        return len(list(d.glob("*.csv")))
    return -1


def work(ref):
    v = version_of(ref)
    if not v:
        return ref, v, {}
    return ref, v, {vv: fetch(ref, vv) for vv in range(v, max(0, v - a.versions), -1)}


with ThreadPoolExecutor(a.workers) as ex:
    for ref, v, res in ex.map(work, refs):
        print(f"{ref} v{v}: csv files per version {res}", flush=True)
