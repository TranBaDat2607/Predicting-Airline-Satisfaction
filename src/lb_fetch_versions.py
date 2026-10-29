"""Download the submission*.csv of every OLDER version of the scored public notebooks.

A notebook's search-API score is its best version's; the output in data/tilt/k is its latest version's. When they
differ the data point is wrong (src/tilt_eb.py down-weights it). Each version's output is a candidate for the
scored file; src/lb_match.py picks the one the robust tilt posterior predicts closest to the listed score.

Order: notebooks the last robust fit trusted least first (data/tilt/fit_eb_report.csv weight), then the rest.
    python src/lb_fetch_versions.py [--max-refs N]      # -> data/tilt/kv/<owner>__<slug>/v<k>/submission*.csv
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
ap.add_argument("--max-refs", type=int, default=10_000)
ap.add_argument("--max-versions", type=int, default=25, help="only the newest N older versions of a notebook")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--max-w", type=float, default=99.0, help="only notebooks the last robust fit weighted below this")
a = ap.parse_args()

OUT = Path("data/tilt/kv")
k = pd.read_csv("data/tilt/kernels.csv")
k = k[k.best_public_score > 0]
ver = pd.read_csv("data/tilt/versions.csv").dropna().set_index("ref").version.to_dict() \
    if Path("data/tilt/versions.csv").exists() else {}
try:
    w = pd.read_csv("data/tilt/fit_eb_report.csv").set_index("name").w.to_dict()
except FileNotFoundError:
    w = {}
refs = sorted((r for r in k.ref if w.get(r, 0.95) < a.max_w), key=lambda r: w.get(r, 0.95))[: a.max_refs]
api = KaggleApi()
api.authenticate()


def version_of(ref):
    if ref in ver:
        return int(ver[ref])
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
    if (d / ".done").exists():
        return "ok" if list(d.glob("*.csv")) else "none"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        r = subprocess.run(["kaggle", "kernels", "output", f"{ref}/{v}", "-p", str(d), "--file-pattern",
                            r"(?i)^submission.*\.csv$", "-q"], capture_output=True, text=True, timeout=600)
        msg = (r.stdout + r.stderr).lower()
        if "rate limit" in msg or "429" in msg:
            time.sleep(20 + 20 * i)
            continue
        (d / ".done").write_text(msg[-300:])
        return "ok" if list(d.glob("*.csv")) else "none"
    return "fail"


def work(ref):
    v = version_of(ref)
    if not v or v <= 1:
        return ref, v, {}
    res = {}
    for vv in range(v - 1, max(0, v - 1 - a.max_versions), -1):
        res[vv] = fetch(ref, vv)
    return ref, v, res


with ThreadPoolExecutor(a.workers) as ex:
    for ref, v, res in ex.map(work, refs):
        got = sum(1 for s in res.values() if s == "ok")
        print(f"{ref} v{v}: {got}/{len(res)} older versions with a submission file", flush=True)
