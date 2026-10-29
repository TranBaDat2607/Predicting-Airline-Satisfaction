"""Current version number of every scored public notebook (data/tilt/kernels.csv) -> data/tilt/versions.csv.

A notebook's search-API score is its BEST version's public score; the output we download is its LATEST version.
With one version they are the same file; with several the score may belong to another output (a wrong data
point for the tilt).

    python src/lb_versions.py
"""
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

api = KaggleApi()
api.authenticate()
k = pd.read_csv("data/tilt/kernels.csv")
refs = k[k.best_public_score > 0].ref.tolist()
try:
    done = pd.read_csv("data/tilt/versions.csv")
    done = done[done.version.notna()]
except FileNotFoundError:
    done = pd.DataFrame(columns=["ref", "version", "last_run"])
todo = [r for r in refs if r not in set(done.ref)]


def get(ref):
    owner, slug = ref.split("/")
    for i in range(5):
        try:
            with api.build_kaggle_client() as kc:
                q = ApiGetKernelRequest()
                q.user_name, q.kernel_slug = owner, slug
                m = kc.kernels.kernels_api_client.get_kernel(q).metadata
                return {"ref": ref, "version": m.current_version_number, "last_run": str(m.last_run_time)}
        except Exception:
            time.sleep(2 + 3 * i)
    return {"ref": ref, "version": None, "last_run": None}


with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(get, todo))
out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
out.to_csv("data/tilt/versions.csv", index=False)
print(f"{len(out)} notebooks, version known for {out.version.notna().sum()}")
print(out.version.describe())
