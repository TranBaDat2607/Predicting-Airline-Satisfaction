"""Inventory of public notebooks of the competition with their best public score (Kaggle search API).

    python src/lb_files.py            # -> data/tilt/kernels.csv (ref, title, best_public_score, last_run)
"""
import time

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.search.types.search_api_service import ListEntitiesFilters, ListEntitiesRequest
from kagglesdk.search.types.search_enums import DocumentType

COMP = "playground-series-s6e10"
QUERIES = ["s6e10", "s6e10 airline", "airline satisfaction", "airline passenger satisfaction", "playground s6e10",
           "ps s6e10", "playground series s6e10", "airline", "satisfaction", "s6 e10", "season 6 episode 10"]

api = KaggleApi()
api.authenticate()

refs = {}
for page in range(1, 60):
    for attempt in range(5):
        try:
            ks = api.kernels_list(competition=COMP, page=page, page_size=100, sort_by="dateCreated")
            break
        except Exception as e:
            time.sleep(3)
    else:
        break
    if not ks:
        break
    for k in ks:
        refs[k.ref] = {"ref": k.ref, "title": k.title, "last_run": str(k.last_run_time)}
    print("competition list page", page, len(refs), flush=True)

scores = {}
with api.build_kaggle_client() as kc:
    for q in QUERIES:
        token = ""
        for _ in range(40):
            req = ListEntitiesRequest()
            f = ListEntitiesFilters()
            f.query = q
            f.document_types = [DocumentType.KERNEL]
            req.filters = f
            req.page_size = 100
            if token:
                req.page_token = token
            for attempt in range(5):
                try:
                    resp = kc.search.search_api_client.list_entities(req)
                    break
                except Exception:
                    time.sleep(3)
            else:
                break
            for d in resp.documents:
                if d.kernel_document and d.owner_user:
                    scores[f"{d.owner_user.user_name}/{d.slug}"] = d.kernel_document.best_public_score
            token = resp.next_page_token
            if not token:
                break
        print("query", repr(q), "scored refs so far", len(scores), flush=True)

df = pd.DataFrame(refs.values())
df["best_public_score"] = df.ref.map(scores)
df.to_csv("data/tilt/kernels.csv", index=False)
s = df.best_public_score
print(f"competition kernels {len(df)}, with a score lookup {s.notna().sum()}, score > 0: {(s > 0).sum()}")
print(s[s > 0].describe())
