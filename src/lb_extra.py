"""Per-version public scores that notebooks document: a version table in the markdown, or a score in a version's title
(Meta Kaggle, data/tilt/meta_versions.csv; titles often lag one commit, so no version is assumed for them).

A listed notebook score gives one (file, score) pair; a version history gives one per version, and versions of a notebook
are often real model changes, i.e. new directions. Each claim is checked against the robust posterior: the version's output
must be predicted within --zmax of the claimed score (a CV number in a title fails this).

    python src/lb_extra.py fetch            # download the outputs of the needed versions -> data/tilt/kv/...
    python src/lb_extra.py match            # -> data/tilt/mined_extra.csv + data/tilt/mined/x_*.npy, read by src/lb_collect.py
"""
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

KV = Path("data/tilt/kv")
# (ref, {version: score}) from the notebooks' own tables, read by hand on 2026-10-30
TABLES = {
    "rabari9999/s6e10-lb-0-96209-rank-10": {21: .96195, 26: .96199, 27: .96200, 28: .96200, 29: .96201, 30: .96201, 31: .96201, 34: .96207, 35: .96206,
                                            36: .96206, 37: .96206, 38: .96207, 39: .96206, 40: .96207, 41: .96207, 42: .96207, 43: .96207, 45: .96208,
                                            46: .96208, 47: .96208, 48: .96208, 49: .96207, 50: .96208, 51: .96208, 52: .96208, 53: .96208, 54: .96209},
    "sarunpm/airline-satisfaction-lgbm-0-96059-lb": {2: .95844, 3: .95765, 6: .95787, 8: .96025, 9: .96027, 13: .96016, 43: .96031, 44: .96035, 47: .96045,
                                                     48: .96057, 46: .96059},
    "anhadmahajan06/s6e10-are-you-satisfied": {1: .95808, 2: .96076, 13: .96092, 15: .96095, 16: .96096, 17: .96086, 27: .96100, 30: .96110, 31: .96110},
    "nursrijan/s6e10-airline-passenger-satisfaction-0-96184-lb": {1: .96017, 2: .96068, 3: .96115, 4: .96152, 5: .96178, 6: .96184},
    "yeashusemwal/secrets-to-satisfaction-it-s-stacking": {1: .96171, 2: .96176, 4: .96181, 13: .96190, 14: .96213},
    "yeashusemwal/trust-the-cv": {6: .96181},
    "sadamtorres/s6e10-14-experiments-what-worked-lb-0-96108": {1: .95851, 2: .95963, 3: .96108},
    "areebbb/s6e10-what-each-idea-is-worth-stack-0-96186": {2: .95986, 3: .96186},
    "soukeaizenz/lb-0-96124-top-25-all-according-to-plan-meta": {1: .96115, 2: .96124},
    "chinzorigtganbat/s6e10-lightgbm-oof": {9: .96046, 12: .96042, 13: .96042, 14: .96040},
    "miickey/s6e10-fast-oof-ensemble-meta-stacking": {6: .96130},
    "ou20040313/s6e10-orange-blend": {15: .96187},
    "amanatar/s6e10-frontier-megastack-original-brain": {6: .96176},
    "amanatar/s6e10-the-consensus-forge": {3: .96027, 4: .95971, 5: .95973},
    "sarunpm/airline-satisfaction-0-95974-lb": {4: .95810, 8: .95781, 11: .95961, 13: .95974, 14: .95974},
    "kava1/s6e10-predicting-airline-satisfaction": {2: .95854, 5: .95848, 8: .95828, 10: .95981, 12: .95954, 16: .95910, 18: .95984, 21: .95958,
                                                    26: .95919, 28: .95986},
    "hittmg/261010playground-s6e10day2-3-lightgbm": {1: .95745, 2: .95746},
}
# scores a notebook documents without a reliable version number: matched over all its versions like a title
CLAIMS = {}


def title_claims():
    v = pd.read_csv("data/tilt/meta_versions.csv")
    v = v[(v.best > 0) & v.t_score.notna() & v.VersionNumber.notna()]
    v = v[~v.Title.str.contains(r"(?i)\bcv\b", regex=True)]  # a CV number in a title is not a public score
    out = {}
    for ref, g in v.groupby("ref"):
        best = g.best.iloc[0]
        s = sorted({float(x) for x in g.t_score if abs(x - best) > 1e-9})
        if s:
            out[ref] = s
    for ref, sc in CLAIMS.items():
        out[ref] = sorted(set(out.get(ref, [])) | set(sc))
    return out


def need():
    """(ref, version) pairs whose outputs the claims need."""
    v = pd.read_csv("data/tilt/meta_versions.csv")
    v = v[v.VersionNumber.notna()]
    pairs = set()
    for ref, tab in TABLES.items():
        pairs |= {(ref, int(k)) for k in tab}
    for ref in title_claims():
        pairs |= {(ref, int(k)) for k in v[v.ref == ref].VersionNumber}
    return sorted(pairs)


def fetch(pair):
    ref, ver = pair
    d = KV / ref.replace("/", "__") / f"v{ver}"
    if (d / ".done").exists() or (d / ".done_all").exists():
        return "cached"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        try:
            r = subprocess.run(["kaggle", "kernels", "output", f"{ref}/{ver}", "-p", str(d), "--file-pattern", r"(?i)^submission.*\.csv$", "-q"],
                               capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            continue
        msg = (r.stdout + r.stderr).lower()
        if "rate limit" in msg or "429" in msg:
            time.sleep(20 + 20 * i)
            continue
        (d / ".done").write_text(msg[-300:])
        return "ok" if list(d.glob("*.csv")) else "none"
    return "fail"


def run_fetch():
    pairs = need()
    print(len(pairs), "versions needed", flush=True)
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(fetch, pairs))
    print(pd.Series(res).value_counts().to_string())


def read(path, ids):
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    if "id" not in d.columns or len(d) != len(ids):
        return None
    col = "satisfaction" if "satisfaction" in d.columns else d.columns[-1]
    v = pd.to_numeric(d.set_index("id")[col], errors="coerce").reindex(ids).to_numpy(float)
    if not np.isfinite(v).all() or np.unique(v[:5000]).size < 50:
        return None
    return (rankdata(v) - 0.5) / len(v)


def run_match(zmax_tab=3.0, zmax_title=1.5):
    sys.path.insert(0, "src")
    from tilt_eb import TiltEB  # noqa: E402
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
    qs = np.sort(oof if oof.max() <= 1 else expit(oof))
    b = pd.read_csv("subs/0044_lbb-settf8-10.csv").set_index("id")["satisfaction"].reindex(ids).to_numpy(float)
    rb = (rankdata(b) - 0.5) / len(b)
    p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
    R = np.load("data/tilt/ranks.npy", mmap_mode="r")
    files = pd.read_csv("data/tilt/files.csv")
    corr = np.array([np.corrcoef(R[i], rb)[0, 1] for i in range(len(files))])
    sel = np.flatnonzero((files.score >= 0.955).to_numpy() & (corr >= 0.90) & (np.abs(corr - 1) > 1e-7))
    T = TiltEB(p).fit([np.asarray(R[i], dtype=np.float64) for i in sel], files.score.to_numpy()[sel], 0.96214)
    T.fit_noise(nu=2, exact=files.source.to_numpy()[sel] == "own")
    q = np.linalg.inv(T._cov(T.noise_var))
    print("posterior ready", flush=True)
    out_dir = Path("data/tilt/mined")
    rows = []

    def cands(ref, versions=None):
        d = KV / ref.replace("/", "__")
        res = []
        if d.exists():
            for vd in sorted(d.glob("v*"), key=lambda x: int(x.name[1:])):
                if versions is not None and int(vd.name[1:]) not in versions:
                    continue
                for f in sorted(vd.glob("*.csv")):
                    if re.search(r"(?i)oof|train", f.name):
                        continue
                    r = read(f, ids)
                    if r is not None:
                        res.append((f"{vd.name}/{f.name}", r))
        return res

    def key(ref, ver, s):
        return "x_" + re.sub(r"[^A-Za-z0-9]+", "_", ref)[:60] + f"_{ver}_{str(s)[2:]}"

    for ref, tab in TABLES.items():
        for ver, s in tab.items():
            cs = cands(ref, {ver})
            if not cs:
                print(f"{ref} v{ver}: no output", flush=True)
                continue
            best = None
            for name, r in cs:
                mu = T.predict(r)
                if best is None or abs(mu - s) < abs(best[1] - s):
                    best = (name, mu, r)
            mu, sd = T.predict_dist(best[2], q)
            z = (s - mu) / sd
            ok = abs(z) < zmax_tab
            print(f"{ref} v{ver} {s:.5f}: {best[0]} pred {mu:.5f} sd {sd * 1e5:.1f}u z {z:+.2f} {'ACCEPT' if ok else 'reject'}", flush=True)
            if ok:
                k = key(ref, ver, s)
                np.save(out_dir / f"{k}.npy", best[2].astype(np.float32))
                rows.append({"name": k, "score": s, "note": f"table {ref} {best[0]}"})
    for ref, scores in title_claims().items():
        cs = cands(ref)
        if not cs:
            continue
        mus = np.array([T.predict(r) for _, r in cs])
        for s in scores:
            near = np.flatnonzero(np.abs(mus - s) < 12e-5)
            if not len(near):
                print(f"{ref} title {s:.5f}: nothing within 12u of it", flush=True)
                continue
            zs = []
            for i in near:
                mu, sd = T.predict_dist(cs[i][1], q)
                zs.append(((s - mu) / sd, i, mu, sd))
            zs.sort(key=lambda x: abs(x[0]))
            z, i, mu, sd = zs[0]
            fit = [x for x in zs if abs(x[0]) < zmax_title]
            same = all(np.corrcoef(cs[x[1]][1], cs[i][1])[0, 1] > 0.9995 for x in fit)
            ok = abs(z) < zmax_title and same
            print(f"{ref} title {s:.5f}: {cs[i][0]} pred {mu:.5f} sd {sd * 1e5:.1f}u z {z:+.2f}, {len(fit)} fit, same {same} {'ACCEPT' if ok else 'reject'}", flush=True)
            if ok:
                k = key(ref, "t", s)
                np.save(out_dir / f"{k}.npy", cs[i][1].astype(np.float32))
                rows.append({"name": k, "score": s, "note": f"title {ref} {cs[i][0]}"})
    pd.DataFrame(rows).to_csv("data/tilt/mined_extra.csv", index=False)
    print(len(rows), "accepted")


if __name__ == "__main__":
    (run_fetch if sys.argv[1] == "fetch" else run_match)()
