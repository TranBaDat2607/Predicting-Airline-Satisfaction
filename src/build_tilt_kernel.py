"""Run the public-label tilt (src/tilt_eb.py) on Kaggle CPU: stage the inputs as private datasets and write
kernels/tilt-cpu/ (a script kernel embedding src/tilt.py + src/tilt_eb.py verbatim, so the remote fit is the local
code). Two datasets so that a new own score only re-uploads the small one:
  tranbadat/s6e10-tilt-public : ranks of the scored public files (data/tilt/ranks.npy minus own), base, prior, ids
  tranbadat/s6e10-tilt-own    : ranks + scores of our own scored submissions (from the ledger)

    python src/build_tilt_kernel.py --stage-public --stage-own --nu 2 --scan 0.8 1 1.25 --write 1.0 --name tilt-v5
    kaggle kernels push -p kernels/tilt-cpu        # then: kaggle kernels output tranbadat/s6e10-tilt-cpu -p <dir>
Outputs: <name>-eta<eta>.csv candidates ([public-LB fitted], never final picks) + fit_report.csv + summary.json.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ap = argparse.ArgumentParser()
ap.add_argument("--stage-public", action="store_true")
ap.add_argument("--stage-own", action="store_true")
ap.add_argument("--base", default="subs/0044_lbb-settf8-10.csv")
ap.add_argument("--base-score", type=float, default=0.96214)
ap.add_argument("--nu", type=float, default=2.0)
ap.add_argument("--min-score", type=float, default=0.955)
ap.add_argument("--min-corr", type=float, default=0.90)
ap.add_argument("--scan", type=float, nargs="*", default=[0.8, 1.0, 1.25])
ap.add_argument("--write", type=float, nargs="*", default=[1.0])
ap.add_argument("--name", default="tilt")
ap.add_argument("--exclude", nargs="*", default=[])
a = ap.parse_args()

USER = "tranbadat"
DS = Path("data/kaggle_ds")
files = pd.read_csv("data/tilt/files.csv")
R = np.load("data/tilt/ranks.npy", mmap_mode="r")
own = (files.source == "own").to_numpy()


def push_dataset(d: Path, slug: str, title: str):
    meta = {"title": title, "id": f"{USER}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{USER}/{slug}"], capture_output=True, text=True)
    cmd = (["kaggle", "datasets", "version", "-p", str(d), "-m", "update", "-q"] if exists.returncode == 0
           and "ready" in exists.stdout.lower() else ["kaggle", "datasets", "create", "-p", str(d), "-q"])
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(slug, " ".join(cmd[1:3]), "->", (r.stdout + r.stderr).strip()[-200:])
        if r.returncode == 0:
            return
    raise SystemExit(f"upload of {slug} failed")


if a.stage_public:
    d = DS / "s6e10-tilt-public"
    d.mkdir(parents=True, exist_ok=True)
    idx = np.flatnonzero(~own)
    np.save(d / "ranks.npy", np.asarray(R[idx], dtype=np.float32))
    files.iloc[idx].to_csv(d / "files.csv", index=False)
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    b = pd.read_csv(a.base).set_index("id")["satisfaction"].reindex(ids).to_numpy(np.float64)
    np.save(d / "base.npy", b)
    np.save(d / "ids.npy", ids)
    oof = np.load("artifacts/0020_stack-lr-pool319-kaggle/oof.npy").astype(float)
    np.save(d / "qs.npy", np.sort(oof if oof.max() <= 1 else expit(oof)))
    push_dataset(d, "s6e10-tilt-public", "s6e10 tilt public scored files")
if a.stage_own:
    d = DS / "s6e10-tilt-own"
    d.mkdir(parents=True, exist_ok=True)
    idx = np.flatnonzero(own)
    np.save(d / "ranks.npy", np.asarray(R[idx], dtype=np.float32))
    files.iloc[idx].to_csv(d / "files.csv", index=False)
    push_dataset(d, "s6e10-tilt-own", "s6e10 tilt own scored files")

MODS = {m: base64.b64encode(Path(f"src/{m}").read_bytes()).decode() for m in ("tilt.py", "tilt_eb.py")}
CFG = {k: getattr(a, k) for k in ("base_score", "nu", "min_score", "min_corr", "scan", "write", "name", "exclude")}
SCRIPT = r'''
import base64, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
T0 = time.time()
CFG = __CFG__
MODS = __MODS__
mods = Path("/tmp/mods"); mods.mkdir(exist_ok=True)
for k, v in MODS.items():
    (mods / k).write_bytes(base64.b64decode(v))
sys.path.insert(0, str(mods))
from tilt_eb import TiltEB
def log(m): print(f"[{(time.time() - T0) / 60:5.1f}m] {m}", flush=True)
def find(name):
    for p in Path("/kaggle/input").rglob("files.csv"):
        if name in str(p.parent):
            return p.parent
    raise SystemExit(f"input {name} not found")
pub, own = find("s6e10-tilt-public"), find("s6e10-tilt-own")
Rp, Ro = np.load(pub / "ranks.npy", mmap_mode="r"), np.load(own / "ranks.npy")
files = pd.concat([pd.read_csv(pub / "files.csv"), pd.read_csv(own / "files.csv")], ignore_index=True)
npub = len(Rp)
def row(i):
    return np.asarray(Rp[i] if i < npub else Ro[i - npub], dtype=np.float64)
b, ids, qs = np.load(pub / "base.npy"), np.load(pub / "ids.npy"), np.load(pub / "qs.npy")
N = len(b)
rb = (rankdata(b) - 0.5) / N
p = np.interp(rb, (np.arange(len(qs)) + 0.5) / len(qs), qs)
keep = (files.score >= CFG["min_score"]).to_numpy().copy()
for s in CFG["exclude"]:
    keep &= ~files.name.astype(str).str.contains(s, regex=False).to_numpy()
corr = np.array([np.corrcoef(row(i), rb)[0, 1] if keep[i] else 0 for i in range(len(files))])
keep &= (corr >= CFG["min_corr"]) & (np.abs(corr - 1) > 1e-7)
sel = np.flatnonzero(keep)
log(f"files {len(files)} (public {npub}, own {len(Ro)}), used {len(sel)}")
t = TiltEB(p).fit([row(i) for i in sel], files.score.to_numpy()[sel], CFG["base_score"])
log("system built")
t.fit_noise(nu=CFG["nu"], exact=files.source.to_numpy()[sel] == "own")
scale = 1e5 / (t.n1 * t.n0)
rep = files.iloc[sel][["name", "source", "score"]].copy()
rep["corr"], rep["surprise_u"], rep["loo_resid_u"] = corr[sel], t.y * scale, t.loo * scale
rep["noise_sd_u"], rep["w"] = np.sqrt(t.noise_var) * scale, t.w
rep.to_csv("/kaggle/working/fit_report.csv", index=False)
summ = {"n_files": int(len(sel)), "k0": float(t.k0), "k1": float(t.k1), "nu": CFG["nu"], "loo_r2": float(t.loo_r2),
        "w_lt_0.2": int((t.w < 0.2).sum()), "pred": {}, "written": {}}
for eta in sorted(set(CFG["scan"]) | set(CFG["write"])):
    s = t.score(eta)
    summ["pred"][str(eta)] = float(t.predict(s))
    log(f"eta {eta}: predicted public {summ['pred'][str(eta)]:.5f}")
    if eta in CFG["write"]:
        rs = (rankdata(s) - 0.5) / N
        f = f"{CFG['name']}-eta{eta:g}.csv"
        pd.DataFrame({"id": ids, "satisfaction": s}).to_csv(f"/kaggle/working/{f}", index=False, float_format="%.9f")
        summ["written"][f] = {"eta": eta, "pred": summ["pred"][str(eta)], "rank_corr_base": float(np.corrcoef(rs, rb)[0, 1])}
Path("/kaggle/working/summary.json").write_text(json.dumps(summ, indent=1))
log("done")
'''.replace("__CFG__", repr(CFG)).replace("__MODS__", repr(MODS))
kd = Path("kernels/tilt-cpu")
kd.mkdir(parents=True, exist_ok=True)
(kd / "s6e10-tilt-cpu.py").write_text(SCRIPT)
meta = {"id": f"{USER}/s6e10-tilt-cpu", "title": "s6e10-tilt-cpu", "code_file": "s6e10-tilt-cpu.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [f"{USER}/s6e10-tilt-public", f"{USER}/s6e10-tilt-own"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}
(kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {kd} (cfg {CFG})")
