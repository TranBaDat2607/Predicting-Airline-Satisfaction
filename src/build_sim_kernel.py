"""Kaggle CPU kernel for tilt simulations (no upload: inputs are the competition data and our own kernel output
tranbadat/s6e10-honest-stack-v3, which holds the 321-member OOF pool and the stack OOF). Embeds src/tilt.py,
src/tilt_eb.py and the simulation script verbatim and runs several argument sets in parallel.

    python src/build_sim_kernel.py --script tilt_probe_sim.py --runs "--strategy maxvar" "--strategy random" ...
    kaggle kernels push -p kernels/tilt-sim      # outputs: run<i>.log
"""
import argparse
import base64
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--script", default="tilt_probe_sim.py")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--slug", default="s6e10-tilt-sim")
a = ap.parse_args()

MODS = {m: base64.b64encode(Path(f"src/{m}").read_bytes()).decode() for m in ("tilt.py", "tilt_eb.py", a.script)}
SCRIPT = r'''
import base64, subprocess, sys, time
from pathlib import Path
MODS = __MODS__
RUNS = __RUNS__
SCRIPT = __SCRIPTNAME__
src = Path("/kaggle/working/src"); src.mkdir(parents=True, exist_ok=True)
for k, v in MODS.items():
    (src / k).write_bytes(base64.b64decode(v))
def find(name, under="/kaggle/input"):
    hits = sorted(Path(under).rglob(name))
    if not hits:
        raise SystemExit(f"{name} not found under {under}")
    return str(hits[0])
train = find("train.csv")
pool = [str(p) for p in Path("/kaggle/input").rglob("oof.npy") if p.parent.name == "pool"][0]
base = find("oof_stack.npy")
print("inputs:", train, pool, base, flush=True)
procs = []
for i, r in enumerate(RUNS):
    cmd = [sys.executable, f"src/{SCRIPT}", "--train", train, "--pool", pool, "--base", base] + r.split()
    procs.append(subprocess.Popen(cmd, cwd="/kaggle/working", stdout=open(f"/kaggle/working/run{i}.log", "w"),
                                  stderr=subprocess.STDOUT))
    time.sleep(30)
for p in procs:
    p.wait()
for i in range(len(RUNS)):
    print(f"== run{i}: {RUNS[i]}"); print(open(f"/kaggle/working/run{i}.log").read()[-3000:])
'''.replace("__MODS__", repr(MODS)).replace("__RUNS__", repr(a.runs)).replace("__SCRIPTNAME__", repr(a.script))
kd = Path("kernels/tilt-sim")
kd.mkdir(parents=True, exist_ok=True)
(kd / f"{a.slug}.py").write_text(SCRIPT)
meta = {"id": f"tranbadat/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "competition_sources": ["playground-series-s6e10"],
        "kernel_sources": ["tranbadat/s6e10-honest-stack-v3"], "model_sources": []}
(kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {kd}: {len(a.runs)} runs of {a.script}")
