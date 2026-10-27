"""Write kernels/stack-cpu/: a Kaggle CPU kernel that rebuilds the public OOF pool from attached inputs and runs the
honest LR stack (src/pool.py + src/stack.py, embedded verbatim, so the remote run is the local code).

    python src/build_stack_kernel.py [--C 0.01 0.03 0.1] [--kernel-extra owner/slug ...]
    kaggle kernels push -p kernels/stack-cpu
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

DATASETS = [
    "cdeotte/s6e10-124-public-oof-and-test-predictions", "cdeotte/s6e10-tfm-oof-and-test-predictions",
    "goodpjw2008/s6e10-goodpjw2008-extra-members-oof", "dariushafshar/s6e10-golem-oof-library",
    "abdullahsafwan333/oof-dataset", "sadamtorres/ps-s6e10-airline-satisfaction-artifacts",
    "kratosyan/s6e10-busyaprime-five-members-oof", "najiama/s6e10-oof", "sachith7/s6e10-stack-oof-predictions",
    "kagankoral/s6e10-ablation-members-oof", "goodpjw2008/s6e10-tabpfn-member", "koumeimaki/s6e10-route-tfm-nn-oof",
    # new since the 246-member pool (2026-10-17)
    "amanatar/s6e10-forge-own-pairs", "dmitriykolmikov/s6e10-catboost-all-categorical-oof-and-test",
    "vigneshalugunulla/srim-s6e10-bespoke-artifacts",
]
KERNELS = [
    "mitudru/s6e10-analysis-first-equality-blocks-stack", "kagankoral/s6e10-encodings-catboost-realmlp-lb-0-961",
    "amanatar/s6e10-the-original-prior-stack-v5", "megayak/s6e10-honest-oof-stack",
    "thisray/s6e10-xgb-realmlp-catboost-lb-0-96123", "kratosyan/s6e10-xgboost-realmlp-route-id-teacher",
    "yekenot/ps-s6-e10-realmlp-pytabkit", "wangxintong111/s6e10-catboost-rating-crosses-cv-0-96080",
    "wangxintong111/s6e10-lightgbm-context-stats-cv-0-96065", "hemingweb/s6e10-exp01-tabpfn35-full-context",
    # samanyu1808/s6e10-tabpfn-route-og, samanyu1808/s6e10-tabpfn-seeds and kiza123123/s6e10-winning-multi-model-pipeline
    # are no longer public (2026-10-27): their 6 members come from tranbadat/s6e10-own-members (--dataset-extra)
    # new since the 246-member pool
    "hermengardo/single-catboost-eda-ps6e10", "goodpjw2008/s6e10-more-aux-carriers-lb-0-96163",
    "goodpjw2008/s6e10-auxiliary-task-features-lb-0-96161",
    # used by yeashusemwal/trust-the-cv (nested CV 0.962260), missing from our v1 pool
    "mojjeed/s6e10-cv-96180-stacking", "ewtherka/s6e10-factorization-machine-w-more-variants",
    "koumeimaki/s6e10-feature-ladder-what-helps-lightgbm",
    "thethiago28/notebook4ec6a89dc3",
]

RUNNER = r'''
import base64, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

CFG = __CFG__
T0 = time.time()
INPUT, OUT, RUN = (Path(os.environ.get(k, v)) for k, v in (("KAGGLE_INPUT", "/kaggle/input"), ("KAGGLE_WORKING", "/kaggle/working"), ("KG_RUN_DIR", "/tmp/run")))
def log(m): print(f"[{(time.time() - T0) / 60:5.1f}m] {m}", flush=True)

def find(slug, depth=4):
    name, frontier = slug.split("/")[-1].lower(), [INPUT]
    for _ in range(depth):
        nxt = []
        for d in frontier:
            for k in sorted(p for p in d.iterdir() if p.is_dir()):
                if k.name.lower() == name:
                    return k
                nxt.append(k)
        frontier = nxt
    return None

for d in ("src", "data/ext", "artifacts"):
    (RUN / d).mkdir(parents=True, exist_ok=True)
for fn, b64 in CFG["files"].items():
    (RUN / fn).write_bytes(base64.b64decode(b64))
comp = find("playground-series-s6e10")
for f in ("train.csv", "test.csv"):
    os.symlink(comp / f, RUN / "data" / f)
for kind, prefix in (("datasets", ""), ("kernels", "kout-")):
    for ref in CFG[kind]:
        d = find(ref)
        if d is None:
            log(f"MISSING input {ref}")
            continue
        os.symlink(d, RUN / "data" / "ext" / (prefix + ref.split("/")[-1]), target_is_directory=True)
        log(f"linked {ref} ({sum(1 for _ in d.rglob('*'))} entries)")
y = pd.read_csv(RUN / "data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).to_numpy()
folds = np.zeros(len(y), dtype=int)
for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    folds[v] = k
pd.DataFrame({"fold": folds}).to_csv(RUN / "data/folds.csv", index=False)
for cmd in CFG["cmds"]:
    log("$ " + cmd)
    subprocess.run(cmd, shell=True, cwd=RUN, check=True)
log("done")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", nargs="+", default=["0.01", "0.03", "0.1"])
    ap.add_argument("--kernel-extra", nargs="*", default=[])
    ap.add_argument("--dataset-extra", nargs="*", default=[])
    ap.add_argument("--slug", default="s6e10-honest-stack")
    ap.add_argument("--stack-args", default="", help="extra src/stack.py arguments, e.g. --raw")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    files = {f"src/{n}": base64.b64encode((root / "src" / n).read_bytes()).decode() for n in ("pool.py", "stack.py")}
    cfg = dict(files=files, datasets=DATASETS + args.dataset_extra, kernels=KERNELS + args.kernel_extra,
               cmds=["python src/pool.py",
                     "mkdir -p /kaggle/working/pool && cp artifacts/pool/*.csv /kaggle/working/pool/",
                     f"python src/stack.py --C {' '.join(args.C)} {args.stack_args} --save /kaggle/working/stack",
                     "cp artifacts/pool/*.npy /kaggle/working/pool/"])
    out = root / "kernels" / ("stack-cpu" if args.slug == "s6e10-honest-stack" else args.slug)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.py").write_text("# S6E10 honest LR stack over every public OOF library (generated by src/build_stack_kernel.py)\n"
                                         + RUNNER.replace("__CFG__", json.dumps(cfg)))
    meta = dict(id=f"tranbadat/{args.slug}", title=args.slug, code_file=f"{args.slug}.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                dataset_sources=cfg["datasets"], competition_sources=["playground-series-s6e10"],
                kernel_sources=cfg["kernels"], model_sources=[])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({len(cfg['datasets'])} datasets, {len(cfg['kernels'])} kernels)")


if __name__ == "__main__":
    main()
