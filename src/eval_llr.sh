#!/usr/bin/env bash
# Evaluate a collected LLR kernel output as stack members: label-free AUC of each column, then the honest LR stack
# (nested CV, frozen folds) on the 319-member pool + the LLR columns, against 0020's 0.962249.
#   bash src/eval_llr.sh data/ext/kout-s6e10-llr llr1
set -euo pipefail
d=$1; tag=$2
P=${PY:-/home/nammovuivui/anaconda3/envs/airline-satisfaction/bin/python}
"$P" - "$d" "$tag" <<'EOF'
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
d, tag = sys.argv[1], sys.argv[2]
tr, te = np.load(f"{d}/llr_train.npy"), np.load(f"{d}/llr_test.npy")
y = pd.read_csv("data/train.csv", usecols=["satisfaction"]).satisfaction.astype(int).to_numpy()
for j, n in enumerate(["last", "first"]):
    np.save(f"{d}/oof_{tag}_{n}.npy", tr[:, j].astype(np.float64)); np.save(f"{d}/test_{tag}_{n}.npy", te[:, j].astype(np.float64))
    print(f"{tag}_{n}: label-free AUC on train {roc_auc_score(y, tr[:, j]):.6f}  test mean {te[:, j].mean():+.3f} train mean {tr[:, j].mean():+.3f}")
EOF
"$P" src/pool_add.py --base artifacts/pool --out "artifacts/pool_$tag" \
  --add "own:${tag}_last:$d/oof_${tag}_last.npy:$d/test_${tag}_last.npy" \
  --add "own:${tag}_first:$d/oof_${tag}_first.npy:$d/test_${tag}_first.npy"
mv artifacts/pool artifacts/pool.v2link
ln -sfn "pool_$tag" artifacts/pool
trap 'rm -f artifacts/pool; mv artifacts/pool.v2link artifacts/pool' EXIT
"$P" src/stack.py --C 0.03 --name "stack_lr_pool321_$tag" --notes "LR C=0.03 on stack-v2 pool (319) + $tag LLR members (last, first)"
