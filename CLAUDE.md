# Kaggle competition: playground-series-s6e10

Workspace managed by the **kaggle-grandmaster** plugin. Read this before every session.

## Facts (keep updated — `.kaggle-gm/competition.json` is the source of truth)

- Metric: **ROC AUC** (higher is better). The exact implementation lives in `src/metric.py` and is unit-tested against a known value.
- Task: binary — target: `satisfaction` (bool, 44.4% True) — id column: `id`
- Submission: `id,satisfaction` with a **probability** per row (299,844 test rows)
- Code competition: no. Limits: **10 submissions/day**, teams up to 3
- Final deadline: **2026-10-31 23:59 UTC** (always confirm on the competition page)
- Data: train 699,635 × 22 features, test 299,844. 4 categoricals (Gender, Customer Type, Type of Travel, Class), Age,
  Flight Distance, 14 service ratings 0–5, 2 delays. Only `Arrival Delay in Minutes` has NaNs (0.04%).
- Original dataset: `data/original/` (`teejmahal20/airline-passenger-satisfaction`, 129,880 rows, target
  `satisfied` / `neutral or dissatisfied`, has an extra `Inflight service` column and an `Unnamed: 0` index).
- Adversarial validation train vs test: AUC 0.5004 → no shift, plain stratified CV is trustworthy.

## Layout

```
data/        raw competition files (never edited, never committed); data/original/ = original dataset
src/         training / inference code (committed)
notebooks/   exploration only — anything that matters gets moved into src/
artifacts/   per-experiment OOF + test predictions + models (artifacts/<exp_id>/)
subs/        submission CSVs, named after the experiment id
reports/     EDA, recon, adversarial validation, post-mortems
.kaggle-gm/  competition.json (facts), ledger.jsonl (every experiment, committed)
```

## Non-negotiable rules

1. **The CV scheme is frozen**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `satisfaction`,
   written to `data/folds.csv`. This is the split the whole community uses, so our OOFs stack with the public
   OOF libraries. Never change it.
2. **Every run is logged** with `kgkit.experiment.Ledger().log(...)` — CV, fold scores, params, features, OOF and test predictions.
3. **One change per experiment, compared and decided** against the accepted baseline (`python -m kgkit ledger baseline`):
   `python -m kgkit ledger compare <new> --truth data/train.csv:satisfaction --folds data/folds.csv:fold`, then
   `python -m kgkit ledger decide <new> keep|discard|inconclusive "<why>" --parent <baseline>`. Ideas live in `python -m kgkit backlog`.
4. **Trust CV, verify with LB.** Log every public LB score with `python -m kgkit ledger lb <id> <score>`. The public LB
   is packed at ~0.9619; 5th-decimal differences there are noise.
5. **No target leakage**: target encodings out-of-fold; encoders fit inside the fold. Never pseudo-label from public
   consensus predictions (their test preds contain every fold's labels — CV up, LB down).
6. **Validate every submission** with `python -m kgkit validate subs/<file>.csv data/sample_submission.csv`, and **ask the
   user before submitting**.
7. **Commit code before long runs** so the ledger's git hash points at the exact code.

## Local GPU: how to train here (long-term rule — all training is local)

**Never use Kaggle GPUs for this competition.** Do not run `/kg-gpu`, `kgkit gpu push`, or `kaggle kernels push`.
Every experiment runs on this machine. When a plugin skill says "without a local GPU, run on Kaggle", it does not
apply here.

### Hardware and environment
- GPU: **NVIDIA GeForce RTX 5060, 8 GB VRAM** (Blackwell, sm_120). CPU: 12 threads. Disk: ~50 GB free.
- Python env: conda **`airline-satisfaction`** (Python 3.12). Use it for everything:
  - Python: `C:/Users/Admin/anaconda3/envs/airline-satisfaction/python.exe`
  - Kaggle CLI: `C:/Users/Admin/anaconda3/envs/airline-satisfaction/Scripts/kaggle.exe`
  - `python` on PATH is base Anaconda (3.13, **no** lightgbm/xgboost/catboost). Use the full path unless the session
    was started from `conda activate airline-satisfaction`.
- kgkit: `PYTHONPATH="<kgkit home from session brief>" <env python> -m kgkit <cmd>`.
- Windows console is cp1252: when printing notebook text or emoji, set `PYTHONIOENCODING=utf-8` (or
  `sys.stdout.reconfigure(encoding="utf-8")` when running with `-I`).

### Which model runs where (verified 2026-10-17)
| Model | Device | How |
|---|---|---|
| XGBoost 3.4 | **GPU** ✅ | `device="cuda", tree_method="hist"`. Template: `--params '{"device":"cuda"}'`. Raw 5-fold baseline: 0.95879 CV in 31 s |
| CatBoost 1.2.10 | **GPU** ✅ | `task_type="GPU"`. Template: `--params '{"task_type":"GPU"}'` |
| LightGBM 4.7 | **CPU only** | The pip wheel has no GPU/CUDA learner. Use `n_jobs=6` when a GPU job runs alongside |
| sklearn (linear, kNN, SVM, MLP), stacking LR | CPU | cuML is not available on native Windows; use sklearn |
| RealMLP / TabM (pytabkit), TabPFN, TabICL | GPU | **Not installed yet**: needs `torch` built for CUDA ≥ 12.8 (required for RTX 50xx) + `pytabkit` / `tabpfn` |

The plugin template `train_gbdt.py` defaults to CPU for every model. After `kg-start` copies it to `src/`, add a
`--device {cpu,cuda}` flag that sets the XGBoost/CatBoost GPU params, and use `--device cuda` by default.

### GPU scheduling rules (8 GB, one card)
1. **One GPU job at a time.** Do not launch parallel `experiment-runner` agents that both use the GPU. A GPU job plus one
   CPU job (LightGBM, sklearn, stacking) in parallel is fine.
2. Long runs (> ~2 min) go in the background (`run_in_background`), logging to `artifacts/<exp>/train.log`. Check progress
   with the log and `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`, not tight polling loops.
3. Before starting a GPU job, check `nvidia-smi` that no other process is holding VRAM.
4. Always smoke-test first (`--smoke`: 1 fold, few rounds) before a full 5-fold run.
5. Free memory between folds/models in torch code: `del model; gc.collect(); torch.cuda.empty_cache()`.
6. On CUDA out-of-memory: lower batch size / context size and retry once. Don't fall back to Kaggle.
7. **TabPFN on 8 GB**: public runs used the full 560k-row context (17.5 GB) and are not reproducible here. Subsample the
   context (start ≤ 50k rows, stratified, per fold), predict the validation/test rows in batches, small `n_estimators`.
   Treat it as a stack member, not a main model. Its licence is non-commercial (allowed for Kaggle use).

### Throughput budget
- XGBoost GPU 5-fold run on raw features: ~30 s → feature experiments are cheap, so screen many.
- Deadline 2026-10-31: plan finals (stack + 2 picks) by 2026-10-29; no new model families after 2026-10-27.

## Strategy (from scouting, 2026-10-17)

- Public LB top 0.96284; a crowd at ~0.9619 comes from stacking ~245 public OOF/test pairs (nested CV ~0.96222).
- Best public single models: tuned LightGBM + out-of-fold target encoding (3 smoothings) + pair features + `Flight Distance`
  "route" profile stats + original-dataset lookup stats ≈ **0.9611**; RealMLP ≈ 0.961; TabPFN-3.5 (n_est=4, full context) 0.9615.
- Biggest lever: stack public OOF libraries (`cdeotte/s6e10-124-public-oof-and-test-predictions`,
  `cdeotte/s6e10-tfm-oof-and-test-predictions`, `goodpjw2008/s6e10-goodpjw2008-extra-members-oof`,
  `dariushafshar/s6e10-golem-oof-library`) with our own diverse members. Use L2 logistic regression on logits, with a separate
  penalty per model family.
- Plain appending of the original rows to training: +0.00007 on fold 0, i.e. negligible. Use it for lookup features instead.

## Commands

- `/kaggle-grandmaster:kg-status` — where we stand
- `/kaggle-grandmaster:kg-experiment <idea>` — run one disciplined experiment (locally)
- `/kaggle-grandmaster:kg-grind` — autonomous improvement loop (local GPU only, one GPU job at a time)
- `/kaggle-grandmaster:kg-ensemble` — blend the best diverse models
- `/kaggle-grandmaster:kg-submit <file>` — validate, submit, log (ask first)
- `/kaggle-grandmaster:kg-final` — choose the final submissions

## Current plan / notes

<!-- Keep a short running log here: what worked, what did not, what to try next. -->
- 2026-10-17: env + plugin 1.3.1 ready; data downloaded; scouting done. Next: restart in the conda env, run
  `/kaggle-grandmaster:kg-start playground-series-s6e10`, then add `--device` to `src/train_gbdt.py`, install torch (CUDA ≥ 12.8)
  + pytabkit, write `src/train_nn.py`, import public OOFs and build the stack.
