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
6. **Validate every submission** with `python -m kgkit validate subs/<file>.csv data/sample_submission.csv` (or
   `bash src/submit.sh subs/<id>.csv "<msg>"`, which validates, submits and logs the LB). Standing approval from the
   user (2026-10-24): submit without asking, up to the daily limit; decide open questions by measuring, not asking.
7. **Commit code before long runs** so the ledger's git hash points at the exact code.

## Compute (from 2026-10-24: Linux machine, no local GPU — user rule: use Kaggle GPU **and** CPU)

- Local: Linux, 12 CPU threads, 15 GB RAM, **no GPU**. Python env: `/home/nammovuivui/anaconda3/envs/airline-satisfaction/bin/python`
  (3.12: pandas, sklearn, lightgbm, xgboost, catboost, torch-cpu). `kaggle` CLI on PATH. kgkit 1.8.0 is vendored in
  `src/kgkit` (scripts need no PYTHONPATH); the plugin copy is `~/.claude/plugins/cache/claude-code-kaggle-grandmaster/kaggle-grandmaster/1.8.0`.
  The plugin is installed for another project, so its skills are not auto-loaded here: read them from that path.
- **Kaggle GPU (2x T4)** for every model that trains: `python -m kgkit gpu build --name <exp> --cmd "python src/<x>.py --name <exp> --folds {fold}" ...`
  then `python -m kgkit gpu push kernels/<slug>`; hand-written kernels (TabPFN: `kernels/tabpfn-*`) go through
  `kgkit gpu push <dir> --hours H`. Max 2 GPU sessions at once; weekly quota resets Saturday 00:00 UTC.
  Use valid machine shapes only (`NvidiaTeslaT4`, `NvidiaTeslaP100`).
- **Kaggle CPU** for the big stack over all public OOF libraries: `python src/build_stack_kernel.py` → `kaggle kernels push -p kernels/stack-cpu`
  (links each library to `data/ext/<slug>`, runs `src/pool.py` + `src/stack.py`, saves `stack/` + `pool/`).
- Local CPU: LB-layer analysis (`src/lbcand.py`, `src/tfm_pool.py`), smoke tests (`--smoke --device cpu`).
- Network here is flaky (DNS): downloads go through `src/pull_datasets.sh` (retries) or retry loops.

### Throughput budget
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

- 2026-10-24 (session 3, /goal: public top 10 with Kaggle GPU+CPU, commit progress, file plugin UAT issues):
  - LB moved: 10th = 0.96214, crowd at 0.96213 = 0.85 nina 0.96210 + 0.10 Mehran full-train TabPFN + 0.05 trust-the-cv.
    LB probes 0012-0017 (ledger, tag lb-blend): every move away from those weights lost 1-3e-5 (see memory s6e10-lb-layer).
  - New members in flight on Kaggle GPU: TabPFN-3.5 aux+catfd (kernels/tabpfn-aux), set transformer (src/train_settf.py,
    kernels/s6e10-settf-v1), honest 5-fold TabPFN on Mehran's view (kernels/tabpfn-mehranview). Honest stack on Kaggle CPU.
  - UAT issues filed: #43 (gpu push: unknown machine_shape -> bare 400), #44 (runner flattens inputs, collisions dropped).

- 2026-10-17 (session 2, /goal: public top 10 or 10 subs today): repo github.com/TranBaDat2607/Predicting-Airline-Satisfaction (private).
  - Pool: `bash src/pull_datasets.sh <ds...>` (whole download, per-file fallback on 404) + `kaggle kernels output` into
    `data/ext/kout-*`; `python src/pool.py` → 246 deduped members (stack-level/blend OOFs banned, incl. `124:nj_01`).
  - `src/stack.py` LR C=0.03: 161 members 0.962139, 233 → 0.962194 (**LB 0.96185**), 246 → 0.962200.
  - Residual XGB on the LR margin (members or raw features) and pairwise-rank residual: +0 on fold 0 → the linear stack is saturated.
  - LB blends (`src/blend_public.py`, tagged lb-blend): 0.3 S246 + 0.3 nina 0.96188.txt + 0.4 nina 0.96183 → **LB 0.96189**.
    The 0.96190 public files = ~0.6 honest stack + 0.4 Den Pugovkin family (rank corr 0.996 with honest stacks).
  - TabPFN needs a Prior Labs licence token (TABPFN_TOKEN) — ask the user. TabICL v2 works on 8 GB with ≤150k context
    (`src/train_tabicl.py`, 0.959 fold 0). TabM (pytabkit) is slow here (~70 min/fold) and weak (0.958).
  - Always validate with `bash src/submit.sh <file> "<msg>"` (validates, submits, logs LB). On Windows PYTHONPATH uses `;`;
    kgkit is vendored in `src/kgkit` so scripts need no PYTHONPATH.

<!-- Keep a short running log here: what worked, what did not, what to try next. -->
- 2026-10-17: env + plugin 1.3.1 ready; data downloaded; scouting done. Next: restart in the conda env, run
  `/kaggle-grandmaster:kg-start playground-series-s6e10`, then add `--device` to `src/train_gbdt.py`, install torch (CUDA ≥ 12.8)
  + pytabkit, write `src/train_nn.py`, import public OOFs and build the stack.
