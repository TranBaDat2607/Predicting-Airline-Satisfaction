# Next steps — playground-series-s6e10

_Written 2026-10-17 ~16:40 UTC. Goal: public LB top 10 (line is **0.96193**, so we need **≥ 0.96194**; ties rank by
earlier submission)._

## Where we stand

| | |
|---|---|
| Best public LB | **0.96189** (`0005_blend-s246-n88t-n83`, also `0008`) |
| Best honest stack | `0004_stack-lr-pool246`: LR C=0.03 on 246 members, nested CV **0.962200** (accepted baseline) |
| Submissions today | 4 used / 10 (0003 → 0.96185, 0005 → 0.96189, 0008 → 0.96189, 0009 → 0.96188). Quota resets 00:00 UTC |
| Kaggle GPU quota | 19.5 h left this week (resets 2026-10-24) |

Full history: `python -m kgkit ledger list` (run from `src/`, kgkit is vendored there) and the notes in `CLAUDE.md`.

## 1. Run the TabPFN-3.5 aux member on Kaggle GPU (highest expected gain)

The kernel is ready in `kernels/tabpfn-aux/` (private, 2×T4, internet on, inputs: competition +
`sachith7/s6e10-stack-oof-predictions`). It is the public full-context TabPFN-3.5 recipe adapted to a **new view**:
raw columns + Flight Distance as a categorical + sachith7's 13 auxiliary expected values, 1 seed, 400k-row context,
frozen folds (`StratifiedKFold(5, shuffle=True, random_state=42)`, rebuilt in the script). Local smoke test passes.
Expected runtime ~4.5 h wall / ~4.5 quota hours.

The push was blocked: **"Maximum batch GPU session count of 2 reached"** — `tranbadat/v2-train` and
`tranbadat/v2-dev-train` (another competition) hold both slots. Once one frees:

```bash
cd kernels/tabpfn-aux
kaggle kernels push -p . -t 27000
kaggle kernels status tranbadat/s6e10-tabpfn35-aux-catfd
# when COMPLETE:
kaggle kernels output tranbadat/s6e10-tabpfn35-aux-catfd -p ../../data/ext/kout-s6e10-tabpfn35-aux-catfd
```

It cannot run locally: TabPFN prediction OOMs on the 8 GB card even at 120k context and 1,024-row batches.
TabDPT also needs > 8 GB (14 GB at batch 2048 × context 1024); a Kaggle T4 run is the next option for it too
(`src/train_tabdpt.py`).

## 2. Re-stack with the new member

```bash
C:/Users/Admin/anaconda3/envs/airline-satisfaction/python.exe src/pool.py      # picks up data/ext/kout-*/oof_*.csv + test_*.csv
C:/Users/Admin/anaconda3/envs/airline-satisfaction/python.exe src/stack.py --C 0.03 --name stack_lr_pool248 --log
```

Keep it only if nested CV beats 0.962200 on most folds (`kgkit ledger compare`). Reference: our own TabICL member
(`0010`, OOF 0.95697) moved the stack by just +0.000001, so a member needs to be strong *and* different.

## 3. Submit (validate first, always through the helper)

```bash
python src/blend_public.py --base <new stack id> \
    --add data/ext/ps-s6e10-7/0.96188.txt:0.3 data/ext/ps-s6e10-7/0.96183.csv:0.4 --name blend_<...>
bash src/submit.sh subs/<id>.csv "<message>"          # validates, submits, waits, logs the LB score
```

LB evidence for the blend weights: honest stack + 0.4 × the Den Pugovkin family (`0.96183.csv`) ≈ +0.00004 over the
stack alone. Swapping in Den's current output (`0.96188`) or adding 10 % of the 0.96160 TabPFN file did not help.

## What was tried and discarded (don't repeat)

- Family-wise L2 penalties (`0007`): gain only on the tuning fold; folds 1–4 all −1e-5.
- XGBoost residual on the LR margin (members, raw features, or `rank:pairwise`): +0 on fold 0.
- LightGBM stacker (`0006`): 0.961915 alone; 0.9 LR + 0.1 LGB = +4e-6 (noise).
- Pruning near-duplicate members (188 kept): same CV; 111 kept: worse.
- Own TabM (pytabkit): ~70 min/fold here and only 0.958; stopped after 3 folds.
- No leak: no exact train/test duplicates, original-dataset matches only on 27 % of rows with 94 % label agreement
  (already used by public lookup features).

## Other ideas, if time allows

1. TabPFN-3.5 with more seeds or a 15-fold split on the same view (public 15-fold TabPFN was the largest single
   measured gain, +0.00015).
2. TabDPT (retrieval in-context learner, a new family) on a Kaggle T4.
3. Keep one CV-only candidate (`0004` or its successor) as a final pick; the LB blends' weights are LB-tuned.

## Plugin UAT issues filed this session

TranBaDat2607/claude-code-kaggle-grandmaster #22 (discussions sync cp1252 crash), #23 (dataset download 404 →
per-file fallback), #24 (train_gbdt template name / no `--device`), #25 (ledger needs LB-only records),
#26 (credential hook blocks the username; no preflight for the 2-GPU-session limit).
