# Next steps — playground-series-s6e10

_Updated 2026-10-23 ~18:30 UTC (session 3). Goal: public LB top 10. The line is **0.96214** (10th place) → we need
**≥ 0.96215**; ties rank by earlier submission. Our best public: 0.96212 (0016, 0017); 0.96189 before this session._

## Where we stand

| | |
|---|---|
| Public crowd file "A" | 0.96213 = 0.85 nina2025 0.96210 + 0.10 Mehran full-train TabPFN + 0.05 trust-the-cv (prob space) |
| LB probes today | 0012 0.96210 · 0013 0.96209 · 0014 0.96210 · 0016 0.96212 · 0017 0.96212 (all lb-blend, see ledger) |
| Lesson | A's weights are a local optimum; generic novel files and the honest TabPFN mean do not reproduce Mehran's +3e-5 |
| Honest stack (reference) | trust-the-cv: 253 members, nested CV 0.962260; ours 0004: 246 members 0.962200 |

## Running on Kaggle

| Kernel | Resource | What | Output → |
|---|---|---|---|
| `tranbadat/s6e10-honest-stack` | CPU | LR stack over 15 datasets + 15 kernel outputs (`src/build_stack_kernel.py`) | `data/ext/kout-s6e10-honest-stack/` |
| `tranbadat/s6e10-tabpfn35-aux-catfd` | GPU 2xT4 | TabPFN-3.5, raw + catFD + 13 aux expected ratings, 400k ctx | `data/ext/kout-s6e10-tabpfn35-aux-catfd/` |

Queued for the GPU quota reset (2026-10-24 00:00 UTC, 30 h):
- `kernels/s6e10-settf-v1` (kgkit gpu build): set transformer, value-embedding tokens + side features + RTD pretraining
  (`src/train_settf.py`; cdeotte's Q-FT v2 reached OOF 0.9608 with this design; no public OOF of this family exists).
  Rebuild it after any change to `src/train_settf.py` (`python -m kgkit gpu build --name settf_v1 ...`, see CLAUDE.md).
- `kernels/tabpfn-mehranview`: honest 5-fold TabPFN on Mehran's view (47 in-fold TE combos + KMeans group), 200k ctx.

## Then

1. Stack v2 on Kaggle CPU: `python src/build_stack_kernel.py --kernel-extra tranbadat/s6e10-tabpfn35-aux-catfd ...`
   (+5 sources trust-the-cv has and v1 lacked). Keep a member only if nested CV goes up on most folds.
2. LB layer: new honest members as directions on top of A (`src/lbcand.py`), small weights, one probe per idea.
3. Finals (by 2026-10-29): pick 1 = best LB file; pick 2 = best honest stack (CV-only), rank corr < 0.999 with pick 1.

## Plugin UAT issues filed this session
TranBaDat2607/claude-code-kaggle-grandmaster #43 (gpu push: unknown machine_shape → bare 400), #44 (gpu build runner
flattens inputs; same-named files from different sources silently dropped).
