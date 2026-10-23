# Next steps — playground-series-s6e10

_Updated 2026-10-23 ~20:45 UTC (session 3). Goal: public LB top 10. The line is **0.96214** (10th place) → we need
**≥ 0.96215**; ties rank by earlier submission. Our best public: **0.96212** (several LB-layer probes); 0.96189 before._

## Where we stand

| | |
|---|---|
| Public crowd file "A" | 0.96213 = 0.85 nina2025 0.96210 + 0.10 Mehran full-train TabPFN + 0.05 trust-the-cv (prob space) |
| LB probes 2026-10-23 | 10/10 used: every move off A lost 1–4e-5 (table in the ledger, tag lb-blend; memory s6e10-lb-layer) |
| What helps on the LB | only Mehran's full-train TabPFN (his 47 TE combos + KMeans group, 10 estimators): +3e-5 at 10 %, peak at 10 % |
| What does not | honest TabPFN mean, own TabPFN aux, novel single models, Mehran's GBDT, more/less of the honest stack |
| Honest stack | 0020: 319 members (Kaggle CPU kernel v2), nested CV **0.962249**, public 0.96185 — baseline |

## Running / queued on Kaggle

| Kernel | Resource | Status |
|---|---|---|
| `tranbadat/s6e10-settf-v1` | GPU | running (pushed 20:26 UTC on the expiring quota): set transformer d512/σ32 + d256/σ32, 3 epochs |
| `kernels/tabpfn-mehranview-full` | GPU | queued by `src/gpu_queue.sh` for 2026-10-24 00:03 UTC: independent full-train re-draw of Mehran's file (300k ctx, ≤8 draws) |
| `kernels/tabpfn-mehranview` | GPU | queued: honest 5-fold Mehran view (200k ctx, 2 seeds) → OOF for the stack |

## Plan for 2026-10-24 (10 new probes)

1. settf: `python -m kgkit gpu collect tranbadat/s6e10-settf-v1` (or `kaggle kernels output`), add to the pool with
   `src/pool_add.py`, stack locally (`src/stack.py --C 0.03`), keep if nested CV > 0.962249 on most folds.
2. Mehran re-draw M2: probes 0.85 N + 0.05 M + 0.05 M2 + 0.05 S, then 0.85 N + 0.10 M2 + 0.05 S, then line search.
3. Mehran-view 5-fold OOF: stack member test (+ honest stack v3 on Kaggle CPU: `python src/build_stack_kernel.py --kernel-extra ...`).
4. Finals (by 2026-10-29): pick 1 = best LB file; pick 2 = best honest stack (CV-only), rank corr < 0.999 with pick 1.

## Plugin UAT issues filed this session
#43 (gpu push: unknown machine_shape → bare 400), #44 (runner flattens inputs; same-named files dropped), #45 (vendored
kgkit cannot gpu build: templates missing), #46 (folds hash = file bytes; CRLF vs LF), #47 (gpu build embeds folds.csv →
2.5 MB script → bare 400).
