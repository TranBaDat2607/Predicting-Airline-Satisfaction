# Next steps — playground-series-s6e10

_Updated 2026-10-23 ~20:45 UTC (session 3). Goal: public LB top 10. The line is **0.96214** (10th place) → we need
**≥ 0.96215**; ties rank by earlier submission. Our best public: **0.96212** (several LB-layer probes); 0.96189 before._

## Update 2026-10-24 06:30 UTC

- **Public 0.96214** (0044/0045): 0.80/0.775 nina210 + 0.10 Mehran + 0.10/0.125 **8-model set-transformer mean**
  (OOF 0.96122). 11th, tied with 10th (Aurrog 0.96214) but listed below. Equal rounded scores are ordered by exact
  score, so the 2-model probes (0.96213) already ranked above the A crowd. Need 0.96215.
- The LB rewards settf *quality*: 2 models (OOF 0.96087) 0.96213 → 8 models (0.96122) 0.96214. On OOF it adds nothing to
  the honest stack (5-mean member: 0.962247; SRIM LGB member: 0.962249), so it is an LB-layer direction only.
- GPU now: `s6e10-settf-v4` (d384 s7/s8, d256 s9/s10) and `s6e10-settf-v5` (d512 s11/s12, d384 s13, d256 s14) → 16 models.
- Next probes (2026-10-25): 16-model mean at 0.10 / 0.125 / 0.15; then keep trust-cv (0.05) next to it.

## Update 2026-10-24 00:35 UTC

- **Our best public: 0.96213** (0032/0033/0034): 0.85-0.75 nina210 + 0.10 Mehran + 0.05-0.15 **own set transformer**
  (mean of settf d512/d256 test preds). It is the only direction that does not lose (20 % → 0.96212); every other
  probe (19, incl. 4 re-draws of Mehran's file) lost 1-4e-5. Need 0.96215 for top 10.
- Lever: a stronger/cleaner set-transformer direction. `kernels/s6e10-settf-v2` (d512 seed 1, d768, d384) is queued by
  `src/gpu_queue.sh` for the first free GPU slot; a d192 CPU run is training locally (`artifacts/settf_cpu192/`).
  Then average all settf models and re-probe 0.10 / 0.125.
- GPU now: `tranbadat/s6e10-tabpfn35-mehranview` (5-fold, OOF) and `tranbadat/s6e10-tabpfn35fast-catfd10` (fast checkpoint).
- `src/lb_surface.py`: LB-surface fit over 61 scored files has no predictive power near A (do not spend probes on it).

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
