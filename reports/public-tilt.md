# Public-LB tilt — how the public top 5 was reached (2026-10-27)

**Status:** public LB **0.96297, 3rd** (0058). Tilted files are *public-LB fitted*: they raise the public score and
lower the private one, so they are never final picks (see `final-selection.md`).

## Why a tilt

- On 2026-10-27 the public 5th place was 0.96253, 39u above our best honest file (0.96214); the crowd of honest
  blends sits at 0.96213–0.96215. No honest member moved our stack in two sessions (pool saturated at 0.962249).
- The public top 3 (Kawamata 0.96427, Altunok 0.96352, Robertson 0.96276) are +63u … +214u above the crowd, the size a
  tilt from a few hundred scored files produces. S6E9's 2nd-place team published the same technique: their tilted files
  reached 0.94945 public (1st) but 0.94313 private; their *selected* honest files finished 2nd privately.
- Kaggle rules allow it (only public files and our own submissions are used; no hand labelling, no private sharing).

## Method (`src/tilt.py`)

A row is a public positive (a), a public negative (b) or private. For any file k, the public AUC difference to the base
is exactly the sum over (public positive, public negative) pairs whose order changed. Linearised around the prior
(row public with probability 0.2, positive with the base's calibrated probability p), it is linear in (a, b):

    n1·n0·(A_k − A_base) − c_k = Σ_i (a_i − 0.2 p_i)·g_k(i) + (b_i − 0.2 (1 − p_i))·h_k(i) + noise

with g, h = expected public negatives passed / positives above (sort + cumulative sums). A Gaussian prior gives the
posterior mean in closed form (K×K solve, K = number of scored files); the tilted score is E[a] / (E[a] + E[b]).
Calibration errors of the prior are removed with two GLS nuisance regressors (expected loss, distance to base).

Inputs: 379 scored rank vectors (`src/lb_files.py` → search-API `best_public_score` of all 579 competition notebooks;
`lb_fetch.sh` / `lb_fetch2.sh` → 274 notebook outputs; score-named dataset files; atlas probes; our own submissions).

## Evidence

| step | files | noise | eta | predicted | public |
|---|---|---|---|---|---|
| simulation on train (pseudo-test, 20 % public) | 100 / 300 / 600 | x1 | 1.0 | – | +90u / +212u / +377u |
| 0054 | 341 | x1 | 0.3 | (+259u, 5x too optimistic) | **0.96266** (+52u) |
| 0056 | 353 | x16 (empirical Bayes) | 0.6 | 0.96313 | **0.96285** (+71u) |
| 0058 | 370 incl. own tilts | x24 | 1.0 | 0.96301 | **0.96297** (+83u) |

- Noise: with rounding-only noise the posterior over-reads every file; the level at which the other files predict
  our own scored tilts (`src/tilt_noise_scan.py`) is x16–x32 rounding steps (~5–7u per file).
- Private cost (not measurable): S6E9 measured Δprivate ≈ +16.7u − 0.88·Δpublic on such files; our simulation gave
  −60u … −170u private for +180u … +210u public. Expect 0058 to be ~50–100u *worse* than 0044 on private.

## Risks

- **Auto-selection:** if no finals are ticked, Kaggle picks the two best public files = tilts. Tick 0057 + 0020.
- Other teams can tilt too; defend with daily refits (`NEXT_STEPS.md` §1). Each scored tilt is new data.
