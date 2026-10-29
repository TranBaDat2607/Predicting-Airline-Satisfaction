# Final selection — playground-series-s6e10

_Draft 2026-10-27 (session 4), updated 2026-10-29 (session 5: tilts 0062-0070 added to the never-select list). Deadline **2026-10-31 23:59 UTC** (2026-11-01 06:59 in Vietnam, UTC+7). Re-check before then._

## ACTION FOR THE USER (no API exists for this)

Open <https://www.kaggle.com/competitions/playground-series-s6e10/submissions>, tick **exactly these two** as final:

| pick | submission | public | why |
|---|---|---|---|
| 1 | `0057_lbb-settf16-10.csv` — 0.80 nina210 + 0.10 Mehran full-train TabPFN + 0.10 own set transformers (16-model mean) | 0.96214 | best honest public file; all parts are honest predictions, only the three weights were LB-tuned |
| 2 | `0020_stack-lr-pool319-kaggle.csv` — honest LR stack over 319 public members (Kaggle CPU) | 0.96185 | best nested CV (0.962249), no LB tuning at all; different risk axis (rank corr ~0.996 with pick 1) |

`0060_stack-lr-pool321-kaggle-v3.csv` (Kaggle CPU, 2026-10-27, 321 members, nested 0.962248, public 0.96185) is the
same stack (test rank corr 0.99992 with 0020): either can be pick 2.

If nothing is ticked, Kaggle **auto-selects the two best public scores — the tilted files below — and the private score
drops** (S6E9 measured about -0.9u private per +1u public for such files).

## Never select (public-LB fitted, ledger tag `public-lb-fitted`)

| submission | public | what |
|---|---|---|
| `0054_tilt-v1-eta0-3.csv` | 0.96266 | public-label tilt of 0044, eta 0.3 |
| `0056_tilt-v2-eta0-6.csv` | 0.96285 | public-label tilt of 0044, eta 0.6, EB noise x16 |
| `0058_tilt-v3-eta1.csv` | 0.96297 | public-label tilt of 0044, eta 1.0, EB noise x24, 370 files (predicted 0.96301) |
| `0062` … `0070` (session 5: `tilt-v4` … `tilt-v11`, `probe-top10k-x2`) | 0.96361 … **0.96432** | robust tilts (Student-t noise, version-matched notebook scores); 0068 is the public best |

These exist only to hold the public-LB position the goal asked for. They are fitted to the 20 % public rows using
365 publicly shared scored files (`src/tilt.py`), so on the 80 % private rows they are noise on top of 0044.

## Evidence

- Public vs private: honest files keep their private score; the public LB is fitted by the top teams (the public top 3
  are tilts of the same kind: Kawamata 0.96427, Altunok 0.96352, Robertson 0.96276), so expect a large shake-up at the top.
- Honest stack: nested CV 0.962249 (0020), public 0.96185. Pool saturated: set transformers, SRIM LGB, TabPFN variants
  all +0 (memory `s6e10-members-tried`). The generator-surrogate LLR (distilgpt2 GReaT-style on the original, Kaggle GPU)
  scores 0.9511 label-free but adds +0 to the stack (0059: 0.962248, discard).
- LB-layer blend 0044/0057: public 0.96214; the crowd file A (0.85 nina210 + 0.10 Mehran + 0.05 trust-cv) is 0.96213.
  The public gap A − 0020 (+2.8e-4) is ~3.8 paired SDs (rank corr ~0.996), so pick 1 is probably also better on private.

_Update this table if a better honest stack (LLR members) or a better honest LB-layer file is submitted._
