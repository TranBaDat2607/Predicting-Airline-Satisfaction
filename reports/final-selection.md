# Final selection — playground-series-s6e10

_Decided 2026-10-30 (session 6, `/kg-final`). Deadline **2026-10-31 23:59 UTC** = 2026-11-01 06:59 in Vietnam (UTC+7). If the competition's Overview page gives another time, that one counts._

**Status: SELECTED.** `python -m kgkit final` on 2026-10-31 read back 0057 and 0020 as the 2 finals. Re-checked on 2026-10-31 17:45 UTC: 0020 is still the best nested CV, 0057 ties the best unfitted public score. The steps below are kept for the record.

## Website steps (done by the user)

Kaggle has an API to list selected submissions, none to select them: the Kaggle CLI 2.2.5, the `kagglesdk` competition service and the plugin all confirm it. So neither the plugin nor Claude can tick them.

1. Open <https://www.kaggle.com/competitions/playground-series-s6e10/submissions>, signed in as the team (account `tranbadat`).
2. Mark **exactly these two** as the final score (use the file name; both are older rows, so they may need paging down):

| pick | file name | description on the page | date (UTC) | ref | public |
|---|---|---|---|---|---|
| 1 | `0020_stack-lr-pool319-kaggle.csv` | 0020 honest LR stack, 319 public members, nested CV 0.962249 (Kaggle CPU kernel) | 2026-10-23 18:48 | 57569800 | 0.96185 |
| 2 | `0057_lbb-settf16-10.csv` | 0057 HONEST 0.80 nina210 + 0.10 Mehran + 0.10 own set transformers (16-model mean) - final-pick candidate | 2026-10-27 07:19 | 57693156 | 0.96214 |

3. Check it: `PYTHONPATH=src /home/nammovuivui/anaconda3/envs/airline-satisfaction/bin/python -m kgkit final` must print `2 selected as finals`, these two files and no WARNING. It also records them as `final_picks`.

Do not record the picks with `kgkit config final_picks=...` before ticking them: that replaces the auto-selection warning with a one-line note while nothing is selected on Kaggle.

## What happens if nothing is ticked

Kaggle takes the two best public scores: `0078_tilt-v16-exp-t1k18-eta1` (0.96473) and one of `0079`/`0080`/`0081` (nudge probes of 0078, tied at 0.96467). All are public-LB fitted, so their public score says nothing about the private one. Expected cost: about 0.001-0.002 of private AUC against an honest file. That is a rough estimate, not measurable here: S6E9 measured about -0.88 private per +1 public on such files (0078 is +0.00259 public over its 0.96214 base), and our simulation gave -60u to -170u private for +180u to +210u public.

## Why these two

| | 0020 (pick 1) | 0057 (pick 2, hedge) |
|---|---|---|
| what it is | L2 logistic stack on 319 members, every weight chosen in nested CV | 0.80 nina2025 0.96210 + 0.10 Mehran TabPFN + 0.10 own set transformers |
| nested CV | **0.962249**, best honest CV, accepted baseline | none (LB-only record, tag `lb-blend`) |
| public | 0.96185 | 0.96214 |
| protects against | the public-chosen chain not carrying over to private | the CV stack being beaten by the crowd's file, if the public gap is real |
| risk | public gap of 0.00029 behind 0057 | its 0.96210 parent grew from a chain of moves chosen on the public LB, and its weights were LB-tuned too (no row-level fitting) |

- An LB-only blend can only be the hedge, so 0057 is pick 2, not pick 1 as in the earlier draft of this file.
- Different risk axes: rank corr 0.99616, rank-diff sd 0.0253. Noise by the community rule (public sd = 0.003 x rank-diff sd, private = 0.0012 x): public 7.6e-5, private 3.0e-5. The public gap of 29e-5 is about 3.8 public sd, so 0057 may well be better on private too, but the 0.96210 parent's public selection makes that gap optimistic.
- `0060_stack-lr-pool321-kaggle-v3.csv` (public 0.96185) is the same stack as 0020: rank corr 0.99992, CV gap +1.2e-6 with paired SE 4.6e-6. Not a third bet.

## Robustness checks (2026-10-30)

- Kaggle's stored copies (`kaggle competitions submission-download`) are **byte-identical** to `subs/` for both picks.
- 0020 rebuilds from Kaggle kernel `tranbadat/s6e10-honest-stack-v2`, `stack/test_stack.npy`: rank corr 1.0000000.
- 0057 rebuilds from its parts (ledger params; raw-value blend of `data/ext/ps-s6e10-02/0.96210.csv`, `data/lb/1-airline-s6e10-tabpfn/submission.csv`, `data/lb/settf_mean16_test.csv`): rank corr 1.00000000.
- `kgkit validate`: both 299,844 rows, columns `id`, `satisfaction`.
- CV-LB correlation is not reported by kgkit (3 subs with their own CV). Adversarial validation AUC 0.5004: no train/test shift, so nested CV is trustworthy.

## Display only: never select (ledger tag `public-lb-fitted`)

| submission | public | what |
|---|---|---|
| `0054_tilt-v1-eta0-3.csv` | 0.96266 | public-label tilt of 0044, eta 0.3 |
| `0056_tilt-v2-eta0-6.csv` | 0.96285 | public-label tilt of 0044, eta 0.6, EB noise x16 |
| `0058_tilt-v3-eta1.csv` | 0.96297 | public-label tilt of 0044, eta 1.0, EB noise x24, 370 files |
| `0062` … `0071` (session 5: `tilt-v4` … `tilt-v12`, `probe-top10k-x2`) | 0.96361 … 0.96432 | robust tilts (Student-t noise, version-matched notebook scores) |
| `0072` … `0081` (session 6: `tilt-v13` … `tilt-v16`, `probe-n*`) | 0.96451 … **0.96473** | exponential-posterior tilts with mined exact pairs and nudge probes; 0078 is the public best |

These exist only to hold the public-LB position the goal asked for. They are fitted to the 20 % public rows, so on the 80 % private rows they are noise on top of 0044. As far as we know, the selection does not change the public board, which shows each team's best public score.

## Evidence

- The public top is fitted by several teams (us 0.96473, Kawamata 0.96451, Altunok 0.96446): expect a large shake-up. Kawamata wrote in the forum that his probing gains will not carry over to private.
- Honest stack saturated: set transformers, SRIM LGB, TabPFN variants and the generator-surrogate LLR all add +0 (`0059`, memory `s6e10-members-tried`).
- The public crowd now sits at 0.96223 (tkms0816's file), which its own author says includes row-block noise selected on the public LB.

## Outcome (private board, read 2026-11-06 with `kgkit final`)

| final | public | private |
|---|---|---|
| `0057_lbb-settf16-10` | 0.96214 | 0.96222 |
| `0020_stack-lr-pool319-kaggle` | 0.96185 | **0.96236** (team score) |

Rank 426 / 3,735, no medal (bronze 0.96237, silver 0.96238, gold 0.96244; 48 teams tied at 0.96236). Best private of any
submission: 0.96238 (`0009`, 2e-5 higher). The public-best tilt `0078` scored 0.96027 private (it would have ranked
~1,981), and every tilt scored between 0.9600 and 0.9618. The LB-weighted hedge 0057 was the weaker final. Full analysis: `reports/postmortem.md`.
