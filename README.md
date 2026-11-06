# Predicting Airline Satisfaction: Kaggle Playground S6E10

Binary classification of airline passenger satisfaction on synthetic tabular data (700k train rows, 22 features),
scored by ROC AUC. [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e10)

| | |
|---|---|
| **Private leaderboard** | **0.96236, rank 426 / 3,735 (top 11.4%)**, no medal (bronze cut-off 0.96237) |
| Public leaderboard | 0.96473, rank 1, from a public-LB-fitted file that was deliberately **not** a final |
| Winning private score | 0.96318 |
| Finals | `0020` honest LR stack (private **0.96236**) and `0057` stack + public blend (private 0.96222) |
| Dates | joined 2026-10-17, ended 2026-10-31; file submissions, 10 per day |

Standings read on 2026-11-06 with `kgkit final`. The medal zone is computed from Kaggle's progression table. 48 teams
share 0.96236 (ranks 390–437), and 315 teams sit at 0.96237–0.96238 (ranks 75–389), so ranks 75–437 are separated by
two score units, the size of the noise.

**The problem.** The data is synthetic and the public leaderboard is packed. Hundreds of teams stack the same public
out-of-fold libraries and land within 1e-5 of each other. The test is split 20% public / 80% private, so with ~670
publicly scored files you can fit the public 20%, and that fit carries no information about the private 80%. This
repo documents both tracks: the honest stack that decided the result, and a public-LB-fitting experiment that reached
public 1st and showed exactly how much fitting costs.

![Public vs private score of all 49 submissions. The 3 honest stacks score highest on private (0.96236–0.96237). The 23 blends with public files score 0.96220–0.96238. The 23 files tilted toward the public labels reach public 0.96473 but fall to 0.9600–0.9618 on private.](docs/figures/public_vs_private_all.png)

## My work and other people's work

The final score is a stack. Most of its ~320 members are **public OOF/test pairs published by other Kagglers**,
listed under Credit. My contribution is the pipeline that collects, audits and stacks them honestly, plus my own
members. My own models alone reached 0.9607 OOF AUC (set transformer) and 0.9612 as a 16-model mean, against 0.96225
for the full stack. They added +0 to a stack that was already saturated, so the score is mostly the public libraries
combined well.

**Mine**
- **OOF pool builder** ([`src/pool.py`](src/pool.py), [`src/pull_datasets.sh`](src/pull_datasets.sh)): downloads about 30
  public OOF libraries with retries, reads their six file layouts, converts to logits, checks row order and AUC, and
  removes exact duplicates and stack-level members (which would leak).
- **Honest stack** ([`src/stack.py`](src/stack.py), [`src/stack_family.py`](src/stack_family.py),
  [`src/build_stack_kernel.py`](src/build_stack_kernel.py)): L2 logistic regression on standardised logits, with
  per-family penalties and every hyperparameter chosen in **nested** CV on the frozen folds. Runs as a Kaggle CPU
  kernel (`kernels/s6e10-honest-stack-v2`, `-v3`).
- **Own members, trained on Kaggle 2xT4** (all on the same frozen folds):
  - set transformer with RTD pre-training, 16 seeds/sizes ([`src/train_settf.py`](src/train_settf.py), `kernels/s6e10-settf-v*`)
  - TabPFN-3.5 with auxiliary and route features (`kernels/tabpfn-*`)
  - TabICL v2 ([`src/train_tabicl.py`](src/train_tabicl.py))
  - GBDTs ([`src/train_gbdt.py`](src/train_gbdt.py))
  - a distilgpt2 GReaT-style replica of the data generator used as a likelihood-ratio member ([`src/train_llr.py`](src/train_llr.py), `kernels/s6e10-llr*`)
- **Public-LB fitting study** (`src/lb_*.py`, `src/tilt*.py`, `src/probe_*.py`, [`reports/public-tilt.md`](reports/public-tilt.md)):
  an empirical-Bayes model of the public labels fitted to ~670 publicly scored files, with Student-t noise,
  per-version score matching and an exponential-tilt posterior. It reached public 1st (0.96473). It was always
  tagged display-only and was never a final.
- **Experiment discipline**: every run is logged in a ledger (`.kaggle-gm/ledger.jsonl`, 81 experiments) with its CV,
  fold scores and decision, using the [kaggle-grandmaster](https://github.com/TranBaDat2607/claude-code-kaggle-grandmaster) plugin (`src/kgkit`).

**From public work**
- The ~320 stack members: public OOF/test libraries by the authors under Credit, used as stack inputs.
- `0057` blends 0.80 of a public submission file (nina2025) and 0.10 of a public TabPFN notebook's output (mehrankazeminia).

## What moved the score

| Step | CV (nested) | Public | Private |
|---|---:|---:|---:|
| XGBoost on raw features (`0001`) | 0.95883 | – | – |
| LR stack, 233 public + own members (`0003`) | 0.96219 | 0.96185 | 0.96237 |
| + blend with public files (`0009`, best private of any submission) | – | 0.96188 | **0.96238** |
| LR stack, 319 members on Kaggle CPU (**`0020`, final**) | 0.96225 | 0.96185 | **0.96236** |
| + own set transformers / TabPFN / GReaT LLR (`0027`–`0061`) | 0.96225 (+0) | 0.96185 | 0.96236 |
| Stack + public blends, weights picked on public (`0057`, final) | – | 0.96214 | 0.96222 |
| Tilt toward the public labels (`0078`, not a final) | – | **0.96473** | 0.96027 |

Stacking public libraries got to the crowd within a few days. After that, nothing honest moved the score, and every
step that moved public up moved private down.

## What I learned

- **The public leaderboard was anti-correlated with private.** Over the 26 non-fitted submissions, Spearman
  public→private was −0.59 (95% CI −0.79 to −0.20). Picking finals by best CV gave 0.96236; picking by best public gave 0.96222.
- **Fitting the public LB has a precise cost.** It took public from 0.96214 to 0.96473 (+0.0026) and private from
  0.96222 to 0.96027 (−0.0020). Had `0078` been a final, the rank would have been about 1,981 instead of 426.
- **In a crowd of identical stacks, only new information counts.** Every top-10 write-up I read added members the
  public libraries lacked. The 1st, 4th and 7th places trained a class-conditional generative model of the GReaT generator on
  the *labelled* rows and scored it with Bayes' rule (+4e-4 to +6e-4). My generator replica was trained label-free and
  added +0, one step short of the winning idea.
- **A flat baseline is the trigger for a new model family, not for leaderboard work.** The honest CV did not move after
  Oct 24, and the last six days went into public fitting.

Full analysis: [`reports/postmortem.md`](reports/postmortem.md). Every submission: [`docs/results.md`](docs/results.md).

## Repo guide

| Path | What |
|---|---|
| [`reports/postmortem.md`](reports/postmortem.md) | Start here: result, what predicted private, what winners did |
| [`docs/results.md`](docs/results.md) | Every submission (CV / public / private) and every own model |
| [`src/pool.py`](src/pool.py), [`src/stack.py`](src/stack.py) | The honest pipeline: build the OOF pool, nested-CV LR stack |
| `src/train_*.py`, `kernels/` | Own members and the Kaggle GPU/CPU kernels that trained them |
| `src/lb_*.py`, `src/tilt*.py`, `src/probe_*.py` | Public-LB fitting study (display only) |
| [`reports/final-selection.md`](reports/final-selection.md) | How the two finals were chosen, and their outcome |
| `.kaggle-gm/ledger.jsonl` | All 81 experiments with CV, fold scores, tags and decisions |
| `src/kgkit/` | Vendored [kaggle-grandmaster](https://github.com/TranBaDat2607/claude-code-kaggle-grandmaster) toolkit (ledger, comparisons, Kaggle GPU runner) |

## Reproduce

- Hardware: a 12-thread CPU with 15 GB RAM locally. Kaggle 2xT4 kernels for the neural and TabPFN members (about 14 GPU-h
  in total). A Kaggle CPU kernel for the 320-member stack.
- Environment: Python 3.12 with pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, torch, matplotlib, plus the
  `kaggle` CLI.
- Data: accept the competition rules, then `kaggle competitions download -c playground-series-s6e10 -p data`. The
  competition data and the downloaded OOF libraries are **not** in this repository.
- The honest stack:
  ```bash
  bash src/pull_datasets.sh <library slugs...>   # public OOF libraries -> data/ext/
  python src/pool.py                             # -> artifacts/pool/{oof,test}.npy + members.csv
  python src/stack.py --C 0.03 --name stack_lr --log
  # or on Kaggle CPU: python src/build_stack_kernel.py && kaggle kernels push -p kernels/stack-cpu
  ```
- The README figure: `python src/plot_showcase.py`.

## Credit

The stack stands on public OOF/test libraries and notebooks shared during the competition. Thank you to their authors:

- cdeotte: [s6e10-124-public-oof-and-test-predictions](https://www.kaggle.com/datasets/cdeotte/s6e10-124-public-oof-and-test-predictions), [s6e10-tfm-oof-and-test-predictions](https://www.kaggle.com/datasets/cdeotte/s6e10-tfm-oof-and-test-predictions)
- goodpjw2008: [s6e10-goodpjw2008-extra-members-oof](https://www.kaggle.com/datasets/goodpjw2008/s6e10-goodpjw2008-extra-members-oof), [s6e10-tabpfn-member](https://www.kaggle.com/datasets/goodpjw2008/s6e10-tabpfn-member)
- dariushafshar: [s6e10-golem-oof-library](https://www.kaggle.com/datasets/dariushafshar/s6e10-golem-oof-library)
- abdullahsafwan333, amanatar, dmitriykolmikov, ewtherka, hemingweb, hermengardo, kagankoral, kiza123123, koumeimaki,
  kratosyan, megayak, mitudru, mojjeed, najiama, sachith7, sadamtorres, samanyu1808, thethiago28, thisray,
  vigneshalugunulla, wangxintong111, yekenot: OOF/test pairs and notebook outputs. The exact list for the final stack
  is in `dataset_sources` / `kernel_sources` of
  [`kernels/s6e10-honest-stack-v2/kernel-metadata.json`](kernels/s6e10-honest-stack-v2/kernel-metadata.json).
- nina2025 ([ps-s6e10-04](https://www.kaggle.com/code/nina2025/ps-s6e10-04)) and mehrankazeminia
  ([1-airline-s6e10-tabpfn](https://www.kaggle.com/code/mehrankazeminia/1-airline-s6e10-tabpfn)): public submission
  files in the `0057` blend.
- Original dataset: [teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
- Winners' write-ups that shaped the post-mortem: [1st](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754683),
  [4th](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754814),
  [7th](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754646).

## License

My code: MIT ([LICENSE](LICENSE)). Competition data: governed by the competition's rules and not redistributed here.
Third-party OOF libraries, notebooks and datasets keep their authors' licenses (see each Kaggle page). They are
referenced, not included.
