# Post-mortem: playground-series-s6e10 (Predicting Airline Satisfaction)

Written 2026-11-06 with kaggle-grandmaster 1.20.0 (`kgkit ledger sync-lb`, `kgkit final`, `kgkit ledger predictiveness`,
`kgkit discussions solutions`). Every score below comes from Kaggle's submission list or private board, read on 2026-11-06.

## 1. Result

| | |
|---|---|
| Private | **0.96236, rank 426 / 3,735** (top 11.4%), no medal |
| Public (at close) | 0.96473, rank 1 (from a public-LB-fitted file that was **not** a final) |
| Finals | `0057_lbb-settf16-10` (public 0.96214 / private 0.96222), `0020_stack-lr-pool319-kaggle` (0.96185 / **0.96236**) |
| Best private of any submission | 0.96238 (`0009_blend-s246-n88t-den`, 2026-10-17), 2e-5 above the finals |
| Cut-offs (computed from Kaggle's progression table) | gold 0.96244 (rank 17), silver 0.96238 (186), bronze 0.96237 (373); winner 0.96318 |

**We did not lose 400 places to overfitting in the finals.** The finals were the honest picks, and the guard against
LB-fitted finals worked. The public 1st place was never a real position. Our honest work sat at about rank 400 the
whole time:

| What | n | public | private | private rank if it had been the final |
|---|---|---|---|---|
| Tilts toward the public labels (best: 0078) | 24 | 0.96266 → **0.96473** | 0.96177 → **0.96027** | 1,981 |
| Blends with weights picked by public probes (0057) | 13 | 0.96209–0.96214 | 0.96220–0.96223 | ~854 |
| Honest LR stack (0020 / 0060) | 3 | 0.96185 | 0.96236 | 390–437 (tie) |
| Early public-OOF blend (0009) | 4 | 0.96188 | 0.96238 | 75–224 (tie) |

The board was a crowd: **315 teams scored 0.96237–0.96238 (ranks 75–389)**. Those are the public-OOF stacks, all
separated by 1–2e-5, which is at the noise floor (paired SE ~4.6e-6, gains under ~9e-6 are noise). The bronze and
silver cut-offs fell inside this crowd, so getting a medal from inside it was a lottery. Getting clearly above it
(≥0.96240, top 36) required members that the public libraries did not have.

## 2. What predicted private (`kgkit ledger predictiveness`)

| set | n | public → private (Spearman, 95% CI) | CV → private |
|---|---|---|---|
| every honest scored submission | 26 | **−0.59 [−0.79, −0.20]** | – |
| records with their own CV | 3 | – | −0.87 (n too small, private ties) |
| public-LB fitted (left out by the tool) | 23 | public +0.0026, private −0.0021 vs honest | – |

- Picking finals by best CV: 0.96236. By best (non-fitted) public: 0.96222. Ours: 0.96236. Best possible: 0.96238.
- Public − private on honest files: mean −0.0002 (the public 20% slice was harder). On tilted files: **+0.0029 to +0.0045**.
- Public was **anti-correlated** with private even among honest submissions: every move tuned on public (the lbb weight
  probes, nina/Mehran mixes) lost 1–1.5e-5 on private against the plain stack. Winners report the same thing (16th:
  a public 0.96213 file gave +190u public and −50u private; 55th: −2u; 7th: their best-public file scored 0.96223).

Figures: `docs/figures/public_vs_private.png`, `private_over_time.png`, `cv_vs_private.png`, and `public_vs_private_all.png` (all 49 submissions, including the fitted ones).

## 3. What worked

- Frozen SKF 5/42 folds plus an L2 logistic-regression stack on logits over ~320 public OOF/test pairs. It is the same
  backbone the 1st, 11th, 29th and 55th places used, and it reached the crowd in 6 days (0.96219 nested CV by Oct 17).
- Family-specific penalties (the 55th did the same with TabPFN x3).
- The finals guard: tilts were tagged `public-lb-fitted`, never accepted as a baseline and never selected. Without it,
  Kaggle's auto-selection would have finished at about rank 1,981.

## 4. What did not work

- **Public-LB fitting (Oct 27–30, 24 of 49 submissions, about 4 sessions).** Public went from 0.96214 to 0.96473 (rank 1).
  It produced a lot of tooling (EB/Student-t noise, version mining, exponential posterior) and **zero private value**.
  Every tilt scored 0.9600–0.9618 private.
- **LB-probed blend weights (Oct 23–24, 13 submissions)** gave 0 on private, and one of them became a final (0057, the
  weaker final).
- **Own members** (TabPFN aux, set transformer, TabICL, label-free GReaT LLR) gave +0 to a 320-member stack. The honest
  baseline did not move after Oct 24 (0.962249 nested), and no new family was tried after Oct 27.
- Kaggle GPU: about 14 h used over two weekly quotas (about 60 h available).

## 5. What the winners did that we did not (`solution-researcher`, 10 write-ups)

1. **A class-conditional generative model scored with Bayes' rule (LLR)**, trained on the competition rows with the
   label in the sequence, with random column orders and order-averaged scoring. The 1st, 4th and 7th places used it
   three different ways: a fine-tuned GPT-2 replica of the GReaT generator with **target-first** fine-tuning
   (+4.9e-4 CV), an AO-ARM transformer from scratch (0.96258 private alone), and GenPure (16 seeds). It was the
   largest lever reported, +3.7e-4 to +6e-4 private.
   - **Our near-miss:** we trained a distilgpt2 GReaT replica on the original data (0059/0061) but used it
     *label-free*, which gave +0. The winning step was to continue training on the labelled train fold with the
     target first, then score `log P(x|T) − log P(x|F) + prior`. The infrastructure was already in place.
2. **Self-supervised pre-training (RTD/MLM) on train+test, then fine-tuning** (9th; the 11th's Q-FT gave +2e-4 private;
   it fits 2xT4 at about 1 h 10 min per task).
3. **Auditing members that early-stop on the scored fold** (29th: removed +2e-4 of fake OOF; the 1st warns about it in
   the public pools) and re-tuning C as the pool grows.
4. **Picking finals by CV.** The big climbs (33rd up 960 places, 7th from 318th, 16th from 515th) all came from this.

## 6. Validation lessons

- On a synthetic Playground dataset with a 20% public slice, the public score of near-identical files is noise or worse:
  here it was anti-correlated with private (−0.59). Nested CV over 700k rows was the right ruler (55th: 0.996).
- An LB-fitted file's public score measures how well it fitted the public labels, not the model. Its private score
  can only fall.
- When the board is a crowd at the noise floor, the rank inside the crowd is luck. Only a gain several times the paired
  SE above the crowd changes the outcome.

## 7. Process lessons

- The sessions' `/goal` was a **public rank** ("public top 10 / top 5 / top 1"). That goal steered 6 of the 14 days
  into public probing. The goal should have been "nested CV +X above the public-OOF crowd".
- The honest plateau (Oct 24 onwards) was the signal to start a new model family, not to switch to LB work. The
  generative-LLR direction was already half-built.
- GPU quota was left unused while the stack was saturated. The target-first GPT needs about 10 A100-h per pair (about
  30–50 T4-h), which was within reach over two weeks of quota if started by Oct 24.

## 8. Five reusable takeaways

See `~/.kaggle-gm/lessons.md` (section `playground-series-s6e10`).

## 9. Plugin issues filed

TranBaDat2607/claude-code-kaggle-grandmaster #85 (public-rank goals and LB-fitting), #86 (public-tuned blends treated as honest), #87 (post-mortem hides what fitting cost), #88 (plateau and crowd detector), #89 (model-the-generator playbook).
