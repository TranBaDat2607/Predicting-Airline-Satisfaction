# Full results: playground-series-s6e10

Every scored submission, read from Kaggle on 2026-11-06 (`kgkit ledger sync-lb`). Finals are in **bold**. CV is nested 5-fold ROC AUC on the frozen `StratifiedKFold(5, shuffle=True, random_state=42)` split, where the submission has its own OOF.

## Submissions (49)

| id | type | CV | public | private | public − private |
|---|---|---:|---:|---:|---:|
| 0003_stack-lr-pool233 | honest | 0.962194 | 0.96185 | 0.96237 | -0.00052 |
| 0005_blend-s246-n88t-n83 | blend with public files | 0.962200 | 0.96189 | 0.96237 | -0.00048 |
| 0008_blend-s246-n88t-n83-n60 | blend with public files | 0.962200 | 0.96189 | 0.96237 | -0.00048 |
| 0009_blend-s246-n88t-den | blend with public files | 0.962200 | 0.96188 | 0.96238 | -0.00050 |
| 0012_lbb-pfnmean10 | blend with public files | – | 0.96210 | 0.96223 | -0.00013 |
| 0013_lbb-pfnmean20 | blend with public files | – | 0.96209 | 0.96223 | -0.00014 |
| 0014_lbb-novel6-w10 | blend with public files | – | 0.96210 | 0.96222 | -0.00012 |
| 0016_lbb-m15 | blend with public files | – | 0.96212 | 0.96221 | -0.00009 |
| 0017_lbb-s10 | blend with public files | – | 0.96212 | 0.96223 | -0.00011 |
| 0018_lbb-mehran-gbdt5 | blend with public files | – | 0.96212 | 0.96220 | -0.00008 |
| **0020_stack-lr-pool319-kaggle** | honest | 0.962249 | 0.96185 | 0.96236 | -0.00051 |
| 0022_lbb-taux5 | blend with public files | – | 0.96212 | 0.96221 | -0.00009 |
| 0023_lbb-taux10 | blend with public files | – | 0.96211 | 0.96220 | -0.00009 |
| 0024_lbb-m07 | blend with public files | – | 0.96212 | 0.96223 | -0.00011 |
| 0028_lbb-m5-m2q5 | blend with public files | – | 0.96212 | 0.96222 | -0.00010 |
| 0029_lbb-m2q10 | blend with public files | – | 0.96211 | 0.96222 | -0.00011 |
| 0030_lbb-m10-m2q5 | blend with public files | – | 0.96211 | 0.96221 | -0.00010 |
| 0031_lbb-m75-m2q75 | blend with public files | – | 0.96211 | 0.96221 | -0.00010 |
| 0032_lbb-settf5 | blend with public files | – | 0.96213 | 0.96222 | -0.00009 |
| 0033_lbb-settf10 | blend with public files | – | 0.96213 | 0.96221 | -0.00008 |
| 0034_lbb-settf15 | blend with public files | – | 0.96213 | 0.96221 | -0.00008 |
| 0035_lbb-settf20 | blend with public files | – | 0.96212 | 0.96220 | -0.00008 |
| 0044_lbb-settf8-10 | blend with public files | – | 0.96214 | 0.96222 | -0.00008 |
| 0045_lbb-settf8-125 | blend with public files | – | 0.96214 | 0.96222 | -0.00008 |
| 0054_tilt-v1-eta0-3 | tilt (public-LB fitted) | – | 0.96266 | 0.96177 | +0.00089 |
| 0056_tilt-v2-eta0-6 | tilt (public-LB fitted) | – | 0.96285 | 0.96183 | +0.00102 |
| **0057_lbb-settf16-10** | blend with public files | – | 0.96214 | 0.96222 | -0.00008 |
| 0058_tilt-v3-eta1 | tilt (public-LB fitted) | – | 0.96297 | 0.96148 | +0.00149 |
| 0060_stack-lr-pool321-kaggle-v3 | honest | 0.962248 | 0.96185 | 0.96236 | -0.00051 |
| 0062_tilt-v4-t2-eta1 | tilt (public-LB fitted) | – | 0.96361 | 0.96073 | +0.00288 |
| 0063_tilt-v5-t2-vfix13-eta1 | tilt (public-LB fitted) | – | 0.96365 | 0.96083 | +0.00282 |
| 0064_tilt-v6-t2-vfix45-eta1 | tilt (public-LB fitted) | – | 0.96378 | 0.96063 | +0.00315 |
| 0065_probe-top10k-x2 | tilt (public-LB fitted) | – | 0.96373 | 0.96046 | +0.00327 |
| 0066_tilt-v7-t2k80-eta1 | tilt (public-LB fitted) | – | 0.96369 | 0.96071 | +0.00298 |
| 0067_tilt-v8-vfix90-eta1 | tilt (public-LB fitted) | – | 0.96414 | 0.96050 | +0.00364 |
| 0068_tilt-v9-reml-eta1 | tilt (public-LB fitted) | – | 0.96432 | 0.96025 | +0.00407 |
| 0069_tilt-v10-clean-eta1 | tilt (public-LB fitted) | – | 0.96391 | 0.95999 | +0.00392 |
| 0070_tilt-v11-t1-eta1 | tilt (public-LB fitted) | – | 0.96428 | 0.96025 | +0.00403 |
| 0071_tilt-v12-bag-eta1 | tilt (public-LB fitted) | – | 0.96423 | 0.96023 | +0.00400 |
| 0072_tilt-v13-exp-eta1 | tilt (public-LB fitted) | – | 0.96452 | 0.96028 | +0.00424 |
| 0073_tilt-v14-exp-eta1 | tilt (public-LB fitted) | – | 0.96462 | 0.96022 | +0.00440 |
| 0074_probe-n1 | tilt (public-LB fitted) | – | 0.96461 | 0.96019 | +0.00442 |
| 0075_probe-n2 | tilt (public-LB fitted) | – | 0.96453 | 0.96014 | +0.00439 |
| 0076_probe-n3 | tilt (public-LB fitted) | – | 0.96457 | 0.96020 | +0.00437 |
| 0077_tilt-v15-exp-eta1 | tilt (public-LB fitted) | – | 0.96451 | 0.96025 | +0.00426 |
| 0078_tilt-v16-exp-t1k18-eta1 | tilt (public-LB fitted) | – | 0.96473 | 0.96027 | +0.00446 |
| 0079_probe-n11 | tilt (public-LB fitted) | – | 0.96467 | 0.96018 | +0.00449 |
| 0080_probe-n12 | tilt (public-LB fitted) | – | 0.96467 | 0.96021 | +0.00446 |
| 0081_probe-n13 | tilt (public-LB fitted) | – | 0.96467 | 0.96021 | +0.00446 |

## Own models (OOF ROC AUC, not submitted alone)

| id | model | CV | notes |
|---|---|---:|---|
| 0001_xgb-raw | xgb | 0.958829 | XGBoost GPU, raw features, template defaults |
| 0010_tabicl-aux-c150k | tabicl | 0.956972 | TabICL v2, raw + aux_ev, 150k stratified context per fold |
| 0021_tabpfn35-auxcatfd-c400k | external | 0.959932 | own Kaggle GPU kernel tranbadat/s6e10-tabpfn35-aux-catfd: TabPFN-3.5, raw + Flight Distance categorical + 13 a |
| 0025_settf-d512 | settf | 0.960581 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0026_settf-d256 | settf | 0.960612 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0036_tabpfn35-mehranview-5f | external | 0.960619 | own Kaggle GPU kernel: TabPFN-3.5 on Mehran's view (47 in-fold TE combos + KMeans group), 200k ctx, 2 seeds |
| 0037_tabpfn35fast-catfd10-5f | external | 0.960289 | own Kaggle GPU kernel: TabPFN-3.5 FAST checkpoint, raw + catFD + catFD10, full fold context, 2 seeds |
| 0038_settf-d512s1 | settf | 0.960566 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0039_settf-d768 | settf | 0.960364 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0040_settf-d384 | settf | 0.960706 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0041_settf-d512s4 | settf | 0.960744 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0042_settf-d512s5 | settf | 0.960676 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0043_settf-d768s6 | settf | 0.960479 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0046_settf-d384s7 | settf | 0.960503 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0047_settf-d384s8 | settf | 0.960475 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0048_settf-d256s9 | settf | 0.960454 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0049_settf-d256s10 | settf | 0.960662 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0050_settf-d512s11 | settf | 0.960573 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0051_settf-d512s12 | settf | 0.960630 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0052_settf-d384s13 | settf | 0.960656 | set transformer: value-embedding tokens + side features, RTD pretraining |
| 0053_settf-d256s14 | settf | 0.960707 | set transformer: value-embedding tokens + side features, RTD pretraining |

## Stacks (nested CV)

| id | CV | notes |
|---|---:|---|
| 0002_stack-lr-pool161 | 0.962139 | L2 LR on standardised logits, 160 public members + xgb_raw |
| 0003_stack-lr-pool233 | 0.962194 | L2 LR on standardised logits, 233 public+own members (13 libraries + kernel outputs) |
| 0004_stack-lr-pool246 | 0.962200 | L2 LR C=0.03, 246 members (+kernel outputs) |
| 0006_stack-lgb246 | 0.961915 |  |
| 0007_stack-lr-family246 | 0.962206 | family-wise L2 scales tuned on fold 0, 246 members |
| 0011_stack-lr-pool247-tabicl | 0.962201 | LR C=0.03, 246 + own TabICL aux c150k |
| 0019_stack-lr-pool300-kaggle | 0.962200 | Kaggle CPU kernel tranbadat/s6e10-honest-stack v1: LR C=0.01 on 300 deduped public members (30 sources), nested CV on fr |
| 0020_stack-lr-pool319-kaggle | 0.962249 | Kaggle CPU kernel tranbadat/s6e10-honest-stack-v2: LR C=0.03 on 319 members (v1 + mojjeed 15-fold GBDTs, ewtherka FM, ki |
| 0027_stack-lr-pool321-settf | 0.962246 | LR C=0.03 on stack-v2 pool (319) + own set transformers d512/d256 (Kaggle GPU, OOF 0.96058/0.96061) |
| 0059_stack-lr-pool321-llr | 0.962248 | LR C=0.03 on stack-v2 pool (319) + generator-surrogate LLR (distilgpt2 GReaT-style on the original, label-free AUC 0.951 |
| 0060_stack-lr-pool321-kaggle-v3 | 0.962248 | Kaggle CPU kernel tranbadat/s6e10-honest-stack-v3: LR C=0.02 on 321 members (v2 pool + settf16 + 6 members of vanished p |
| 0061_stack-lr-pool321-llrmix | 0.962249 | LLR recipe mix (Kaggle GPU s6e10-llr + s6e10-llr-v2, label-free AUC 0.9516/0.9504): nested 0.962249 (-0.02u, 1/5 up); v2 |
