# Exp17 saved-prediction analysis

Keep Exp15 as the deployed solution. Exp17's reader generates useful alternatives, but its score margin does not reliably identify a better annotated span. No retraining or model calls were performed for this analysis.

## What failed

Across 195 annotated positives, the reader accepted 54 choices: 25 improved tIoU, 23 worsened it, and six tied. Of those choices:

| Replacement group | Count | Improved | Regressed | Sum of tIoU changes |
|---|---:|---:|---:|---:|
| No overlap with Exp15 | 20 | 6 | 13 | −6.92346 |
| Some overlap with Exp15 | 34 | 19 | 10 | +1.50546 |

The net loss is therefore primarily occurrence switching, not a blanket inability to refine boundaries. Some nonoverlapping switches helped, so forbidding all of them is a tradeoff, not a universal correctness rule.

Examples from the saved predictions:

- `sample_6_yes_q02`: sinus symptoms. Exp15 tIoU 0.991; reader 0. The reader moved from 9.05–12.30 s to 32.77–35.08 s, selecting “It points to an acute sinus infection.” Its replacement margin was still 9.118.
- `sample_19_yes_q03`: Airomir among renewed medicines. Exp15 tIoU 0.860; reader 0. The reader moved from 68.70–71.66 s to 19.68–21.08 s, selecting “Activel and Aromir,” with margin 6.336.
- `sample_23_yes_q05`: informed about scarring and infection. Exp15 tIoU 0.913; reader 0. The reader selected the following phrase “you should know both before we book it,” instead of overlapping the reference-bearing passage.

Large QA logit margins do not establish correct occurrence, complete evidence, or high tIoU. Raising one confidence threshold cannot reliably fix this.

## CPU-only replay results

Every score below includes all 390 questions and all 195 annotated positives, with classification frozen. Candidate timestamps are already calibrated; calibration was not applied twice.

| Selection policy | tIoU | Score |
|---|---:|---:|
| Keep Exp15 | 0.634262 | 0.776455 |
| Original Exp17 | 0.606478 | 0.759784 |
| Accept original top candidate only if it overlaps Exp15 | 0.641982 | 0.781087 |
| Highest-scoring qualifying **overlapping candidate among all five**, otherwise Exp15 | **0.649671** | **0.785700** |
| Same pool, change start only | 0.643764 | 0.782156 |
| Same pool, change end only | 0.640032 | 0.779917 |
| Same pool, accept only a contained span | 0.634761 | 0.776754 |

The best tested rule filters all five candidates before choosing one. It retains the original minimum null margin of 2 and minimum baseline-score advantage of 2. It improves 22 positive spans and worsens 10; the other 163 tie.

This is a **post-hoc development result**, not a new validation score. The paired-conversation bootstrap interval for its score improvement is **[−0.004148, +0.022984]**. Its per-reader-fold tIoU deltas are −0.03356, +0.03493, +0.01660, +0.03962, +0.01586. Four folds improve, but the first regresses.

I also tested a small family of overlap thresholds, score margins and boundary blends. Choosing a rule on the other four reader folds and applying it to the held-out fold scored **0.770788**, below Exp15. This check does not constitute fully nested retraining: reader models and the prior demonstration bank introduce dependencies. Nevertheless, it provides no support for another threshold-tuning cycle.

The provided script exposes all 23 alternative/keep rules, plus original Exp17, rather than hiding unsuccessful variants.

## Remaining headroom

The gold-assisted oracle over reader top-five candidates plus Exp15 is 0.770358 tIoU. Restricting candidates to ones overlapping Exp15 still leaves an oracle of **0.752325 tIoU**, equivalent to score 0.847292 at the current accuracy.

Those are ceilings obtained using the reference spans to choose candidates, not achievable scores demonstrated by a deployable ranker. They show that useful boundary alternatives exist. They do not establish that this small development set can teach a reliable ranker.

A local score of 0.800 needs tIoU 0.673504 at current accuracy. The best replay here reaches only 0.649671. Validation 0.800 is not supported by these results.

## Recommendation

1. Keep Exp15 for official validation. Do not force the Exp17 final-model gate open on the basis of this replay.
2. Preserve these reader candidates; another ASR or Qwen replay is unnecessary for their analysis.
3. If another architecture experiment is worth the time, focus on **boundary selection within the occurrence already located by Exp15**. Train on contextual windows around that occurrence and supervise alternative boundary quality. Keep the Exp15 span as a real candidate. This is a proposed next experiment, not a measured improvement. It will not by itself solve Exp15's 19 wrong-occurrence cases.

The current confidence-only gate is unsuitable for unrestricted occurrence replacement. A separately evaluated occurrence selector would be needed to address those cases without losing strong predictions.

## Reproduce the audit

Extract `medical-exp17-selection-audit.zip`. From your `medical-appointment` directory, point to the extracted script:

```bash
python /path/to/audit_exp17_selection.py \
  --trace reports/exp17-oof/trace.jsonl \
  --csv data/question_train.csv \
  --out reports/exp17-selection-audit
```

It uses Python and NumPy only. No GPU, servers, model checkpoints, training or code installation is required. Use a new output directory when repeating. The script does not modify the inference pipeline or `api.py`.

It writes `summary.json`, `policies.csv`, and `pool_changes.csv`. The latter identifies the exact improved/regressed questions and selected timestamps. The ZIP also contains the results computed from your uploaded files.

This audit does not create the missing Exp17 deployment checkpoint. Your failed promotion gate remains the reason `--experiment exp17` cannot currently start.
