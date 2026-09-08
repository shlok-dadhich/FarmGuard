# Evaluation

## Metrics

`src/evaluation/metrics.py` (scikit-learn based):

- accuracy
- macro precision / macro recall / **macro F1 (primary)**
- weighted F1
- per-class precision / recall / F1
- confidence statistics (mean confidence, fraction below 0.5)

`src/evaluation/confusion_matrix.py` returns raw counts; the UI and report generator
render both raw counts and row-normalized percentages.

## Dataset validation

`scripts/prepare_data.py` and the Model Evaluation page verify, before evaluating:

- images exist (missing files counted)
- class labels exist and match the manifest
- no invalid classes
- corrupt images detected (sample-verified with PIL `verify()`)
- leakage between splits reported

The pre-evaluation summary shows: number of samples, number of classes, class
distribution.

## Evaluation entry points

| Entry point | Scope |
|---|---|
| `scripts/evaluate.py --crop X --model Y --checkpoint P` | one model, one split |
| `scripts/evaluate_all.py --crop X --split test` | all models for a crop |
| `scripts/evaluate_all.py --all --split test` | all models × all crops |
| `scripts/run_experiments.py --crop X` | full pipeline (prepare → evaluate → compare) |
| UI **Model Evaluation** page | interactive, with downloads |

Batch evaluation (`evaluate_all.py`) is resilient: a broken model is recorded with
its failure reason and evaluation continues; the summary JSON/CSV under
`outputs/reports/` shows status per (crop, model).

## Per-model artifacts

For every evaluated model:

```
outputs/metrics/eval_<crop>_<arch>.json        # metrics + CM + efficiency + latency
outputs/confusion_matrices/cm_<crop>_<arch>_<split>.{png,json}
outputs/metrics/runs.csv                        # appended experiment run (kind=eval)
```

## Efficiency

`src/evaluation/efficiency.py` computes per model: accuracy per M parameters,
macro F1 per M parameters, macro F1 per FLOP, and performance-vs-latency.
`scripts/generate_comparison.py` turns these into scatter plots
(parameters/FLOPs/latency vs macro F1) that answer:

- "Which model performs best?" → best macro F1.
- "Which model gives the best performance for the cost?" → best efficiency score.

## Report

`python scripts/generate_comparison.py --crop X` produces
`outputs/reports/model_comparison_<crop>.{csv,json,md,html}` plus
`outputs/figures/<crop>/*.png` — see `docs/experiment_tracking.md` and the README
for the section-by-section description.