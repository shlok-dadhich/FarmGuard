# Architecture

AgroVision is a modular, config-driven platform. All experiment settings live in
`configs/*.yaml`; the only code a user writes is for *new architectures* (adapters),
never for adding a checkpoint.

## Module map

| Module | Responsibility |
|---|---|
| `src/core` | configuration loading + validation, device/seed resolution, WHAT/WHY/HOW exceptions, logging |
| `src/models` | `interface.py` (ModelAdapter contract), `registry.py` (arch key → adapter path), `factory.py` (build + checkpoint loading), `metadata.py` (model *instances* from YAML), `adapters/` (torchvision/timm backends), `mock.py` (demo) |
| `src/data` | ImageFolder scan → stratified split → CSV manifests → datasets → loaders; validation + dataset reports |
| `src/evaluation` | metrics (macro F1 primary), confusion matrix, efficiency, evaluator loop, figure helpers, reports |
| `src/explainability` | Grad-CAM++, Score-CAM, Saliency, target-layer resolution, overlay/heatmap, pointing-game faithfulness, `xai_pipeline` |
| `src/inference` | predictor, preprocessing, model loader, ensemble voting + agreement |
| `src/tracking` | run schema (file = source of truth) + tracker (optional MLflow/TensorBoard) |
| `src/training` | trainer used only by `scripts/train.py` (demo training; model training is NOT the platform's job) |
| `src/services` | thin wrappers combining evaluation/prediction/ensemble/explanation |
| `src/ai` | optional AI text provider with offline fallback (assistive only) |
| `src/utils` | image helpers, filesystem, environment/reproducibility report |
| `ui/` | Streamlit pages + shared cached helpers (`ui/common.py`) + components |
| `scripts/` | CLI entry points (see README) |

## Data flow

```
configs/*.yaml
   │  load_config()  (src/core/config.py)
   ▼
src/models/metadata.py ── instances: id, architecture, crop, checkpoint, classes_file
   ▼
UI / scripts resolve (arch, checkpoint, input_size, class_names) via ui/common.model_info()
   ▼
inference: preprocess_pil -> Predictor  |  evaluation: loaders_from_splits -> evaluate_model
   ▼
metrics + confusion matrix + efficiency   ->   outputs/metrics, outputs/confusion_matrices
   ▼
tracking: append RunRecord (runs.csv + run_<id>.json)
   ▼
comparison: scripts/generate_comparison.py -> outputs/reports + outputs/figures/<crop>/
```

## Configuration layers

- `models.yaml` — architecture registry (`models:`) + optional instances (`instances:`)
- `datasets.yaml` — crops, roots, class lists, split ratios, preprocessing
- `project.yaml` — seed, device, output dir, demo mode
- `training.yaml` — trainer settings (only used by `scripts/train.py`)
- `explainability.yaml` — XAI method settings, overlay, faithfulness mask pattern
- `tracking.yaml` — experiment-tracking backend
- `ai.yaml` — optional AI text provider

## Resilience rules

- One broken model never stops a batch run (`evaluate_all.py` records failures).
- Missing data → UNAVAILABLE RESULT, never fabricated.
- `runs.csv` is rebuilt from authoritative per-run JSONs when schema drift or
  corruption is detected.
- All UI pages degrade gracefully: unsupported XAI methods, absent checkpoints and
  missing manifests show explanations, not tracebacks.