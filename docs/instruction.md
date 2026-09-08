# AgroVision Development Instructions

*Main development and maintenance instruction document.*
Read this before modifying the repository. Companion docs: `required.md`
(what the user must supply), `HOW_TO_RUN.md` (operational guide),
`documentation_validation.md` (what was verified), plus `architecture.md`,
`model_integration.md`, `evaluation.md`, `experiment_tracking.md`, `xai.md`.

---

## 1. Project Purpose

AgroVision is a **research-oriented multi-crop plant disease classification
evaluation and explainability platform**. It does **not** train models — models are
trained externally and supplied as checkpoints. The platform provides:

- multiple crops (configurable per crop in `configs/datasets.yaml`)
- multiple trained classification models (registered in `configs/models.yaml`)
- single-image inference (Streamlit Diagnosis page + `src/inference/`)
- dataset evaluation with scikit-learn metrics (macro F1 primary)
- experiment tracking (file backend mandatory, MLflow/TensorBoard optional)
- model comparison with sorting, best-model verdicts and exports
- confusion matrices (raw + normalized) and classification reports
- metric visualization and efficiency analysis (params/latency/FLOPs where available)
- ensemble evaluation (majority / soft / weighted vote, agreement, cost-benefit)
- Grad-CAM++, Score-CAM, Saliency explainability
- optional pointing-game faithfulness when lesion masks are supplied
- optional AI-generated text explanation (assistive only, never the classifier)
- Streamlit frontend (`app.py` + `ui/pages/`)

## 2. Architectural Principles

1. **Models are externally supplied.** The platform never trains, rewrites, or
   silently substitutes models. `models/` is user-owned.
2. **Model internals are isolated behind adapters.** Everything (inference,
   evaluation, XAI, UI) talks to a `ModelAdapter` (`src/models/interface.py`);
   nothing imports a specific model's internals.
3. **Evaluation must be reproducible.** Deterministic stratified splits
   (`data/splits/<crop>/*.csv`), deterministic val/test preprocessing, seeds and
   environment captured per run.
4. **Experiments never overwrite one another.** Every run gets a unique `run_id`
   and is appended (`outputs/metrics/runs.csv` + `run_<id>.json`); per-run JSONs
   are authoritative and the CSV is rebuilt from them on schema drift.
5. **Metrics are stored persistently** in the run records and per-eval JSONs.
6. **XAI is model-aware.** Target layers come from each adapter's
   `target_layers()`; unsupported methods degrade gracefully.
7. **Streamlit is the primary frontend**; CLI scripts cover the same workflows.
8. **CSV/JSON experiment tracking is mandatory**; MLflow is optional.
9. **scikit-learn is used for standard metrics.**
10. **Research results and demo results never mix.** Demo output is labelled
    `status=demo` / DEMO banner and is never used as a research result.

## 3. Source of Truth

| Concern | Source |
|---|---|
| Model architecture definitions | `configs/models.yaml -> models:` |
| Model instances (checkpoint ↔ crop ↔ classes) | `configs/models.yaml -> instances:` |
| Dataset/crop configuration | `configs/datasets.yaml -> crops:` |
| Class mapping | manifest labels (`data/splits/<crop>/*.csv`) → `classes_file` → `datasets.yaml classes` |
| Experiment tracking | file store `outputs/metrics/` (`configs/tracking.yaml -> file_store`) |
| XAI configuration | `configs/explainability.yaml` |
| Application configuration | `configs/project.yaml` |
| Training settings (demo `train.py` only) | `configs/training.yaml` |
| AI text provider | `configs/ai.yaml` + `.env` |

## 4. Folder Ownership

**USER MANAGED (the user edits these):**

| Path | Content |
|---|---|
| `models/checkpoints/` | trained weight files (`<crop>_<arch>.pt` or per-instance paths) |
| `models/custom/` | custom model Python code (only for non-built-in architectures) |
| `data/raw/<crop>/` | ImageFolder datasets: `<crop>/<class>/images...` |
| `data/raw/<crop>/masks/` | optional lesion masks for faithfulness |
| `configs/models.yaml` | register models / instances |
| `configs/datasets.yaml` | crops, roots, class lists |
| `configs/classes/` | per-crop class-mapping YAML files |
| `.env` | optional AI keys |

**SYSTEM MANAGED (generated — do not hand-edit):**

| Path | Content |
|---|---|
| `data/splits/<crop>/` | train/val/test CSV manifests (from `prepare_data.py`) |
| `data/metadata/` | dataset reports |
| `outputs/metrics/` | experiment history (`runs.csv`, `run_<id>.json`, `eval_*.json`) |
| `outputs/confusion_matrices/` | CM PNG + JSON |
| `outputs/figures/<crop>/` | comparison plots |
| `outputs/reports/` | comparison reports + pipeline summaries |
| `outputs/xai/` | explanation images + meta JSON |
| `outputs/logs/`, `outputs/checkpoints/` | logs; demo-train checkpoints |

**SHARED (source code; modify with care):** `src/`, `ui/`, `scripts/`, `tests/`,
`docs/`, `configs/` (schema must stay backward compatible), `app.py`, `README.md`.

## 5. Model Integration Rules

Adding a model = **config + checkpoint, no Python** (for built-in architectures).

Checkpoint location: `models/checkpoints/<crop>_<arch>.pt` (auto-discovery), or any
path referenced from an instance's `checkpoint:` field. Format: a `state_dict`
(or a dict with a `state_dict` key). Loading is `strict=False`; mismatches raise a
clear `CheckpointError` listing the configured path and class counts.

Required metadata per model: architecture key (must exist in `models:`), crop,
checkpoint path, and (for the class mapping) a `classes_file` or dataset classes.
Optional: `input_size` (defaults to adapter size), `target_layer` (default `auto`).

Complete example — **Model A** (`efficientnet_v2_s`) for **Crop A** (`tomato`):

```yaml
# configs/models.yaml
instances:
  - id: efficientnet_crop1
    name: EfficientNetV2-S (crop1)
    architecture: efficientnet_v2_s   # key from models.yaml -> models:
    crop: tomato                      # key from configs/datasets.yaml -> crops:
    checkpoint: models/checkpoints/tomato/efficientnet_v2_s.pt
    classes_file: configs/classes/tomato.yaml
    input_size: 224
    target_layer: auto
```

```yaml
# configs/classes/tomato.yaml   (order = model output order)
classes:
  - Bacterial_spot
  - Early_blight
  - Healthy
  - Late_blight
```

```
models/checkpoints/tomato/efficientnet_v2_s.pt
```

Then: `python scripts/validate_model.py --model efficientnet_crop1`.

**Adding another model later** (simplest workflow): place checkpoint + add one
`instances:` entry + (optional) classes file. No source edits.

**Custom architectures** (not one of the built-ins): place code in `models/custom/`,
write an adapter in `src/models/adapters/`, add the architecture key to
`configs/models.yaml -> models:` and the registry `src/models/registry.py`.
Full procedure: `docs/model_integration.md` §6.

## 6. Dataset Integration Rules

- Directory: `data/raw/<crop>/<class_name>/images...` (ImageFolder layout).
- Image formats: jpg/jpeg/png/bmp/webp. Corrupt images are detected and reported.
- Class naming: folder names become labels; the class order in `datasets.yaml`
  `classes:` (or a `classes_file`) must match the model output order.
- Splits: `python scripts/prepare_data.py --crop <crop>` creates deterministic,
  stratified, leakage-checked 75/10/15 manifests (`data/splits/<crop>/*.csv`).
  CSV format: columns `path, label, label_id`.
- Optional lesion masks: `data/raw/<crop>/masks/*.png` (binary, white = lesion)
  enable pointing-game faithfulness. Without them faithfulness reports
  UNAVAILABLE — never fabricated.

## 7. Experiment Rules

Lifecycle:

```
MODEL → CONFIGURATION (instances) → DATASET (manifests) → EVALUATION
     → METRICS → EXPERIMENT LOG → XAI → COMPARISON → REPORT
```

- Every evaluation (CLI or UI) appends a `RunRecord` — results are never
  overwritten; historical runs stay queryable (Experiments page, `runs.csv`).
- Why not overwrite? Research needs an auditable, append-only history for
  reproducibility and honest comparison.
- `scripts/run_experiments.py` chains prepare → evaluate_all → comparison.

## 8. Metrics Rules

Recorded per run/eval: **accuracy**, macro precision, macro recall, **macro F1
(primary research comparison metric)**, weighted F1, per-class precision/recall/F1,
confusion matrix (raw + normalized), inference latency, throughput, parameter
count, FLOPs (0 unless an adapter implements `flops()`), and efficiency scores
(accuracy/F1 per M params, F1 per FLOP, performance vs latency).
Accuracy alone is never sufficient — macro F1 drives ranking.

## 9. XAI Rules

- **Grad-CAM++** (`src/explainability/gradcam.py`) — default; requires a resolvable
  target layer from the adapter.
- **Score-CAM** (`scorecam.py`) — perturbation cross-check (subsampled for CPU).
- **Saliency** (`saliency.py`) — input gradients; needs gradient flow to input.
- Target layers are per-architecture (`target_layers()`), never assumed.
- Qualitative explanation always works for supported models (heatmap/overlay).
- Quantitative faithfulness = pointing game (hit rate) and **requires lesion masks**;
  absent masks → explicitly unavailable.
- Unsupported method on a model → clear "XAI method unavailable for this model"
  message with the reason; the app never crashes. Do not claim a method works
  for a model unless tested (see `tests/unit/test_xai.py`, `test_saliency.py`).

## 10. Streamlit Rules

| Page | Responsibility |
|---|---|
| Overview (home) | status, dataset manifests, recent runs, reproducibility, quickstart |
| Diagnosis | crop → model → image → prediction/top-k/confidence → XAI overlay → optional AI text |
| Model Evaluation | run dataset evaluation in-app; metrics, per-class, CM raw+normalized, classification report, downloads |
| Model Comparison | per-crop ranking from logged runs; sort, best verdicts, per-model detail |
| Experiments | filter history; compare ≥2 runs (metrics, per-class recall, CMs, latency); downloads |
| Explainability | Grad-CAM++/Score-CAM/Saliency tabs; overlay controls; pointing-game faithfulness |
| Ensemble | majority/soft/weighted voting; agreement; dataset-level cost-benefit |
| About | methodology, architectures, datasets, reproducibility, limitations |

All pages read from config/tracking data — nothing is hard-coded per crop/model.

## 11. Testing Rules

- Unit tests (`tests/unit/`) — config, registry, instances, metrics, tracking,
  split, seed, ensemble, XAI, comparison, batch evaluation, model validation.
- Integration/smoke — `tests/smoke/test_smoke.py` + `scripts/smoke_test.py`
  (10 checks: config → registry → model → preprocessing → prediction → metrics →
  experiment logging → Grad-CAM++ → comparison report → app import).
- End-to-end — `docs/end_to_end_validation.md` records the full acceptance run.
- Before considering a change complete: `python -m pytest` and
  `python scripts/smoke_test.py` must pass, and any new command must be executed
  (not just written).

## 12. Research Integrity Rules (never)

- Fabricated metrics / fake checkpoints / fake datasets / fake XAI results / fake graphs.
- Overwriting historical experiments.
- Silently changing preprocessing or evaluation splits.
- Presenting DEMO results as research results.
- Hiding failed model evaluations (failures are recorded with reasons).

## 13. Change Management

Before changing an architecture/adapter/data pipeline:
1. inspect dependents (adapters feed XAI target layers, input size, inference, UI)
2. preserve backward compatibility (config schema, run schema, CSV handling)
3. update the relevant docs
4. add/update tests
5. run `python -m pytest` + `python scripts/smoke_test.py`

After implementation:
1. update status tables in `docs/required.md` and `docs/documentation_validation.md`
2. update README if user-facing behaviour changed
3. verify generated outputs (evaluate one model, generate a comparison, open the app)