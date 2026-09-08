# Required Inputs From User

*What exactly must YOU provide for AgroVision to work completely?*
This is a checklist, not prose. Status legend used throughout:
**IMPLEMENTED** · **PARTIALLY IMPLEMENTED** · **NOT IMPLEMENTED** ·
**BLOCKED** · **REQUIRES USER INPUT** · **OPTIONAL**

---

## A. TRAINED MODELS

For **every model** you provide, you must know / provide:

- [ ] **Model name** (how you want it shown, e.g. "EfficientNetV2-S crop1")
- [ ] **Architecture** — must match one of the registered keys:
      `efficientnet_v2_s`, `resnet50`, `convnext_tiny`, `regnet_y_4gf`,
      `densenet121` (built-in), or provide custom code for anything else
- [ ] **Crop** the model was trained for (must exist in `configs/datasets.yaml -> crops:`)
- [ ] **Checkpoint file** (state_dict or dict containing `state_dict`)
- [ ] **Checkpoint format** (PyTorch; confirm torch version compatibility)
- [ ] **Model class/code** if custom (place in `models/custom/`, adapter needed)
- [ ] **Number of classes**
- [ ] **Exact class names**
- [ ] **Class ordering** (index i must equal classes[i])
- [ ] **Input image size** (defaults to the architecture's size if omitted)
- [ ] **Preprocessing/normalization** (defaults are ImageNet mean/std + resize(256)/center-crop(size))
- [ ] **Channel order** (assumed RGB; images are converted)
- [ ] **Target layer for Grad-CAM++** (default `auto` → adapter resolves; provide
      a layer name only if auto-discovery fails)
- [ ] **Whether gradients are supported** (needed for Saliency)
- [ ] **Whether Score-CAM is supported** (needs a feature layer + forward passes)
- [ ] **Whether the model outputs logits** (assumed; probabilities are derived by softmax)
- [ ] **Framework/version** if relevant (torch/torchvision versions captured per run)

### Model information template (copy per model)

```text
Model:            (e.g. EfficientNetV2-S crop1)
Architecture:     (e.g. efficientnet_v2_s)
Crop:             (e.g. tomato)
Checkpoint:       (e.g. models/checkpoints/tomato/efficientnet_v2_s.pt)
Custom code:      (path in models/custom/ — only for non-built-in architectures)
Input size:       (e.g. 224)
Classes:          (comma-separated, in output order)
Normalization:    (mean/std; defaults: 0.485/0.456/0.406, 0.229/0.224/0.225)
Target layer:     (auto, or explicit module name if auto fails)
Checkpoint format: (state_dict / dict-with-state_dict)
```

## B. DATASET

For every crop:

- [ ] **Dataset path** (`data/raw/<crop>/<class>/...`)
- [ ] **Dataset name** (crop key, e.g. `tomato`)
- [ ] **Dataset version** if known (recorded per run)
- [ ] **Class names** (folder names)
- [ ] **Class-to-index mapping** (folders sorted alphabetically; or `classes:` list
      in `configs/datasets.yaml` / `configs/classes/<crop>.yaml` for explicit order)
- [ ] **Image folder structure** (ImageFolder layout, jpg/jpeg/png/bmp/webp)
- [ ] **Labels** (derived from folder names)
- [ ] **Split information** — created by `python scripts/prepare_data.py --crop <crop>`
      (deterministic stratified 75/10/15; manifests are the eval source of truth)
- [ ] **Test dataset** (split manifest; evaluation defaults to `test`)
- [ ] **Validation dataset** (split manifest; optional for eval, used by ensemble weights)
- [ ] **Optional lesion masks** (`data/raw/<crop>/masks/` — binary, white = lesion)
      for pointing-game faithfulness

Mandatory: dataset path, class names, enough images per class for a meaningful split.
Optional: explicit class mapping, masks, version string.

## C. CROP INFORMATION

For each crop, fill:

```text
Crop name:            (e.g. tomato)
Disease classes:      (e.g. Bacterial_spot, Early_blight, Healthy, Late_blight)
Dataset location:     (e.g. data/raw/tomato/)
Models available:     (e.g. efficientnet_crop1, resnet_crop1)
Class mapping:        (e.g. configs/classes/tomato.yaml)
Dataset split:        (e.g. data/splits/tomato/test.csv — REQUIRED for evaluation)
Evaluation available: (yes, once the manifest exists)
XAI mask available:   (yes/no — determines quantitative faithfulness)
```

## D. XAI REQUIREMENTS

**Required for QUALITATIVE XAI (always available for supported models):**
- [ ] an image
- [ ] a working (loadable) model
- [ ] a supported target layer (auto from the adapter; Saliency needs gradients)

**Required for QUANTITATIVE XAI (pointing game):**
- [ ] an image
- [ ] a trained model
- [ ] an explanation (Grad-CAM++ etc.)
- [ ] a **ground-truth lesion/region mask**

> If masks are unavailable, quantitative faithfulness is **explicitly reported as
> unavailable** — results are never invented.

## E. AI INTEGRATION (OPTIONAL)

Only if you enable the AI text layer (`configs/ai.yaml`):

- [ ] provider (`openai_compatible`; default is `fallback` which works offline)
- [ ] API key (env var, default `AI_API_KEY`; see `.env.example`)
- [ ] environment variable set in `.env`
- [ ] model name (default `gpt-4o-mini`)

**AI is an explanation layer, NOT the disease classifier.** The system remains
fully usable without an API key (fallback provider).

## F. SYSTEM REQUIREMENTS

- Python ≥ 3.11 (confirmed by `pyproject.toml`)
- OS: Windows / Linux / macOS (project developed and tested on Windows + POSIX bash)
- GPU: **optional** (`configs/project.yaml -> device: auto|cpu|cuda`; CPU works)
- CUDA: only needed if you want GPU inference; falls back to CPU automatically
- RAM: not benchmarked — **estimate**: 8 GB is comfortable for inference/evaluation
  of the built-in torchvision models; more for large datasets (label as estimate)
- Disk: depends on datasets/checkpoints; outputs are small text/PNG artifacts
  (estimate only)
- Dependencies: `pip install -r requirements.txt`

## G. CONFIGURATION REQUIRED

| File | Purpose | Required fields | Optional fields |
|---|---|---|---|
| `configs/models.yaml` | architecture registry + model instances | `models:` map exists | `instances:` list (recommended for real models) |
| `configs/datasets.yaml` | crops | `crops:<crop>:root` | `classes`, `description` |
| `configs/classes/<crop>.yaml` | class mapping | `classes:` list | — |
| `configs/project.yaml` | app-wide | exists (defaults fine) | `device`, `demo_mode` |
| `configs/explainability.yaml` | XAI settings | exists (defaults fine) | overlay colormap/alpha, mask dir |
| `configs/tracking.yaml` | tracking backend | `backend: file` | `mlflow_uri`, `file_store` |
| `configs/ai.yaml` | AI provider | `provider: fallback` | API settings |
| `.env` | secrets | — (empty works) | `AI_API_KEY`, `AI_BASE_URL`, `AI_MODEL` |

Every file ships with working defaults — you only edit what your project needs.

## H. WHAT THE AGENT IMPLEMENTS

| Item | Status |
|---|---|
| project structure | IMPLEMENTED |
| model adapters (5 built-in + mock) | IMPLEMENTED |
| model registry | IMPLEMENTED |
| model instances config (YAML) | IMPLEMENTED |
| checkpoint loading with clear errors | IMPLEMENTED |
| dataset loader + validation | IMPLEMENTED |
| preprocessing | IMPLEMENTED |
| inference | IMPLEMENTED |
| evaluation | IMPLEMENTED |
| metrics (macro F1 primary) | IMPLEMENTED |
| experiment tracking (file) | IMPLEMENTED |
| Grad-CAM++ | IMPLEMENTED |
| Score-CAM | IMPLEMENTED |
| Saliency | IMPLEMENTED |
| faithfulness (pointing game) | IMPLEMENTED (requires masks) |
| model comparison | IMPLEMENTED |
| graph generation | IMPLEMENTED |
| confusion matrices | IMPLEMENTED |
| ensemble | IMPLEMENTED |
| Streamlit frontend (8 pages) | IMPLEMENTED |
| report generation | IMPLEMENTED |
| tests + smoke test | IMPLEMENTED |
| README + documentation | IMPLEMENTED |
| FLOPs measurement | PARTIALLY IMPLEMENTED (adapter `flops()` hook exists, returns 0 unless implemented; efficiency uses params/latency which are measured) |
| Integrated Gradients | NOT IMPLEMENTED (optional in spec; Saliency covers gradient-based needs) |
| MLflow / TensorBoard | OPTIONAL (works when installed) |
| AI text explanation | OPTIONAL (default fallback works offline) |
| model training | NOT IMPLEMENTED by design (models are externally supplied; `scripts/train.py` is demo-only) |
| `outputs/experiments/` live store | NOT IMPLEMENTED as a directory — experiment history lives in `outputs/metrics/` (configurable via `configs/tracking.yaml -> file_store`) |

## I. CURRENT PROJECT STATUS (live checklist)

| Component | Status | What is required | How to verify | Blocking issue |
|---|---|---|---|---|
| Config loading | IMPLEMENTED | nothing | `python -m pytest tests/unit/test_config.py` | none |
| Model registry/adapters | IMPLEMENTED | nothing | `python -m pytest tests/unit/test_registry.py` | none |
| Model instances | IMPLEMENTED | valid `instances:` entries | `python scripts/validate_model.py --model <id>` | none |
| Checkpoint loading | IMPLEMENTED | real checkpoint files | `validate_model.py` shows load ok | **REQUIRES USER INPUT: your checkpoints** |
| Dataset pipeline | IMPLEMENTED | `data/raw/<crop>/` + manifests | `python scripts/prepare_data.py --crop tomato` | **REQUIRES USER INPUT: your dataset** |
| Evaluation | IMPLEMENTED | manifests + model | `python scripts/evaluate_all.py --crop tomato --models <id>` | your data/weights |
| Experiment tracking | IMPLEMENTED | nothing | smoke check 7; Experiments page | none |
| Comparison + graphs | IMPLEMENTED | logged runs | `python scripts/generate_comparison.py --crop tomato` | needs real runs for meaningful output |
| XAI (Grad-CAM++/Score-CAM/Saliency) | IMPLEMENTED | loadable model + image | `python scripts/generate_xai.py --image <img> --model <id>` | your model/weights |
| Faithfulness | IMPLEMENTED | lesion masks | Explainability page mask upload | **REQUIRES USER INPUT: masks** (optional) |
| Ensemble | IMPLEMENTED | ≥2 models + manifest | Ensemble page | your models |
| Streamlit | IMPLEMENTED | nothing | `streamlit run app.py` | none |
| Reports | IMPLEMENTED | logged runs | `generate_comparison.py` | real runs |
| Tests | IMPLEMENTED | nothing | `python -m pytest` (34 pass) + `python scripts/smoke_test.py` | none |

## J. FINAL COMPLETION CHECKLIST

The project is **not** complete merely because code exists. Mark COMPLETE only when:

- [ ] all required models are available (checkpoints supplied)
- [ ] all checkpoints load (`validate_model.py` → load OK)
- [ ] class mappings validated (order matches model output)
- [ ] all datasets accessible (manifests exist, no missing/corrupt images)
- [ ] evaluation succeeds on at least one real model + split
- [ ] metrics generated (accuracy, macro P/R/F1, weighted F1, per-class)
- [ ] experiment tracking works (runs appended, history intact)
- [ ] model comparison works (report + figures generated from real runs)
- [ ] graphs generated (`outputs/figures/<crop>/`)
- [ ] confusion matrices generated (`outputs/confusion_matrices/`)
- [ ] XAI works (Grad-CAM++ on a real model)
- [ ] additional XAI methods tested (Score-CAM, Saliency)
- [ ] faithfulness works when masks exist (and is honestly UNAVAILABLE otherwise)
- [ ] Streamlit works (all pages render without tracebacks)
- [ ] ensemble works (voting on ≥2 real models)
- [ ] reports generated (comparison CSV/JSON/MD/HTML)
- [ ] tests pass (`python -m pytest`)
- [ ] smoke test passes (`python scripts/smoke_test.py`)
- [ ] documentation complete (this package + README)

Status as of the last validation run: see `docs/documentation_validation.md`
and `docs/end_to_end_validation.md`. Items above that depend on your checkpoints
and datasets remain **REQUIRES USER INPUT** until you supply them.