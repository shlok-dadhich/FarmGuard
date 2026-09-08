# 🌱 AgroVision

Multi-crop plant disease **classification evaluation and explainability platform**.
AgroVision does **not train your models for you** — you supply trained checkpoints,
and the platform handles everything around them: loading, preprocessing, inference,
dataset evaluation, metrics, experiment tracking, model comparison, ensembles,
Grad-CAM++ / Score-CAM / Saliency explainability, and a Streamlit research dashboard.

**Research integrity rules (enforced by design):**
- No invented metrics, no fake checkpoints, no fabricated results.
- Missing data is reported as **UNAVAILABLE RESULT**, never filled in.
- Demo/test output is always labelled **DEMO / TEST RESULT** and never mixed with research experiments.
- Experiment history is append-only: old runs are never overwritten.

---

## QUICK START

```bash
# 1. Clone and enter the repository
git clone <your-repo-url> AgroVision
cd AgroVision

# 2. Create a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2b. ...or Linux/macOS
# python3 -m venv .venv
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put your trained checkpoints into models/
#    models/checkpoints/tomato/my_model.pt   (see "MODEL INTEGRATION" below)

# 5. Configure your model (one YAML entry — no Python changes)
#    edit configs/models.yaml -> instances:

# 6. Point the dataset config at your data and prepare splits
#    edit configs/datasets.yaml -> crops.<crop>.root
python scripts/prepare_data.py --crop tomato

# 7. Run the whole evaluation pipeline for one crop
python scripts/run_experiments.py --crop tomato --split test

# 8. Launch the dashboard
streamlit run app.py
```

Everything is config-driven: adding another trained model means editing YAML
(and dropping a checkpoint file), **not** editing Python.

---

## PROJECT OVERVIEW

| What | How |
|---|---|
| Load a configured model + class mapping | `src/models/` (registry → factory → adapters) |
| Preprocess + predict on an image | `src/inference/` |
| Evaluate a dataset (accuracy, macro F1, per-class, confusion matrix) | `src/evaluation/` |
| Track experiments (file backend, optional MLflow/TensorBoard) | `src/tracking/` |
| Grad-CAM++, Score-CAM, Saliency + pointing-game faithfulness | `src/explainability/` |
| Ensembles (majority / soft / weighted vote) | `src/inference/ensemble.py` |
| Comparison reports + figures | `scripts/generate_comparison.py` |
| Streamlit dashboard | `app.py` + `ui/pages/` |

The **primary comparison metric is macro F1** — accuracy alone is never enough.

---

## SYSTEM ARCHITECTURE

```
AgroVision/
├── app.py                  # Streamlit entry point (navigation)
├── README.md
├── requirements.txt
├── .env.example
│
├── configs/                # ALL configuration (YAML, no Python edits for new models)
│   ├── models.yaml         #   architecture registry + optional model instances
│   ├── datasets.yaml       #   crops, data roots, class lists, split ratios
│   ├── classes/            #   per-crop class-mapping files (classes_file)
│   ├── project.yaml        #   seed, device, output dir, demo mode
│   ├── training.yaml       #   training hyperparameters (used by scripts/train.py)
│   ├── explainability.yaml #   XAI settings (colormap, alpha, faithfulness)
│   ├── tracking.yaml       #   experiment-tracking backend
│   └── ai.yaml             #   optional AI text provider
│
├── models/                 # EXTERNALLY SUPPLIED — your checkpoints + custom code
│   ├── README.md
│   ├── checkpoints/        #   <crop>_<arch>.pt  (or per-instance paths)
│   └── custom/             #   custom model Python code (see docs/model_integration.md)
│
├── data/
│   ├── raw/<crop>/         #   ImageFolder-style datasets
│   ├── splits/<crop>/      #   train/val/test CSV manifests (auto-generated)
│   └── metadata/           #   per-crop dataset reports
│
├── src/
│   ├── core/               # config, device, exceptions (WHAT/WHY/HOW), seed, logging
│   ├── models/             # interface, registry, factory, metadata, adapters/
│   ├── data/               # scan, split, datasets, loaders, validation, metadata
│   ├── evaluation/         # metrics, confusion matrix, efficiency, figures, evaluator
│   ├── explainability/     # gradcam, scorecam, saliency, faithfulness, xai_pipeline
│   ├── inference/          # predictor, preprocessing, model_loader, ensemble
│   ├── tracking/           # run schema + tracker (file/MLflow/TensorBoard)
│   ├── training/           # trainer (used only by scripts/train.py)
│   ├── services/           # thin service layer over evaluation/prediction/ensemble
│   ├── ai/                 # optional AI text provider (fallback works offline)
│   └── utils/              # images, files, reproducibility
│
├── ui/                     # Streamlit pages + shared helpers
│   ├── common.py
│   ├── theme.py
│   ├── components/
│   └── pages/              # 1_Diagnosis, 2_Model_Evaluation, 2_Model_Comparison,
│                           # 3_Explainability, 4_Ensemble, 5_About, 6_Experiments
│
├── scripts/
│   ├── prepare_data.py     #   build split manifests
│   ├── evaluate.py         #   evaluate ONE model on a split
│   ├── evaluate_all.py     #   batch evaluate (continue-on-error)
│   ├── run_experiments.py  #   full pipeline: prepare -> evaluate_all -> comparison
│   ├── generate_comparison.py  # comparison report + figures
│   ├── generate_xai.py     #   CLI Grad-CAM++/Score-CAM/Saliency
│   ├── train.py            #   OPTIONAL demo training (mock) — not the focus
│   └── smoke_test.py       #   end-to-end smoke test (10 checks)
│
├── outputs/                # ALL research outputs (gitignored)
│   ├── experiments/        #   reserved per the spec layout (live history is outputs/metrics/)
│   ├── metrics/            #   RUN HISTORY: runs.csv, run-<id>.json, eval_<crop>_<arch>.json
│   ├── figures/<crop>/     #   comparison plots
│   ├── confusion_matrices/ #   CM PNG + JSON per (crop, model, split)
│   ├── xai/                #   explanation images + meta JSON
│   ├── predictions/
│   ├── reports/            #   comparison reports + pipeline summaries
│   └── logs/
│
├── tests/                  # pytest suite (unit + smoke)
└── docs/
    ├── architecture.md
    ├── model_integration.md
    ├── evaluation.md
    ├── experiment_tracking.md
    ├── xai.md
    └── end_to_end_validation.md
```

---

## INSTALLATION

### Virtual environment

**Windows (PowerShell or cmd):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Dependencies

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.11. PyTorch, torchvision, timm, OpenCV, Pillow, NumPy, pandas,
scikit-learn, matplotlib, Streamlit, PyYAML, pytest. MLflow / TensorBoard / OpenAI are
optional — the platform works fully without them.

---

## MODEL INTEGRATION

> "I already trained my model. Where do I put it?"

**Step 1 — place the checkpoint.**

```
models/
├── README.md
├── checkpoints/
│   └── tomato/
│       └── my_model.pt          <- trained weights
└── custom/
    └── my_model.py              <- ONLY if your model needs custom Python code
```

Two supported formats:
- **Standard torchvision/timm architectures** (EfficientNetV2-S, ResNet-50, ConvNeXt-Tiny,
  RegNetY-4GF, DenseNet-121, mock_demo): no custom code needed. The checkpoint must be a
  `state_dict` (or a dict containing a `state_dict` key). Head size must match your class count.
- **Custom externally supplied models**: put the model class in `models/custom/` and register
  an adapter — see `docs/model_integration.md`.

**Step 2 — configure the model (YAML only).**

Add an entry under `instances:` in `configs/models.yaml`:

```yaml
instances:
  - id: my_tomato_model
    name: My Tomato Model
    architecture: efficientnet_v2_s   # key from models.yaml -> models:
    crop: tomato                      # crop key from configs/datasets.yaml
    checkpoint: models/checkpoints/tomato/my_model.pt
    classes_file: configs/classes/tomato.yaml   # optional
    input_size: 224                   # optional; defaults to the architecture's size
    target_layer: auto                # optional; adapters resolve their own layers
```

`configs/classes/tomato.yaml`:
```yaml
classes:
  - Bacterial_spot
  - Early_blight
  - Healthy
  - Late_blight
```

The list order defines the model's output index order and must match training.

**Step 3 — done.** The model now appears in the UI (Diagnosis, Model Evaluation,
Explainability, Ensemble) and in the CLI (`evaluate_all.py`, `generate_xai.py`).

> No instances configured? The platform falls back to offering every registered
> architecture and auto-discovers checkpoints named `<crop>_<arch>.pt` under
> `models/checkpoints/` and `outputs/checkpoints/`.

**Registering a brand-new architecture** (only needed when your architecture is not
one of the built-ins) requires adapter code — see `docs/model_integration.md`.

---

## DATASET SETUP

Datasets are ImageFolder-style:

```
data/raw/tomato/
├── Bacterial_spot/
│   ├── img_001.jpg
│   └── ...
├── Early_blight/
└── ...
```

1. Put your images under `data/raw/<crop>/<class>/`.
2. Configure the crop in `configs/datasets.yaml` (root + optional `classes:` list).
3. Generate deterministic, stratified split manifests (75/10/15 by default, no leakage):

```bash
python scripts/prepare_data.py --crop tomato
```

This writes `data/splits/tomato/{train,val,test}.csv` (path, label, label_id) plus a
dataset report under `data/metadata/`. Splits are reused across all models and seeds.
Missing/corrupt images and class imbalance are reported — never silently ignored.

CSV manifests can also be written by any external tool, as long as they contain
`path`, `label`, `label_id` columns.

---

## RUNNING THE STREAMLIT APP

```bash
streamlit run app.py
```

| Page | What it does |
|---|---|
| **Overview** | crops/models/runs at a glance, dataset status, reproducibility info |
| **Diagnosis** | select crop → model → upload image → predict → top-k → Grad-CAM++/Score-CAM/Saliency overlay → optional AI text |
| **Model Evaluation** | run dataset evaluation from the UI: metrics, per-class table, confusion matrix (raw + normalized), classification report, downloads (CSV/JSON/PNG/HTML) |
| **Model Comparison** | per-crop ranking from logged runs, sorting, best-accuracy / best-F1 / best-efficiency verdicts, per-model details |
| **Experiments** | filter history (crop, model, kind, split, date, run id), compare 2+ runs, per-class recall, confusion matrices, downloads |
| **Explainability** | Grad-CAM++ vs Score-CAM vs Saliency side-by-side, overlay controls, pointing-game faithfulness when masks are supplied |
| **Ensemble** | majority / soft / weighted voting, agreement rate, dataset-level cost-benefit (measured only) |
| **About** | methodology, candidate architectures, datasets, reproducibility, limitations |

### Diagnosis page step-by-step

1. Select **Crop** — only models configured for that crop appear.
2. Select **Model**.
3. Upload a leaf image.
4. Click **Predict** — predicted disease + confidence + top-k.
5. Inspect the probability chart and model metadata.
6. Select an **XAI method** (Grad-CAM++, Score-CAM, or Saliency).
7. Inspect the heatmap / overlay (opacity + colormap controls).
8. **Download** the explanation (saved to `outputs/xai/` with meta JSON).

---

## HOW TO EVALUATE ONE MODEL

```bash
python scripts/evaluate.py --crop tomato --model my_tomato_model --checkpoint models/checkpoints/tomato/my_model.pt
```

Writes `outputs/metrics/eval_tomato_<arch>.json` and logs the run to experiment history.

## HOW TO EVALUATE MULTIPLE MODELS (batch)

```bash
python scripts/evaluate_all.py --crop tomato --split test
python scripts/evaluate_all.py --crop tomato --models model_a,model_b --limit 64
```

One broken model never destroys the rest — failures are recorded with reasons and
the evaluation continues.

## HOW TO EVALUATE ALL MODELS × ALL CROPS

```bash
python scripts/evaluate_all.py --all --split test
```

## RUN THE ENTIRE EVALUATION PIPELINE

```bash
python scripts/run_experiments.py --crop tomato --split test   # one crop
python scripts/run_experiments.py --all --split test           # everything
```

Pipeline: prepare data → evaluate all models → generate comparison report.

---

## HOW EXPERIMENT TRACKING WORKS

Every evaluation (CLI or UI) creates a unique `run_id` and **appends** a record —
history is never overwritten.

- **Source of truth (file backend, always on):**
  - `outputs/metrics/runs.csv` — one row per run (all metrics + metadata)
  - `outputs/metrics/run_<id>.json` — full per-run record
- **Optional backends:** MLflow (`mlflow ui --backend-store-uri ./outputs/mlruns`)
  and TensorBoard (`tensorboard --logdir outputs/tb`) — see `configs/tracking.yaml`.
- **Recorded per run:** run_id, timestamp, crop, architecture, dataset version/split,
  image size, batch size, checkpoint path + sha256 hash, accuracy, macro precision/
  recall/F1, weighted F1, per-class recall, params, FLOPs, latency, throughput,
  evaluation samples, confusion-matrix path, device, software version, git commit,
  kind (train/eval/ensemble/demo), status (ok/failed/demo).
- **Inspect old runs:** the **Experiments** page filters and compares any runs;
  `scripts/generate_comparison.py` consumes the same data.

## HOW TO COMPARE MODELS

```bash
python scripts/generate_comparison.py --crop tomato   # or --all
```

Generates:
```
outputs/reports/model_comparison_tomato.csv
outputs/reports/model_comparison_tomato.json
outputs/reports/model_comparison_tomato.md
outputs/reports/model_comparison_tomato.html   (self-contained, open in a browser)
outputs/figures/tomato/*.png                   (accuracy, F1, PRF, params/FLOPs/latency vs F1, radar, per-class recall)
```

The report contains: executive summary, best by accuracy, best by macro F1, best by
efficiency, per-class analysis, confusion-matrix analysis, model agreement, XAI results,
latency/computational analysis, and full raw results. The Streamlit **Model Comparison**
page shows the same rankings interactively with sorting and best-model highlights.

---

## HOW TO GENERATE GRAD-CAM++ AND OTHER XAI

```bash
# Grad-CAM++ + Score-CAM + Saliency
python scripts/generate_xai.py --image path/to/image.jpg --model my_tomato_model --method both

# one method only
python scripts/generate_xai.py --image path/to/image.jpg --model my_tomato_model --method gradcam_pp
python scripts/generate_xai.py --image path/to/image.jpg --model my_tomato_model --method scorecam
python scripts/generate_xai.py --image path/to/image.jpg --model my_tomato_model --method saliency
```

Outputs go to `outputs/xai/` with meaningful filenames plus a meta JSON sidecar
(prediction, confidence, target class, model, method, timestamp).

- **Grad-CAM++** — default; uses model-specific target layers from each adapter.
- **Score-CAM** — perturbation-based cross-check (subsampled for CPU speed).
- **Saliency** — input-gradient map; works for any differentiable model.
- If a method is not supported by a model, the UI shows
  *"XAI method unavailable for this model"* with the reason — it never crashes.

**Faithfulness (pointing game):** with binary lesion masks at
`data/raw/<crop>/masks/` (or uploaded per image), AgroVision computes
pointing-game hit rate = hits / evaluated samples. Without masks it reports
*"Quantitative faithfulness evaluation unavailable because lesion masks are not provided."* —
masks are never fabricated.

---

## HOW TO RUN AN EXPERIMENT (end-to-end example)

1. **Add a model** — place `models/checkpoints/tomato/my_model.pt`.
2. **Configure** — add an `instances:` entry in `configs/models.yaml`
   (id, name, architecture, crop, checkpoint, classes_file, input_size).
3. **Configure the dataset** — `configs/datasets.yaml`, then
   `python scripts/prepare_data.py --crop tomato`.
4. **Run evaluation** — `python scripts/run_experiments.py --crop tomato --split test`.
5. **Inspect metrics** — the **Model Evaluation** page or `outputs/metrics/eval_*.json`.
6. **Generate XAI** — `python scripts/generate_xai.py --image ... --model my_tomato_model`.
7. **Compare models** — `python scripts/generate_comparison.py --crop tomato`
   or the **Model Comparison** page.
8. **Export the report** — `outputs/reports/model_comparison_tomato.html` (print to PDF).

---

## HOW TO ADD A NEW MODEL

1. **Place checkpoint** — `models/checkpoints/<crop>/<file>.pt` (state_dict).
2. **Add metadata** — classes file `configs/classes/<crop>.yaml` (optional).
3. **Configure** — one `instances:` entry in `configs/models.yaml`.
4. **Test** — `python scripts/smoke_test.py`, then
   `python scripts/evaluate.py --crop <crop> --model <id> --checkpoint <path>`.
5. **Run evaluation** — `python scripts/evaluate_all.py --crop <crop>` and
   `python scripts/generate_comparison.py --crop <crop>`.

No source files need to be edited unless you are adding a **new architecture**
(see `docs/model_integration.md`).

---

## DEMO / TEST MODE

`configs/project.yaml` sets `demo_mode: true` (default) and `mock_demo` is a tiny
deterministic model for testing the frontend, pipeline, tracking and XAI plumbing.
Demo output is **always** visibly labelled **DEMO / TEST RESULT** (banner + status
`demo` in runs + meta JSON) and never mixed with research experiments.
To go research mode: set `demo_mode: false`, place real checkpoints, and configure
instances — real checkpoints load with status `ok`.

---

## TROUBLESHOOTING

| Problem | Cause / fix |
|---|---|
| **Checkpoint loading error** | path wrong, architecture ≠ checkpoint, or class count mismatch. Errors list the configured path, expected classes and model classes. Fix the instance config or re-export the weights as `{"state_dict": ...}`. |
| **Class mismatch** | classes_file order must match the training order; the manifest labels are the ground truth used in evaluation. |
| **Missing target layer** | the adapter's `target_layers()` is used; if your architecture isn't supported, implement it in `models/custom/` + adapter (see docs/model_integration.md). |
| **CUDA unavailable** | `configs/project.yaml -> device: auto` falls back to CPU; set `cpu` explicitly to silence messages. |
| **Corrupt image** | `prepare_data.py` reports missing/corrupt files; re-export them. |
| **Wrong preprocessing** | instance `input_size` overrides the architecture default; mean/std are ImageNet-standard unless you supply a custom adapter. |
| **Missing dataset** | run `prepare_data.py` for the crop; manifests live in `data/splits/<crop>/`. |
| **Missing configuration** | run from the repository root; `configs/*.yaml` must exist. |
| **Unsupported XAI method** | the UI shows why (e.g., no target layer, no gradient path) and keeps working — choose another method. |

---

## RESEARCH OUTPUTS

| What | Where |
|---|---|
| Experiment history | `outputs/metrics/runs.csv` + `outputs/metrics/run_<id>.json` |
| Per-evaluation metrics | `outputs/metrics/eval_<crop>_<arch>.json` |
| Confusion matrices | `outputs/confusion_matrices/` (PNG + JSON) |
| Comparison plots | `outputs/figures/<crop>/` |
| Comparison reports | `outputs/reports/model_comparison_<crop>.{csv,json,md,html}` |
| XAI images + meta | `outputs/xai/` |
| Pipeline summaries | `outputs/reports/evaluate_all_*.{json,csv}` |
| Dataset reports | `data/metadata/<crop>_report.json` |

---

## REPRODUCIBILITY

- **Configuration** is captured in `configs/*.yaml` (seed, device, split ratios,
  augmentation version, tracking backend, XAI settings).
- **Model metadata** (architecture, checkpoint hash, classes file, input size) is
  recorded per run.
- **Environment** (Python version, platform, torch/torchvision/timm/streamlit/sklearn
  versions, git commit) is captured via `src/utils/reproducibility.py` and stored
  with each run.
- Deterministic stratified splits and deterministic val/test preprocessing mean the
  same model + same manifest reproduce the same evaluation numbers.

---

## TESTS

```bash
python -m pytest            # full unit + smoke suite
python scripts/smoke_test.py  # 10-check end-to-end smoke (config -> tracking -> XAI -> report -> app)
```

See `docs/end_to_end_validation.md` for the full acceptance walkthrough.

## DOCUMENTATION

- `docs/instruction.md` — development & maintenance instructions (read before modifying)
- `docs/required.md` — what the user must supply (checklists + status tables)
- `docs/HOW_TO_RUN.md` — practical operational guide (start here after cloning)
- `docs/documentation_validation.md` — validation report of this documentation package
- `docs/architecture.md` — module map and data flow
- `docs/model_integration.md` — checkpoint placement, instances, custom models
- `docs/evaluation.md` — metrics, evaluation scripts, confusion matrices
- `docs/experiment_tracking.md` — run schema, backends, UI comparison
- `docs/xai.md` — Grad-CAM++, Score-CAM, Saliency, faithfulness
- `docs/end_to_end_validation.md` — acceptance test results