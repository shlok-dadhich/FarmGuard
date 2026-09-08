# AgroVision — How To Run

*"I have received the completed repository. What exactly do I do?"*
Every command in this document exists in the repository and was executed during
validation (see `docs/documentation_validation.md`). Example crop: **tomato**
(one of the configured crops: tomato, potato, pepper, corn, grape).

---

## 1. PREREQUISITES

- Python ≥ 3.11 (`python --version`)
- Git (to clone; not required afterwards)
- Optional: NVIDIA GPU + CUDA (CPU works; `configs/project.yaml -> device: auto` picks CUDA when available)
- Environment variables: none required. Optional AI keys go in `.env` (copy from `.env.example`)
- Model files: **your trained checkpoints** (you supply these)
- Datasets: **your images** (you supply these)

## 2. INSTALLATION

```bash
git clone <your-repo-url> AgroVision
cd AgroVision
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 3. VERIFY INSTALLATION

```bash
python scripts/smoke_test.py
```

Expected: `SMOKE PASSED` after 10 numbered checks (config → registry → model →
preprocessing → prediction → metrics → experiment logging → Grad-CAM++ →
comparison report → app import).

```bash
python -m pytest
```

Expected: all tests pass (34 at last validation; summary line shows no failures).

## 4. ADD TRAINED MODELS

Place checkpoints:

```
models/
└── checkpoints/
    └── tomato/
        ├── efficientnet_v2_s.pt
        ├── resnet50.pt
        └── convnext_tiny.pt
```

Checkpoint format: a PyTorch `state_dict` (or a dict containing a `state_dict` key).

Register each model in `configs/models.yaml` (under `instances:`):

```yaml
instances:
  - id: efficientnet_crop1
    name: EfficientNetV2-S (crop1)
    architecture: efficientnet_v2_s
    crop: tomato
    checkpoint: models/checkpoints/tomato/efficientnet_v2_s.pt
    classes_file: configs/classes/tomato.yaml
    input_size: 224
    target_layer: auto
```

Registered architecture keys you can reference: `efficientnet_v2_s`, `resnet50`,
`convnext_tiny`, `regnet_y_4gf`, `densenet121`, `mock_demo` (demo only).
If `instances:` is left empty, the app auto-discovers `<crop>_<arch>.pt` files and
offers every registered architecture.

## 5. ADD CLASS MAPPINGS

Create `configs/classes/tomato.yaml`:

```yaml
classes:
  - Bacterial_spot
  - Early_blight
  - Healthy
  - Late_blight
```

**Class order must match the model's output index order.** If omitted, class names
come from the split manifests, then from `datasets.yaml -> crops.<crop>.classes`.

## 6. ADD DATASET

ImageFolder layout:

```
data/
└── raw/
    └── tomato/
        ├── Bacterial_spot/
        │   ├── img_001.jpg
        │   └── ...
        ├── Early_blight/
        └── ...
```

Point `configs/datasets.yaml` at it and build the deterministic split manifests:

```bash
python scripts/prepare_data.py --crop tomato
```

This writes `data/splits/tomato/{train,val,test}.csv` (75/10/15, stratified,
leakage-checked) plus `data/metadata/tomato_report.json`.

## 7. VALIDATE MODEL

```bash
python scripts/validate_model.py --model efficientnet_crop1
# or for an architecture key (needs --crop):
python scripts/validate_model.py --model resnet50 --crop tomato
```

Checks, in order: config resolution → checkpoint found → checkpoint loads →
class names resolve → input size → XAI target layers → one forward pass.
Missing weights are reported as **REQUIRES USER INPUT** (model runs as DEMO until
supplied) and do not fail the validation; unloadable checkpoints or broken configs
exit non-zero.

## 8. RUN SINGLE-MODEL EVALUATION

```bash
python scripts/evaluate.py --crop tomato --model efficientnet_crop1
# optional explicit checkpoint:
python scripts/evaluate.py --crop tomato --model efficientnet_crop1 --checkpoint models/checkpoints/tomato/efficientnet_v2_s.pt
```

Evaluates the `test` split (falls back to `val` if absent). Generated files:

- `outputs/metrics/eval_tomato_efficientnet_v2_s.json` — metrics + per-class recall
- experiment run appended to `outputs/metrics/runs.csv` (kind=eval)

## 9. RUN MULTI-MODEL EVALUATION

```bash
python scripts/evaluate_all.py --crop tomato --split test
python scripts/evaluate_all.py --crop tomato --models efficientnet_crop1,resnet_crop1 --limit 64
```

One broken model never stops the batch — failures are recorded with reasons and
evaluation continues. Outputs per model: eval JSON, confusion matrix PNG+JSON, a
logged run, and a summary under `outputs/reports/evaluate_all_<timestamp>.{json,csv}`.

## 10. RUN ALL EXPERIMENTS

```bash
python scripts/run_experiments.py --crop tomato --split test
python scripts/run_experiments.py --all --split test
```

Pipeline: prepare data → evaluate all models → generate comparison report.
`--skip-prepare` skips data preparation; `--limit N` caps samples per split.
Every evaluation is **appended** to experiment tracking — history is never overwritten.

## 11. EXPERIMENT TRACKING

- **Records live in:** `outputs/metrics/runs.csv` (one row per run) and
  `outputs/metrics/run_<id>.json` (full record, authoritative). The store path is
  configurable via `configs/tracking.yaml -> file_store`.
- **Per-run fields:** run_id, timestamp, crop, architecture, seed, dataset
  version/split, image size, batch size, checkpoint path + hash, accuracy, macro
  P/R/F1, weighted F1, per-class recall, params, FLOPs, latency, throughput,
  evaluation samples, CM path, device, software version, git commit, kind, status.
- **Find previous runs:** the **Experiments** page (filter by crop/model/kind/split/
  date/run-id) or open `outputs/metrics/runs.csv` in any spreadsheet tool.
- **Compare runs:** Experiments page (select ≥2) or `scripts/generate_comparison.py`.
- **Avoid overwriting:** you don't — every run gets a new `run_id` and is appended;
  if the CSV ever drifts, it is rebuilt from the per-run JSONs automatically.
- **Optional backends:** `configs/tracking.yaml -> backend: mlflow|both`, then
  `mlflow ui --backend-store-uri ./outputs/mlruns`; TensorBoard via
  `tensorboard --logdir outputs/tb`.

## 12. GENERATE MODEL COMPARISON

```bash
python scripts/generate_comparison.py --crop tomato
python scripts/generate_comparison.py --all
```

Outputs:

- `outputs/reports/model_comparison_tomato.csv` — full comparison table
- `outputs/reports/model_comparison_tomato.json` — machine-readable report
- `outputs/reports/model_comparison_tomato.md` — markdown report
- `outputs/reports/model_comparison_tomato.html` — self-contained report (print to PDF)
- `outputs/figures/tomato/` — accuracy by model, macro F1 by model, PRF grouped,
  params/FLOPs/latency vs F1, radar (≥3 models), per-class recall heatmap

The report contains: executive summary, best by accuracy, best by macro F1, best by
efficiency, per-class analysis, confusion-matrix analysis, model agreement (honestly
UNAVAILABLE without stored per-image probabilities), XAI results inventory, latency/
computational analysis, and full raw results.

## 13. GENERATE XAI

```bash
# any image works; outputs/demo_leaf.jpg is generated by the smoke test
python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model efficientnet_crop1
python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model efficientnet_crop1 --method gradcam_pp
python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model efficientnet_crop1 --method scorecam
python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model efficientnet_crop1 --method saliency
```

- `--method both` (default) runs Grad-CAM++ + Score-CAM + Saliency.
- Outputs: `outputs/xai/xai_<timestamp>_<method>_c<class>.jpg` + a meta JSON
  sidecar (prediction, confidence, target class, model, method).
- A method that a model cannot support produces a clear "unavailable" error and
  does not affect the others.

## 14. STREAMLIT APPLICATION

```bash
streamlit run app.py
```

UI flow: **select crop → select model → upload image → analyze → prediction →
confidence → XAI → explanation**.

| Page | What to do there |
|---|---|
| Overview | status of crops/models/runs; quickstart |
| Diagnosis | crop → model → image → Predict; inspect top-k, confidence, Grad-CAM++/Score-CAM/Saliency overlay, optional AI text |
| Model Evaluation | crop → model → split → Run evaluation; metrics, per-class table, confusion matrix (raw/normalized), classification report, downloads |
| Model Comparison | crop → rankings, sorting, best verdicts, per-model detail |
| Experiments | filter runs; select ≥2 to compare; download |
| Explainability | method tabs, overlay controls, faithfulness (upload a mask) |
| Ensemble | pick 2–3 models, single-image consensus or dataset-level agreement + cost-benefit |
| About | methodology, architectures, datasets, reproducibility |

## 15. HOW TO COMPARE MODELS THROUGH THE UI

1. Open **Model Comparison**.
2. Select the **crop** (models are grouped per crop).
3. Sort by metric (macro F1 default; accuracy, efficiency, latency available).
4. Inspect the ranking table and best-model verdicts.
5. Expand a model for its eval record and per-class recall.
6. For confusion matrices and per-class recall side-by-side: **Experiments** page
   (select ≥2 runs) or open `outputs/reports/model_comparison_<crop>.html`.
7. Export: comparison HTML/CSV/JSON from `outputs/reports/`.

## 16. HOW TO RUN AN EXPERIMENT FROM START TO FINISH

1. **Add model** — `models/checkpoints/tomato/efficientnet_v2_s.pt`
2. **Configure** — `instances:` entry in `configs/models.yaml` + class mapping
3. **Configure dataset** — `configs/datasets.yaml -> crops.tomato.root`
4. **Prepare data** — `python scripts/prepare_data.py --crop tomato`
5. **Validate model** — `python scripts/validate_model.py --model efficientnet_crop1`
6. **Run evaluation** — `python scripts/run_experiments.py --crop tomato --split test`
7. **Check experiment log** — `outputs/metrics/runs.csv` / Experiments page
8. **Generate XAI** — `python scripts/generate_xai.py --image <img> --model efficientnet_crop1`
9. **Generate comparison** — `python scripts/generate_comparison.py --crop tomato`
10. **Open Streamlit** — `streamlit run app.py`
11. **Export final results** — `outputs/reports/model_comparison_tomato.html` (print to PDF)

## 17. OUTPUT DIRECTORY GUIDE

| Directory | What goes there | Who creates it | Should you edit it? |
|---|---|---|---|
| `outputs/metrics/` | experiment history: `runs.csv`, `run_<id>.json`, `eval_<crop>_<arch>.json`, `smoke_result.json` | tracker + evaluators | No — read-only (this IS the tracking store) |
| `outputs/experiments/` | reserved per the project layout; the live store is `outputs/metrics/` (configurable via `tracking.yaml -> file_store`) | — | No |
| `outputs/figures/<crop>/` | comparison plots | `generate_comparison.py` | No |
| `outputs/confusion_matrices/` | CM PNG + JSON per (crop, model, split) | evaluators | No |
| `outputs/xai/` | explanation images + meta JSON | XAI pipeline / UI | No |
| `outputs/reports/` | `model_comparison_<crop>.*`, `evaluate_all_*.{json,csv}`, `summary.md` | report scripts | No |
| `outputs/predictions/` | reserved for future batch prediction artifacts (per-image predictions are currently shown in the UI; XAI artifacts are saved under `outputs/xai/`) | — | No |
| `outputs/checkpoints/` | demo-training checkpoints (`scripts/train.py`) | train script | No |
| `outputs/logs/` | log files | app/scripts | No |

## 18. TROUBLESHOOTING

| Problem | Cause | Fix | Verify |
|---|---|---|---|
| Checkpoint not found | path wrong / file missing | fix `checkpoint:` in the instance or place `<crop>_<arch>.pt` under `models/checkpoints/` | `python scripts/validate_model.py --model <id>` |
| Checkpoint incompatible | arch ≠ weights, or class count mismatch | re-export as `{"state_dict": ...}` matching the architecture and class count | `validate_model.py` load check |
| Class mismatch | `classes_file` order ≠ model output order | reorder `configs/classes/<crop>.yaml` to match training | compare per-class eval results |
| Wrong preprocessing | input size / normalization mismatch | set `input_size:` on the instance; defaults are ImageNet mean/std + resize(256)/center-crop | prediction sanity on a known image |
| Dataset missing | no `data/raw/<crop>` or no manifests | add data, run `python scripts/prepare_data.py --crop tomato` | check `data/splits/tomato/test.csv` |
| Image corrupt | damaged file | re-export; `prepare_data.py` reports corrupt files | `prepare_data.py` output |
| CUDA unavailable | no GPU/driver | leave `device: auto` (falls back to CPU) or set `cpu` | `get_device` in Overview page |
| XAI target layer unavailable | adapter lacks `target_layers()` for the arch | implement it in the adapter (custom models) — see `docs/model_integration.md` | `validate_model.py` target_layers check |
| Score-CAM unsupported | no feature layer / very slow on CPU | use Grad-CAM++ or Saliency; the UI explains why | Explainability page |
| AI API key missing | `.env` not set | not required — `provider: fallback` works offline | Diagnosis AI checkbox |
| Streamlit import error | run from outside repo root | run from the repository root | `python scripts/smoke_test.py` (check 10) |
| Evaluation failure | manifest empty / model broken | check `evaluate_all.py` summary rows for `reason` | `outputs/reports/evaluate_all_*.json` |

## 19. RESEARCH WORKFLOW (recommended)

```
dataset → model → evaluation → experiment tracking → comparison → efficiency → ensemble → XAI → report
```

1. Prepare data (`prepare_data.py`) — ground truth for all evaluation.
2. Register + validate models (`models.yaml` + `validate_model.py`).
3. Evaluate everything (`evaluate_all.py` / `run_experiments.py`) — this logs runs.
4. Track/audit runs (Experiments page, `outputs/metrics/`).
5. Compare (`generate_comparison.py` + Model Comparison page) with macro F1 primary.
6. Efficiency: `outputs/figures/<crop>/params_vs_f1.png`, `latency_vs_f1.png`,
   `flops_vs_f1.png`, and F1-per-M-params scores in the comparison table.
7. Ensemble: Ensemble page for measured agreement and cost-benefit (claim superiority
   only from measurements).
8. XAI: `generate_xai.py` / Explainability page; faithfulness only with masks.
9. Report: `outputs/reports/model_comparison_<crop>.html` is the research deliverable.

## 20. CLEANUP / RESET

> Warnings: the commands below delete generated output. Experiment history lives in
> `outputs/metrics/` — do not delete it unless you intend to start tracking fresh.

- **Delete generated figures only:**
  `rm -rf outputs/figures outputs/confusion_matrices` (safe to rerun via `generate_comparison.py`)
- **Delete generated XAI artifacts:**
  `rm -rf outputs/xai` (regenerate via `generate_xai.py`)
- **Delete reports:**
  `rm -rf outputs/reports` (regenerate via `generate_comparison.py` / `evaluate_all.py`)
- **Reset experiment history (irreversible):**
  `rm -rf outputs/metrics` — then the next evaluation starts a fresh history
- **Reset demo data (mock images/manifests):**
  `rm -rf data/raw/tomato data/splits/tomato` (they are recreated by the demo `train.py`)
- **Clear Streamlit cache:** `streamlit cache clear`
- **Reset demo checkpoints:** `rm -rf outputs/checkpoints` (only demo-training artifacts)

Never delete `models/` (your weights) or `data/raw` (your data) with these commands.

## 21. CHEAT SHEET

```bash
# Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify
python scripts/smoke_test.py
python -m pytest

# Data
python scripts/prepare_data.py --crop tomato

# Model validation
python scripts/validate_model.py --model efficientnet_crop1

# Evaluate
python scripts/evaluate.py --crop tomato --model efficientnet_crop1
python scripts/evaluate_all.py --crop tomato --split test
python scripts/evaluate_all.py --all --split test

# Full pipeline
python scripts/run_experiments.py --crop tomato --split test

# Comparison
python scripts/generate_comparison.py --crop tomato

# XAI
python scripts/generate_xai.py --image img.jpg --model efficientnet_crop1 --method both

# Streamlit
streamlit run app.py
```