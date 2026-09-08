# Documentation Validation

This document records the documentation validation pass (per the project's
DOCUMENTATION VALIDATION RULE): every command, path and configuration example in
`README.md`, `docs/instruction.md`, `docs/required.md`, `docs/HOW_TO_RUN.md` and
the other `docs/*.md` files was compared against the actual repository and, where
possible, executed.

Validation date: 2026-09-08 · Branch: `main` · 34 pytest tests passing · smoke test passing

---

## 1. Files verified

| File | Verified against | Status |
|---|---|---|
| `README.md` | repository structure, scripts, configs, outputs | ✅ matches (paths/commands re-checked this pass) |
| `docs/instruction.md` | `src/`, `configs/`, `ui/`, `scripts/` | ✅ new, written from the actual repository |
| `docs/required.md` | config schemas, model registry, status tables | ✅ new, written from the actual repository |
| `docs/HOW_TO_RUN.md` | every command executed or arg-checked | ✅ new, all commands exist |
| `docs/architecture.md` | `src/` module map | ✅ updated |
| `docs/model_integration.md` | `src/models/metadata.py`, adapters | ✅ new |
| `docs/evaluation.md` | `src/evaluation/`, scripts | ✅ new |
| `docs/experiment_tracking.md` | `src/tracking/` | ✅ new |
| `docs/xai.md` | `src/explainability/` | ✅ updated (Saliency added) |
| `docs/end_to_end_validation.md` | full acceptance run | ✅ regenerated |
| `docs/tracking.md`, `docs/model_contract.md` | legacy notes | ✅ still accurate (tracking.md points at outputs/metrics; model_contract.md matches auto-discovery) |

## 2. Commands verified (executed or arg-checked)

| Command | Script exists | Executed | Result |
|---|---|---|---|
| `pip install -r requirements.txt` | `requirements.txt` ✅ | yes (earlier env) | ✅ |
| `python scripts/smoke_test.py` | ✅ | yes | SMOKE PASSED (10 checks) |
| `python -m pytest` | ✅ | yes | 34 passed, 0 failed |
| `python scripts/prepare_data.py --crop tomato` | ✅ (`--crop` required) | yes | manifests + report created |
| `python scripts/validate_model.py --model <id>` | ✅ (implemented this pass) | yes | exit 0; REQUIRES USER INPUT when weights missing |
| `python scripts/validate_model.py --model resnet50 --crop tomato` | ✅ | yes | exit 0, VALIDATION OK WITH WARNINGS (no weights) |
| `python scripts/evaluate.py --crop tomato --model mock_demo` | ✅ (`--crop --model --checkpoint`) | yes | eval JSON saved |
| `python scripts/evaluate_all.py --crop tomato --split test` | ✅ (`--crop --split --models --limit --all`) | yes | 6 models evaluated, runs appended |
| `python scripts/evaluate_all.py --all --split test` | ✅ | arg-checked | — |
| `python scripts/run_experiments.py --crop tomato --split test --skip-prepare` | ✅ (`--crop --all --split --skip-prepare --limit`) | yes | PIPELINE OK |
| `python scripts/run_experiments.py --all --split test` | ✅ | arg-checked | — |
| `python scripts/generate_comparison.py --crop tomato` | ✅ (`--crop --all`) | yes | CSV/JSON/MD/HTML + 8 figures |
| `python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model mock_demo --method both` | ✅ (`--image --model --method both\|gradcam_pp\|scorecam\|saliency`) | yes | 3 artifacts + meta JSON |
| `streamlit run app.py` | ✅ | yes (headless, port 8505) | HTTP 200, no tracebacks |
| `mlflow ui --backend-store-uri ./outputs/mlruns` | optional backend | not run (mlflow optional) | documented as optional |
| `tensorboard --logdir outputs/tb` | optional backend | not run | documented as optional |
| `python scripts/train.py --crop tomato --model mock_demo --seed 42` | ✅ (demo only) | earlier session | demo checkpoint created |
| `python scripts/benchmark.py` | ✅ | arg-checked | — |

Removed/avoided: no fictional commands are documented. The spec's example
`validate_model.py` was implemented so the documented command is real.

## 3. Paths verified

| Path | Exists / created by | Notes |
|---|---|---|
| `configs/{models,datasets,project,training,explainability,tracking,ai}.yaml` | ✅ repo | note: XAI config is `explainability.yaml` (not `xai.yaml`) |
| `configs/classes/<crop>.yaml` | user-created | format: `classes:` list; example documented |
| `models/checkpoints/<crop>_<arch>.pt` | user-created | auto-discovery pattern; per-instance paths also supported |
| `models/custom/` | user-created (custom models only) | adapter required |
| `data/raw/<crop>/<class>/` | user-created | ImageFolder layout |
| `data/raw/<crop>/masks/` | user-created (optional) | faithfulness |
| `data/splits/<crop>/{train,val,test}.csv` | `prepare_data.py` | columns path,label,label_id |
| `data/metadata/<crop>_report.json` | `prepare_data.py` | ✅ |
| `outputs/metrics/runs.csv` | tracker | live experiment history (source of truth) |
| `outputs/metrics/run-*.json` | tracker | per-run records (authoritative) |
| `outputs/metrics/eval_<crop>_<arch>.json` | evaluators | ✅ |
| `outputs/confusion_matrices/cm_<crop>_<arch>_<split>.{png,json}` | evaluators | ✅ |
| `outputs/figures/<crop>/*.png` | `generate_comparison.py` | 8 plots (radar only with ≥3 models) |
| `outputs/reports/model_comparison_<crop>.{csv,json,md,html}` | `generate_comparison.py` | ✅ |
| `outputs/reports/evaluate_all_*.{json,csv}` | `evaluate_all.py` | ✅ |
| `outputs/reports/summary.md` | `generate_report.py` (legacy) | ✅ |
| `outputs/xai/xai_<ts>_<method>_c<cls>.jpg` + `*_meta.json` | XAI pipeline/UI | ✅ |
| `outputs/experiments/` | **not a live store** | spec-layout name; live store is `outputs/metrics/` (configurable via `tracking.yaml -> file_store`) — documented as such |
| `outputs/predictions/` | **reserved** | batch prediction not implemented; per-image predictions shown in UI — documented as such |

## 4. Configuration examples verified

- `instances:` schema (`id, name, architecture, crop/crops, checkpoint,
  classes_file, input_size, target_layer`) — matches `src/models/metadata.py`
  `load_model_specs()` exactly (validated by `tests/unit/test_model_specs.py`).
- `classes:` file schema — matches `src/models/metadata.py` `_load_classes_file`.
- Architecture keys in docs (`efficientnet_v2_s`, `resnet50`, `convnext_tiny`,
  `regnet_y_4gf`, `densenet121`, `mock_demo`) — match `src/models/registry.py`
  exactly.
- Crop keys (`tomato, potato, pepper, corn, grape`) — match
  `configs/datasets.yaml -> crops:`.
- `device: auto|cpu|cuda` — matches `configs/project.yaml` + `src/core/device.py`.
- `tracking.yaml -> backend: file|mlflow|both`, `file_store` — matches
  `src/tracking/tracker.py`.
- `.env.example` variables (`AI_API_KEY`, `AI_BASE_URL`, `AI_MODEL`,
  `MLFLOW_TRACKING_URI`) — match `configs/ai.yaml` env expansion.

## 5. Status separation

**IMPLEMENTED (verified by tests/smoke):**
config loading · model registry + 5 built-in adapters + mock · model instances ·
checkpoint loading with WHAT/WHY/HOW errors · dataset scan/split/validation ·
preprocessing · inference · evaluation · metrics (macro F1 primary) ·
confusion matrices (raw + normalized) · efficiency scores · experiment tracking
(file; append-only with schema-drift recovery) · Grad-CAM++ · Score-CAM ·
Saliency · pointing-game faithfulness · model comparison page + report generator ·
8 figures · ensemble voting + agreement · Streamlit (8 pages) · 3 CLI evaluation
paths + pipeline wrapper · model validation script · 10-check smoke test ·
34 unit tests · README + docs package.

**USER MUST PROVIDE:**
- [x] trained checkpoint(s) (`models/checkpoints/` or instance `checkpoint:`)
- [x] model architecture/code if custom (`models/custom/` + adapter)
- [x] crop name (config key in `configs/datasets.yaml`)
- [x] class names + exact class ordering (folder names / `classes:` list)
- [x] number of classes (derived from checkpoint head + manifest)
- [x] dataset path + images (`data/raw/<crop>/<class>/`)
- [x] dataset labels (folder names)
- [x] evaluation/test data (manifests generated from it)
- [x] input image size (optional; defaults to architecture size)
- [x] normalization/preprocessing (optional; ImageNet defaults)
- [x] target layer for XAI if auto-discovery fails (instance `target_layer:`)

**OPTIONAL:**
- [ ] lesion masks (`data/raw/<crop>/masks/`) for faithfulness
- [ ] AI API key (`.env`) for the assistive text layer
- [ ] GPU/CUDA (CPU fallback)
- [ ] MLflow server / TensorBoard

**NOT IMPLEMENTED (honestly reported):**
- Integrated Gradients (optional in spec; Saliency covers gradient-based needs)
- FLOPs auto-measurement (adapter hook returns 0 unless implemented; efficiency
  uses measured params/latency)
- `outputs/experiments/` as a live directory (history lives in `outputs/metrics/`,
  configurable) — documented deviation
- `outputs/predictions/` batch prediction artifacts — reserved

**BLOCKED / REQUIRES USER INPUT:**
- Real evaluation numbers, real XAI on trained weights, faithful comparison
  reports — everything downstream of **your checkpoints and datasets**.
  The platform's plumbing is verified with the demo model; research results are
  impossible to produce until real weights are supplied (by design).

## 6. Known documentation gaps / future work

- FLOPs column is present but 0 until an adapter implements `flops()` — the
  comparison report shows it as measured data, so implement before relying on it.
- Confidence-distribution comparison across models is not persisted (per-image
  probabilities are not written to disk); the UI shows them live.
- `docs/tracking.md` / `docs/model_contract.md` are legacy notes kept for
  compatibility; the canonical references are `docs/experiment_tracking.md` and
  `docs/model_integration.md`.