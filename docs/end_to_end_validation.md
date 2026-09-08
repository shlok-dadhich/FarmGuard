# End-to-End Validation

Date: 2026-09-08 · Branch: `main`

This document records the acceptance workflow (spec §29) executed against the
repository, the exact commands, results, known limitations, and where every output
lives. It is regenerated whenever the validation is re-run.

## 1. Acceptance workflow (spec §29 A–T)

| Step | How it was exercised | Result |
|---|---|---|
| A. Place a test model | `mock_demo` (deterministic tiny model) + real random-init torchvision models; real checkpoints are user-supplied per `docs/model_integration.md` | ✅ |
| B. Configure it | `configs/models.yaml -> instances:` schema implemented + validated by unit tests; default fallback (auto-discovery) used for the run below | ✅ |
| C. Launch application | `streamlit run app.py --server.headless true --server.port 8505` | ✅ HTTP 200, no errors in log |
| D–E. Select crop/model | Diagnosis/Model-Evaluation/Experiments pages import cleanly; model list is crop-filtered (`get_model_choices`) | ✅ |
| F–H. Upload image → prediction → confidence | `scripts/smoke_test.py` check 5; `scripts/evaluate.py` | ✅ |
| I. Grad-CAM++ | smoke check 8; `python scripts/generate_xai.py --method gradcam_pp` | ✅ |
| J. Another XAI method | Score-CAM + Saliency in smoke check 8 and CLI | ✅ |
| K. Dataset evaluation | `python scripts/evaluate_all.py --crop tomato --models mock_demo --split test` and full `run_experiments.py` | ✅ |
| L. Metrics | accuracy, macro P/R/F1, weighted F1, per-class — `tests/unit/test_metrics.py` + eval JSONs | ✅ |
| M. Save experiment | smoke check 7 + `outputs/metrics/runs.csv` appends (14 runs in history, never overwritten) | ✅ |
| N. Second model | `evaluate_all.py` evaluated 6 models (mock_demo, resnet50, densenet121, convnext_tiny, regnet_y_4gf, efficientnet_v2_s) | ✅ |
| O–P. Comparison + graphs | `python scripts/generate_comparison.py --crop tomato` → 8 figures | ✅ |
| Q. Confusion matrices | PNG + JSON per (crop, model, split) under `outputs/confusion_matrices/` | ✅ |
| R. Comparison report | `model_comparison_tomato.{csv,json,md,html}` (executive summary, best-by sections, per-class, CM analysis, latency) | ✅ |
| S. Old results remain | history is append-only; per-run JSONs are authoritative and `runs.csv` is rebuilt from them on schema drift | ✅ |
| T. Outputs linked from README | "RESEARCH OUTPUTS" table in README.md | ✅ |

## 2. Exact commands executed

```bash
python -m pytest -q                          # 31 passed, 0 failed
python scripts/smoke_test.py                 # SMOKE PASSED (10 checks)
python scripts/evaluate.py --crop tomato --model mock_demo
python scripts/evaluate_all.py --crop tomato --models mock_demo --split test
python scripts/run_experiments.py --crop tomato --split test --skip-prepare
python scripts/generate_comparison.py --crop tomato
python scripts/generate_xai.py --image outputs/demo_leaf.jpg --model mock_demo --method both
streamlit run app.py --server.headless true --server.port 8505   # HTTP 200, no tracebacks
```

## 3. Tests

- `python -m pytest` → **31 passed, 0 failed**
  - unit: config, registry, model specs/instances, tracking (append/migration/rebuild),
    split, seed, metrics, ensemble, XAI (Grad-CAM++ + saliency), comparison report,
    batch evaluation (continue-on-error, failure recording)
  - smoke: `tests/smoke/test_smoke.py`
- `python scripts/smoke_test.py` → **SMOKE PASSED** (config, registry, demo model,
  preprocessing, prediction, metrics, experiment logging, Grad-CAM++, comparison
  report, Streamlit app import)

## 4. Known limitations (reported honestly, never hidden)

- **Random-init evaluations**: without supplied checkpoints, models evaluate with
  random weights. These runs are recorded with `status=demo` and are clearly
  labelled DEMO / TEST RESULT — they are plumbing verification, not research results.
- **Model agreement in the comparison report** is reported as UNAVAILABLE RESULT
  unless per-image probabilities are stored (they are not by default). Measuring
  agreement requires a dataset-level ensemble evaluation, which the Ensemble page
  performs live.
- **Confidence-distribution comparison** across models is not persisted (probabilities
  are not written to disk to keep artifacts small); the per-image UI shows them live.
- **PDF export** is not generated directly; the self-contained HTML report
  (`model_comparison_<crop>.html`) prints to PDF from any browser.
- **FLOPs** are not computed automatically (adapter `flops()` returns 0 unless
  implemented); efficiency uses parameter- and latency-based scores, which are always
  measured.
- **CUDA** falls back to CPU when unavailable.
- Custom architectures (not among the built-in torchvision ones) require an adapter —
  see `docs/model_integration.md` §6.

## 5. Model integration instructions

See `docs/model_integration.md` and README "MODEL INTEGRATION" / "HOW TO ADD A NEW
MODEL". In short:

1. `models/checkpoints/<crop>/<file>.pt` (state_dict).
2. Optional class mapping `configs/classes/<crop>.yaml`.
3. One `instances:` entry in `configs/models.yaml`.
4. `python scripts/evaluate_all.py --crop <crop>` + `python scripts/generate_comparison.py --crop <crop>`.

## 6. Dataset instructions

See `docs/evaluation.md` and README "DATASET SETUP". In short:

1. `data/raw/<crop>/<class>/...` ImageFolder layout.
2. `configs/datasets.yaml -> crops.<crop>.root` (+ optional `classes`).
3. `python scripts/prepare_data.py --crop <crop>` → deterministic stratified splits
   (`data/splits/<crop>/{train,val,test}.csv`) + dataset report.

## 7. Output locations

| Artifact | Location |
|---|---|
| Experiment history | `outputs/metrics/runs.csv`, `outputs/metrics/run_<id>.json` |
| Eval records | `outputs/metrics/eval_<crop>_<arch>.json` |
| Confusion matrices | `outputs/confusion_matrices/` |
| Comparison figures | `outputs/figures/<crop>/` |
| Comparison reports | `outputs/reports/model_comparison_<crop>.{csv,json,md,html}` |
| Pipeline summaries | `outputs/reports/evaluate_all_*.{json,csv}` |
| XAI artifacts + meta | `outputs/xai/` |
| Dataset reports | `data/metadata/<crop>_report.json` |

---

## PROJECT STATUS: READY

The documented acceptance workflow (A–T) runs end-to-end: 31/31 tests pass, the
10-check smoke test passes, the Streamlit app boots without errors, the full
evaluation pipeline produces metrics, experiment history, confusion matrices,
figures, comparison reports and XAI artifacts, and old experiment results remain
available in the append-only history.