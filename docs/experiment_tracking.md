# Experiment Tracking

## Design

- **File backend is the source of truth and always on** — zero dependencies.
- Every evaluation/training creates a unique `run_id`
  (`run-YYYYMMDD-HHMMSS-<hex>`); records are **appended, never overwritten**.
- MLflow and TensorBoard are optional mirrors; the platform works fully without them.

## Where results live

```
outputs/metrics/
├── runs.csv                 # one row per run (all recorded fields)
├── run_<id>.json            # full per-run record (authoritative)
└── eval_<crop>_<arch>.json  # latest eval payload for a (crop, architecture)
```

The `runs.csv` is a convenience view. Per-run JSONs are authoritative: if the CSV is
ever inconsistent (e.g. schema drift between versions), `src/tracking/run_schema.py`
rebuilds it from the JSONs automatically.

## Recorded fields (`src/tracking/run_schema.py`)

timestamp, run_id, crop, architecture, seed, dataset_version, split_version,
dataset_name, split (train/val/test), image_size, batch_size, epochs, optimizer,
learning_rate, scheduler, weight_decay, augmentation_version, train_loss, val_loss,
accuracy, macro_f1, weighted_f1, per_class_recall, parameter_count, flops,
training_time_s, inference_latency_ms, throughput, evaluation_samples,
confusion_matrix_path, checkpoint_path, checkpoint_hash (sha256 prefix),
device, xai_method, software_version, git_commit, kind (train/eval/ensemble/demo),
status (ok/failed/demo), notes.

## Inspecting history

- **Experiments page** (`streamlit run app.py`): filter by crop, architecture, kind,
  split, date range, run id; select 2+ runs for a side-by-side comparison (metrics,
  per-class recall, confusion matrices, latency/params/FLOPs) and download CSV/JSON.
- `python scripts/generate_comparison.py --crop X` reads the same data for the
  per-crop report.
- Raw inspection: `outputs/metrics/run_<id>.json` or any spreadsheet tool on
  `runs.csv`.

## Backends (`configs/tracking.yaml`)

| backend | behaviour |
|---|---|
| `file` (default) | runs.csv + JSONs under `outputs/metrics` |
| `mlflow` | `mlflow ui --backend-store-uri ./outputs/mlruns` |
| `both` | file + MLflow |

TensorBoard scalars are written when torch's TensorBoard integration is installed
(`tensorboard --logdir outputs/tb`).

## Honesty rules

- Demo runs are recorded with `kind=demo` / `status=demo` and are never used as
  research results; the comparison report prefers `kind=eval` runs over `kind=train`
  and labels everything from measured data.
- Failed evaluations are recorded with `status=failed` and a reason — never hidden.
- If a number was not measured it is stored as 0/empty and displayed as
  UNAVAILABLE RESULT, never invented.