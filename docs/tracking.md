# Experiment tracking / monitoring (MLflow alternative included)
## Backends (configs/tracking.yaml + configs/project.yaml:tracking)
- file (default, zero-dep): outputs/metrics/<run_id>.json + runs.csv
- mlflow: set backend mlflow|both, mlflow_uri ./outputs/mlruns; view with `mlflow ui --backend-store-uri ./outputs/mlruns`
- tensorboard: auto if torch tensorboard installed; view `tensorboard --logdir outputs/tb`
##logged per run
timestamp, run_id, crop, architecture, seed, dataset/split versions, image_size, batch_size, epochs, optimizer/lr/scheduler/wd, augmentation_version, train/val loss, accuracy, macro_f1, per-class recall, params, flops, training_time, latency, checkpoint_path.
## UI
Model Comparison page reads runs.csv; scripts/generate_report.py builds outputs/reports/summary.md.
