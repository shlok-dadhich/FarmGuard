"""Optional MLflow backend. Zero hard dependency: disabled gracefully if mlflow missing."""
from __future__ import annotations
from pathlib import Path

class MLflowBackend:
    def __init__(self, uri="./outputs/mlruns", experiment="agrovision"):
        self.uri = uri; self.experiment = experiment; self._mlflow = None
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
        except Exception as e:
            print(f"[tracking] MLflow unavailable ({e}); file backend only.")

    @property
    def available(self): return self._mlflow is not None

    def log_run(self, params: dict, metrics: dict, artifacts: list | None = None):
        if not self.available: return None
        ml = self._mlflow
        with ml.start_run():
            for k, v in params.items():
                try: ml.log_param(k, v)
                except Exception: pass
            for k, v in metrics.items():
                try:
                    if isinstance(v, (int, float)): ml.log_metric(k, float(v))
                except Exception: pass
            for a in artifacts or []:
                try:
                    if a and Path(a).exists(): ml.log_artifact(str(a))
                except Exception: pass
        return True
