"""Unified tracker: file backend always works; mlflow + tensorboard optional."""
from __future__ import annotations
import time, uuid
from datetime import datetime, timezone
from pathlib import Path
from src.tracking.run_schema import RunRecord, append_run
from src.tracking.mlflow_tracker import MLflowBackend

class ExperimentTracker:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.backend = cfg.get("backend", "file")
        self.file_store = Path(cfg.get("file_store", "outputs/metrics"))
        self.tb_dir = Path(cfg.get("tensorboard_dir", "outputs/tb"))
        self.experiment = cfg.get("experiment", "agrovision")
        self.ml = MLflowBackend(cfg.get("mlflow_uri", "./outputs/mlruns"), self.experiment) \
            if self.backend in ("mlflow", "both") else MLflowBackend.__new__(MLflowBackend)
        if self.backend in ("mlflow", "both") and not hasattr(self.ml, "uri"):
            self.ml = MLflowBackend(cfg.get("mlflow_uri", "./outputs/mlruns"), self.experiment)
        self._tb = None

    def _tb_writer(self):
        if self._tb is not None: return self._tb
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(str(self.tb_dir))
        except Exception: self._tb = False
        return self._tb if self._tb else None

    def new_run_id(self): return f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def log_epoch(self, run_id, epoch, metrics: dict):
        w = self._tb_writer()
        if w:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    try: w.add_scalar(f"{run_id}/{k}", float(v), epoch)
                    except Exception: pass

    def finish_run(self, record: RunRecord, artifacts=None):
        append_run(record, self.file_store)
        if self.backend in ("mlflow", "both") and getattr(self.ml, "available", False):
            d = record.to_dict()
            params = {k: d[k] for k in ("crop","architecture","seed","optimizer","scheduler","image_size","batch_size","epochs") if k in d}
            metrics = {k: d[k] for k in ("train_loss","val_loss","accuracy","macro_f1","training_time_s","inference_latency_ms") if isinstance(d.get(k),(int,float))}
            try: self.ml.log_run(params, metrics, artifacts)
            except Exception as e: print(f"[tracking] mlflow log failed: {e}")
        w = self._tb_writer()
        if w:
            try: w.flush()
            except Exception: pass
        return record

def get_tracker(cfg: dict | None = None) -> ExperimentTracker:
    return ExperimentTracker(cfg or {})
