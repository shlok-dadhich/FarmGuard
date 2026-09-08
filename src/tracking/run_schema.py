"""Run record schema + CSV/JSON persistence."""
from __future__ import annotations
import csv, json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REQUIRED = ["timestamp","run_id","crop","architecture","seed","image_size","batch_size","epochs",
"optimizer","learning_rate","scheduler","weight_decay","augmentation_version","train_loss","val_loss",
"accuracy","macro_f1","parameter_count","flops","training_time_s","inference_latency_ms","checkpoint_path"]

@dataclass
class RunRecord:
    timestamp: str = ""
    run_id: str = ""
    crop: str = ""
    architecture: str = ""
    seed: int = 0
    dataset_version: str = "v1"
    split_version: str = "v1"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 0
    optimizer: str = ""
    learning_rate: float = 0.0
    scheduler: str = ""
    weight_decay: float = 0.0
    augmentation_version: str = ""
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class_recall: dict = field(default_factory=dict)
    parameter_count: int = 0
    flops: float = 0.0
    training_time_s: float = 0.0
    inference_latency_ms: float = 0.0
    checkpoint_path: str = ""

    def to_dict(self):
        return asdict(self)

def append_run(record: RunRecord, out_dir: Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d = record.to_dict()
    (out_dir / f"{record.run_id}.json").write_text(json.dumps(d, indent=2))
    csv_path = out_dir / "runs.csv"
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if new: w.writeheader()
        w.writerow(flat)
