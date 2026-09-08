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
    dataset_name: str = ""
    split: str = ""           # train | val | test | subset
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
    weighted_f1: float = 0.0
    per_class_recall: dict = field(default_factory=dict)
    parameter_count: int = 0
    flops: float = 0.0
    training_time_s: float = 0.0
    inference_latency_ms: float = 0.0
    throughput: float = 0.0     # images/second (measured, 0 when not measured)
    evaluation_samples: int = 0
    confusion_matrix_path: str = ""
    checkpoint_path: str = ""
    checkpoint_hash: str = ""  # sha256 of checkpoint bytes when available
    device: str = ""
    xai_method: str = ""
    software_version: str = ""
    git_commit: str = ""
    kind: str = "train"         # train | eval | ensemble | demo
    status: str = "ok"          # ok | failed | demo
    notes: str = ""

    def to_dict(self):
        return asdict(self)

def _read_csv(csv_path: Path):
    """Return (header, rows) or (None, []) when the file is unreadable/inconsistent."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
        if header is None:
            return None, []
        if any(len(r) != len(header) for r in rows):
            return None, []
        return header, rows
    except Exception:
        return None, []


def _rebuild_from_json(out_dir: Path, csv_path: Path) -> bool:
    """Rebuild runs.csv from the per-run JSON files (authoritative history).

    Used when the CSV is inconsistent or corrupted. The new run's JSON is always
    written before this is called, so it is included in the rebuilt file.
    """
    records = []
    for j in sorted(out_dir.glob("run-*.json")):
        try:
            records.append(json.loads(j.read_text()))
        except Exception:
            continue
    if not records:
        return False
    header: list[str] = []
    for r in records:
        for k in r:
            if k not in header:
                header.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in header})
    return True


def append_run(record: RunRecord, out_dir: Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d = record.to_dict()
    (out_dir / f"{record.run_id}.json").write_text(json.dumps(d, indent=2))
    csv_path = out_dir / "runs.csv"
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            w.writeheader()
            w.writerow(flat)
        return
    header, rows = _read_csv(csv_path)
    if header is None:
        # inconsistent widths (schema drift or corruption): rebuild from JSONs
        _rebuild_from_json(out_dir, csv_path)
        return
    missing = [k for k in flat if k not in header]
    if not missing:
        # write in the FILE's column order (may differ from the dataclass order
        # when the file was migrated/rebuilt from older-version JSONs)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writerow(flat)
        return
    # schema migration: rewrite with the union of old and new columns
    new_header = header + [k for k in missing if k not in header]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in zip(header, r)})
        w.writerow({k: flat.get(k, "") for k in new_header})
