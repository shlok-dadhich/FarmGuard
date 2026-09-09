"""Batch evaluation: ALL MODELS x ONE CROP, ONE MODEL x ALL CROPS, or ALL x ALL.

Usage:
    python scripts/evaluate_all.py --crop tomato --split test
    python scripts/evaluate_all.py --crop tomato --models resnet50,mock_demo --limit 64
    python scripts/evaluate_all.py --all --split val

Behaviour:
- discovers configured models (model instances from configs/models.yaml when
  present, otherwise all registered architectures)
- evaluates each model on the requested split manifest (data/splits/<crop>/<split>.csv)
- continues when one model fails and records the failure reason
- logs every evaluation as an experiment run (kind=eval) — history is appended,
  never overwritten
- saves per-model metrics JSON, confusion matrices (PNG+JSON) and a summary report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.core.config import load_config
from src.core.device import resolve_device
from src.data.datamodule import loaders_from_splits
from src.evaluation.confusion_matrix import compute_cm
from src.evaluation.efficiency import efficiency_row
from src.evaluation.metrics import compute_metrics, confidence_stats
from src.inference.model_loader import load_model
from src.models.factory import get_adapter
from src.models.metadata import load_model_specs, models_for_crop
from src.models.registry import MODEL_REGISTRY
from src.tracking.run_schema import RunRecord
from src.tracking.tracker import get_tracker
from src.utils.reproducibility import env_report


def _checkpoint_hash(path: str | None, max_bytes: int = 200 * 1024 * 1024) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        if p.stat().st_size > max_bytes:
            return "skipped-large-file"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


def resolve_targets(crop: str | None, models_arg: str | None, cfg) -> list[dict]:
    """Model targets: (key, architecture, checkpoint, input_size, name)."""
    specs = load_model_specs(cfg.models_cfg)
    targets: list[dict] = []
    if crop is not None:
        specs_crop = models_for_crop(specs, crop)
        if specs_crop:
            # one target per configured instance (instances may share an architecture)
            for spec in specs_crop:
                ckpt = spec.checkpoint if (spec.checkpoint and Path(spec.checkpoint).exists()) else None
                targets.append({
                    "key": spec.id, "architecture": spec.architecture, "checkpoint": ckpt,
                    "input_size": spec.input_size or get_adapter(spec.architecture).input_size(),
                    "name": spec.display_name, "crop": crop,
                })
        else:
            # classic behaviour: every registered architecture for this crop
            for arch in MODEL_REGISTRY:
                targets.append({
                    "key": arch, "architecture": arch, "checkpoint": None,
                    "input_size": get_adapter(arch).input_size(), "name": arch, "crop": crop,
                })
    else:
        for spec in specs:
            for c in spec.crops:
                ckpt = spec.checkpoint if (spec.checkpoint and Path(spec.checkpoint).exists()) else None
                targets.append({
                    "key": spec.id, "architecture": spec.architecture, "checkpoint": ckpt,
                    "input_size": spec.input_size or get_adapter(spec.architecture).input_size(),
                    "name": spec.display_name, "crop": c,
                })
        if not specs:
            for arch in MODEL_REGISTRY:
                for c in cfg.datasets.get("crops", {}):
                    targets.append({
                        "key": arch, "architecture": arch, "checkpoint": None,
                        "input_size": get_adapter(arch).input_size(), "name": arch, "crop": c,
                    })
    if models_arg:
        wanted = {m.strip() for m in models_arg.split(",") if m.strip()}
        # Explicitly requested registered architectures (e.g. --models mock_demo for
        # plumbing tests) are still resolvable even when instances are configured.
        covered = {t["key"] for t in targets} | {t["architecture"] for t in targets}
        for arch in MODEL_REGISTRY:
            if arch in wanted and arch not in covered:
                targets.append({
                    "key": arch, "architecture": arch, "checkpoint": None,
                    "input_size": get_adapter(arch).input_size(), "name": arch, "crop": crop,
                })
        targets = [t for t in targets if t["key"] in wanted or t["architecture"] in wanted]
    return targets


def run_evaluations(crop: str | None, split: str, models_arg: str | None = None,
                    limit: int | None = None, cfg=None, device: str | None = None,
                    out_root: Path | None = None) -> dict:
    """Evaluate targets and return the summary dict. Never raises for a single
    broken model — failures are recorded and evaluation continues."""
    cfg = cfg or load_config()
    device = device or resolve_device(cfg.device)
    out_root = Path(out_root) if out_root else Path(cfg.project.get("output_dir", "outputs"))
    tracker = get_tracker(cfg.tracking)
    env = env_report()

    metrics_dir = out_root / "metrics"
    cm_dir = out_root / "confusion_matrices"
    reports_dir = out_root / "reports"
    figures_dir = out_root / "figures"
    for d in (metrics_dir, cm_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    targets = resolve_targets(crop, models_arg, cfg)
    if not targets:
        return {"status": "no-targets", "rows": [], "note": "No models/crops matched."}

    summaries = []
    for t in targets:
        row = {"crop": t["crop"], "model_key": t["key"], "architecture": t["architecture"],
               "checkpoint": t["checkpoint"] or "none (random init)", "status": "failed", "reason": ""}
        try:
            split_dir = out_root.parent / "data" / "splits" / t["crop"]
            manifest = split_dir / f"{split}.csv"
            if not manifest.exists():
                raise FileNotFoundError(
                    f"No split manifest data/splits/{t['crop']}/{split}.csv — run "
                    "scripts/prepare_data.py --crop <crop> first."
                )
            df = pd.read_csv(manifest)
            if df.empty:
                raise ValueError(f"Split manifest {manifest} is empty.")
            if limit:
                df = df.groupby("label_id", group_keys=False).apply(
                    lambda g: g.head(max(1, limit // max(1, df["label_id"].nunique())))
                ).reset_index(drop=True)
            nc = int(df["label_id"].nunique())
            loader = loaders_from_splits(split_dir, image_size=t["input_size"],
                                         batch_size=32)[split]
            model = load_model(t["architecture"], nc, t["checkpoint"], device)
            adapter = get_adapter(t["architecture"])
            params = adapter.parameter_count(model)

            ys, ps, probs = [], [], []
            model.eval()
            t0 = time.time()
            n = 0
            with torch.no_grad():
                for xb, yb in loader:
                    out = model(xb.to(device))
                    p = torch.softmax(out, 1).cpu().tolist()
                    probs += p
                    ps += torch.argmax(out, 1).cpu().tolist()
                    ys += yb.tolist()
                    n += len(yb)
            wall = max(time.time() - t0, 1e-6)
            latency_ms = wall / max(1, n) * 1000
            throughput = n / wall

            metrics = compute_metrics(ys, ps)
            cm = compute_cm(ys, ps, labels=list(range(nc)))
            eff = efficiency_row(metrics["accuracy"], metrics["macro_f1"], params, 0.0, latency_ms)

            demo = not t["checkpoint"]
            run_id = tracker.new_run_id()
            cm_json = cm_dir / f"cm_{t['crop']}_{t['architecture']}_{split}.json"
            cm_png = cm_dir / f"cm_{t['crop']}_{t['architecture']}_{split}.png"
            cm_json.write_text(json.dumps({"confusion_matrix": cm, "labels": list(range(nc)),
                                           "crop": t["crop"], "architecture": t["architecture"],
                                           "split": split, "demo": demo}, indent=2))
            try:
                from src.evaluation.figures import plot_confusion_matrix
                plot_confusion_matrix(cm, list(range(nc)), t["architecture"], cm_png, normalize=False)
            except Exception:
                pass

            rec = RunRecord(
                timestamp=datetime.now(timezone.utc).isoformat(), run_id=run_id,
                crop=t["crop"], architecture=t["architecture"], dataset_name=t["crop"],
                split=split, image_size=t["input_size"], batch_size=32,
                accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                weighted_f1=metrics["weighted_f1"], per_class_recall=metrics.get("per_class_recall", {}),
                parameter_count=params, flops=0.0, inference_latency_ms=latency_ms,
                throughput=throughput, evaluation_samples=n, confusion_matrix_path=str(cm_json),
                checkpoint_path=t["checkpoint"] or "", checkpoint_hash=_checkpoint_hash(t["checkpoint"]),
                device=device,                software_version=str(env.get("torch", "")),
                git_commit=str(env.get("commit", "")), kind="eval",
                status="demo" if demo else "ok",
            )
            tracker.finish_run(rec, artifacts=[cm_png] if cm_png.exists() else None)

            eval_rec = {"metrics": metrics, "confusion_matrix": cm, "efficiency": eff,
                        "latency_ms": latency_ms, "throughput": throughput, "n": n,
                        "split": split, "crop": t["crop"], "architecture": t["architecture"],
                        "run_id": run_id, "status": rec.status}
            (metrics_dir / f"eval_{t['crop']}_{t['architecture']}.json").write_text(
                json.dumps(eval_rec, indent=2))
            (figures_dir / t["crop"]).mkdir(parents=True, exist_ok=True)

            row.update({"status": rec.status, "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"], "macro_recall": metrics["macro_recall"],
                        "macro_precision": metrics["macro_precision"], "weighted_f1": metrics["weighted_f1"],
                        "n": n, "latency_ms": round(latency_ms, 3), "throughput": round(throughput, 2),
                        "params": params, "run_id": run_id, "cm_json": str(cm_json),
                        "conf_stats": confidence_stats(probs)})
        except Exception as e:
            row["reason"] = f"{type(e).__name__}: {e}"
        summaries.append(row)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    summary = {"timestamp": ts, "crop": crop or "all", "split": split, "total": len(summaries),
               "ok": sum(1 for r in summaries if r["status"] in ("ok", "demo")),
               "failed": sum(1 for r in summaries if r["status"] == "failed"), "rows": summaries}
    (reports_dir / f"evaluate_all_{ts}.json").write_text(json.dumps(summary, indent=2, default=str))
    try:
        pd.DataFrame(summaries).to_csv(reports_dir / f"evaluate_all_{ts}.csv", index=False)
    except Exception:
        pass
    return summary


def main():
    ap = argparse.ArgumentParser(description="Batch evaluation (continue-on-error).")
    ap.add_argument("--crop", default=None, help="crop key from configs/datasets.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--models", default=None, help="comma-separated model keys/architectures")
    ap.add_argument("--limit", type=int, default=None, help="cap samples per split (subset eval)")
    ap.add_argument("--all", action="store_true", help="all models x all crops")
    a = ap.parse_args()
    crop = None if a.all else (a.crop or "tomato")
    summary = run_evaluations(crop, a.split, a.models, a.limit)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for r in summary["rows"]:
        mark = "OK  " if r["status"] in ("ok", "demo") else "FAIL"
        print(f"[{mark}] {r['crop']}/{r['model_key']}: {r.get('accuracy', 'n/a')} acc "
              f"{r.get('macro_f1', 'n/a')} f1" + (f"  reason={r['reason']}" if r["reason"] else ""))
    sys.exit(0 if summary.get("failed", 0) == 0 or summary.get("ok", 0) > 0 else 1)


if __name__ == "__main__":
    main()