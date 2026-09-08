"""Comparison report generator tests (scripts/generate_comparison.py)."""

import json

import pandas as pd

from scripts.generate_comparison import build_comparison


def _runs():
    return pd.DataFrame([
        {"timestamp": "2026-01-01T00:00:00+00:00", "run_id": "run-1", "crop": "tomato",
         "architecture": "mock_demo", "kind": "train", "split": "val", "accuracy": 0.5,
         "macro_f1": 0.4, "macro_precision": 0.4, "macro_recall": 0.4, "weighted_f1": 0.4,
         "per_class_recall": '{"0": 0.5, "1": 0.5}', "parameter_count": 1000, "flops": 0.0,
         "inference_latency_ms": 1.2, "throughput": 500.0, "evaluation_samples": 4,
         "checkpoint_path": "", "confusion_matrix_path": ""},
        {"timestamp": "2026-01-02T00:00:00+00:00", "run_id": "run-2", "crop": "tomato",
         "architecture": "mock_demo", "kind": "eval", "split": "test", "accuracy": 0.45,
         "macro_f1": 0.35, "macro_precision": 0.35, "macro_recall": 0.35, "weighted_f1": 0.35,
         "per_class_recall": '{"0": 0.4, "1": 0.3}', "parameter_count": 1000, "flops": 0.0,
         "inference_latency_ms": 1.1, "throughput": 510.0, "evaluation_samples": 4,
         "checkpoint_path": "", "confusion_matrix_path": ""},
        {"timestamp": "2026-01-03T00:00:00+00:00", "run_id": "run-3", "crop": "tomato",
         "architecture": "resnet50", "kind": "eval", "split": "test", "accuracy": 0.6,
         "macro_f1": 0.55, "macro_precision": 0.55, "macro_recall": 0.55, "weighted_f1": 0.55,
         "per_class_recall": '{"0": 0.6, "1": 0.5}', "parameter_count": 25000000, "flops": 0.0,
         "inference_latency_ms": 3.0, "throughput": 200.0, "evaluation_samples": 4,
         "checkpoint_path": "", "confusion_matrix_path": ""},
    ])


def test_build_comparison_files(tmp_path):
    s = build_comparison(_runs(), "tomato", tmp_path / "metrics", tmp_path, ["c0", "c1"])
    for f in ("csv", "json", "md", "html"):
        assert (tmp_path / "reports" / f"model_comparison_tomato.{f}").exists(), f
    assert (tmp_path / "figures" / "tomato" / "macro_f1_by_model.png").exists()
    assert (tmp_path / "figures" / "tomato" / "per_class_recall.png").exists()
    assert s["best_macro_f1"]["model"] == "resnet50"
    assert s["best_accuracy"]["model"] == "resnet50"


def test_latest_eval_preferred_over_train(tmp_path):
    s = build_comparison(_runs(), "tomato", tmp_path / "metrics", tmp_path, ["c0", "c1"])
    mock_row = next(r for r in s["models"] if r["architecture"] == "mock_demo")
    assert mock_row["split"] == "test"          # eval run won over the earlier train run
    assert abs(mock_row["accuracy"] - 0.45) < 1e-9


def test_build_comparison_empty(tmp_path):
    s = build_comparison(pd.DataFrame(), "tomato", tmp_path / "metrics", tmp_path)
    assert s["status"] == "no-runs"
    assert "UNAVAILABLE" in (tmp_path / "reports" / "model_comparison_tomato.html").read_text()


def test_report_json_roundtrip(tmp_path):
    build_comparison(_runs(), "tomato", tmp_path / "metrics", tmp_path, ["c0", "c1"])
    data = json.loads((tmp_path / "reports" / "model_comparison_tomato.json").read_text())
    assert len(data["models"]) == 2
    assert "executive_summary" in data
    assert data["model_agreement"]["status"] == "UNAVAILABLE"