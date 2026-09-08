"""End-to-end smoke test.

Verifies, in order:
 1. configuration loads and validates
 2. model registry loads
 3. demo model builds and loads
 4. image preprocessing works
 5. prediction works (label + probs + confidence)
 6. metrics work
 7. experiment result is logged (file backend, temp dir)
 8. Grad-CAM++ executes
 9. comparison report generation works (temp dir)
10. Streamlit application imports successfully

Checks 7 and 9 run against temporary directories so demo output never mixes
with research experiments. XAI artifacts land under outputs/xai/ (clearly
labelled demo output via meta JSON).

Usage: python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # 1. configuration
    from src.core.config import load_config

    cfg = load_config()
    cfg.validate()
    print("1. config OK")

    # 2. registry
    from src.models.factory import build_model, get_adapter
    from src.models.registry import MODEL_REGISTRY

    assert "mock_demo" in MODEL_REGISTRY and len(MODEL_REGISTRY) >= 6
    ad = get_adapter("mock_demo")
    print(f"2. registry OK ({len(MODEL_REGISTRY)} architectures)")

    # 3. demo model loads
    m = build_model("mock_demo", 4)
    assert ad.parameter_count(m) > 0
    print(f"3. demo model OK params={ad.parameter_count(m)}")

    # 4. demo image + preprocessing
    from PIL import Image

    from src.inference.preprocessing import preprocess_pil
    from src.utils.image import make_demo_image

    ip = make_demo_image(Path("outputs/demo_leaf.jpg"))
    x = preprocess_pil(Image.open(ip).convert("RGB"), ad.input_size())
    assert x.shape == (1, 3, ad.input_size(), ad.input_size())
    print("4. preprocessing OK", tuple(x.shape))

    # 5. prediction
    from src.inference.predictor import Predictor

    pred = Predictor(m, [f"c{i}" for i in range(4)], demo=True).predict(Image.open(ip).convert("RGB"))
    assert "label" in pred and "probs" in pred and "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0
    print(f"5. prediction OK {pred['label']} conf={pred['confidence']:.3f}")

    # 6. metrics
    from src.evaluation.metrics import compute_metrics

    mm = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert 0.0 <= mm["macro_f1"] <= 1.0 and mm["accuracy"] == 0.75
    print(f"6. metrics OK macro_f1={mm['macro_f1']:.3f}")

    # 7. experiment logging (temp dir - never pollutes research outputs)
    with tempfile.TemporaryDirectory() as tmp:
        from datetime import datetime, timezone

        from src.tracking.run_schema import RunRecord
        from src.tracking.tracker import ExperimentTracker

        t = ExperimentTracker({"backend": "file", "file_store": tmp})
        rid = t.new_run_id()
        rec = RunRecord(timestamp=datetime.now(timezone.utc).isoformat(), run_id=rid,
                        crop="tomato", architecture="mock_demo", kind="demo", status="demo",
                        accuracy=mm["accuracy"], macro_f1=mm["macro_f1"], split="test",
                        evaluation_samples=4, per_class_recall=mm.get("per_class_recall", {}))
        t.finish_run(rec)
        assert (Path(tmp) / f"{rid}.json").exists() and (Path(tmp) / "runs.csv").exists()
        print(f"7. experiment logging OK run={rid}")

    # 8. Grad-CAM++ (+ Score-CAM + saliency execute too)
    from src.explainability.xai_pipeline import run_xai

    r = run_xai(m, ad, Image.open(ip).convert("RGB"), x, methods=("gradcam_pp", "scorecam", "saliency"))
    assert "gradcam_pp" in r
    print(f"8. Grad-CAM++ OK (also ran scorecam+saliency): {list(r)}")

    # 9. comparison report generation (temp dir - never pollutes research outputs)
    with tempfile.TemporaryDirectory() as tmp:
        import pandas as pd

        from scripts.generate_comparison import build_comparison

        runs = pd.DataFrame([
            {"timestamp": "2026-01-01T00:00:00+00:00", "run_id": "run-1", "crop": "tomato",
             "architecture": "mock_demo", "kind": "eval", "split": "test", "accuracy": 0.5,
             "macro_f1": 0.4, "macro_precision": 0.4, "macro_recall": 0.4, "weighted_f1": 0.4,
             "per_class_recall": '{"0": 0.5, "1": 0.5}', "parameter_count": 1000, "flops": 0.0,
             "inference_latency_ms": 1.2, "throughput": 500.0, "evaluation_samples": 4,
             "checkpoint_path": "", "confusion_matrix_path": ""},
            {"timestamp": "2026-01-02T00:00:00+00:00", "run_id": "run-2", "crop": "tomato",
             "architecture": "resnet50", "kind": "eval", "split": "test", "accuracy": 0.6,
             "macro_f1": 0.5, "macro_precision": 0.5, "macro_recall": 0.5, "weighted_f1": 0.5,
             "per_class_recall": '{"0": 0.6, "1": 0.4}', "parameter_count": 25000000, "flops": 0.0,
             "inference_latency_ms": 3.0, "throughput": 200.0, "evaluation_samples": 4,
             "checkpoint_path": "", "confusion_matrix_path": ""},
        ])
        out_root = Path(tmp)
        s = build_comparison(runs, "tomato", out_root / "metrics", out_root, ["c0", "c1"])
        for f in ("csv", "json", "md", "html"):
            assert (out_root / "reports" / f"model_comparison_tomato.{f}").exists(), f
        assert s["best_macro_f1"]["model"] == "resnet50"
        assert (out_root / "figures" / "tomato" / "macro_f1_by_model.png").exists()
        print("9. comparison report OK -> reports/model_comparison_tomato.{csv,json,md,html} + figures/")

    # 10. Streamlit app imports
    import app

    assert app.__name__ == "app"
    print("10. Streamlit app import OK")

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/smoke_result.json").write_text(
        json.dumps({"pred": pred["label"], "xai": r,
                    "status": "DEMO — plumbing output, not a research result"},
                   indent=2, default=str))
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()