"""Batch evaluation tests (scripts/evaluate_all.py)."""

from dataclasses import replace

import pandas as pd

from src.core.config import load_config
from src.utils.image import make_demo_image


def _make_manifest(tmp_path, n_per_class=3):
    split_dir = tmp_path / "data" / "splits" / "tomato"
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in range(4):
        for i in range(n_per_class):
            p = tmp_path / "data" / "raw" / "tomato" / f"class{c}" / f"img{i}.jpg"
            make_demo_image(p, (64, 64))
            rows.append({"path": str(p), "label": f"class{c}", "label_id": c})
    pd.DataFrame(rows).to_csv(split_dir / "test.csv", index=False)
    return split_dir


def test_run_evaluations_ok_and_logged(tmp_path):
    from scripts.evaluate_all import run_evaluations

    _make_manifest(tmp_path)
    cfg = replace(load_config(),
                  tracking={"backend": "file", "file_store": str(tmp_path / "metrics")},
                  project={**load_config().project, "output_dir": str(tmp_path / "out")})
    s = run_evaluations("tomato", "test", models_arg="mock_demo", cfg=cfg,
                        device="cpu", out_root=tmp_path / "out")
    row = s["rows"][0]
    assert row["status"] == "demo"  # no checkpoint -> clearly labelled demo, never fabricated
    assert row["accuracy"] is not None and 0 <= row["accuracy"] <= 1
    assert row["macro_f1"] is not None
    assert (tmp_path / "metrics" / "runs.csv").exists()          # experiment appended (tracker store)
    assert (tmp_path / "out" / "metrics" / "eval_tomato_mock_demo.json").exists()  # eval record (out_root)
    assert (tmp_path / "out" / "confusion_matrices").exists()


def test_continue_on_error_and_failure_recorded(tmp_path):
    from scripts.evaluate_all import run_evaluations

    _make_manifest(tmp_path)
    bad_ckpt = tmp_path / "corrupt.pt"
    bad_ckpt.write_bytes(b"this is not a torch checkpoint")
    cfg = replace(load_config(),
                  tracking={"backend": "file", "file_store": str(tmp_path / "metrics")},
                  project={**load_config().project, "output_dir": str(tmp_path / "out")},
                  models_cfg={"models": load_config().models_cfg.get("models", {}),
                              "instances": [
                                  {"id": "ok_model", "architecture": "mock_demo", "crop": "tomato"},
                                  {"id": "broken_model", "architecture": "mock_demo", "crop": "tomato",
                                   "checkpoint": str(bad_ckpt)},
                              ]})
    s = run_evaluations("tomato", "test", models_arg="ok_model,broken_model", cfg=cfg,
                        device="cpu", out_root=tmp_path / "out")
    rows = {r["model_key"]: r for r in s["rows"]}
    assert rows["ok_model"]["status"] == "demo"
    assert rows["broken_model"]["status"] == "failed"
    assert "reason" in rows["broken_model"] and rows["broken_model"]["reason"]
    assert s["total"] == 2 and s["ok"] == 1 and s["failed"] == 1


def test_no_manifest_is_failure_not_crash(tmp_path):
    from scripts.evaluate_all import run_evaluations

    cfg = replace(load_config(),
                  tracking={"backend": "file", "file_store": str(tmp_path / "metrics")},
                  project={**load_config().project, "output_dir": str(tmp_path / "out")})
    s = run_evaluations("tomato", "test", models_arg="mock_demo", cfg=cfg,
                        device="cpu", out_root=tmp_path / "out")
    assert s["rows"][0]["status"] == "failed"
    assert "prepare_data" in s["rows"][0]["reason"]