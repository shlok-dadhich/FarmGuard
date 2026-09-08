"""Model validation tests (scripts/validate_model.py)."""

from dataclasses import replace

from scripts.validate_model import validate_model


def test_validate_unknown_model(tmp_path):
    from src.core.config import load_config

    cfg = replace(load_config(),
                  tracking={"backend": "file", "file_store": str(tmp_path / "m")})
    report = validate_model("does_not_exist", "tomato", cfg)
    assert report["status"] == "error"
    assert report["checks"]["resolve"]["ok"] is False


def test_validate_mock_demo_missing_weights(tmp_path, monkeypatch):
    from src.core.config import load_config
    from scripts import validate_model as vm

    monkeypatch.setattr(vm, "discover_checkpoint", lambda *a, **k: None)  # force no weights
    cfg = replace(load_config(),
                  tracking={"backend": "file", "file_store": str(tmp_path / "m")})
    report = validate_model("mock_demo", "tomato", cfg)
    # no supplied checkpoint -> clearly reported, not failed
    assert report["status"] == "ok"
    assert report["checks"]["forward"]["ok"] is True
    assert report["checks"]["target_layers"]["ok"] is True
    assert any("DEMO" in w or "weights not supplied" in w for w in report["warnings"])


def test_validate_corrupt_checkpoint_fails(tmp_path):
    from src.core.config import load_config

    bad = tmp_path / "corrupt.pt"
    bad.write_bytes(b"not a torch checkpoint")
    cfg = replace(load_config(),
                  tracking={"backend": "file", "file_store": str(tmp_path / "m")},
                  models_cfg={"models": load_config().models_cfg.get("models", {}),
                              "instances": [
                                  {"id": "broken", "architecture": "mock_demo",
                                   "crop": "tomato", "checkpoint": str(bad)},
                              ]})
    report = validate_model("broken", "tomato", cfg)
    assert report["status"] == "error"
    assert report["checks"]["load"]["ok"] is False