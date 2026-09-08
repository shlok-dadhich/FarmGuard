"""Model instance configuration tests (configs/models.yaml -> instances:)."""

import pytest

from src.models.metadata import class_names_for_crop, load_model_specs, models_for_crop


def _cfg(instances):
    return {"models": {"resnet50": {}, "mock_demo": {}}, "instances": instances}


def test_load_instances_and_crop_filtering():
    specs = load_model_specs(_cfg([
        {"id": "r1", "architecture": "resnet50", "crop": "tomato", "checkpoint": "x.pt"},
        {"id": "r2", "architecture": "mock_demo", "crops": ["tomato", "potato"], "input_size": 96},
    ]))
    assert len(specs) == 2
    assert [s.id for s in models_for_crop(specs, "tomato")] == ["r1", "r2"]
    assert [s.id for s in models_for_crop(specs, "potato")] == ["r2"]
    assert specs[1].input_size == 96
    assert specs[1].display_name == "r2"


def test_unknown_architecture_raises():
    with pytest.raises(Exception, match="unknown architecture"):
        load_model_specs(_cfg([{"id": "x", "architecture": "does_not_exist", "crop": "tomato"}]))


def test_duplicate_id_raises():
    with pytest.raises(Exception, match="[Dd]uplicate"):
        load_model_specs(_cfg([
            {"id": "a", "architecture": "resnet50", "crop": "tomato"},
            {"id": "a", "architecture": "mock_demo", "crop": "tomato"},
        ]))


def test_missing_id_and_crop_raise():
    with pytest.raises(Exception, match="id"):
        load_model_specs(_cfg([{"architecture": "resnet50", "crop": "tomato"}]))
    with pytest.raises(Exception, match="crop"):
        load_model_specs(_cfg([{"id": "a", "architecture": "resnet50"}]))


def test_class_names_priority(tmp_path):
    classes_file = tmp_path / "tomato.yaml"
    classes_file.write_text("classes:\n  - Bacterial_spot\n  - Healthy\n")
    specs = load_model_specs(_cfg([
        {"id": "r1", "architecture": "resnet50", "crop": "tomato", "classes_file": str(classes_file)},
    ]))
    names = class_names_for_crop("tomato", specs, {"crops": {"tomato": {"classes": ["a", "b"]}}})
    assert names == ["Bacterial_spot", "Healthy"]  # classes_file beats datasets.yaml


def test_class_names_fallback():
    assert class_names_for_crop("tomato", [], {"crops": {"tomato": {"classes": ["x", "y"]}}}) == ["x", "y"]
    assert class_names_for_crop("tomato", [], {"crops": {"tomato": {}}})[0].startswith("class")