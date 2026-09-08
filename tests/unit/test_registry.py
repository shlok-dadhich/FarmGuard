def test_registry():
    from src.models.registry import MODEL_REGISTRY
    from src.models.factory import get_adapter, build_model
    assert set(["efficientnet_v2_s","resnet50","convnext_tiny","regnet_y_4gf","densenet121","mock_demo"]) <= set(MODEL_REGISTRY)
    m = build_model("mock_demo", 3)
    assert get_adapter("mock_demo").parameter_count(m) > 0
