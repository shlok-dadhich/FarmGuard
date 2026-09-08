def test_config_loads():
    from src.core.config import load_config
    cfg = load_config(); cfg.validate()
    assert cfg.seed == 42 and "mock_demo" in cfg.models_cfg["models"]
