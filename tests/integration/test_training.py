def test_train_loop_cpu():
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.factory import build_model
    from src.training.trainer import train_one_run
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(24,3,64,64), torch.randint(0,4,(24,)))
    loaders = {"train": DataLoader(ds, batch_size=8), "val": DataLoader(ds, batch_size=8)}
    m = build_model("mock_demo", 4)
    res = train_one_run(m, loaders, {"training":{"optimizer":{"family":"adamw","lr":1e-3,"weight_decay":0.0},"scheduler":{"family":"cosine"},"batch_size":8,"early_stopping":{"patience":5},"mixed_precision":False},"epochs_override":1}, device="cpu")
    assert "checkpoint" in res
