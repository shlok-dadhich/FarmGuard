"""Checkpoint loading compatibility tests."""

import torch
from torch import nn

from src.models.factory import load_checkpoint_weights


def test_load_checkpoint_weights_accepts_model_state_dict(tmp_path):
    source = nn.Linear(3, 2)
    checkpoint = tmp_path / "cotton.pth"
    torch.save({"model_state_dict": source.state_dict(), "epoch": 4}, checkpoint)

    target = nn.Linear(3, 2)
    load_checkpoint_weights(target, checkpoint)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)