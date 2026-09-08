from __future__ import annotations
def build_optimizer(params, family="adamw", lr=3e-4, weight_decay=0.05):
    import torch
    family = family.lower()
    if family == "adamw": return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if family == "sgd": return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if family == "adam": return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {family}")
