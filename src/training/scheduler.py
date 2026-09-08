from __future__ import annotations
def build_scheduler(opt, family="cosine", epochs=25, min_lr=1e-6):
    import torch
    if family == "cosine": return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)
    if family in ("none","constant"): return torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0)
    if family == "step": return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1,epochs//3))
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=min_lr)
