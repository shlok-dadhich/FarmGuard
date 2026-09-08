from __future__ import annotations
import torch
from src.evaluation.metrics import compute_metrics
@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
    model.eval().to(device); ys, ps, probs = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        p = torch.softmax(out, 1).cpu().tolist()
        probs += p; ps += torch.argmax(out, 1).cpu().tolist(); ys += y.tolist()
    return {"metrics": compute_metrics(ys, ps), "y_true": ys, "y_pred": ps, "probs": probs}
