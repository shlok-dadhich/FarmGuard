from __future__ import annotations
from src.evaluation.evaluator import evaluate_model
from src.evaluation.efficiency import efficiency_row
def evaluate(model, loader, device="cpu", params=0, flops=0.0, latency_ms=0.0):
    r = evaluate_model(model, loader, device)
    m = r["metrics"]
    r["efficiency"] = efficiency_row(m["accuracy"], m["macro_f1"], params, flops, latency_ms)
    return r
