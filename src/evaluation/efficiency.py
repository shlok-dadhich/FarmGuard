"""Efficiency-normalized analysis."""
from __future__ import annotations
def efficiency_row(accuracy, macro_f1, params, flops, latency_ms):
    m = max(1, params) / 1e6
    f = max(flops, 1e-6)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "params": params, "flops": flops,
        "latency_ms": latency_ms, "acc_per_mparams": accuracy / m, "f1_per_mparams": macro_f1 / m,
        "acc_per_gflop": accuracy / (f / 1e9), "f1_per_gflop": macro_f1 / (f / 1e9)}
def rank_models(rows: list, key="macro_f1"):
    return sorted(rows, key=lambda r: r.get(key, 0), reverse=True)
