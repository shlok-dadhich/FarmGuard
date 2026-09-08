"""Benchmark latency/params across registry (mock-safe)."""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import time, torch
from src.core.config import load_config
from src.models.registry import MODEL_REGISTRY
from src.models.factory import build_model, get_adapter
def main():
    cfg = load_config()
    for arch in MODEL_REGISTRY:
        try:
            ad = get_adapter(arch); m = build_model(arch, 4).eval()
            s = ad.input_size(); d = torch.randn(1,3,s,s)
            _ = m(d); t=time.time()
            for _ in range(10): _ = m(d)
            print(f"{arch}: params={ad.parameter_count(m)} latency={(time.time()-t)/10*1000:.1f}ms")
        except Exception as e: print(f"{arch}: FAILED {e}")
if __name__ == "__main__": main()

