"""End-to-end smoke: config, demo image, mock inference, gradcam, explanation, json, app import."""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import json
from pathlib import Path
def main():
    from src.core.config import load_config
    cfg = load_config(); cfg.validate(); print("1. config OK")
    from src.utils.image import make_demo_image
    ip = make_demo_image(Path("outputs/demo_leaf.jpg")); print("2. demo image OK", ip)
    from src.models.factory import build_model, get_adapter
    ad = get_adapter("mock_demo"); m = build_model("mock_demo", 4); print("3. mock model OK params=", ad.parameter_count(m))
    from PIL import Image
    from src.inference.predictor import Predictor
    pred = Predictor(m, [f"c{i}" for i in range(4)], demo=True).predict(Image.open(ip).convert("RGB"))
    assert "label" in pred and "probs" in pred; print("4-5. inference OK", pred["label"], f"{pred['confidence']:.2f}")
    import torch
    from src.inference.preprocessing import preprocess_pil
    x = preprocess_pil(Image.open(ip).convert("RGB"), ad.input_size())
    from src.explainability.xai_pipeline import run_xai
    r = run_xai(m, ad, Image.open(ip).convert("RGB"), x, methods=("gradcam_pp","scorecam"))
    print("6. XAI OK", r)
    from src.ai.explanation import explain_prediction
    txt = explain_prediction({"label": pred["label"], "confidence": pred["confidence"], "top_k": pred["top_k"], "crop": "tomato", "model": "mock_demo"}, cfg.ai)
    assert len(txt) > 20; print("7. explanation OK")
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/smoke_result.json").write_text(json.dumps({"pred": pred["label"], "xai": r}, indent=2, default=str)); print("8. result JSON OK")
    import app; print("9. app import OK")
    print("SMOKE PASSED")
if __name__ == "__main__": main()

