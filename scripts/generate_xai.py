"""python scripts/generate_xai.py --image path --model mock_demo [--method gradcam_pp|scorecam|both]"""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse
from PIL import Image
from src.core.config import load_config
from src.models.factory import build_model, get_adapter
from src.inference.preprocessing import preprocess_pil
from src.explainability.xai_pipeline import run_xai
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--model", default="mock_demo");    ap.add_argument("--method", default="both",
                    choices=["both", "gradcam_pp", "scorecam", "saliency"],
                    help="XAI method; 'both' runs gradcam_pp + scorecam + saliency")
    a = ap.parse_args()
    cfg = load_config()
    ad = get_adapter(a.model); model = build_model(a.model, 4)
    img = Image.open(a.image).convert("RGB")
    x = preprocess_pil(img, ad.input_size())
    methods = ("gradcam_pp", "scorecam", "saliency") if a.method == "both" else (a.method,)
    print(run_xai(model, ad, img, x, methods=methods, meta={"model": a.model}))
if __name__ == "__main__": main()

