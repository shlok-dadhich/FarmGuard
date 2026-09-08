from __future__ import annotations
from PIL import Image
from src.inference.model_loader import load_model
from src.inference.predictor import Predictor
def predict_image(arch, ckpt, image: Image.Image, class_names, device="cpu", demo=False, crop="", top_k=3):
    m = load_model(arch, max(1, len(class_names) or 4), ckpt, device)
    return Predictor(m, class_names, device, demo=demo, arch=arch, crop=crop).predict(image, top_k)
