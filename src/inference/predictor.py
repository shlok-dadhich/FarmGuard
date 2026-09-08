from __future__ import annotations
import torch
from PIL import Image
from src.inference.preprocessing import preprocess_pil
class Predictor:
    def __init__(self, model, class_names, device="cpu", demo=False, arch="", crop=""):
        self.model = model; self.class_names = class_names; self.device = device
        self.demo = demo; self.arch = arch; self.crop = crop
    @torch.no_grad()
    def predict(self, img: Image.Image, top_k=3):
        from src.inference.preprocessing import preprocess_pil
        size = 64 if self.demo else 224
        try: size = self.model_input_size()
        except Exception: pass
        x = preprocess_pil(img, size).to(self.device)
        probs = torch.softmax(self.model(x), 1)[0].cpu().tolist()
        idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        return {"label": self.class_names[idx[0]] if self.class_names else str(idx[0]),
            "confidence": probs[idx[0]], "top_k": [(self.class_names[i] if self.class_names else str(i), probs[i]) for i in idx],
            "probs": probs, "demo": self.demo}
    def model_input_size(self):
        return 224
