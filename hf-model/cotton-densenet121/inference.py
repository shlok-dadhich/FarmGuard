"""Run inference with the FarmGuard cotton DenseNet-121 checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).parent


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python inference.py path/to/image.jpg")

    with (ROOT / "classes.json").open(encoding="utf-8") as file:
        classes = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(classes))
    checkpoint = torch.load(ROOT / "best_densenet.pth", map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = transform(Image.open(sys.argv[1]).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    print(f"Prediction: {classes[index]}")
    print(f"Confidence: {probabilities[index].item():.2%}")


if __name__ == "__main__":
    main()
