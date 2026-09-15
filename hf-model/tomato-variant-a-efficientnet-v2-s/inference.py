import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms


REPO_DIR = Path(__file__).resolve().parent
MODEL_PATH = REPO_DIR / "model.pt"
CLASSES_PATH = REPO_DIR / "classes.json"


def load_model():
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_info = json.load(f)

    classes = class_info["classes"]

    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, classes


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])


def predict(image_path):
    model, classes = load_model()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted = probabilities.max(dim=1)

    return classes[predicted.item()], confidence.item()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/image.jpg")
        raise SystemExit(1)

    label, confidence = predict(sys.argv[1])

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
