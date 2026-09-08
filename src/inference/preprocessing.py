from __future__ import annotations
from PIL import Image
def preprocess_pil(img: Image.Image, size=224):
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return tf(img.convert("RGB")).unsqueeze(0)
