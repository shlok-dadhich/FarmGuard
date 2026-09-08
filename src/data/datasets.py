"""Dataset abstraction: ImageFolder-style crop datasets + CSV manifests."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def scan_imagefolder(root: Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["path", "label", "label_id"])
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    for i, c in enumerate(classes):
        for p in sorted((root / c).rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                rows.append({"path": str(p), "label": c, "label_id": i})
    df = pd.DataFrame(rows, columns=["path", "label", "label_id"])
    return df


class CropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_id"])
