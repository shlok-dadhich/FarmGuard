"""Dataloaders from manifests."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from src.data.datasets import CropDataset
from src.data.preprocessing import build_transforms

def loaders_from_splits(split_dir: Path, image_size=224, batch_size=32, num_workers=0):
    from src.data.preprocessing import build_transforms
    split_dir = Path(split_dir)
    out = {}
    for split in ("train","val","test"):
        p = split_dir / f"{split}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        tf = build_transforms(image_size, train=(split=="train"))
        ds = CropDataset(df, transform=tf)
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)
    return out
