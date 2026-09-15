import os
import shutil
import random
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# Root where the script expects the data
ROOT = Path(r"C:\Users\shlok\OneDrive\Documents\FarmGuard")
DATA_ROOT = ROOT / "data" / "raw" / "Cotton_data"

# Existing folder containing all images (augmented/original collection)
SRC = DATA_ROOT / "Cotton-Original-Augmented"

# Desired split ratios (train / val / test)
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Ensure split directories exist
for split in SPLITS:
    (DATA_ROOT / split).mkdir(parents=True, exist_ok=True)

# Iterate over each class folder inside the source directory
for class_dir in SRC.iterdir():
    if not class_dir.is_dir():
        continue
    # Gather all image files recursively (any extension) under the class directory
    image_files = [p for p in class_dir.rglob("*") if p.is_file()]
    random.shuffle(image_files)
    n = len(image_files)
    train_end = int(n * SPLITS["train"]) 
    val_end = train_end + int(n * SPLITS["val"])

    split_mapping = {
        "train": image_files[:train_end],
        "val":   image_files[train_end:val_end],
        "test":  image_files[val_end:],
    }

    for split, files in split_mapping.items():
        dest_dir = DATA_ROOT / split / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest_dir / f.name)

print("[OK] Data split created under:", DATA_ROOT)
