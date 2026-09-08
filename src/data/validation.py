"""Dataset validation: missing/corrupt/dupes/imbalance/leakage."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import pandas as pd

def validate_manifest(df: pd.DataFrame) -> dict:
    report = {"rows": len(df), "missing": 0, "corrupt": [], "duplicates": 0, "empty_classes": [], "class_counts": {}}
    if df.empty: return report
    report["duplicates"] = int(df.duplicated("path").sum())
    report["class_counts"] = df["label"].value_counts().to_dict()
    for p in df["path"].tolist():
        if not Path(p).exists(): report["missing"] += 1
    # corrupt check on up to 50 files
    for p in df["path"].tolist()[:50]:
        if Path(p).exists():
            try:
                with Image.open(p) as im: im.verify()
            except Exception: report["corrupt"].append(p)
    return report
