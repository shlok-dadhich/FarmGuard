"""Dataset report generation."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def dataset_report(df: pd.DataFrame, out: Path) -> dict:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    rep = {"rows": len(df), "classes": sorted(df["label"].unique().tolist()) if not df.empty else [],
           "counts": df["label"].value_counts().to_dict() if not df.empty else {}}
    out.write_text(json.dumps(rep, indent=2))
    return rep
