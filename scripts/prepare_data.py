"""python scripts/prepare_data.py --crop tomato [--data-root ...]"""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
from src.core.config import load_config
from src.core.device import set_seed
from src.data.datasets import scan_imagefolder
from src.data.splitting import stratified_split, check_leakage
from src.data.validation import validate_manifest
from src.data.metadata import dataset_report
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", required=True); ap.add_argument("--data-root", default=None); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config(); set_seed(a.seed)
    info = cfg.datasets["crops"].get(a.crop, {})
    root = Path(a.data_root or info.get("root", f"data/raw/{a.crop}"))
    df = scan_imagefolder(root)
    print(f"scanned {len(df)} images from {root}")
    if df.empty:
        print("No images found - creating empty manifests so pipeline stays executable.")
        import pandas as pd
        df = pd.DataFrame(columns=["path","label","label_id"])
        out = Path(f"data/splits/{a.crop}"); out.mkdir(parents=True, exist_ok=True)
        for s in ("train","val","test"): df.to_csv(out / f"{s}.csv", index=False)
        return
    print(validate_manifest(df))
    ratios = cfg.datasets.get("split_ratios", {"train":0.75,"val":0.10,"test":0.15})
    splits = stratified_split(df, seed=a.seed, train_r=ratios["train"], val_r=ratios["val"], test_r=ratios["test"], out_dir=Path(f"data/splits/{a.crop}"))
    print({k: len(v) for k, v in splits.items()})
    print("leakage:", check_leakage(splits) or "none")
    dataset_report(df, Path(f"data/metadata/{a.crop}_report.json"))
if __name__ == "__main__": main()

