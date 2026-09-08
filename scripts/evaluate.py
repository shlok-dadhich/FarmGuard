"""python scripts/evaluate.py --crop tomato --model mock_demo --checkpoint path"""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
import pandas as pd
from src.core.config import load_config
from src.core.device import resolve_device
from src.data.datamodule import loaders_from_splits
from src.inference.model_loader import load_model
from src.models.factory import get_adapter
from src.services.evaluation_service import evaluate
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", default="tomato"); ap.add_argument("--model", default="mock_demo"); ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()
    cfg = load_config(); device = resolve_device(cfg.device)
    split_dir = Path(f"data/splits/{a.crop}")
    size = get_adapter(a.model).input_size()
    loaders = loaders_from_splits(split_dir, image_size=size, batch_size=32)
    if "test" not in loaders and "val" not in loaders:
        print("WHAT: no eval split found\nWHY: prepare-data/train not run\nHOW: python scripts/prepare_data.py --crop "+a.crop); return
    loader = loaders.get("test", loaders.get("val"))
    df = pd.read_csv(split_dir / ("test.csv" if "test" in loaders else "val.csv"))
    nc = int(df["label_id"].nunique()) if not df.empty else 4
    model = load_model(a.model, nc, a.checkpoint, device)
    r = evaluate(model, loader, device, get_adapter(a.model).parameter_count(model))
    out = Path(f"outputs/metrics/eval_{a.crop}_{a.model}.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r["metrics"], indent=2))
    print(json.dumps(r["metrics"], indent=2)); print(f"saved {out}")
if __name__ == "__main__": main()

