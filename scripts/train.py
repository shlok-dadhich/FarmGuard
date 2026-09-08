"""python scripts/train.py --crop tomato --model mock_demo --seed 42 [--epochs 2]"""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import argparse, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.core.config import load_config
from src.core.device import set_seed, resolve_device
from src.data.datamodule import loaders_from_splits
from src.data.datasets import scan_imagefolder
from src.data.splitting import stratified_split
from src.models.factory import build_model, get_adapter
from src.training.trainer import train_one_run
from src.tracking.tracker import get_tracker
from src.tracking.run_schema import RunRecord
from src.utils.reproducibility import env_report
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--crop", default="tomato"); ap.add_argument("--model", default="mock_demo")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(); set_seed(a.seed)
    device = resolve_device(cfg.device)
    split_dir = Path(f"data/splits/{a.crop}")
    if not (split_dir / "train.csv").exists():
        print("No manifests - building synthetic demo manifests so training stays executable (DEMO).")
        from src.utils.image import make_demo_image
        import pandas as pd
        rows = []
        for c in range(4):
            for i in range(20):
                p = Path(f"data/raw/{a.crop}/class{c}/img{i}.jpg"); make_demo_image(p, (96,96))
                rows.append({"path": str(p), "label": f"class{c}", "label_id": c})
        df = pd.DataFrame(rows)
        ratios = cfg.datasets.get("split_ratios", {"train":0.75,"val":0.10,"test":0.15})
        stratified_split(df, seed=a.seed, train_r=ratios["train"], val_r=ratios["val"], test_r=ratios["test"], out_dir=split_dir)
    bs = a.batch_size or cfg.training.get("batch_size", 16)
    size = get_adapter(a.model).input_size()
    loaders = loaders_from_splits(split_dir, image_size=size, batch_size=bs)
    # num classes from train manifest
    df = pd.read_csv(split_dir / "train.csv")
    nc = int(df["label_id"].nunique()) if not df.empty else 4
    model = build_model(a.model, nc)
    tracker = get_tracker(cfg.tracking)
    run_id = tracker.new_run_id()
    tcfg = {"training": cfg.training, "epochs_override": a.epochs or cfg.training.get("screening",{}).get("epochs",2), "batch_size_override": bs}
    res = train_one_run(model, loaders, tcfg, device, tracker, run_id, a.model, a.crop, a.seed)
    vm = res.get("val_metrics", {})
    # latency
    import torch, time
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, size, size).to(device)
        _ = model(dummy)
        s = time.time()
        for _ in range(20): _ = model(dummy)
        lat = (time.time()-s)/20*1000
    rec = RunRecord(timestamp=datetime.now(timezone.utc).isoformat(), run_id=run_id, crop=a.crop, architecture=a.model, seed=a.seed,
        image_size=size, batch_size=bs, epochs=tcfg["epochs_override"], optimizer=cfg.training.get("optimizer",{}).get("family","adamw"),
        learning_rate=float(cfg.training.get("optimizer",{}).get("lr",3e-4)), scheduler=cfg.training.get("scheduler",{}).get("family","cosine"),
        weight_decay=float(cfg.training.get("optimizer",{}).get("weight_decay",0.05)), augmentation_version=cfg.datasets.get("augmentation_version","aug-v1"),
        train_loss=res["train_loss"], val_loss=0.0, accuracy=vm.get("accuracy",0), macro_f1=vm.get("macro_f1",0),
        parameter_count=get_adapter(a.model).parameter_count(model), flops=0.0, training_time_s=res["training_time_s"],
        inference_latency_ms=lat, checkpoint_path=res["checkpoint"])
    tracker.finish_run(rec, artifacts=[res["checkpoint"]])
    print(f"Done run={run_id} acc={rec.accuracy:.3f} f1={rec.macro_f1:.3f} ckpt={res['checkpoint']}")
    print("env:", env_report().get("commit"))
if __name__ == "__main__": main()

