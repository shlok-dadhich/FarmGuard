"""Reusable training loop with tracking hooks."""
from __future__ import annotations
import time
from pathlib import Path
import torch
from src.evaluation.metrics import compute_metrics
from src.training.checkpointing import save_checkpoint
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.losses import build_loss
from src.training.callbacks import EarlyStopping

def train_one_run(model, loaders, cfg, device="cpu", tracker=None, run_id="demo", arch="mock_demo", crop="tomato", seed=42):
    t = cfg.get("training", {}); epochs = int(cfg.get("epochs_override", t.get("screening", {}).get("epochs", 2)))
    bs = int(cfg.get("batch_size_override", t.get("batch_size", 16)))
    opt_cfg, sch_cfg = t.get("optimizer", {}), t.get("scheduler", {})
    model.to(device)
    opt = build_optimizer(model.parameters(), opt_cfg.get("family","adamw"), float(opt_cfg.get("lr",3e-4)), float(opt_cfg.get("weight_decay",0.05)))
    sch = build_scheduler(opt, sch_cfg.get("family","cosine"), epochs)
    loss_fn = build_loss()
    es = EarlyStopping(t.get("early_stopping", {}).get("patience", 7))
    train_loader, val_loader = loaders.get("train"), loaders.get("val")
    best_f1, best_state, start = -1, None, time.time()
    history = []
    use_amp = device.startswith("cuda") and t.get("mixed_precision", True)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    for ep in range(epochs):
        model.train(); tot, n = 0.0, 0
        for x, y in (train_loader or []):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x); loss = loss_fn(out, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                out = model(x); loss = loss_fn(out, y); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sch.step()
        # val
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in (val_loader or []):
                out = model(x.to(device))
                ps += out.argmax(1).cpu().tolist(); ys += y.tolist()
        m = compute_metrics(ys, ps) if ys else {"accuracy":0,"macro_f1":0,"macro_precision":0,"macro_recall":0,"per_class_recall":{}}
        tl = tot / max(1, n)
        history.append({"epoch": ep, "train_loss": tl, **m})
        if tracker: tracker.log_epoch(run_id, ep, {"train_loss": tl, **m})
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if es.step(m["macro_f1"]): break
    if best_state: model.load_state_dict(best_state)
    ckpt = save_checkpoint(model, Path("outputs/checkpoints") / f"{crop}_{arch}_{run_id}.pt",
                           {"crop": crop, "arch": arch, "seed": seed})
    return {"history": history, "best_macro_f1": best_f1, "checkpoint": ckpt,
            "train_loss": history[-1]["train_loss"] if history else 0.0,
            "val_metrics": {k: history[-1][k] for k in ("accuracy","macro_f1") if history} if history else {},
            "training_time_s": time.time() - start}
