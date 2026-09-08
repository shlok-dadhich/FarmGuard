"""Model Evaluation page: run dataset-level evaluation from the UI.

Workflow: select crop -> model -> split -> inspect dataset summary -> Run evaluation.
Results: headline metrics, per-class precision/recall/F1, confusion matrix (raw +
row-normalized), sklearn classification report, downloads (CSV/JSON/PNG/HTML).
Every evaluation is appended to the experiment history (kind=eval) — never overwrites.
"""
from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.common import (
    demo_banner,
    error_box,
    get_class_names,
    get_crops,
    get_device,
    get_model,
    get_model_choices,
    load_app_config,
    model_info,
    unavailable_banner,
)
from ui.theme import inject

inject()
st.title("🧪 Model Evaluation")
st.caption("Dataset-level evaluation with macro F1 as the primary metric. Results are appended to experiment history.")

cfg = load_app_config()
crops = get_crops()
device = get_device()

c1, c2, c3 = st.columns(3)
with c1:
    crop = st.selectbox("Crop", crops, key="eval_crop")
with c2:
    choices = get_model_choices(crop)
    key = st.selectbox("Model", choices, key="eval_model")
with c3:
    split_dir = Path(f"data/splits/{crop}")
    splits = [s for s in ("train", "val", "test") if (split_dir / f"{s}.csv").exists()]
    if not splits:
        unavailable_banner(f"no split manifests for crop '{crop}'. Run: python scripts/prepare_data.py --crop {crop}")
        st.stop()
        raise SystemExit
    split = st.selectbox("Dataset split", splits, key="eval_split")

info = model_info(crop, key)
class_names = get_class_names(crop)

# ---------------------------------------------------------------- dataset summary
manifest = split_dir / f"{split}.csv"
try:
    mdf = pd.read_csv(manifest)
except Exception as e:
    error_box(e)
    st.stop()
n_samples = len(mdf)
n_classes = int(mdf["label_id"].nunique()) if n_samples else 0

if mdf.empty:
    unavailable_banner(f"split manifest {split}.csv for crop '{crop}' is empty.")
    st.stop()

with st.expander(f"📋 Dataset summary — {n_samples} samples, {n_classes} classes", expanded=False):
    dist = mdf.groupby(["label_id", "label"]).size().reset_index(name="count")
    st.dataframe(dist, width="stretch", hide_index=True)
    st.json({"crop": crop, "split": split, "samples": n_samples, "classes": n_classes,
             "class_names": class_names, "manifest": str(manifest)})

if info["demo"]:
    demo_banner()
if info["is_instance"]:
    st.caption(f"Configured instance: **{info['name']}** (architecture {info['architecture']})")

# ---------------------------------------------------------------- run evaluation
if st.button("▶️ Run evaluation", type="primary", key="eval_go"):
    try:
        import torch

        from src.data.datamodule import loaders_from_splits
        from src.evaluation.confusion_matrix import compute_cm
        from src.evaluation.evaluator import evaluate_model
        from src.evaluation.figures import plot_confusion_matrix
        from src.evaluation.metrics import compute_metrics
        from src.tracking.run_schema import RunRecord
        from src.tracking.tracker import get_tracker
        from src.utils.reproducibility import env_report

        with st.spinner(f"Evaluating {info['name']} on {split} ({n_samples} images)…"):
            loaders = loaders_from_splits(split_dir, image_size=info["input_size"], batch_size=32)
            loader = loaders[split]
            model = get_model(info["architecture"], n_classes, info["checkpoint"], device)
            t0 = time.time()
            res = evaluate_model(model, loader, device)
            wall = max(time.time() - t0, 1e-6)
            metrics = res["metrics"]
            latency_ms = wall / max(1, n_samples) * 1000
            throughput = n_samples / wall
            ys, ps = res["y_true"], res["y_pred"]
            cm = compute_cm(ys, ps, labels=list(range(n_classes)))

            # per-class precision/recall/F1
            from sklearn.metrics import f1_score, precision_score, recall_score

            pcls = precision_score(ys, ps, average=None, labels=list(range(n_classes)), zero_division=0)
            rcls = recall_score(ys, ps, average=None, labels=list(range(n_classes)), zero_division=0)
            fcls = f1_score(ys, ps, average=None, labels=list(range(n_classes)), zero_division=0)
            per_class = pd.DataFrame({
                "class": class_names[:n_classes], "precision": pcls, "recall": rcls, "f1": fcls,
            })

            # save artifacts
            out = Path(cfg.project.get("output_dir", "outputs"))
            metrics_dir = out / "metrics"
            cm_dir = out / "confusion_matrices"
            reports_dir = out / "reports"
            for d in (metrics_dir, cm_dir, reports_dir):
                d.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            cm_png = cm_dir / f"cm_{crop}_{info['architecture']}_{split}_{ts}.png"
            cm_norm_png = cm_dir / f"cm_{crop}_{info['architecture']}_{split}_{ts}_norm.png"
            plot_confusion_matrix(cm, class_names[:n_classes], f"{info['name']} ({crop}/{split})", cm_png)
            plot_confusion_matrix(cm, class_names[:n_classes], f"{info['name']} ({crop}/{split})", cm_norm_png, normalize=True)
            cm_json = cm_dir / f"cm_{crop}_{info['architecture']}_{split}_{ts}.json"
            cm_json.write_text(json.dumps({"confusion_matrix": cm, "labels": class_names[:n_classes],
                                           "crop": crop, "architecture": info["architecture"],
                                           "split": split, "demo": info["demo"]}, indent=2))

            # log experiment run (append-only history)
            tracker = get_tracker(cfg.tracking)
            env = env_report()
            run_id = tracker.new_run_id()
            rec = RunRecord(
                timestamp=datetime.now(timezone.utc).isoformat(), run_id=run_id,
                crop=crop, architecture=info["architecture"], dataset_name=crop, split=split,
                image_size=info["input_size"], batch_size=32, accuracy=metrics["accuracy"],
                macro_f1=metrics["macro_f1"], weighted_f1=metrics["weighted_f1"],
                per_class_recall=metrics.get("per_class_recall", {}),
                parameter_count=int(sum(p.numel() for p in model.parameters())),
                inference_latency_ms=latency_ms, throughput=throughput,
                evaluation_samples=n_samples, confusion_matrix_path=str(cm_json),
                checkpoint_path=info["checkpoint"] or "", device=device,
                software_version=str(env.get("torch", "")),
                git_commit=str(env.get("commit", "")), kind="eval",
                status="demo" if info["demo"] else "ok",
            )
            tracker.finish_run(rec, artifacts=[str(cm_png)])

            eval_rec = {"metrics": metrics, "confusion_matrix": cm, "per_class": per_class.to_dict("records"),
                        "latency_ms": latency_ms, "throughput": throughput, "n": n_samples,
                        "split": split, "crop": crop, "architecture": info["architecture"],
                        "run_id": run_id, "status": rec.status, "demo": info["demo"]}
            (metrics_dir / f"eval_{crop}_{info['architecture']}.json").write_text(
                json.dumps(eval_rec, indent=2, default=str))

            st.session_state["eval"] = {"metrics": metrics, "per_class": per_class, "cm": cm,
                                        "cm_png": str(cm_png), "cm_norm_png": str(cm_norm_png),
                                        "latency_ms": latency_ms, "throughput": throughput,
                                        "n": n_samples, "run_id": run_id, "class_names": class_names[:n_classes],
                                        "ys": ys, "ps": ps, "status": rec.status}
    except Exception as e:
        error_box(e)

res = st.session_state.get("eval")
if res is None:
    st.info("Select a model and split, then press **Run evaluation**. No metrics are shown until an evaluation has actually run.")
    st.stop()
    raise SystemExit

# ---------------------------------------------------------------- results
m = res["metrics"]
if res["status"] == "demo":
    demo_banner()
st.markdown(f"#### Results — run `{res['run_id']}` on {res['n']} images")
cols = st.columns(5)
cols[0].metric("Accuracy", f"{m['accuracy']:.4f}")
cols[1].metric("Macro Precision", f"{m['macro_precision']:.4f}")
cols[2].metric("Macro Recall", f"{m['macro_recall']:.4f}")
cols[3].metric("Macro F1", f"{m['macro_f1']:.4f}", delta=None)
cols[4].metric("Weighted F1", f"{m['weighted_f1']:.4f}")
c1, c2 = st.columns(2)
c1.metric("Latency / image", f"{res['latency_ms']:.2f} ms")
c2.metric("Throughput", f"{res['throughput']:.1f} img/s")

# per-class table
st.markdown("#### Per-class metrics")
st.dataframe(res["per_class"].style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}),
             width="stretch", hide_index=True)

# confusion matrix
st.markdown("#### Confusion matrix")
mode = st.radio("View", ["raw counts", "normalized (%)"], horizontal=True, key="eval_cm_mode")
img_path = res["cm_png"] if mode == "raw counts" else res["cm_norm_png"]
st.image(img_path, width="stretch", caption=f"Confusion matrix — {mode}")

# classification report
with st.expander("📄 Classification report (sklearn)", expanded=False):
    from sklearn.metrics import classification_report

    st.text(classification_report(res["ys"], res["ps"], labels=list(range(len(res["class_names"]))),
                                  target_names=res["class_names"], zero_division=0, digits=4))

# downloads
st.markdown("#### Downloads")
csv_bytes = res["per_class"].to_csv(index=False).encode()
json_bytes = json.dumps({"metrics": m, "per_class": res["per_class"].to_dict("records"),
                         "confusion_matrix": res["cm"], "n": res["n"], "run_id": res["run_id"]},
                        indent=2, default=str).encode()
html_bytes = (f"<h1>AgroVision evaluation — {crop}/{info['architecture']}/{split}</h1>"
              f"<p>run {res['run_id']} · {res['n']} samples · status {res['status']}</p>"
              f"<table border=1 cellpadding=5>{pd.Series(m).to_frame('value').to_html()}</table>"
              f"<img src='data:image/png;base64,{base64.b64encode(Path(res['cm_png']).read_bytes()).decode()}' "
              f"style='max-width:100%;'/></body></html>").encode()
d1, d2, d3, d4 = st.columns(4)
d1.download_button("CSV (per-class)", csv_bytes, f"eval_{crop}_{info['architecture']}_{split}.csv",
                   "text/csv", key="eval_dl_csv")
d2.download_button("JSON (full)", json_bytes, f"eval_{crop}_{info['architecture']}_{split}.json",
                   "application/json", key="eval_dl_json")
d3.download_button("PNG (confusion matrix)", Path(res["cm_png"]).read_bytes(),
                   f"cm_{crop}_{info['architecture']}_{split}.png", "image/png", key="eval_dl_png")
d4.download_button("HTML report", html_bytes, f"eval_{crop}_{info['architecture']}_{split}.html",
                   "text/html", key="eval_dl_html")
st.caption("Artifacts also saved under outputs/metrics/, outputs/confusion_matrices/ and logged to experiment history (outputs/metrics/runs.csv).")