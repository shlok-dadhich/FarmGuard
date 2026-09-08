"""Model comparison report generator.

Usage:
    python scripts/generate_comparison.py --crop tomato
    python scripts/generate_comparison.py --all

Reads experiment tracking data (outputs/metrics/runs.csv + eval_*.json) and
writes per-crop comparison artifacts:

    outputs/reports/model_comparison_<crop>.{csv,json,md,html}
    outputs/figures/<crop>/accuracy_by_model.png, macro_f1_by_model.png,
        prf_grouped.png, params_vs_f1.png, flops_vs_f1.png, latency_vs_f1.png,
        radar.png (>=3 models), per_class_recall.png

The report contains: EXECUTIVE SUMMARY, BEST BY ACCURACY, BEST BY MACRO F1,
BEST BY EFFICIENCY, PER-CLASS ANALYSIS, CONFUSION MATRIX ANALYSIS, MODEL
AGREEMENT, XAI RESULTS, LATENCY / COMPUTATIONAL ANALYSIS, FULL RAW RESULTS.

Only measured results are reported; anything absent is marked UNAVAILABLE RESULT.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.core.config import load_config
from src.evaluation.efficiency import efficiency_row


def _latest_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Latest run per (crop, architecture), preferring eval runs over train runs."""
    if df.empty:
        return df
    df = df.copy()
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["_ts"] = pd.Timestamp(0)
    if "kind" in df.columns:
        # eval runs carry the most recent *measured* numbers; train runs are training history
        df["_kind_rank"] = df["kind"].map({"eval": 0, "train": 1}).fillna(2)
        df = df.sort_values(["_kind_rank", "_ts"], ascending=[True, False])
        df = df.groupby(["crop", "architecture"], as_index=False).head(1)
    else:
        df = df.sort_values("_ts").groupby(["crop", "architecture"], as_index=False).tail(1)
    if "_kind_rank" in df.columns:
        return df.drop(columns=["_ts", "_kind_rank"])
    return df.drop(columns=["_ts"])


def _fig_embed(path: Path) -> str:
    if not Path(path).exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _xai_inventory(out_root: Path) -> dict:
    xai_dir = out_root / "xai"
    if not xai_dir.exists():
        return {"count": 0, "latest": [], "faithfulness": []}
    imgs = sorted(xai_dir.glob("*.jpg")) + sorted(xai_dir.glob("*.png"))
    metas = sorted(xai_dir.glob("*_meta.json"))
    ff = sorted(xai_dir.glob("*faithfulness*.json")) + sorted(xai_dir.glob("*_faithfulness*.json"))
    return {"count": len(imgs), "latest": [p.name for p in imgs[-6:]], "faithfulness": [p.name for p in ff]}


def build_comparison(runs_df: pd.DataFrame, crop: str, metrics_dir: Path,
                     out_root: Path, class_names: list[str] | None = None) -> dict:
    """Generate all comparison artifacts for one crop from a runs DataFrame.

    Returns a summary dict. Writes files under out_root (reports/figures/...).
    """
    out_root = Path(out_root)
    metrics_dir = Path(metrics_dir)
    reports_dir = out_root / "reports"
    figures_dir = out_root / "figures" / crop
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if runs_df is None or runs_df.empty:
        note = {"crop": crop, "status": "no-runs",
                "note": "UNAVAILABLE RESULT — no runs logged. Run scripts/evaluate.py or scripts/evaluate_all.py first."}
        (reports_dir / f"model_comparison_{crop}.json").write_text(json.dumps(note, indent=2))
        (reports_dir / f"model_comparison_{crop}.md").write_text(
            f"# Model comparison: {crop}\n\nUNAVAILABLE RESULT — no runs logged yet.\n")
        (reports_dir / f"model_comparison_{crop}.html").write_text(
            f"<h1>Model comparison: {crop}</h1><p>UNAVAILABLE RESULT — no runs logged yet.</p>")
        return note

    df = runs_df[runs_df["crop"] == crop].copy() if "crop" in runs_df.columns else runs_df.copy()
    df = _latest_per_model(df)
    if df.empty:
        return build_comparison(pd.DataFrame(), crop, metrics_dir, out_root, class_names)

    rows = []
    for _, r in df.iterrows():
        eff = efficiency_row(
            float(r.get("accuracy", 0) or 0), float(r.get("macro_f1", 0) or 0),
            int(r.get("parameter_count", 0) or 0), float(r.get("flops", 0) or 0),
            float(r.get("inference_latency_ms", 0) or 0))
        pcr = r.get("per_class_recall")
        if isinstance(pcr, str):
            try:
                pcr = json.loads(pcr)
            except Exception:
                pcr = {}
        rows.append({
            "name": r.get("architecture", "?"), "architecture": r.get("architecture", "?"),
            "crop": crop, "accuracy": eff["accuracy"], "macro_precision": eff.get("macro_precision", float(r.get("macro_precision", 0) or 0)),
            "macro_recall": eff.get("macro_recall", float(r.get("macro_recall", 0) or 0)),
            "macro_f1": eff["macro_f1"], "weighted_f1": float(r.get("weighted_f1", 0) or 0),
            "parameter_count": eff["params"], "flops": eff["flops"], "latency_ms": eff["latency_ms"],
            "throughput": float(r.get("throughput", 0) or 0), "acc_per_mparams": eff["acc_per_mparams"],
            "f1_per_mparams": eff["f1_per_mparams"], "acc_per_gflop": eff["acc_per_gflop"],
            "f1_per_gflop": eff["f1_per_gflop"], "per_class_recall": pcr or {},
            "run_id": r.get("run_id", ""), "split": r.get("split", ""), "kind": r.get("kind", ""),
            "checkpoint_path": r.get("checkpoint_path", ""), "timestamp": str(r.get("timestamp", "")),
            "evaluation_samples": int(r.get("evaluation_samples", 0) or 0),
            "confusion_matrix_path": r.get("confusion_matrix_path", ""),
        })

    best_acc = max(rows, key=lambda r: r["accuracy"])
    best_f1 = max(rows, key=lambda r: r["macro_f1"])
    best_eff = max(rows, key=lambda r: r["f1_per_mparams"])
    fastest = min(rows, key=lambda r: r["latency_ms"] if r["latency_ms"] > 0 else 1e18)

    # ---------------- figures ----------------
    from src.evaluation.figures import (plot_metric_bar, plot_per_class_recall, plot_prf_grouped,
                                        plot_radar, plot_scatter)
    fig_paths = {}
    fig_paths["accuracy"] = plot_metric_bar(rows, "accuracy", figures_dir / "accuracy_by_model.png", "Accuracy by model")
    fig_paths["macro_f1"] = plot_metric_bar(rows, "macro_f1", figures_dir / "macro_f1_by_model.png", "Macro F1 by model")
    fig_paths["prf"] = plot_prf_grouped(rows, figures_dir / "prf_grouped.png")
    fig_paths["params_f1"] = plot_scatter(rows, "parameter_count", "macro_f1", figures_dir / "params_vs_f1.png",
                                          "Parameters (M)", "Macro F1")
    fig_paths["flops_f1"] = plot_scatter(rows, "flops", "macro_f1", figures_dir / "flops_vs_f1.png",
                                         "FLOPs", "Macro F1")
    fig_paths["latency_f1"] = plot_scatter(rows, "latency_ms", "macro_f1", figures_dir / "latency_vs_f1.png",
                                           "Latency (ms)", "Macro F1")
    if len(rows) >= 3:
        try:
            fig_paths["radar"] = plot_radar(rows, ["accuracy", "macro_f1", "f1_per_mparams", "throughput"],
                                            figures_dir / "radar.png")
        except Exception:
            pass
    if class_names and any(r.get("per_class_recall") for r in rows):
        try:
            fig_paths["per_class"] = plot_per_class_recall(rows, class_names, figures_dir / "per_class_recall.png")
        except Exception:
            pass

    # ---------------- confusion matrix gallery ----------------
    gallery = []
    for r in rows:
        found = []
        cm_path = r.get("confusion_matrix_path")
        if cm_path and Path(cm_path).exists():
            found.append(str(cm_path))
        else:
            for cand in (metrics_dir.parent / "confusion_matrices").glob(f"cm_{crop}_{r['architecture']}_*.png"):
                found.append(str(cand))
            for cand in (metrics_dir.parent / "confusion_matrices").glob(f"cm_{crop}_{r['architecture']}_*.json"):
                found.append(str(cand))
        gallery.append({"model": r["architecture"], "artifacts": found})

    # ---------------- XAI inventory ----------------
    xai = _xai_inventory(out_root)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "crop": crop, "n_models": len(rows),
        "executive_summary": (f"Compared {len(rows)} model(s) on crop '{crop}'. "
                              f"Best by accuracy: {best_acc['name']} ({best_acc['accuracy']:.4f}). "
                              f"Best by macro F1: {best_f1['name']} ({best_f1['macro_f1']:.4f}). "
                              f"Best by efficiency (F1 per M params): {best_eff['name']} "
                              f"({best_eff['f1_per_mparams']:.4f}). "
                              f"Fastest inference: {fastest['name']} ({fastest['latency_ms']:.2f} ms)."),
        "best_accuracy": {"model": best_acc["name"], "value": best_acc["accuracy"]},
        "best_macro_f1": {"model": best_f1["name"], "value": best_f1["macro_f1"]},
        "best_efficiency": {"model": best_eff["name"], "f1_per_mparams": best_eff["f1_per_mparams"],
                            "f1_per_gflop": best_eff["f1_per_gflop"]},
        "fastest": {"model": fastest["name"], "latency_ms": fastest["latency_ms"]},
        "models": rows,
        "confusion_matrix_gallery": gallery,
        "model_agreement": {"status": "UNAVAILABLE",
                            "note": "Agreement requires per-image probability storage. "
                                    "Run a dataset-level ensemble evaluation to measure agreement."},
        "xai_results": xai,
        "figures": {k: str(v) for k, v in fig_paths.items()},
    }

    # ---------------- CSV / JSON ----------------
    flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows]
    pd.DataFrame(flat).to_csv(reports_dir / f"model_comparison_{crop}.csv", index=False)
    (reports_dir / f"model_comparison_{crop}.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---------------- Markdown ----------------
    md = _markdown(summary, rows, gallery, xai, fig_paths, crop)
    (reports_dir / f"model_comparison_{crop}.md").write_text(md)

    # ---------------- HTML (self-contained) ----------------
    (reports_dir / f"model_comparison_{crop}.html").write_text(_html(summary, rows, gallery, fig_paths))
    return summary


def _markdown(s, rows, gallery, xai, fig_paths, crop) -> str:
    lines = [f"# Model comparison report — crop: {crop}", "",
             f"Generated: {s['generated_at']}", ""]
    lines += ["## 1. EXECUTIVE SUMMARY", "", s["executive_summary"], "",
              f"## 2. BEST MODEL BY ACCURACY", "",
              f"**{s['best_accuracy']['model']}** — accuracy {s['best_accuracy']['value']:.4f}", "",
              "## 3. BEST MODEL BY MACRO F1", "",
              f"**{s['best_macro_f1']['model']}** — macro F1 {s['best_macro_f1']['value']:.4f}", "",
              "## 4. BEST MODEL BY EFFICIENCY", "",
              f"**{s['best_efficiency']['model']}** — F1 per M params "
              f"{s['best_efficiency']['f1_per_mparams']:.4f}", "",
              "## 5. PER-CLASS ANALYSIS", "",
              "Per-class recall per model (from logged eval runs):", ""]
    for r in rows:
        pcr = r.get("per_class_recall") or {}
        if pcr:
            lines.append(f"- {r['name']}: " + ", ".join(f"{k}={v:.2f}" for k, v in pcr.items()))
        else:
            lines.append(f"- {r['name']}: UNAVAILABLE RESULT (no per-class recall logged)")
    lines += ["", "## 6. CONFUSION MATRIX ANALYSIS", ""]
    if gallery:
        for g in gallery:
            lines.append(f"- {g['model']}: " + ("; ".join(g["artifacts"]) if g["artifacts"] else "no matrix artifact"))
    else:
        lines.append("UNAVAILABLE RESULT — no confusion matrices found.")
    lines += ["", "## 7. MODEL AGREEMENT", "",
              "UNAVAILABLE RESULT — " + s["model_agreement"]["note"], "",
              "## 8. XAI RESULTS", "",
              f"{xai['count']} artifact(s) under outputs/xai/."]
    if xai["latest"]:
        lines.append("Latest: " + ", ".join(xai["latest"]))
    lines += ["", "## 9. LATENCY / COMPUTATIONAL ANALYSIS", ""]
    for r in sorted(rows, key=lambda r: r["latency_ms"]):
        lines.append(f"- {r['name']}: {r['latency_ms']:.2f} ms/image · "
                     f"{r['throughput']:.1f} img/s · {r['parameter_count'] / 1e6:.1f} M params")
    lines += ["", "## 10. FULL RAW RESULTS", "",
              "| Model | Accuracy | Macro F1 | Macro P | Macro R | W-F1 | Params(M) | Latency(ms) | Split | Run id |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["macro_f1"]):
        lines.append(f"| {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['macro_precision']:.4f} | {r['macro_recall']:.4f} | {r['weighted_f1']:.4f} | "
                     f"{r['parameter_count'] / 1e6:.2f} | {r['latency_ms']:.2f} | {r['split'] or '-'} | "
                     f"{r['run_id']} |")
    lines += ["", "---", "Generated by scripts/generate_comparison.py — AgroVision."]
    return "\n".join(lines)


def _html(s, rows, gallery, fig_paths) -> str:
    def img(src):
        return f'<img src="{src}" style="max-width:100%;margin:8px 0;" alt="figure"/>' if src else ""

    figs = "".join(img(_fig_embed(p)) for p in fig_paths.values())
    table = "<table border=1 cellpadding=6 cellspacing=0 style='border-collapse:collapse;font-size:13px;'><tr>" \
            "<th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Macro P</th><th>Macro R</th><th>W-F1</th>" \
            "<th>Params (M)</th><th>FLOPs (G)</th><th>Latency (ms)</th><th>Split</th><th>Run id</th></tr>"
    for r in sorted(rows, key=lambda r: -r["macro_f1"]):
        table += (f"<tr><td>{r['name']}</td><td>{r['accuracy']:.4f}</td><td><b>{r['macro_f1']:.4f}</b></td>"
                  f"<td>{r['macro_precision']:.4f}</td><td>{r['macro_recall']:.4f}</td>"
                  f"<td>{r['weighted_f1']:.4f}</td><td>{r['parameter_count'] / 1e6:.2f}</td>"
                  f"<td>{r['flops'] / 1e9:.2f}</td><td>{r['latency_ms']:.2f}</td>"
                  f"<td>{r['split'] or '-'}</td><td>{r['run_id']}</td></tr>")
    table += "</table>"
    gallery_html = "<ul>" + "".join(
        f"<li><b>{g['model']}</b>: " + (", ".join(g["artifacts"]) if g["artifacts"] else "no matrix artifact") + "</li>"
        for g in gallery) + "</ul>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>AgroVision comparison — {s['crop']}</title></head>
<body style="font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;color:#0f172a;">
<h1>Model comparison report — crop: {s['crop']}</h1>
<p><small>Generated: {s['generated_at']}</small></p>
<h2>1. Executive summary</h2><p>{s['executive_summary']}</p>
<h2>2. Best by accuracy</h2><p><b>{s['best_accuracy']['model']}</b> — {s['best_accuracy']['value']:.4f}</p>
<h2>3. Best by macro F1</h2><p><b>{s['best_macro_f1']['model']}</b> — {s['best_macro_f1']['value']:.4f}</p>
<h2>4. Best by efficiency</h2><p><b>{s['best_efficiency']['model']}</b> — F1/M params {s['best_efficiency']['f1_per_mparams']:.4f}</p>
<h2>5. Figures</h2>{figs}
<h2>6. Full results</h2>{table}
<h2>7. Confusion matrix gallery</h2>{gallery_html}
<h2>8. Model agreement</h2><p>UNAVAILABLE RESULT — {s['model_agreement']['note']}</p>
<h2>9. XAI results</h2><p>{s['xai_results']['count']} artifact(s) under outputs/xai/.</p>
<p style="color:#64748b;font-size:12px;">Generated by scripts/generate_comparison.py — AgroVision. Only measured results are reported.</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Generate per-crop model comparison reports.")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    out_root = Path(cfg.project.get("output_dir", "outputs"))
    metrics_dir = out_root / "metrics"
    runs_csv = metrics_dir / "runs.csv"
    if not runs_csv.exists():
        print("UNAVAILABLE RESULT — no runs.csv yet. Run scripts/evaluate_all.py first.")
        return
    runs = pd.read_csv(runs_csv)
    crops = list(cfg.datasets.get("crops", {})) if a.all else ([a.crop] if a.crop else ["tomato"])
    for crop in crops:
        class_names = None
        manifest = Path(f"data/splits/{crop}/test.csv")
        if manifest.exists():
            try:
                mdf = pd.read_csv(manifest)
                if not mdf.empty and {"label", "label_id"} <= set(mdf.columns):
                    class_names = mdf.sort_values("label_id").drop_duplicates("label_id")["label"].astype(str).tolist()
            except Exception:
                pass
        s = build_comparison(runs, crop, metrics_dir, out_root, class_names)
        print(f"[{crop}] models={s.get('n_models', 0)} -> reports/model_comparison_{crop}.{{csv,json,md,html}} "
              f"+ figures/<crop>/")
    print("Report locations:")
    for crop in crops:
        print(f"  outputs/reports/model_comparison_{crop}.html")


if __name__ == "__main__":
    main()