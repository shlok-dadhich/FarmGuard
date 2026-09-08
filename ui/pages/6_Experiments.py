"""Experiments page: browse and compare experiment history.

Filters: crop, architecture, kind, split, date range, run id. Selecting two or
more runs shows a side-by-side comparison (metrics, per-class recall, confusion
matrices, latency/params/FLOPs) with downloads. History is append-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.common import (
    get_crops,
    load_app_config,
    load_runs_table,
    unavailable_banner,
)
from ui.theme import inject

inject()
st.title("🗂️ Experiments")
st.caption("Experiment history is append-only: old runs are never overwritten.")

cfg = load_app_config()
runs = load_runs_table()

if runs.empty:
    unavailable_banner("no runs logged yet. Run scripts/evaluate.py, scripts/evaluate_all.py or the "
                       "Model Evaluation page first.")
    st.stop()

with st.expander("🔍 Filters", expanded=True):
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        crop_f = st.selectbox("Crop", ["(all)"] + [c for c in get_crops() if c in set(runs.get("crop", []))],
                              key="exp_crop")
    with f2:
        archs = sorted(set(runs.get("architecture", []).dropna().astype(str)))
        arch_f = st.multiselect("Architecture", archs, key="exp_arch")
    with f3:
        kinds = sorted(set(runs.get("kind", []).dropna().astype(str))) or ["train", "eval"]
        kind_f = st.multiselect("Kind", kinds, key="exp_kind")
    with f4:
        split_f = st.multiselect("Split", sorted(set(runs.get("split", []).dropna().astype(str))) or ["val", "test"],
                                 key="exp_split")
    f5, f6, f7 = st.columns(3)
    with f5:
        date_from = st.text_input("From (YYYY-MM-DD)", key="exp_from")
    with f6:
        date_to = st.text_input("To (YYYY-MM-DD)", key="exp_to")
    with f7:
        id_search = st.text_input("Run id contains", key="exp_id")

df = runs.copy()
if crop_f != "(all)" and "crop" in df:
    df = df[df["crop"] == crop_f]
if arch_f and "architecture" in df:
    df = df[df["architecture"].isin(arch_f)]
if kind_f and "kind" in df:
    df = df[df["kind"].isin(kind_f)]
if split_f and "split" in df:
    df = df[df["split"].isin(split_f)]
if date_from or date_to:
    ts = pd.to_datetime(df.get("timestamp"), errors="coerce")
    if date_from:
        df = df[ts >= pd.Timestamp(date_from)]
    if date_to:
        df = df[ts <= pd.Timestamp(date_to) + pd.Timedelta(days=1)]
if id_search and "run_id" in df:
    df = df[df["run_id"].astype(str).str.contains(id_search, case=False, na=False)]

if df.empty:
    unavailable_banner("no runs match these filters.")
    st.stop()

df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

st.markdown(f"### {len(df)} run(s)")
show_cols = [c for c in ("timestamp", "run_id", "crop", "architecture", "kind", "split", "seed",
                         "accuracy", "macro_f1", "weighted_f1", "inference_latency_ms", "throughput",
                         "parameter_count", "status", "checkpoint_path") if c in df.columns]
st.dataframe(df[show_cols].style.format({c: "{:.4f}" for c in ("accuracy", "macro_f1", "weighted_f1")},
                                        na_rep="-"),
             width="stretch", hide_index=True)

# ---------------------------------------------------------------- compare selection
st.divider()
st.markdown("### ⚖️ Compare runs")
st.caption("Select two or more runs to compare metrics, per-class recall, latency and confusion matrices.")
selected = st.multiselect("Runs to compare", df["run_id"].astype(str).tolist(), key="exp_sel",
                          format_func=lambda rid: f"{rid} — " + str(df[df['run_id'].astype(str) == rid].iloc[0].get('architecture', '?')))
sel_df = df[df["run_id"].astype(str).isin(selected)]

if len(sel_df) < 2:
    st.info("Select at least 2 runs to compare.")
else:
    cmp_cols = [c for c in ("architecture", "crop", "split", "kind", "accuracy", "macro_f1",
                            "weighted_f1", "parameter_count", "flops", "inference_latency_ms",
                            "throughput", "evaluation_samples", "status", "timestamp", "run_id") if c in sel_df.columns]
    st.markdown("**Metrics comparison**")
    st.dataframe(sel_df[cmp_cols].style.format({c: "{:.4f}" for c in ("accuracy", "macro_f1", "weighted_f1")},
                                               na_rep="-"),
                 width="stretch", hide_index=True)

    # per-class recall comparison
    st.markdown("**Per-class recall**")
    pc_rows = []
    for _, r in sel_df.iterrows():
        pcr = r.get("per_class_recall")
        if isinstance(pcr, str):
            try:
                pcr = json.loads(pcr)
            except Exception:
                pcr = {}
        pc_rows.append({"architecture": r.get("architecture"), **(pcr or {})})
    pcr_df = pd.DataFrame(pc_rows).fillna(0.0)
    if pcr_df.shape[1] > 1:
        st.dataframe(pcr_df.set_index("architecture").T.style.format("{:.3f}"), width="stretch")
    else:
        st.caption("No per-class recall recorded for these runs (UNAVAILABLE RESULT).")

    # confusion matrices
    st.markdown("**Confusion matrices**")
    cols = st.columns(min(3, len(sel_df)))
    for i, (_, r) in enumerate(sel_df.iterrows()):
        cm_path = r.get("confusion_matrix_path")
        cmj = None
        if cm_path and Path(cm_path).exists():
            try:
                cmj = json.loads(Path(cm_path).read_text())
            except Exception:
                cmj = None
        if cmj and cmj.get("confusion_matrix"):
            try:
                import numpy as np

                from src.evaluation.figures import plot_confusion_matrix

                labels = cmj.get("labels") or [str(i) for i in range(len(cmj["confusion_matrix"]))]
                tmp = Path(cfg.project.get("output_dir", "outputs")) / "figures" / "_cmp"
                tmp.mkdir(parents=True, exist_ok=True)
                p = plot_confusion_matrix(cmj["confusion_matrix"], labels,
                                          f"{r.get('architecture')} ({r.get('split', '?')})",
                                          tmp / f"{r.get('run_id')}_cm.png")
                cols[i % 3].image(str(p), width="stretch",
                                  caption=f"{r.get('architecture')} — {r.get('run_id')}")
            except Exception as e:
                cols[i % 3].caption(f"CM render failed for {r.get('run_id')}: {e}")
        else:
            cols[i % 3].caption(f"{r.get('architecture')}: no confusion matrix artifact (UNAVAILABLE).")

    # downloads
    st.markdown("**Downloads**")
    d1, d2 = st.columns(2)
    d1.download_button("CSV (filtered runs)", df.to_csv(index=False).encode(),
                       "experiment_runs_filtered.csv", "text/csv", key="exp_dl_csv")
    d2.download_button("JSON (selected runs)", sel_df.to_json(orient="records", indent=2).encode(),
                       "experiment_runs_selected.json", "application/json", key="exp_dl_json")