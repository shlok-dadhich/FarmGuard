"""Model Comparison page: per-crop ranking from real logged runs (never faked)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui.common import get_crops, load_eval_record, load_runs_table, unavailable_banner
from ui.theme import inject

inject()
st.title("📊 Model Comparison")
st.caption("Primary research metric: macro F1. Accuracy alone is never enough.")

if st.button("🔄 Refresh data", key="cmp_refresh"):
    st.cache_data.clear()
    st.rerun()

runs = load_runs_table()
crops = get_crops()
crop = st.selectbox("Crop", ["(all)"] + crops, key="cmp_crop")
arch_filter = st.multiselect("Architectures", sorted(runs["architecture"].unique().tolist()) if not runs.empty and "architecture" in runs.columns else [],
                             key="cmp_archs")
sort_by = st.selectbox("Sort by", ["macro_f1", "accuracy", "f1_per_mparams", "acc_per_mparams",
                                   "inference_latency_ms", "parameter_count"], key="cmp_sort")
ascending = st.checkbox("Ascending", value=False, key="cmp_asc")

if runs.empty:
    unavailable_banner("no runs logged yet. Train first: python scripts/train.py --crop <crop> --model <model> --seed 42")
    st.stop()

df = runs.copy()
if crop != "(all)" and "crop" in df.columns:
    df = df[df["crop"] == crop]
if arch_filter and "architecture" in df.columns:
    df = df[df["architecture"].isin(arch_filter)]
if df.empty:
    unavailable_banner("no runs match this filter.")
    st.stop()

# latest run per (crop, architecture)
if {"crop", "architecture", "timestamp"} <= set(df.columns):
    df = df.sort_values("timestamp").groupby(["crop", "architecture"], as_index=False).tail(1)

# efficiency columns (computed, not stored)
from src.evaluation.efficiency import efficiency_row


def _eff(r):
    try:
        return efficiency_row(float(r.get("accuracy", 0) or 0), float(r.get("macro_f1", 0) or 0),
                              int(r.get("parameter_count", 0) or 0), float(r.get("flops", 0) or 0),
                              float(r.get("inference_latency_ms", 0) or 0))
    except Exception:
        return {}


eff = df.apply(lambda r: pd.Series(_eff(r)), axis=1)
for c in ("acc_per_mparams", "f1_per_mparams", "acc_per_gflop", "f1_per_gflop"):
    if c in eff.columns:
        df[c] = eff[c]

show_cols = [c for c in ("crop", "architecture", "seed", "accuracy", "macro_f1", "parameter_count",
                         "flops", "inference_latency_ms", "acc_per_mparams", "f1_per_mparams",
                         "training_time_s", "run_id") if c in df.columns]
if sort_by in df.columns:
    df = df.sort_values(sort_by, ascending=ascending)
df = df.reset_index(drop=True)
df.index = df.index + 1
df.index.name = "rank"

st.markdown("### 🏆 Ranking")
st.dataframe(df[show_cols].style.format({c: "{:.4f}" for c in show_cols if c in ("accuracy", "macro_f1")},
                                        na_rep="n/a"),
             width="stretch")

# verdicts (measured only)
try:
    best_raw = df.loc[df["macro_f1"].astype(float).idxmax()]
    st.success(f"Best raw (macro F1): **{best_raw['architecture']}** ({float(best_raw['macro_f1']):.4f})")
    if "f1_per_mparams" in df.columns:
        best_eff = df.loc[df["f1_per_mparams"].astype(float).idxmax()]
        st.info(f"Best efficiency-adjusted (F1 per M params): **{best_eff['architecture']}** "
                f"({float(best_eff['f1_per_mparams']):.4f})")
        if best_raw["architecture"] != best_eff["architecture"]:
            st.caption("Note: best raw ≠ best efficiency-adjusted — report both (see efficiency analysis).")
except Exception:
    pass

# per-architecture detail: per-class recall + eval records
st.markdown("### 🔍 Per-architecture detail")
for _, r in df.iterrows():
    arch = r["architecture"]
    with st.expander(f"{arch} — acc {float(r.get('accuracy', 0)):.3f} · F1 {float(r.get('macro_f1', 0)):.3f}"):
        pcr = r.get("per_class_recall", "")
        if isinstance(pcr, str) and pcr:
            try:
                st.json(json.loads(pcr))
            except Exception:
                st.text(str(pcr))
        elif isinstance(pcr, dict):
            st.json(pcr)
        else:
            st.caption("Per-class recall not recorded for this run.")
        ev = load_eval_record(r.get("crop", ""), arch)
        if ev:
            st.json({k: v for k, v in ev.items() if k != "probs"})
        else:
            st.caption("No eval JSON (outputs/metrics/eval_*.json) for this pair.")
        st.caption(f"run_id={r.get('run_id', '?')} · checkpoint={r.get('checkpoint_path', '?')}")
