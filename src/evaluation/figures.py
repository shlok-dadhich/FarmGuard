"""Matplotlib figure helpers (Agg backend — safe for scripts and Streamlit).

All functions take explicit output paths and never touch a display. Figures are
written under outputs/figures/<crop>/ by the comparison report generator.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, out: Path, tight: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(cm, labels, title: str, out: Path, normalize: bool = False) -> Path:
    """Confusion matrix as raw counts or row-normalized percentages."""
    arr = np.asarray(cm, dtype=float)
    if normalize:
        arr = arr / (arr.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(max(5, arr.shape[0] * 0.7), max(4, arr.shape[1] * 0.6)))
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1.0 if normalize else None)
    fmt = ".0%" if normalize else "d"
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = f"{val:.0%}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (val > (0.5 if normalize else arr.max() / 2)) else "black", fontsize=8)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} — {'row-normalized' if normalize else 'raw counts'}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def plot_metric_bar(rows: list[dict], metric: str, out: Path, title: str = "") -> Path:
    """Bar chart of one metric per model, best value highlighted."""
    names = [r.get("name", r.get("architecture", "?")) for r in rows]
    vals = [float(r.get(metric, 0) or 0) for r in rows]
    order = np.argsort(vals)[::-1]
    names, vals = [names[i] for i in order], [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
    colors = ["#16a34a" if v == max(vals) else "#3b82f6" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by model")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    return _save(fig, out)


def plot_prf_grouped(rows: list[dict], out: Path) -> Path:
    """Grouped bars: precision / recall / macro F1 per model."""
    names = [r.get("name", r.get("architecture", "?")) for r in rows]
    p = [float(r.get("macro_precision", 0) or 0) for r in rows]
    rc = [float(r.get("macro_recall", 0) or 0) for r in rows]
    f1 = [float(r.get("macro_f1", 0) or 0) for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.0), 4.5))
    w = 0.25
    ax.bar(x - w, p, w, label="Precision", color="#3b82f6")
    ax.bar(x, rc, w, label="Recall", color="#f59e0b")
    ax.bar(x + w, f1, w, label="Macro F1", color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Precision / Recall / Macro F1")
    return _save(fig, out)


def plot_scatter(rows: list[dict], x_key: str, y_key: str, out: Path, xlabel: str, ylabel: str) -> Path:
    """Scatter: efficiency analysis (e.g. params/FLOPs/latency vs macro F1)."""
    xs = [float(r.get(x_key, 0) or 0) for r in rows]
    ys = [float(r.get(y_key, 0) or 0) for r in rows]
    names = [r.get("name", r.get("architecture", "?")) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xs, ys, s=60, c="#3b82f6")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}")
    return _save(fig, out)


def plot_radar(rows: list[dict], cols: list[str], out: Path, max_vals: dict | None = None) -> Path:
    """Radar chart of normalized metrics (requires >= 3 models)."""
    if len(rows) < 3:
        raise ValueError("radar chart needs at least 3 models")
    names = [r.get("name", r.get("architecture", "?")) for r in rows]
    maxv = max_vals or {c: max(float(r.get(c, 0) or 0) for r in rows) for c in cols}
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for r, name in zip(rows, names):
        vals = [(float(r.get(c, 0) or 0) / (maxv.get(c, 1) or 1)) for c in cols]
        vals += vals[:1]
        ax.plot(angles, vals, label=name, linewidth=1.4)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=6.5)
    ax.set_title("Normalized metric profile")
    return _save(fig, out)


def plot_per_class_recall(rows: list[dict], class_names: list[str], out: Path) -> Path:
    """Per-class recall heatmap: rows=models, cols=classes."""
    if not rows or not class_names:
        raise ValueError("per-class recall heatmap needs rows and class names")
    arr = np.zeros((len(rows), len(class_names)))
    for i, r in enumerate(rows):
        pcr = r.get("per_class_recall") or {}
        for j, cn in enumerate(class_names):
            arr[i, j] = float(pcr.get(cn, pcr.get(str(j), 0)) or 0)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(3, len(rows) * 0.6)))
    im = ax.imshow(arr, cmap="YlGn", vmin=0, vmax=1)
    names = [r.get("name", r.get("architecture", "?")) for r in rows]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Per-class recall")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)