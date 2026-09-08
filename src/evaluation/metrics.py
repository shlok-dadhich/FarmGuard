"""Classification metrics. Primary: macro F1."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(y_true, y_pred, labels=None) -> dict:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true: return {"accuracy":0.0,"macro_precision":0.0,"macro_recall":0.0,"macro_f1":0.0,"weighted_f1":0.0,"per_class_recall":{}}
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    labs = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    out["per_class_recall"] = {str(l): float(r) for l, r in zip(labs, rec)}
    out["n"] = len(y_true)
    return out

def confidence_stats(probs) -> dict:
    import numpy as np
    p = np.asarray(probs)
    top = p.max(axis=1) if p.ndim == 2 else p
    return {"mean_conf": float(top.mean()), "min_conf": float(top.min()), "frac_low": float((top < 0.5).mean())}
