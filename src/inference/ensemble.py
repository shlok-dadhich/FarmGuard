"""Voting ensembles + agreement stats."""
from __future__ import annotations
import numpy as np
def agreement_rate(pred_lists: list) -> float:
    a = np.asarray(pred_lists)
    agree = (a == a[0]).all(axis=0).mean() if a.size else 0.0
    return float(agree)
def majority_vote(pred_lists: list) -> list:
    a = np.asarray(pred_lists); out = []
    for col in a.T:
        vals, counts = np.unique(col, return_counts=True)
        out.append(int(vals[counts.argmax()]))
    return out
def soft_average(prob_lists: list) -> list:
    return (np.asarray(prob_lists).mean(axis=0)).tolist()
def weighted_soft(prob_lists: list, weights: list) -> list:
    w = np.asarray(weights, dtype=float); w = w / w.sum()
    return (np.asarray(prob_lists) * w[:, None, None]).sum(axis=0).tolist() if np.asarray(prob_lists).ndim == 3 else ((np.asarray(prob_lists).T * w).T.sum(axis=0)).tolist()
def diversity_warning(rate: float, thr=0.90) -> str | None:
    if rate > thr: return f"Models agree {rate:.1%} of the time: little diversity benefit expected."
    return None
