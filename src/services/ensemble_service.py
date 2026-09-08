from __future__ import annotations
from src.inference.ensemble import majority_vote, soft_average, weighted_soft, agreement_rate, diversity_warning
from src.evaluation.metrics import compute_metrics
def ensemble_report(y_true, pred_lists, prob_lists, weights=None):
    agg = {"agreement_rate": agreement_rate(pred_lists)}
    agg["diversity_note"] = diversity_warning(agg["agreement_rate"])
    out = {"agreement": agg}
    mv = majority_vote(pred_lists); out["majority_vote"] = compute_metrics(y_true, mv)
    import numpy as np
    avg = np.asarray(soft_average(prob_lists)).argmax(1).tolist()
    out["soft_average"] = compute_metrics(y_true, avg)
    if weights:
        w = np.asarray(weighted_soft(prob_lists, weights)).argmax(1).tolist() if isinstance(weighted_soft(prob_lists, weights), list) and len(weighted_soft(prob_lists, weights)) and isinstance(weighted_soft(prob_lists, weights)[0], list) else None
        if w is not None: out["weighted"] = compute_metrics(y_true, w)
    return out
