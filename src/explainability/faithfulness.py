"""Pointing-game faithfulness: max(heatmap) inside mask?"""
from __future__ import annotations
import numpy as np
def pointing_hit(heatmap: np.ndarray, mask: np.ndarray) -> bool:
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    mh, mw = mask.shape[:2]
    yy = int(y * mh / heatmap.shape[0]); xx = int(x * mw / heatmap.shape[1])
    return bool(mask[yy, xx] > 0)
def faithfulness_score(hits: list) -> dict:
    if not hits: return {"score": None, "n": 0, "note": "No lesion masks available: quantitative lesion-based evaluation cannot be performed."}
    return {"score": float(sum(1 for h in hits if h) / len(hits)), "n": len(hits)}
