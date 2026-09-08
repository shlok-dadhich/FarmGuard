from __future__ import annotations
import numpy as np
def normalize(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    return (h - h.min()) / (h.max() - h.min() + 1e-8)
def overlay(original: np.ndarray, heat: np.ndarray, alpha=0.45, colormap="jet"):
    import cv2
    h = (normalize(heat) * 255).astype("uint8")
    h = cv2.resize(h, (original.shape[1], original.shape[0]))
    colored = cv2.applyColorMap(h, getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET))
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * original + alpha * colored).astype("uint8")
