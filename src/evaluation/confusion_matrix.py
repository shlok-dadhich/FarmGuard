from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix
def compute_cm(y_true, y_pred, labels=None):
    return np.asarray(confusion_matrix(y_true, y_pred, labels=labels)).tolist()
