"""Early stopping."""
from __future__ import annotations
class EarlyStopping:
    def __init__(self, patience=7, mode="max"):
        self.patience = patience; self.mode = mode; self.best = None; self.bad = 0
    def step(self, val):
        better = val > (self.best or float("-inf")) if self.mode=="max" else val < (self.best or float("inf"))
        if self.best is None or better: self.best = val; self.bad = 0; return False
        self.bad += 1; return self.bad >= self.patience
