"""Deterministic stratified splitting with manifests (75/10/15)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.exceptions import DataError


def stratified_split(df: pd.DataFrame, seed: int = 42, train_r=0.75, val_r=0.10, test_r=0.15,
                     label_col="label", out_dir: Path | None = None) -> dict:
    if df.empty:
        raise DataError("Empty dataframe cannot be split", "no rows", "HOW: check dataset root / classes")
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise DataError("Split ratios must sum to 1.0", f"got {train_r+val_r+test_r}", "HOW: fix configs/datasets.yaml")
    try:
        train, rest = train_test_split(df, test_size=1 - train_r, random_state=seed, stratify=df[label_col])
        rel_test = test_r / (val_r + test_r)
        val, test = train_test_split(rest, test_size=rel_test, random_state=seed, stratify=rest[label_col])
    except ValueError:
        # fallback: non-stratified when classes too small
        train, rest = train_test_split(df, test_size=1 - train_r, random_state=seed)
        rel_test = test_r / (val_r + test_r)
        val, test = train_test_split(rest, test_size=rel_test, random_state=seed)
    out = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            v.to_csv(out_dir / f"{k}.csv", index=False)
    return out


def check_leakage(splits: dict, path_col="path") -> list:
    tr, va, te = set(splits["train"][path_col]), set(splits["val"][path_col]), set(splits["test"][path_col])
    leaks = []
    for a, b, name in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        overlap = a & b
        if overlap:
            leaks.append(f"{name}: {len(overlap)} overlapping paths")
    return leaks
