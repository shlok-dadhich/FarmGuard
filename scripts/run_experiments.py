"""Run the full evaluation pipeline: data prep -> batch evaluation -> comparison report.

Usage:
    python scripts/run_experiments.py --crop tomato --split test
    python scripts/run_experiments.py --all --split val

Pipeline steps:
1. (optional) prepare_data  — build/refresh split manifests for the crop
2. evaluate_all            — evaluate every configured model on the split, log runs
3. generate_comparison     — build the per-crop comparison report + figures

Each step runs to completion independently; a failure in one step is reported
without hiding results from the others.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> bool:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run([sys.executable, *cmd], cwd=REPO)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="AgroVision evaluation pipeline.")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true", help="skip data preparation step")
    ap.add_argument("--limit", type=int, default=None, help="cap samples per split (subset eval)")
    a = ap.parse_args()

    crop = None if a.all else (a.crop or "tomato")
    steps = []
    if not a.skip_prepare and crop is not None:
        steps.append(["scripts/prepare_data.py", "--crop", crop])
    if a.limit:
        steps.append(["scripts/evaluate_all.py", "--crop", crop, "--split", a.split, "--limit", str(a.limit)]
                     if crop else ["scripts/evaluate_all.py", "--all", "--split", a.split, "--limit", str(a.limit)])
    else:
        steps.append(["scripts/evaluate_all.py", "--crop", crop, "--split", a.split]
                     if crop else ["scripts/evaluate_all.py", "--all", "--split", a.split])
    steps.append(["scripts/generate_comparison.py", "--crop", crop] if crop else ["scripts/generate_comparison.py", "--all"])

    failed = []
    for step in steps:
        if not _run(step):
            failed.append(step[0])
    if failed:
        print(f"\nPIPELINE COMPLETED WITH FAILURES in: {', '.join(failed)}")
        sys.exit(1)
    print("\nPIPELINE OK — reports under outputs/reports/, figures under outputs/figures/")


if __name__ == "__main__":
    main()