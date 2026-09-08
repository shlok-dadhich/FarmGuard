"""Aggregate runs.csv into markdown report."""
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from pathlib import Path
import pandas as pd
def main():
    p = Path("outputs/metrics/runs.csv")
    out = Path("outputs/reports/summary.md"); out.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        out.write_text("# AgroVision report\n\nUNAVAILABLE RESULT - no runs yet.\n"); print("no runs"); return
    df = pd.read_csv(p)
    try:
        table = df.to_markdown(index=False)
    except Exception:
        table = "```\n" + df.to_string(index=False) + "\n```"
    lines = ["# AgroVision experiment summary", "", table]
    out.write_text("\n".join(lines)); print(f"wrote {out}")
if __name__ == "__main__": main()

