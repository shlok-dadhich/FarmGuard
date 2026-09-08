from __future__ import annotations
import json
from pathlib import Path
def save_report(report: dict, out: Path) -> Path:
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)); return out
def markdown_report(title, metrics: dict) -> str:
    lines = [f"# {title}", ""]
    for k, v in metrics.items():
        if isinstance(v, dict): continue
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)
