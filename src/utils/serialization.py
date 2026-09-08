import json
from pathlib import Path
def save_json(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str)); return p
def load_json(path): return json.loads(Path(path).read_text())
