from __future__ import annotations
from pathlib import Path
from PIL import Image
ALLOWED = {".jpg",".jpeg",".png",".bmp",".webp"}
def validate_image(path: Path) -> Path:
    p = Path(path)
    if p.suffix.lower() not in ALLOWED:
        from src.core.exceptions import ImageError
        raise ImageError(f"Unsupported image format: {p.suffix}", f"allowed={sorted(ALLOWED)}", "HOW: upload JPG/PNG/WEBP")
    try:
        with Image.open(p) as im: im.verify()
    except Exception as e:
        from src.core.exceptions import ImageError
        raise ImageError(f"Corrupt image: {p}", str(e), "HOW: re-export the file")
    return p
def make_demo_image(path: Path, size=(96,96)):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    arr = (rng.random((*size,3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return path
