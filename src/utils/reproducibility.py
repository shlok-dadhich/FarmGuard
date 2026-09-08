from __future__ import annotations
import platform, subprocess, sys
def env_report() -> dict:
    rep = {"python": sys.version, "platform": platform.platform()}
    try: rep["commit"] = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: rep["commit"] = "unknown"
    for pkg in ("torch","torchvision","timm","streamlit","sklearn"):
        try: rep[pkg] = __import__(pkg).__version__
        except Exception: rep[pkg] = "missing"
    return rep
