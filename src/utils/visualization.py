from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def bar_chart(data: dict, out: str, title=""):
    fig, ax = plt.subplots()
    ax.bar(list(data.keys()), list(data.values())); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); return out
