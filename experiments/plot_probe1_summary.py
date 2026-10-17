"""Two-panel summary of probe 1: score vs bands (left) and where the score comes from (right, k=1)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
run = Path(sys.argv[1])
an = json.loads((run / "analysis.json").read_text())
rows = json.loads((run / "rows.json").read_text())
nb = json.loads((ROOT / "results/null_band.json").read_text())["res"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.35))
ks = [1, 2, 4]
for i, k in enumerate(ks):
    y = len(ks) - 1 - i
    b = nb[f"k={k} | all"]
    lin = [b[m]["score"] for m in ("idw", "best-linear", "linear-smooth")] + [0.0]
    a1.barh(y, max(lin) - min(lin), left=min(lin), height=0.5, color=GRID, lw=0)
    a1.plot(b["oracle"]["score"], y, marker="D", ms=8, color=ORANGE, lw=0, zorder=3)
    e = an[f"k={k} | all"]
    lo, hi = e["score_zuna_ci"]
    a1.plot([lo, hi], [y, y], color=BLUE, lw=2, zorder=4)
    a1.plot(e["score_zuna"], y, marker="o", ms=8, color=BLUE, lw=0, zorder=5)
a1.set_yticks(range(3), ["1 source", "2 sources", "4 sources"][::-1], color=INK)
a1.axvline(0, color=AXIS, lw=1)
a1.set_xlim(-0.05, 0.3)
a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)
a1.set_xlabel("Physics-use score (gain over splines: possible vs impossible)")
a1.set_title("A. The pre-registered score", loc="left", fontsize=10, weight="bold")
a1.legend(handles=[Patch(color=GRID, label="Smoothing methods"),
                   Line2D([], [], marker="D", color=ORANGE, lw=0, ms=7, label="Knows the physics"),
                   Line2D([], [], marker="o", color=BLUE, lw=2, ms=7, label="ZUNA (95% CI)")],
          frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3)

methods = [("Splines", "shape_spline"), ("Best linear", "shape_best-linear"), ("ZUNA", "shape_zuna"), ("Knows the\nphysics", None)]
x = np.arange(len(methods))
for j, kind in enumerate(("real", "patterns")):
    vals = []
    for name, key in methods:
        if key is None:
            vals.append(nb["k=1 | all"]["mean_err"][kind]["oracle"])
        else:
            vals.append(np.mean([r[key] for r in rows if r["k"] == 1 and r["kind"] == kind]))
    col = AQUA if kind == "real" else VIOLET
    bars = a2.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=col, lw=0,
                  label="possible patterns" if kind == "real" else "impossible patterns")
    for xi, v in zip(x + (j - 0.5) * 0.36, vals):
        a2.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
a2.set_xticks(x, [m[0] for m in methods], color=INK)
a2.set_ylabel("Shape error (1 − correlation), lower is better")
a2.grid(axis="y", color=GRID, lw=0.6); a2.set_axisbelow(True)
a2.set_title("B. Where ZUNA's score comes from (1 source)", loc="left", fontsize=10, weight="bold")
a2.legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 1.04, "ZUNA is thrown off by impossible patterns, but does not beat simple smoothing on possible ones",
         fontsize=11.5, weight="bold")
fig.savefig(run / "fig_probe1_summary.png", dpi=170, bbox_inches="tight")
print(run / "fig_probe1_summary.png")
