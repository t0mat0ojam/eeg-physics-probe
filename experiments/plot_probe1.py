"""Figure: ZUNA's physics-use score vs the smoother band (H1) and the oracle physics model (H2), per k.

  python experiments/plot_probe1.py results/kaggle/probe1/<run>/analysis.json [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

an = json.loads(Path(sys.argv[1]).read_text())
nb = json.loads((ROOT / "results/null_band.json").read_text())["res"]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).with_name("fig_probe1.png")
ks = sorted({int(k.split("|")[0].strip()[2:]) for k in an if k.endswith("| all")})

fig, ax = plt.subplots(figsize=(8.4, 3.6))
for i, k in enumerate(ks):
    y = len(ks) - 1 - i
    b = nb[f"k={k} | all"]
    lin = [b[m]["score"] for m in ("idw", "best-linear", "linear-smooth")] + [0.0]
    ax.barh(y, max(lin) - min(lin), left=min(lin), height=0.5, color=GRID, lw=0, zorder=1)
    ax.plot(b["oracle"]["score"], y, marker="D", ms=8, color=ORANGE, zorder=3, lw=0)
    e = an[f"k={k} | all"]
    s, (lo, hi) = e["score_zuna"], e["score_zuna_ci"]
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2, zorder=4)
    ax.plot(s, y, marker="o", ms=8, color=BLUE, zorder=5, lw=0)
    ax.text(max(hi, b["oracle"]["score"]) + 0.012, y, f"ZUNA {s:+.3f}  ({s / b['oracle']['score']:.0%} of oracle)",
            va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ks)), [f"{k} source{'s' if k > 1 else ''}" for k in ks][::-1], color=INK)
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlabel("Physics-use score: extra gain over splines on real vs impossible patterns")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRID, label="Range of smoothing methods (H1)"),
                   Line2D([], [], marker="D", color=ORANGE, lw=0, ms=7, label="Model that knows the head physics (H2)"),
                   Line2D([], [], marker="o", color=BLUE, lw=2, ms=7, label="ZUNA 1.1 (95% CI)")],
          frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3)
ax.set_xlim(-0.06, max(0.35, max(an[f'k={k} | all']['score_zuna_ci'][1] for k in ks) + 0.12))
ax.set_title("Does ZUNA use head physics when it fills in hidden electrodes?", loc="left", fontsize=11, weight="bold")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
