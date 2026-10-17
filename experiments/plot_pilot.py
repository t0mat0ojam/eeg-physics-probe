import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "results/pilot_discriminability.json").read_text())
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
methods = [("spline (H1: smoothing)", "Smoothing (splines)", MUTED),
           ("minimum-norm (linear physics)", "Linear physics (min-norm)", BLUE),
           ("best linear (any linear method)", "Best possible linear method", AQUA),
           ("dipole fit (nonlinear physics)", "Knows the head's physics\n(dipole fit)", ORANGE)]
ks = [1, 2, 4, 16]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.45))
ax = axes[0]
for i, (key, lab, col) in enumerate(methods):
    d = r[f"k=1 | patch | {key}"]
    ax.barh(len(methods) - 1 - i, d["gap"], color=col, height=0.6, lw=0)
    lo, hi = d["gap_ci95"]
    ax.plot([lo, hi], [len(methods) - 1 - i] * 2, color=INK2, lw=1.2)
    ax.text(max(hi, d["gap"]) + 0.01, len(methods) - 1 - i, f"{d['gap']:+.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(methods)), [m[1] for m in methods][::-1], fontsize=8.5, color=INK)
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlim(-0.05, 0.45)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.set_xlabel("Physics-use score = error on surrogate − error on real")
ax.set_title("One brain source, a patch of 12 electrodes hidden", loc="left", fontsize=9.5)

ax = axes[1]
for key, lab, col in methods:
    g = [r[f"k={k} | patch | {key}"]["gap"] for k in ks]
    ax.plot(range(len(ks)), g, color=col, lw=2, marker="o", ms=5, label=lab.replace("\n", " "))
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(range(len(ks)), [str(k) for k in ks])
ax.set_xlabel("Number of brain sources active at once")
ax.set_ylabel("Physics-use score")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
ax.set_title("Score shrinks as signals become a blur of many sources", loc="left", fontsize=9.5)
fig.text(0.01, 1.04, "Pilot: the test separates “smoothing” from “knowing the physics” (no AI model yet)",
         fontsize=11.5, weight="bold")
fig.text(0.01, 0.975, "Smoothing methods score ~0 by construction. Only a method that knows how brain sources "
         "spread through the head scores high. Bars: 95% CI.", fontsize=8.5, color=INK2)
fig.savefig(ROOT / "results/fig_pilot.png", dpi=170, bbox_inches="tight")
