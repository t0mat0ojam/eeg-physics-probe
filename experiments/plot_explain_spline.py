"""Explainer: what a spline does vs what knowing head physics adds (one dipole, a patch hidden)."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.interpolate import CubicSpline
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physprobe import synth, probe
from physprobe.music import music_predict
mne.set_log_level("ERROR")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"

lf = synth.build_free_leadfield(); L, pos, info = lf["L"], lf["pos"], lf["info"]
C = L.shape[0]
# pick a source whose negative pole sits under a contiguous patch of electrodes
best = None
rng = np.random.default_rng(3)
for _ in range(4000):
    j = rng.integers(L.shape[1]); o = rng.normal(size=3); o /= np.linalg.norm(o)
    g = L[:, j] @ o
    c = int(np.argmin(g))
    H = list(np.argsort(np.linalg.norm(pos - pos[c], axis=1))[:10])
    V = [i for i in range(C) if i not in H]
    gr = g - g[V].mean()
    if gr.max() <= 0:
        continue
    ratio = -gr[H].min() / gr.max()                        # hidden pole as strong as the visible one
    sp = probe.spline_W(pos, V, H) @ gr[V]
    err = np.abs(sp - gr[H]).max() / np.abs(gr[H]).max()
    if 0.7 < ratio < 1.3 and (best is None or err > best[0]):
        best = (err, j, o, H, V)
_, j, o, H, V = best
g = L[:, j] @ o
T = 512
s = np.sin(2 * np.pi * 10 * np.arange(T) / 256)
X = np.outer(g, s)
Xr = X - X[V].mean(0, keepdims=True)
truth = Xr[:, 64]
spline = truth.copy(); spline[H] = (probe.spline_W(pos, V, H) @ Xr[V])[:, 64]
R = probe.reref_matrix(C, V); Lr = np.einsum("cd,dmi->cmi", R, L)
phys = truth.copy(); phys[H] = music_predict(Lr, V, H, Xr[V], 1)[:, 64]
vmax = np.abs(truth).max()

fig = plt.figure(figsize=(11, 7.6), facecolor="#fcfcfb")
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1], hspace=0.75, wspace=0.1)
# --- row 1: 1-D spline analogy
ax = fig.add_subplot(gs[0, :])
x = np.arange(9.0); y = np.exp(-0.5 * ((x - 4.6) / 1.1) ** 2) * -1.0 + 0.35 * np.exp(-0.5 * ((x - 1.5) / 1.0) ** 2)
xf = np.linspace(0, 8, 400); yf = np.exp(-0.5 * ((xf - 4.6) / 1.1) ** 2) * -1.0 + 0.35 * np.exp(-0.5 * ((xf - 1.5) / 1.0) ** 2)
known = [0, 1, 2, 3, 6, 7, 8]
cs = CubicSpline(x[known], y[known])
ax.plot(xf, yf, color=MUTED, lw=1.5, ls="--", label="true signal along the scalp")
ax.plot(xf, cs(xf), color=BLUE, lw=2.2, label="spline: smoothest curve through the known points")
ax.scatter(x[known], y[known], s=70, color=INK, zorder=5, label="electrodes we can see")
ax.scatter([4, 5], y[[4, 5]], s=90, facecolor="#fcfcfb", edgecolor=ORANGE, lw=2, zorder=5, label="hidden electrodes (true value)")
ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
ax.set_title("A spline = the smoothest curve (or surface) that passes through the electrodes you can see",
             loc="left", fontsize=10.5, weight="bold")
ax.text(4.5, cs(4.5) + 0.1, "the spline cuts\nstraight across the dip", ha="center", fontsize=8.5, color=BLUE)
# --- row 2: head maps
mask = np.zeros(C, bool); mask[H] = True
for i, (vals, ttl, sub) in enumerate((
        (truth, "Truth: one brain source", "a + pole and a − pole (orange circles = hidden)"),
        (spline, "Spline's guess", "smooths over the gap and misses the hidden − pole"),
        (phys, "Physics-aware guess", "works out the source from the visible pole,\nthen predicts the hidden one"))):
    ax = fig.add_subplot(gs[1, i])
    mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax), contours=0,
                         sensors=True, mask=mask,
                         mask_params=dict(marker="o", markerfacecolor="none", markeredgecolor=ORANGE, linewidth=0, markersize=8, markeredgewidth=1.8))
    ax.set_title(ttl, fontsize=10, weight="bold", color=INK, pad=6)
    ax.text(0.5, -0.12, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8.3, color=INK2)
fig.text(0.01, 0.505, "Same idea on the head (seen from above, nose at top; red = positive, blue = negative):",
         fontsize=10.5, weight="bold")
out = ROOT / "results" / "fig_explain_spline_vs_physics.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); print(out)
