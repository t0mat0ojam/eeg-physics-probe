"""Explainer figure: physically possible (dipole) vs impossible (matched random) scalp patterns."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physprobe import synth, probe
mne.set_log_level("ERROR")
lf = synth.build_free_leadfield(); L, info = lf["L"], lf["info"]
S1 = probe.single_source_cov(L)
w, V = np.linalg.eigh(S1); A1 = V * np.sqrt(np.clip(w, 0, None))
rng = np.random.default_rng(4)
fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.0), facecolor="#fcfcfb")
for c in range(4):
    j = rng.integers(L.shape[1]); o = rng.normal(size=3); o /= np.linalg.norm(o)
    real = L[:, j] @ o
    fake = A1 @ rng.normal(size=len(w))
    for r, (x, ttl) in enumerate(((real, "possible"), (fake, "impossible"))):
        ax = axes[r, c]
        mne.viz.plot_topomap(x / np.abs(x).max(), info, axes=ax, show=False, cmap="RdBu_r", vlim=(-1, 1),
                             contours=0, sensors=True)
fig.text(0.01, 0.93, "Made by one brain source\n(physically possible)", fontsize=10, weight="bold", va="top")
fig.text(0.01, 0.47, "Random smooth blob, same statistics\n(physically impossible)", fontsize=10, weight="bold", va="top")
fig.suptitle("Scalp patterns seen from above (nose at top; red = positive, blue = negative)",
             x=0.5, y=1.0, fontsize=11)
plt.subplots_adjust(left=0.22, hspace=0.25, wspace=0.05, top=0.9)
out = ROOT / "results" / "fig_explainer_patterns.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); print(out)
