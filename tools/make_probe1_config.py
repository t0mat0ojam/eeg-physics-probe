"""Write jobs/probe1/config.json with fixed (pre-registered) masks."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physprobe import synth, probe
lf = synth.build_free_leadfield()
rng = np.random.default_rng(2026)
cfg = dict(seed=17, ks=[1, 4, 16], n_trials=10, copies=2, seconds=10, steps=25, cov_samples=2000,
           masks=dict(scattered=probe.masks(lf["pos"], 3, 16, rng, "scattered"),
                      patch=probe.masks(lf["pos"], 3, 12, rng, "patch")))
names = lf["info"].ch_names
cfg["mask_names"] = {k: [[names[i] for i in m] for m in v] for k, v in cfg["masks"].items()}
(ROOT / "jobs/probe1/config.json").write_text(json.dumps(cfg, indent=1))
print(json.dumps(cfg["mask_names"], indent=0))
