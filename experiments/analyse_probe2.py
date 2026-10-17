"""Probe 2 (diagnostic): ZUNA (mean of 4 samples, and single sample) vs splines on realistic recordings.

  python experiments/analyse_probe2.py <rows_A.json> [<rows_B.json>]
"""
import json, sys
from pathlib import Path
import numpy as np

def summarise(rows, label):
    m = {k: np.array([r[f"shape_{k}"] for r in rows]) for k in ("zuna", "zuna_single", "spline", "best-linear", "idw")}
    d = m["zuna"] - m["spline"]
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    out = dict(label=label, n=len(rows), mean_err={k: float(v.mean()) for k, v in m.items()},
               zuna_minus_spline=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
               frac_zuna_better=float((d < 0).mean()),
               averaging_gain=float((m["zuna_single"] - m["zuna"]).mean()))
    print(f"\n{label}  (n={len(rows)})")
    print("   mean shape error: " + "  ".join(f"{k} {v:.3f}" for k, v in out["mean_err"].items()))
    print(f"   ZUNA(mean of 4) - splines: {out['zuna_minus_spline']:+.3f} [{out['ci'][0]:+.3f}, {out['ci'][1]:+.3f}];"
          f" ZUNA better on {out['frac_zuna_better']:.0%}; averaging 4 samples lowers ZUNA's error by {out['averaging_gain']:.3f}")
    return out

res = [summarise(json.loads(Path(p).read_text()), lab) for p, lab in zip(sys.argv[1:], ("A: clean, 1 source", "B: with background activity"))]
Path(sys.argv[1]).parent.parent.joinpath("probe2_summary.json").write_text(json.dumps(res, indent=1))
