"""Analyse a probe run whose rows contain the model, the linear family and the oracle (in-run band).

  python experiments/analyse_inrun.py <rows.json> <model_key> [<model_key2> ...]
Score_m = mean_patterns(E_m - E_spline) - mean_real(E_m - E_spline), all masks pooled; 95% bootstrap CI.
H1 band = [min, max] over linear smoothers (spline = 0, idw, best-linear, linear-smooth); H2 = oracle.
"""
import json
import sys
from pathlib import Path

import numpy as np

LIN = ("idw", "best-linear", "linear-smooth")


def score(R, P, m, iR=None, iP=None):
    dR = np.array([r[f"shape_{m}"] - r["shape_spline"] for r in R])
    dP = np.array([r[f"shape_{m}"] - r["shape_spline"] for r in P])
    if iR is not None:
        dR, dP = dR[iR], dP[iP]
    return float(dP.mean() - dR.mean())


def boot(R, P, m, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = [score(R, P, m, rng.integers(0, len(R), len(R)), rng.integers(0, len(P), len(P))) for _ in range(n)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def analyse(rows, models):
    out = {}
    for k in sorted({r["k"] for r in rows}):
        for mk in ("all", "patch", "scattered"):
            sel = [r for r in rows if r["k"] == k and (mk == "all" or r["mask_kind"] == mk)]
            R = [r for r in sel if r["kind"] == "real"]
            P = [r for r in sel if r["kind"] == "patterns"]
            if not R or not P:
                continue
            e = dict(n=dict(real=len(R), patterns=len(P)))
            lin = [score(R, P, m) for m in LIN] + [0.0]
            e["band"] = [min(lin), max(lin)]
            e["oracle"] = score(R, P, "oracle")
            e["mean_err"] = {kind: {m: float(np.mean([r[f"shape_{m}"] for r in S]))
                                    for m in ("spline",) + LIN + ("oracle",) + tuple(models)}
                             for kind, S in (("real", R), ("patterns", P))}
            for m in models:
                e[f"score_{m}"] = score(R, P, m)
                e[f"score_{m}_ci"] = boot(R, P, m)
            out[f"k={k} | {mk}"] = e
    return out


if __name__ == "__main__":
    p = Path(sys.argv[1])
    models = sys.argv[2:]
    res = analyse(json.loads(p.read_text()), models)
    (p.parent / "analysis.json").write_text(json.dumps(res, indent=1))
    for key, e in res.items():
        if not key.endswith("all"):
            continue
        print(f"\n{key}  n={e['n']}  H1 band [{e['band'][0]:+.3f}, {e['band'][1]:+.3f}]  oracle {e['oracle']:+.3f}")
        for kind, v in e["mean_err"].items():
            print(f"   {kind:9s} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()))
        for m in models:
            lo, hi = e[f"score_{m}_ci"]
            print(f"   score {m:18s} {e[f'score_{m}']:+.3f} [{lo:+.3f}, {hi:+.3f}]  ({e[f'score_{m}'] / e['oracle']:.0%} of oracle)")
