"""Analyse rows.json from a (leak-proof, shape-scored) probe run.

E = shape error (mean 1 - r over hidden channels and 5 s segments), one value per (trial, mask).
Physics-use score of method m (reference = splines, computed on the same recordings):
    score_m = mean_patterns(E_m - E_spline) - mean_real(E_m - E_spline)
> 0 means m gains more over splines on physically possible patterns than on random smooth ones.
Linear smoothers define the H1 band (their scores are estimated with large n offline:
results/null_band.json); the oracle physics model defines the H2 reference.

  python experiments/analyse_probe.py results/kaggle/<job>/<run>/rows.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REF = "spline"


def groups(rows):
    g = {}
    for r in rows:
        g.setdefault((r["k"], r["mask_kind"], r["kind"]), []).append(r)
    return g


def score(G, k, mk, m, idx_real=None, idx_patt=None):
    R, P = G[(k, mk, "real")], G[(k, mk, "patterns")]
    dR = np.array([r[f"shape_{m}"] - r[f"shape_{REF}"] for r in R])
    dP = np.array([r[f"shape_{m}"] - r[f"shape_{REF}"] for r in P])
    if idx_real is not None:
        dR, dP = dR[idx_real], dP[idx_patt]
    return float(dP.mean() - dR.mean())


def boot(G, k, mk, m, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    nR, nP = len(G[(k, mk, "real")]), len(G[(k, mk, "patterns")])
    v = [score(G, k, mk, m, rng.integers(0, nR, nR), rng.integers(0, nP, nP)) for _ in range(n)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def analyse(rows, methods=("zuna", "zuna_single", "best-linear", "linear-smooth", "idw")):
    G = groups(rows)
    out = {}
    combos = sorted({(k, mk) for (k, mk, kind) in G if kind == "real" and (k, mk, "patterns") in G})
    for k, mk in combos + [(k, "all") for k in sorted({k for k, _ in combos})]:
        if mk == "all":
            Gk = {("x", "all", kind): [r for (kk, _, kd), v in G.items() if kk == k and kd == kind for r in v]
                  for kind in ("real", "patterns")}
            key = ("x", "all")
        else:
            Gk, key = G, (k, mk)
        e = dict(n=dict(real=len(Gk[(*key, "real")]), patterns=len(Gk[(*key, "patterns")])))
        e["mean_shape_err"] = {kind: {m: float(np.mean([r[f"shape_{m}"] for r in Gk[(*key, kind)]]))
                                      for m in methods + (REF,)} for kind in ("real", "patterns")}
        for m in methods:
            e[f"score_{m}"] = score(Gk, *key, m)
            e[f"score_{m}_ci"] = boot(Gk, *key, m)
        out[f"k={k} | {mk}"] = e
    return out


if __name__ == "__main__":
    p = Path(sys.argv[1])
    res = analyse(json.loads(p.read_text()))
    (p.parent / "analysis.json").write_text(json.dumps(res, indent=1))
    for key, e in res.items():
        print(f"\n{key}  n={e['n']}")
        for kind, v in e["mean_shape_err"].items():
            print(f"   {kind:9s} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()))
        for m in ("zuna", "zuna_single", "best-linear", "linear-smooth", "idw"):
            lo, hi = e[f"score_{m}_ci"]
            print(f"   physics-use score {m:12s} {e[f'score_{m}']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
