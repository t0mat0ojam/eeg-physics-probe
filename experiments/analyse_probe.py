"""Analyse rows.json from a probe run.

Readouts (pooled NMSE = sum of squared errors / sum of hidden-channel power; 95% bootstrap CIs by
resampling trials within each kind):
  physics-use score (primary) = [NMSE_zuna(patterns) - NMSE_zuna(real)]
                                - [NMSE_bestlin(patterns) - NMSE_bestlin(real)]
      Any linear method ties on real vs patterns in expectation, so subtracting the best-linear
      difference removes sampling noise between the two trial sets. > 0 means ZUNA does better on
      physically possible source patterns than on equally smooth impossible ones.
  beats-best-linear (secondary) = NMSE_bestlin(real) - NMSE_zuna(real)
  surrogate version of the score (secondary): 'surrogate' in place of 'patterns'.

  python experiments/analyse_probe.py results/kaggle/<job>/<run>/rows.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

METHODS = ("zuna", "zuna_single", "spline", "min-norm", "best-linear")


def table(rows):
    """dict[(k, mask_kind, kind)] -> per-trial arrays of err per method and power (masks summed)."""
    out = {}
    for r in rows:
        key = (r["k"], r["mask_kind"], r["kind"])
        d = out.setdefault(key, {})
        t = d.setdefault(r["trial"], {m: 0.0 for m in METHODS} | {"power": 0.0})
        for m in METHODS:
            t[m] += r[f"err_{m}"]
        t["power"] += r["power"]
    return {key: {m: np.array([t[m] for t in d.values()]) for m in METHODS + ("power",)} for key, d in out.items()}


def nmse(d, m, idx=None):
    idx = slice(None) if idx is None else idx
    return d[m][idx].sum() / d["power"][idx].sum()


def score(A, B, idxA=None, idxB=None, m="zuna", ref="best-linear"):
    """[NMSE_m(B) - NMSE_m(A)] - [NMSE_ref(B) - NMSE_ref(A)], A = real, B = control."""
    return (nmse(B, m, idxB) - nmse(A, m, idxA)) - (nmse(B, ref, idxB) - nmse(A, ref, idxA))


def boot(f, nA, nB, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = [f(rng.integers(0, nA, nA), rng.integers(0, nB, nB)) for _ in range(n)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def analyse(rows):
    T = table(rows)
    res = {}
    for (k, mk) in sorted({(k, mk) for (k, mk, _) in T}):
        g = {kind: T[(k, mk, kind)] for (kk, mm, kind) in T if kk == k and mm == mk}
        if "real" not in g:
            continue
        R = g["real"]
        nR = len(R["power"])
        e = dict(n_trials={kind: int(len(v["power"])) for kind, v in g.items()},
                 nmse={kind: {m: float(nmse(v, m)) for m in METHODS} for kind, v in g.items()})
        for ctrl in ("patterns", "surrogate"):
            if ctrl in g:
                B = g[ctrl]
                nB = len(B["power"])
                e[f"score_vs_{ctrl}"] = float(score(R, B))
                e[f"score_vs_{ctrl}_ci"] = boot(lambda a, b: score(R, B, a, b), nR, nB)
                for lin in ("spline", "min-norm"):   # linear methods should score ~0 themselves
                    e[f"linear_check_{lin}_vs_{ctrl}"] = float(score(R, B, m=lin))
        e["beats_best_linear"] = float(nmse(R, "best-linear") - nmse(R, "zuna"))
        e["beats_best_linear_ci"] = boot(lambda a, b: nmse(R, "best-linear", a) - nmse(R, "zuna", a), nR, 1)
        e["beats_spline"] = float(nmse(R, "spline") - nmse(R, "zuna"))
        res[f"k={k} | {mk}"] = e
    return res


if __name__ == "__main__":
    p = Path(sys.argv[1])
    res = analyse(json.loads(p.read_text()))
    (p.parent / "analysis.json").write_text(json.dumps(res, indent=1))
    for key, e in res.items():
        print(f"\n{key}  (trials: {e['n_trials']})")
        for kind, v in e["nmse"].items():
            print(f"   {kind:9s} " + "  ".join(f"{m} {x:.3f}" for m, x in v.items()))
        for c in ("patterns", "surrogate"):
            if f"score_vs_{c}" in e:
                lo, hi = e[f"score_vs_{c}_ci"]
                print(f"   physics-use score vs {c}: {e[f'score_vs_{c}']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                      f"   (spline check {e[f'linear_check_spline_vs_{c}']:+.3f})")
        lo, hi = e["beats_best_linear_ci"]
        print(f"   ZUNA beats best linear on real by {e['beats_best_linear']:+.3f} [{lo:+.3f}, {hi:+.3f}];"
              f" beats splines by {e['beats_spline']:+.3f}")
