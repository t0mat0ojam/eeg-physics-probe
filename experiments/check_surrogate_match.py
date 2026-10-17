"""Check: with many trials, linear infill error is equal on real and surrogate recordings."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physprobe import synth, probe
cfg = json.loads((ROOT / "jobs/probe1/config.json").read_text())
lf = synth.build_free_leadfield(); L, pos = lf["L"], lf["pos"]; C = L.shape[0]
rng = np.random.default_rng(5)
n, secs = int(sys.argv[1]) if len(sys.argv) > 1 else 200, 2.0
for k in cfg["ks"]:
    S1 = probe.single_source_cov(L)
    tot = probe.total_cov(S1)
    acc = {}
    for kind in ("real", "patterns", "surrogate"):
        E = {m: [] for m in ("spline", "best-linear")}
        for i in range(n):
            X = probe.make_trial(lf, k, kind, secs, rng, S1)
            for mk in ("scattered", "patch"):
                for H in cfg["masks"][mk]:
                    V = [c for c in range(C) if c not in H]
                    for m, W in (("spline", probe.spline_W(pos, V, H)), ("best-linear", probe.linear_W(tot, V, H))):
                        E[m].append((mk, ((X[H] - W @ X[V]) ** 2).sum(), (X[H] ** 2).sum()))
        acc[kind] = E
    for m in ("spline", "best-linear"):
        for mk in ("scattered", "patch"):
            r = []
            for kind in ("real", "patterns", "surrogate"):
                a = np.array([(e, p) for (kk, e, p) in acc[kind][m] if kk == mk])
                # bootstrap over trials (3 masks per trial share a trial)
                a = a.reshape(n, 3, 2).sum(1)
                bs = [a[idx, 0].sum() / a[idx, 1].sum() for idx in rng.integers(0, n, (1000, n))]
                r.append((a[:, 0].sum() / a[:, 1].sum(), np.quantile(bs, [.025, .975])))
            print(f"k={k:2d} {m:11s} {mk:9s} " + "  ".join(
                f"{nm} {x[0]:.3f} [{x[1][0]:.3f},{x[1][1]:.3f}]" for nm, x in zip(("real", "patt", "surr"), r)), flush=True)
