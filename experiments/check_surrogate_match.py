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
    sc = probe.trial_cov_samples(L, k, 2000, rng)
    tot = sc / np.trace(sc) + 0.01 * np.eye(C) / C
    acc = {}
    for kind in ("real", "surrogate"):
        E = {m: [] for m in ("spline", "best-linear")}
        for i in range(n):
            X, _ = probe.make_trial(lf, k, kind, secs, rng, sc)
            for mk in ("scattered", "patch"):
                for H in cfg["masks"][mk]:
                    V = [c for c in range(C) if c not in H]
                    for m, W in (("spline", probe.spline_W(pos, V, H)), ("best-linear", probe.linear_W(tot, V, H))):
                        E[m].append((mk, ((X[H] - W @ X[V]) ** 2).sum(), (X[H] ** 2).sum()))
        acc[kind] = E
    for m in ("spline", "best-linear"):
        for mk in ("scattered", "patch"):
            r = []
            for kind in ("real", "surrogate"):
                a = np.array([(e, p) for (kk, e, p) in acc[kind][m] if kk == mk])
                # bootstrap over trials (3 masks per trial share a trial)
                a = a.reshape(n, 3, 2).sum(1)
                bs = [a[idx, 0].sum() / a[idx, 1].sum() for idx in rng.integers(0, n, (1000, n))]
                r.append((a[:, 0].sum() / a[:, 1].sum(), np.quantile(bs, [.025, .975])))
            print(f"k={k:2d} {m:11s} {mk:9s} real {r[0][0]:.3f} [{r[0][1][0]:.3f},{r[0][1][1]:.3f}]  "
                  f"surr {r[1][0]:.3f} [{r[1][1][0]:.3f},{r[1][1][1]:.3f}]", flush=True)
