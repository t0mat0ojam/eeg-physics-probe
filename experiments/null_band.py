"""Offline (CPU) reference scores under the exact probe protocol, large n.

H1 band  : physics-use scores of linear smoothers (splines = reference, IDW, best linear, smoothed linear).
H2 ref   : oracle physics model (RAP-MUSIC with the true lead field, k known) — what full use of
           head physics looks like.
Also per-trial SDs of (E_m - E_spline) for the power calculation.
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from physprobe import synth, probe
from pilot_shape import music_predict

n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4]
cfg = json.loads((ROOT / "jobs/probe1/config.json").read_text())
masks = [("scattered", m) for m in cfg["masks"]["scattered"]] + [("patch", m) for m in cfg["masks"]["patch"]]
lf = synth.build_free_leadfield(); L, pos = lf["L"], lf["pos"]; C = L.shape[0]
S1 = probe.single_source_cov(L)
fam = []
for _, H in masks:
    V = [c for c in range(C) if c not in H]
    R = probe.reref_matrix(C, V)
    fam.append(dict(V=V, H=H, R=R, Lr=np.einsum("cd,dmi->cmi", R, L), **probe.linear_family(pos, S1, V, H)))
rng = np.random.default_rng(99)
out, t0 = {}, time.time()
for k in ks:
    loc_stream = np.concatenate([rng.permutation(L.shape[1]) for _ in range(k * n // L.shape[1] + 1)])
    rec = {}
    for kind in ("real", "patterns"):
        for i in range(n):
            locs = loc_stream[i * k:(i + 1) * k] if kind == "real" else None
            X = probe.make_trial(lf, k, kind, cfg["seconds"], rng, S1, locs=locs)
            mi = i % len(masks); f = fam[mi]; mk = masks[mi][0]
            Xr = probe.highpass(probe.reref_visible(X, f["V"]))
            tH = Xr[f["H"]]
            E = {m: probe.shape_err(f[m] @ Xr[f["V"]], tH) for m in ("spline", "idw", "best-linear", "linear-smooth")}
            E["oracle"] = probe.shape_err(music_predict(f["Lr"], f["V"], f["H"], Xr[f["V"]], k), tH)
            for key in (mk, "all"):
                rec.setdefault((key, kind), []).append(E)
    for mk in ("scattered", "patch", "all"):
        e = {}
        for m in ("idw", "best-linear", "linear-smooth", "oracle"):
            dR = np.array([x[m] - x["spline"] for x in rec[(mk, "real")]])
            dP = np.array([x[m] - x["spline"] for x in rec[(mk, "patterns")]])
            e[m] = dict(score=float(dP.mean() - dR.mean()),
                        se=float(np.sqrt(dP.var(ddof=1) / len(dP) + dR.var(ddof=1) / len(dR))),
                        sd_real=float(dR.std(ddof=1)), sd_patt=float(dP.std(ddof=1)))
        e["mean_err"] = {kind: {m: float(np.mean([x[m] for x in rec[(mk, kind)]])) for m in rec[(mk, kind)][0]}
                         for kind in ("real", "patterns")}
        out[f"k={k} | {mk}"] = e
        print(f"k={k} {mk:9s} " + "  ".join(f"{m} {v['score']:+.3f}±{v['se']:.3f}" for m, v in e.items() if m != "mean_err")
              + f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)
(ROOT / "results" / "null_band.json").write_text(json.dumps(dict(n=n, ks=ks, protocol="leak-proof, shape error", res=out), indent=1))
