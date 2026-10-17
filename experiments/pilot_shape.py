"""CPU pilot v2: shape-based score (leak-proof) — does it still separate smoothing from physics?

Protocol matching the leak-proof ZUNA input: re-reference each recording to the average of the
VISIBLE electrodes, hidden electrodes zero-filled (so ZUNA's per-channel scale and average-reference
offset carry no hidden information). Score = 1 - Pearson r between predicted and true hidden
waveform, per hidden channel and 5 s segment, averaged — insensitive to amplitude.

Stand-ins: splines (H1), best linear (true covariance, visible-reference domain), minimum-norm,
and an oracle physics model: RAP-MUSIC scan over the true lead field for k dipoles + least squares
(nonlinear; knows the head). Reports, per k, each method's score on real vs patterns recordings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physprobe import synth, probe  # noqa: E402

SEG = 1280   # 5 s at 256 Hz


def reref(X, V):
    return X - X[V].mean(0, keepdims=True)


def shape_err(P, T):
    """mean over channels and 5 s segments of 1 - r."""
    out = []
    for s in range(0, T.shape[1] - SEG + 1, SEG):
        p, t = P[:, s:s + SEG], T[:, s:s + SEG]
        p = p - p.mean(1, keepdims=True); t = t - t.mean(1, keepdims=True)
        r = (p * t).sum(1) / np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1) + 1e-30)
        out.append(1 - r)
    return float(np.mean(out))


def music_predict(Lr, V, H, XV, k):
    """RAP-MUSIC with the (re-referenced) lead field Lr (C, M, 3); returns hidden prediction."""
    U, s, _ = np.linalg.svd(XV, full_matrices=False)
    Us = U[:, :k]
    LV, LH = Lr[V], Lr[H]
    M = Lr.shape[1]
    chosen = []                      # list of (loc, orientation-free columns)
    A = np.zeros((len(V), 0))
    for _ in range(k):
        if A.shape[1]:
            Pp = np.eye(len(V)) - A @ np.linalg.pinv(A)
        else:
            Pp = np.eye(len(V))
        Usp = Pp @ Us
        Usp, _ = np.linalg.qr(Usp)
        best, bj = -1, None
        for j in range(M):
            G = Pp @ LV[:, j]
            Q, _ = np.linalg.qr(G)
            val = np.linalg.svd(Q.T @ Usp, compute_uv=False)[0]
            if val > best:
                best, bj = val, j
        chosen.append(bj)
        A = np.concatenate([A, LV[:, bj]], 1)
    S = np.linalg.lstsq(A, XV, rcond=None)[0]                  # (3k, T)
    AH = np.concatenate([LH[:, j] for j in chosen], 1)
    return AH @ S


def main(n_trials=60, ks=(1, 2, 3, 4, 8), secs=5.0):
    lf = synth.build_free_leadfield()
    L, pos = lf["L"], lf["pos"]
    C = L.shape[0]
    cfg = json.loads((ROOT / "jobs/probe1/config.json").read_text())
    masks = cfg["masks"]["scattered"] + cfg["masks"]["patch"]
    kinds_of_mask = ["scattered"] * len(cfg["masks"]["scattered"]) + ["patch"] * len(cfg["masks"]["patch"])
    S1 = probe.single_source_cov(L)
    tot = probe.total_cov(S1)
    rng = np.random.default_rng(11)
    res = {}
    for k in ks:
        acc = {}
        for kind in ("real", "patterns"):
            for i in range(n_trials):
                X = probe.make_trial(lf, k, kind, secs, rng, S1)
                mi = i % len(masks)
                H = masks[mi]; V = [c for c in range(C) if c not in H]
                R = np.eye(C) - np.outer(np.ones(C), np.isin(np.arange(C), V)) / len(V)   # visible-average reference
                Xr = R @ X
                Lr = np.einsum("cd,dmi->cmi", R, L)
                totr = R @ tot @ R.T
                preds = {
                    "spline": probe.spline_W(pos, V, H) @ Xr[V],
                    "best-linear": probe.linear_W(totr, V, H) @ Xr[V],
                    "physics oracle (MUSIC)": music_predict(Lr, V, H, Xr[V], k),
                }
                for m, P in preds.items():
                    acc.setdefault((m, kinds_of_mask[mi], kind), []).append(shape_err(P, Xr[H]))
        for (m, mk, kind), v in acc.items():
            res.setdefault(f"k={k} | {mk} | {m}", {})[kind] = (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
        for mk in ("scattered", "patch"):
            for m in ("spline", "best-linear", "physics oracle (MUSIC)"):
                d = res[f"k={k} | {mk} | {m}"]
                print(f"k={k} {mk:9s} {m:24s} real {d['real'][0]:.3f}±{d['real'][1]:.3f}  "
                      f"patterns {d['patterns'][0]:.3f}±{d['patterns'][1]:.3f}  gap {d['patterns'][0] - d['real'][0]:+.3f}",
                      flush=True)
    (ROOT / "results" / "pilot_shape.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
