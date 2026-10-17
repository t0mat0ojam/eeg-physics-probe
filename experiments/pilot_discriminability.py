"""Pilot (CPU, no foundation model): can the planned test tell "smoothing" from "physics"?

Logic of the test
-----------------
Any LINEAR infill method (spherical splines, minimum-norm, the best possible linear
predictor) has an expected error that depends only on the covariance of the fields.
So if we compare
  A  "real" fields   : made by k dipolar brain sources through a head model, and
  B  "surrogate" fields: Gaussian fields with exactly the same covariance as A,
every linear method must score the same on A and B. A method can only do better on A
by using the structure that comes from physics (a few sources projected through the
head). The gap (error on B - error on A) is therefore a "physics-use" score.

This pilot checks that the score behaves: ~0 for linear stand-ins of H1 (splines) and
linear physics (minimum-norm / optimal linear), clearly > 0 for an oracle single-dipole
fit (a stand-in for a model that has learned the forward physics), and shrinking as the
number of sources k grows (many sources -> fields become Gaussian).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physprobe.leadfield_sphere import EEGMMIDB_CHANNELS  # noqa: E402


def build_free_leadfield(grid_mm=10.0):
    import mne
    mne.set_log_level("ERROR")
    info = mne.create_info(list(EEGMMIDB_CHANNELS), sfreq=160.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    src = mne.setup_volume_source_space(pos=grid_mm, sphere=sphere, mindist=5.0, exclude=20.0)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False, eeg=True)
    L = fwd["sol"]["data"].astype(float)                     # (C, 3M) free orientation
    rr = fwd["src"][0]["rr"][fwd["src"][0]["vertno"]] - sphere["r0"]
    r_in = sphere["layers"][0]["rad"]
    keep = (rr[:, 2] > -0.01) & (np.linalg.norm(rr, axis=1) > 0.35 * r_in)   # upper, not too deep
    idx = np.where(keep)[0]
    L = L.reshape(len(EEGMMIDB_CHANNELS), -1, 3)[:, idx]     # (C, M, 3)
    L = L - L.mean(0, keepdims=True)                          # average reference
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(info["chs"]))]) - sphere["r0"]
    return L, pos, rr[idx], r_in


def make_fields(L, n, k, rng, snr_db=20.0):
    C, M, _ = L.shape
    X = np.zeros((n, C))
    for i in range(n):
        locs = rng.choice(M, k, replace=False)
        for j in locs:
            o = rng.normal(size=3); o /= np.linalg.norm(o)
            X[i] += rng.lognormal(0, 0.5) * L[:, j] @ o
    X /= np.sqrt((X ** 2).mean(1, keepdims=True))                # unit power per field
    noise = rng.normal(size=X.shape) * 10 ** (-snr_db / 20)
    return X + noise


def gaussian_surrogate(X, rng):
    cov = np.cov(X.T, bias=True)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0, None)
    Z = rng.normal(size=X.shape)
    return Z @ (V * np.sqrt(w)).T


def masks(pos, n_masks, n_hidden, rng, kind):
    C = len(pos)
    out = []
    for _ in range(n_masks):
        if kind == "scattered":
            H = rng.choice(C, n_hidden, replace=False)
        else:   # contiguous patch around a random electrode
            c = rng.integers(C)
            d = np.linalg.norm(pos - pos[c], axis=1)
            H = np.argsort(d)[:n_hidden]
        out.append(np.sort(H))
    return out


# ---------------- infill methods ----------------
def spline_predict(pos, V, H, XV):
    from mne.channels.interpolation import _make_interpolation_matrix
    W = _make_interpolation_matrix(pos[V], pos[H])
    return XV @ W.T


def linear_predict(cov, V, H, XV, ridge=1e-6):
    A = cov[np.ix_(V, V)] + ridge * np.trace(cov) / len(cov) * np.eye(len(V))
    W = cov[np.ix_(H, V)] @ np.linalg.inv(A)
    return XV @ W.T


def dipole_predict(L, V, H, XV):
    """Oracle single-dipole scan with the true head model (stand-in for 'learned physics')."""
    LV, LH = L[V], L[H]                              # (|V|, M, 3)
    M = L.shape[1]
    best_res = np.full(len(XV), np.inf)
    pred = np.zeros((len(XV), len(H)))
    for j in range(M):
        Q, R = np.linalg.qr(LV[:, j])                # (|V|,3)
        coef = XV @ Q                                # (n,3)
        res = (XV ** 2).sum(1) - (coef ** 2).sum(1)
        better = res < best_res
        if better.any():
            o = np.linalg.solve(R, coef[better].T).T  # (nb,3)
            pred[better] = o @ LH[:, j].T
            best_res[better] = res[better]
    return pred


def sq_err(Xh, P):
    """Per-field squared error and hidden-channel power; NMSE is pooled = sum(err)/sum(power)."""
    return np.stack([((Xh - P) ** 2).sum(1), (Xh ** 2).sum(1)], 1)


def pooled(E):
    return float(E[:, 0].sum() / E[:, 1].sum())


def boot_gap(EA, EB, rng, n=2000):
    ia = rng.integers(0, len(EA), (n, len(EA)))
    ib = rng.integers(0, len(EB), (n, len(EB)))
    g = EB[ib, 0].sum(1) / EB[ib, 1].sum(1) - EA[ia, 0].sum(1) / EA[ia, 1].sum(1)
    return [float(np.quantile(g, .025)), float(np.quantile(g, .975))]


def main():
    rng = np.random.default_rng(0)
    L, pos, rr, r_in = build_free_leadfield()
    print("lead field:", L.shape, flush=True)
    C = L.shape[0]
    n_fields, n_masks = 400, 8
    res = {}
    for k in (1, 2, 4, 16):
        A = make_fields(L, n_fields, k, rng)
        B = gaussian_surrogate(A, rng)
        covA = np.cov(A.T, bias=True)
        for kind, nh in (("scattered", 16), ("patch", 12)):
            Ms = masks(pos, n_masks, nh, rng, kind)
            for name in ("spline (H1: smoothing)", "best linear (any linear method)",
                         "minimum-norm (linear physics)", "dipole fit (nonlinear physics)"):
                ea, eb = [], []
                for H in Ms:
                    V = np.setdiff1d(np.arange(C), H)
                    for X, e in ((A, ea), (B, eb)):
                        if name.startswith("spline"):
                            P = spline_predict(pos, V, H, X[:, V])
                        elif name.startswith("best linear"):
                            P = linear_predict(covA, V, H, X[:, V])
                        elif name.startswith("minimum-norm"):
                            Lf = L.reshape(C, -1)
                            P = linear_predict(Lf @ Lf.T + 1e-2 * np.trace(Lf @ Lf.T) / C * np.eye(C), V, H, X[:, V])
                        else:
                            P = dipole_predict(L, V, H, X[:, V])
                        e.append(sq_err(X[:, H], P))
                ea, eb = np.concatenate(ea), np.concatenate(eb)
                key = f"k={k} | {kind} | {name}"
                ci = boot_gap(ea, eb, rng)
                res[key] = dict(nmse_real=pooled(ea), nmse_surrogate=pooled(eb),
                                gap=pooled(eb) - pooled(ea), gap_ci95=ci)
                print(f"{key:70s} real {pooled(ea):.3f}  surrogate {pooled(eb):.3f}  "
                      f"gap {pooled(eb) - pooled(ea):+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)
    out = ROOT / "results" / "pilot_discriminability.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
