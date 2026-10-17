"""Real vs covariance-matched surrogate recordings, and the linear reference infill methods.

Real trial      : k dipoles (random location/orientation) with rhythmic + 1/f time courses,
                  through the sphere head model, scaled to `rms_uv`, plus white sensor noise.
Patterns trial  : k sources with the same time courses, but random smooth scalp patterns drawn from
                  the ensemble covariance (same per-trial rank as real; not dipole fields).
Surrogate trial : Gaussian in space with the *ensemble* spatial covariance of the real signal,
                  same temporal process, same scaling and noise. Any fixed linear spatial
                  predictor has the same expected error on both.
"""
from __future__ import annotations

import numpy as np

from . import synth


AMP_SIGMA = 0.3                                   # log-normal source amplitude spread
EA2 = float(np.exp(2 * AMP_SIGMA ** 2))           # E[a^2]


def single_source_cov(L):
    """E[l l^T] for one dipole at a uniformly random grid location with random orientation."""
    C, M, _ = L.shape
    return np.einsum("cmi,dmi->cd", L, L) / (3 * M)


def time_courses(k, T, rng):
    tc = 0.7 * synth.rhythm(k, T, rng) + 0.3 * synth.pink(k, T, rng)
    return tc / tc.std(1, keepdims=True)


def make_trial(lf, k, kind, seconds, rng, S1, rms_uv=20.0, snr_db=20.0, locs=None):
    """One recording. All trials of a given k share ONE global gain (no per-trial rescaling),
    so the three kinds have the same ensemble covariance k*E[a^2]*S1 by construction."""
    L = lf["L"]
    C, M, _ = L.shape
    T = int(seconds * synth.FS)
    w, V = np.linalg.eigh(S1)
    A1 = V * np.sqrt(np.clip(w, 0, None))
    if kind == "surrogate":
        X = np.sqrt(k * EA2) * A1 @ time_courses(C, T, rng)
    else:
        if kind == "real":
            locs = rng.integers(0, M, k) if locs is None else np.asarray(locs)
            o = rng.normal(size=(k, 3))
            o /= np.linalg.norm(o, axis=1, keepdims=True)
            G = np.einsum("cki,ki->ck", L[:, locs], o)
        elif kind == "patterns":
            G = A1 @ rng.normal(size=(C, k))
        else:
            raise ValueError(kind)
        X = (G * rng.lognormal(0, AMP_SIGMA, size=k)) @ time_courses(k, T, rng)
    gain = rms_uv * 1e-6 / np.sqrt(k * EA2 * np.trace(S1) / C)
    noise_sd = rms_uv * 1e-6 * 10 ** (-snr_db / 20)
    return gain * X + noise_sd * rng.normal(size=(C, T))


def total_cov(S1, snr_db=20.0):
    """Covariance (up to scale) of any kind of trial, signal + sensor noise."""
    C = len(S1)
    return S1 / np.trace(S1) + 10 ** (-snr_db / 10) * np.eye(C) / C


def masks(pos, n_masks, n_hidden, rng, kind):
    out = []
    for _ in range(n_masks):
        if kind == "scattered":
            H = rng.choice(len(pos), n_hidden, replace=False)
        else:
            c = rng.integers(len(pos))
            H = np.argsort(np.linalg.norm(pos - pos[c], axis=1))[:n_hidden]
        out.append(sorted(int(h) for h in H))
    return out


def spline_W(pos, V, H):
    from mne.channels.interpolation import _make_interpolation_matrix
    return _make_interpolation_matrix(pos[V], pos[H])


def linear_W(cov, V, H, ridge=1e-6):
    A = cov[np.ix_(V, V)] + ridge * np.trace(cov) / len(cov) * np.eye(len(V))
    return cov[np.ix_(H, V)] @ np.linalg.inv(A)


def highpass(X, fc=0.5):
    import mne
    return mne.filter.filter_data(X, synth.FS, fc, None, verbose="ERROR")


# ---------------- leak-proof protocol (see jobs/leak) ----------------
SEG = 1280   # ZUNA's 5 s segment at 256 Hz


def reref_matrix(C, V):
    """Re-reference to the average of the visible electrodes V."""
    m = np.zeros(C)
    m[list(V)] = 1.0 / len(V)
    return np.eye(C) - np.outer(np.ones(C), m)


def reref_visible(X, V):
    return X - X[list(V)].mean(0, keepdims=True)


def leakproof_input(X, H):
    """Visible-average reference, hidden rows zeroed: ZUNA's per-channel mean/std and its
    average-reference offset then carry no information about the hidden electrodes."""
    C = X.shape[0]
    V = [c for c in range(C) if c not in set(H)]
    Y = reref_visible(X, V)
    Y[list(H)] = 0.0
    return Y


def shape_err(P, T):
    """Mean over channels and 5 s segments of (1 - Pearson r). Amplitude-free."""
    out = []
    for s0 in range(0, T.shape[1] - SEG + 1, SEG):
        p = P[:, s0:s0 + SEG] - P[:, s0:s0 + SEG].mean(1, keepdims=True)
        t = T[:, s0:s0 + SEG] - T[:, s0:s0 + SEG].mean(1, keepdims=True)
        r = (p * t).sum(1) / np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1) + 1e-300)
        out.append(1 - r)
    return float(np.mean(out))


def idw_W(pos, V, H, power=2.0):
    """Inverse-distance-weighted average of visible electrodes (crude smoother)."""
    d = np.linalg.norm(pos[list(H)][:, None] - pos[list(V)][None], axis=2)
    w = 1.0 / np.maximum(d, 1e-6) ** power
    return w / w.sum(1, keepdims=True)


def linear_family(pos, S1, V, H):
    """Linear smoothers spanning the H1 band, all in the visible-reference domain."""
    C = len(S1)
    R = reref_matrix(C, V)
    W = {"spline": spline_W(pos, V, H), "idw": idw_W(pos, V, H)}
    for name, nf in (("best-linear", 0.01), ("linear-smooth", 0.3)):
        cov = S1 / np.trace(S1) + nf * np.eye(C) / C
        W[name] = linear_W(R @ cov @ R.T, V, H)
    return W
