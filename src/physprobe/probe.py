"""Real vs covariance-matched surrogate recordings, and the linear reference infill methods.

Real trial      : k dipoles (random location/orientation) with rhythmic + 1/f time courses,
                  through the sphere head model, scaled to `rms_uv`, plus white sensor noise.
Surrogate trial : Gaussian in space with the *ensemble* spatial covariance of the real signal,
                  same temporal process, same scaling and noise. Any fixed linear spatial
                  predictor has the same expected error on both.
"""
from __future__ import annotations

import numpy as np

from . import synth


def trial_cov_samples(L, k, n, rng):
    """Spatial covariance of the signal part of n random k-source trials (analytic)."""
    C, M, _ = L.shape
    S = np.zeros((C, C))
    for _ in range(n):
        G = np.zeros((C, C))
        for j in rng.choice(M, k, replace=False):
            o = rng.normal(size=3)
            o /= np.linalg.norm(o)
            l = L[:, j] @ o * rng.lognormal(0, 0.3)
            G += np.outer(l, l)
        S += G / np.trace(G)            # each trial scaled to the same total power
    return S / n


def make_trial(lf, k, kind, seconds, rng, sig_cov=None, rms_uv=20.0, snr_db=20.0):
    T = int(seconds * synth.FS)
    if kind == "real":
        X, _ = synth.focal_sources(lf, k, T, rng)
    else:
        w, V = np.linalg.eigh(sig_cov)
        A = V * np.sqrt(np.clip(w, 0, None))
        Z = 0.7 * synth.rhythm(len(w), T, rng) + 0.3 * synth.pink(len(w), T, rng)
        X = A @ Z
    X *= rms_uv * 1e-6 / X.std()
    noise = rng.normal(size=X.shape) * X.std() * 10 ** (-snr_db / 20)
    return X + noise, X.std() ** 2 * 10 ** (-snr_db / 10)


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
