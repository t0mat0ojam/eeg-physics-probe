"""Synthetic EEG from a spherical head model (MNE 4-layer sphere, EEGMMIDB 64-channel cap)."""
from __future__ import annotations

import numpy as np

from .leadfield_sphere import EEGMMIDB_CHANNELS

FS = 256.0


def build_free_leadfield(grid_mm: float = 10.0):
    """Free-orientation lead field (C, M, 3), average reference, plus electrode/source positions."""
    import mne
    mne.set_log_level("ERROR")
    info = mne.create_info(list(EEGMMIDB_CHANNELS), sfreq=FS, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    src = mne.setup_volume_source_space(pos=grid_mm, sphere=sphere, mindist=5.0, exclude=20.0)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False, eeg=True)
    L = fwd["sol"]["data"].astype(float)
    rr = fwd["src"][0]["rr"][fwd["src"][0]["vertno"]] - sphere["r0"]
    r_in = sphere["layers"][0]["rad"]
    keep = (rr[:, 2] > -0.01) & (np.linalg.norm(rr, axis=1) > 0.35 * r_in)
    idx = np.where(keep)[0]
    L = L.reshape(len(EEGMMIDB_CHANNELS), -1, 3)[:, idx]
    L = L - L.mean(0, keepdims=True)
    pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(info["chs"]))]) - sphere["r0"]
    return dict(L=L, pos=pos, rr=rr[idx], info=info)


def pink(n: int, T: int, rng, fmin: float = 0.5) -> np.ndarray:
    """n independent 1/f time series of length T, unit variance."""
    f = np.fft.rfftfreq(T, 1 / FS)
    amp = np.where(f > fmin, 1 / np.sqrt(np.maximum(f, fmin)), 0.0)
    X = np.fft.irfft(amp * (rng.normal(size=(n, len(f))) + 1j * rng.normal(size=(n, len(f)))), n=T)
    return X / X.std(1, keepdims=True)


def rhythm(n: int, T: int, rng, f0=(8, 13)) -> np.ndarray:
    """Band-limited bursts (e.g. alpha): narrow-band noise times a slow envelope, unit variance."""
    f = np.fft.rfftfreq(T, 1 / FS)
    centre = rng.uniform(*f0, size=(n, 1))
    amp = np.exp(-0.5 * ((f[None] - centre) / 1.0) ** 2)
    X = np.fft.irfft(amp * (rng.normal(size=(n, len(f))) + 1j * rng.normal(size=(n, len(f)))), n=T)
    env = 0.5 + np.abs(pink(n, T, rng, fmin=0.2))
    X = X * env
    return X / X.std(1, keepdims=True)


def focal_sources(lf, k: int, T: int, rng):
    """Scalp signals (C, T) from k dipoles with random location/orientation and rhythmic+1/f time courses."""
    L = lf["L"]
    M = L.shape[1]
    X = np.zeros((L.shape[0], T))
    locs = rng.choice(M, k, replace=False)
    tc = 0.7 * rhythm(k, T, rng) + 0.3 * pink(k, T, rng)
    for j, s in zip(locs, tc):
        o = rng.normal(size=3)
        o /= np.linalg.norm(o)
        X += np.outer(L[:, j] @ o, s * rng.lognormal(0, 0.3))
    return X, locs


def recording(lf, k: int, seconds: float, rng, rms_uv: float = 20.0, snr_db: float = 20.0):
    T = int(seconds * FS)
    X, locs = focal_sources(lf, k, T, rng)
    X *= rms_uv * 1e-6 / X.std()
    X += rng.normal(size=X.shape) * X.std() * 10 ** (-snr_db / 20)
    return X, locs


def save_fif(X: np.ndarray, path, info=None):
    import mne
    if info is None:
        info = mne.create_info(list(EEGMMIDB_CHANNELS), sfreq=FS, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    raw = mne.io.RawArray(X, info, verbose="ERROR")
    raw.save(path, overwrite=True, verbose="ERROR")
    return path
