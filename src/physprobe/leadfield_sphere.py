"""Synthetic linear volume-conduction model: X = L z + noise.

The lead field comes from MNE's 4-layer spherical head model (brain / CSF / skull /
scalp) evaluated at the 64 electrode positions of the PhysioNet EEGMMIDB cap, with
radially oriented dipoles on a shell 1 cm below the inner skull surface. No recorded
data is involved; this is the textbook forward model the theory section assumes.
"""
from __future__ import annotations

import numpy as np

EEGMMIDB_CHANNELS = (
    "FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 "
    "Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 "
    "TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz"
).split()


def fibonacci_hemisphere(n: int, z_min: float = -0.1) -> np.ndarray:
    """~Uniform unit vectors with z >= z_min (upper head)."""
    pts = []
    m = int(n * 2 / (1 - z_min)) + 8
    golden = np.pi * (3 - np.sqrt(5))
    for i in range(m):
        z = 1 - 2 * (i + 0.5) / m
        if z < z_min:
            continue
        r = np.sqrt(1 - z * z)
        pts.append((r * np.cos(golden * i), r * np.sin(golden * i), z))
    return np.array(pts)


def build_leadfield(n_sources: int = 600, verbose: bool = False):
    """Return dict(L, ch_names, ch_pos, src_pos, src_dir, info) with an average reference.

    L has shape (C, M), in volts per (A m); it is rescaled to unit mean column norm so
    source and noise variances can be stated in the same arbitrary units.
    """
    import mne

    mne.set_log_level("ERROR" if not verbose else "INFO")
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(list(EEGMMIDB_CHANNELS), sfreq=160.0, ch_types="eeg")
    info.set_montage(montage)

    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    r0 = sphere["r0"]
    r_brain = sphere["layers"][0]["rad"]          # inner-skull radius (m)
    dirs = fibonacci_hemisphere(n_sources)
    src_pos = r0 + (r_brain - 0.010) * dirs       # 1 cm below inner skull
    src = mne.setup_volume_source_space(pos=dict(rr=src_pos, nn=dirs), sphere=sphere,
                                        verbose=False)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False,
                                    eeg=True, verbose=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    L = fwd["sol"]["data"].astype(np.float64)     # (C, M), fixed (radial) orientation
    kept = fwd["src"][0]["vertno"]
    L = L - L.mean(axis=0, keepdims=True)         # average reference
    L = L / np.linalg.norm(L, axis=0).mean()
    ch_pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(info["chs"]))]) - r0
    return dict(L=L, ch_names=list(EEGMMIDB_CHANNELS), ch_pos=ch_pos,
                src_pos=src_pos[kept] - r0, src_dir=dirs[kept], info=info)


def angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(a @ b.T, -1, 1))


def source_covariance(src_dir, ch_pos, ch_names, *, bg_power=1.0, bg_width_deg=15.0,
                      alpha_power=10.0, alpha_at=("O1", "Oz", "O2", "POz"),
                      alpha_width_deg=20.0, task_at="C3", task_width_deg=8.0):
    """Background + strong occipital rhythm, and the spatial pattern of one task source.

    Returns (cov_bg, a_task):
      cov_bg : (M, M) covariance of everything that is *not* the task source
               (spatially smooth background plus a strong posterior patch)
      a_task : (M,) unit-norm spatial pattern of the task source over cortex; the task
               latent t ~ N(0, 1) enters the sources as sqrt(task_power) * a_task * t.
    """
    d = angular_distance(src_dir, src_dir)
    w = np.deg2rad(bg_width_deg)
    cov_bg = bg_power * np.exp(-0.5 * (d / w) ** 2)

    def patch(names, width_deg):
        centre = np.mean([ch_pos[ch_names.index(n)] for n in np.atleast_1d(names)], axis=0)
        dd = angular_distance(src_dir, centre[None])[:, 0]
        p = np.exp(-0.5 * (dd / np.deg2rad(width_deg)) ** 2)
        return p / np.linalg.norm(p)

    a_alpha = patch(alpha_at, alpha_width_deg)
    cov_bg = cov_bg + alpha_power * np.outer(a_alpha, a_alpha) * len(a_alpha) / 10.0
    return cov_bg, patch(task_at, task_width_deg)
