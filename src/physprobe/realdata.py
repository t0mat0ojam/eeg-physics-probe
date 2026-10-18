"""Real EEG segments from EEGMMIDB (PhysioNet), in the 64-channel order used everywhere else."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .leadfield_sphere import EEGMMIDB_CHANNELS
from .synth import FS


def _edf_path(root: Path, subject: int, run: int):
    for p in (root / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf",
              root / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf"):
        if p.exists():
            return p
    return None


def load_segments(subjects, runs, root: Path, start_s=10.0, seconds=10.0, highpass=0.5, download=True):
    """One segment per (subject, run): high-pass the whole run, resample to 256 Hz, cut [start, start+seconds).
    Returns list of dict(name, subject, run, X (64, T) volts). Missing files are skipped (reported)."""
    import mne
    from mne.datasets import eegbci
    mne.set_log_level("ERROR")
    root = Path(root)
    segs, missing = [], []
    for s in subjects:
        for r in runs:
            p = _edf_path(root, s, r)
            if p is None and download:
                try:
                    got = eegbci.load_data(s, [r], path=str(root), update_path=False)
                    p = Path(got[0])
                except Exception as e:  # network or server error
                    missing.append((s, r, str(e)[:120]))
                    continue
            if p is None:
                missing.append((s, r, "not found"))
                continue
            raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            raw.pick(list(EEGMMIDB_CHANNELS))
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
            raw.filter(highpass, None, verbose="ERROR")
            raw.resample(FS, verbose="ERROR")
            a, b = int(start_s * FS), int((start_s + seconds) * FS)
            X = raw.get_data()[:, a:b]
            if X.shape[1] < b - a:
                missing.append((s, r, "too short"))
                continue
            segs.append(dict(name=f"S{s:03d}R{r:02d}", subject=s, run=r, X=X))
    return segs, missing


def loso_covariances(segs):
    """Leave-one-subject-out pooled spatial covariance (all-channel average reference), per subject."""
    C = segs[0]["X"].shape[0]
    covs = {}
    per = {}
    for g in segs:
        Y = g["X"] - g["X"].mean(0, keepdims=True)
        per.setdefault(g["subject"], []).append(Y @ Y.T / Y.shape[1])
    tot = sum(sum(v) for v in per.values())
    n = sum(len(v) for v in per.values())
    for s, v in per.items():
        covs[s] = (tot - sum(v)) / (n - len(v))
    return covs
