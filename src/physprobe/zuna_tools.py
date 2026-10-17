"""Thin wrapper around zuna.reconstruct_fif: seeded, batched over many files in one call."""
from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

HIGHPASS = 0.5


@contextlib.contextmanager
def zuna_seed(seed: int, extra=()):
    """reconstruct_fif has no seed argument; it runs eeg_eval.py with key=value overrides,
    and that script seeds torch from `seed`. Append seed=<n> to that subprocess call."""
    real = subprocess.run

    def patched(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any(str(c).endswith("eeg_eval.py") for c in cmd):
            cmd = list(cmd) + [f"seed={seed}"] + list(extra)
        return real(cmd, *a, **k)

    subprocess.run = patched
    try:
        yield
    finally:
        subprocess.run = real


def reconstruct_batch(fifs, hidden, out: Path, seed: int = 0, steps: int = 25, gpu="0", extra=()):
    """Run ZUNA once over all `fifs` (same hidden channels). Returns {name: (data, ch_names)}."""
    import mne
    from zuna import reconstruct_fif
    out = Path(out)
    inp = out / "input"
    shutil.rmtree(out, ignore_errors=True)
    inp.mkdir(parents=True)
    for f in fifs:
        shutil.copy(f, inp / Path(f).name)
    with zuna_seed(seed, extra):
        reconstruct_fif(input_dir=str(inp), output_dir=str(out / "fif"), figures_dir=str(out / "fig"),
                        tmp_dir=str(out / "tmp"), gpu_device=gpu, highpass_hz=HIGHPASS,
                        montage="standard_1005", repair_channels=list(hidden),
                        use_fif_annotations=False, annotate_infill=False, sample_steps=steps)
    res = {}
    for f in fifs:
        rec = mne.io.read_raw_fif(out / "fif" / "hybrid" / Path(f).name, preload=True, verbose="ERROR")
        res[Path(f).name] = (rec.get_data(), rec.ch_names)
    return res
