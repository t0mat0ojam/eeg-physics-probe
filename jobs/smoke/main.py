"""Smoke test: does ZUNA run on the Kaggle GPU, how fast, and is its infill sane on synthetic EEG?

Writes $JOB_OUT/result.json. Set MOCK=1 to replace ZUNA with spline interpolation (local test).
"""
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("JOB_OUT", "out_smoke"))
OUT.mkdir(parents=True, exist_ok=True)
MOCK = os.environ.get("MOCK") == "1"
res = dict(python=platform.python_version(), mock=MOCK, steps={})


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, default=str))


t0 = time.time()
try:
    res["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if not MOCK:
        res["pip_install"] = sh(f"{sys.executable} -m pip install -q zuna mne 2>&1 | tail -3")
        res["zuna_version"] = sh(f"{sys.executable} -c 'import zuna; print(zuna.__version__)'")
    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None
    res["torch"] = getattr(torch, "__version__", None)
    res["cuda"] = bool(torch and torch.cuda.is_available())
    if res["cuda"]:
        res["gpu"] = torch.cuda.get_device_name(0)
        res["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    res["steps"]["setup_min"] = (time.time() - t0) / 60
    save()

    import mne
    from physprobe import synth
    rng = np.random.default_rng(0)
    lf = synth.build_free_leadfield()
    work = Path("/tmp/smoke") if not MOCK else OUT / "work"
    (work / "in").mkdir(parents=True, exist_ok=True)
    truths, fifs = {}, []
    for i in range(4):
        X, _ = synth.recording(lf, k=1, seconds=10, rng=rng)
        f = work / "in" / f"sim{i}_raw.fif"
        synth.save_fif(X, f, lf["info"])
        truths[f.name] = X
        fifs.append(f)
    hidden = ["C3", "Cz", "Pz", "F4", "O1", "T8"]
    names = list(lf["info"].ch_names)
    H = [names.index(c) for c in hidden]

    def prep(X):  # what ZUNA compares against: same 0.5 Hz high-pass
        return mne.filter.filter_data(X, synth.FS, 0.5, None, verbose="ERROR")

    t1 = time.time()
    if MOCK:
        recs = {}
        for f in fifs:
            raw = mne.io.read_raw_fif(f, preload=True, verbose="ERROR")
            raw.info["bads"] = hidden
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            recs[f.name] = (raw.get_data(), raw.ch_names)
    else:
        from physprobe.zuna_tools import reconstruct_batch
        recs = reconstruct_batch(fifs, hidden, work / "zuna", seed=1, steps=25, gpu="0")
    res["steps"]["zuna_call_min"] = (time.time() - t1) / 60

    nm_z, nm_s = [], []
    for f in fifs:
        truth = prep(truths[f.name])
        data, ch = recs[f.name]
        idx = [ch.index(c) for c in hidden]
        T = min(truth.shape[1], data.shape[1])
        rz = prep(data[:, :T])[idx]
        raw = mne.io.RawArray(truths[f.name], lf["info"], verbose="ERROR")
        raw.info["bads"] = hidden
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        rs = prep(raw.get_data())[H, :T]
        tH = truth[H, :T]
        nm_z.append(float(((tH - rz) ** 2).sum() / (tH ** 2).sum()))
        nm_s.append(float(((tH - rs) ** 2).sum() / (tH ** 2).sum()))
    res["nmse_zuna"] = nm_z
    res["nmse_spline"] = nm_s
    res["ok"] = True
except Exception:
    res["ok"] = False
    res["error"] = traceback.format_exc()
res["steps"]["total_min"] = (time.time() - t0) / 60
save()
print(json.dumps(res, indent=1, default=str))
