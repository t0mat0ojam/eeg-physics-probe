"""Run ZUNA 1.1 inference directly (same call as zuna.reconstruct_fif, minus the per-file figures).

Each input .fif carries its own hidden electrodes in info['bads'] (ZUNA always masks those), so one
call can cover many recordings with different masks. Several calls can run in parallel, one per GPU.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HIGHPASS = 0.5


def _paths():
    import zuna
    app = Path(zuna.__file__).parent / "inference/AY2l/lingua/apps/AY2latent_bci"
    return app, app / "eeg_eval.py", app / "configs/config_infer_fif.yaml", app.parent.parent


def launch(input_dir, output_dir, tmp_dir, gpu: int, seed: int, steps: int = 25, log_path=None):
    """Start one ZUNA inference process on one GPU; returns the Popen handle."""
    app, eeg_eval, config, lingua_root = _paths()
    for d in (output_dir, tmp_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    ov = {
        "config": config, "dump_dir": Path(tmp_dir).absolute(),
        "inference_figures_dir": Path(tmp_dir).absolute(),
        "data.data_dir": Path(input_dir).absolute(), "data.v4_recon_save_fif": "true",
        "data.v4_recon_out_dir": Path(output_dir).absolute(), "data.v4_segment_sec": 5.0,
        "data.v4_montage": "standard_1005", "data.v4_use_fif_annotations": "false",
        "data.v4_recon_annotate_infill": "false", "data.v4_recon_save_preprocessed": "false",
        "data.v4_bad_token_overlap": 0.0, "diffusion_cfg": 1.0, "diffusion_sample_steps": steps,
        "data.v4_highpass_hz": HIGHPASS, "seed": seed,
    }
    cmd = [sys.executable, str(eeg_eval)] + [f"{k}={v}" for k, v in ov.items()]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WANDB_MODE"] = "disabled"
    # Pretend to be a 1-process torchrun job so lingua uses OUR port (it otherwise derives the same
    # port for every process from SLURM_JOB_ID, and two processes on one machine collide).
    env.update(LOCAL_RANK="0", RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(29500 + 7 * int(gpu)))
    env["PYTHONPATH"] = os.pathsep.join([str(lingua_root), str(app), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    for k in [k for k in env if k.startswith("SLURM_")]:
        del env[k]
    out = open(log_path, "w") if log_path else subprocess.DEVNULL
    return subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)


def run_parallel(groups, work: Path, seed: int, steps: int = 25, n_gpus: int = 1, log_dir=None):
    """groups: list of lists of .fif paths, one list per GPU. Returns {file name: (data, ch_names)}."""
    import mne
    procs = []
    for g, fifs in enumerate(groups):
        if not fifs:
            continue
        d = Path(work) / f"gpu{g}"
        shutil.rmtree(d, ignore_errors=True)
        (d / "in").mkdir(parents=True)
        for f in fifs:
            os.link(f, d / "in" / Path(f).name) if hasattr(os, "link") else shutil.copy(f, d / "in" / Path(f).name)
        procs.append((g, d, fifs, launch(d / "in", d / "out", d / "tmp", gpu=g % max(n_gpus, 1),
                                         seed=seed + 1000 * g, steps=steps, log_path=d / "zuna.log")))
    import time as _time
    t0 = _time.time()
    while any(p.poll() is None for *_, p in procs):          # heartbeat so notebooks don't look frozen
        _time.sleep(30)
        el = (_time.time() - t0) / 60
        if int(el * 2) % 10 == 0:
            print(f"  ... ZUNA still running ({el:.0f} min so far)", flush=True)
    res, failures = {}, []
    for g, d, fifs, p in procs:
        rc = p.wait()
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            txt = (d / "zuna.log").read_text(errors="replace")
            (Path(log_dir) / f"zuna_gpu{g}_tail.log").write_text(txt[-20000:])
        if rc != 0:
            tail = (d / "zuna.log").read_text()[-3000:]
            raise RuntimeError(f"ZUNA process on GPU {g} failed (exit {rc}):\n{tail}")
        for f in fifs:
            rec = mne.io.read_raw_fif(d / "out" / "hybrid" / Path(f).name, preload=True, verbose="ERROR")
            res[Path(f).name] = (rec.get_data(), rec.ch_names)
    return res
