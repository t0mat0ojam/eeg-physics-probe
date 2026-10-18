"""Probe 4, BrainOmni part (Kaggle CPU): real EEGMMIDB segments, leak-proof, vs splines and best linear.

Shape errors per 2 s window (BrainOmni's window) AND per 5 s window (ZUNA's), for every method, so the
ZUNA part (GPU, separate job) can be compared on identical segments and masks.
LOCAL_TEST=1: random weights, local EDFs (runs 4, 8 of subjects 21-22).
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("JOB_OUT", "out_real_brainomni")); OUT.mkdir(parents=True, exist_ok=True)
LOCAL = os.environ.get("LOCAL_TEST") == "1"
cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
if LOCAL:
    cfg.update(subjects=[21, 22], runs=[4, 8])
log = dict(config=cfg, local_test=LOCAL)
save_log = lambda: (OUT / "log.json").write_text(json.dumps(log, indent=1, default=str))


def shape_err_w(P, T, W):
    out = []
    for s0 in range(0, T.shape[1] - W + 1, W):
        p = P[:, s0:s0 + W] - P[:, s0:s0 + W].mean(1, keepdims=True)
        t = T[:, s0:s0 + W] - T[:, s0:s0 + W].mean(1, keepdims=True)
        out.append(1 - (p * t).sum(1) / np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1) + 1e-300))
    return float(np.mean(out))


t0 = time.time()
try:
    if not LOCAL:
        subprocess.run(f"{sys.executable} -m pip install -q einops einx vector-quantize-pytorch==1.22.10 "
                       f"huggingface_hub mne", shell=True, check=True)
    import mne
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    from physprobe import probe, realdata, brainomni_tools as bt
    from physprobe.synth import build_free_leadfield
    mne.set_log_level("ERROR")
    root = Path("/home/claude/which-information/data/eegmmidb_v2/eeg2") if LOCAL else Path("/kaggle/temp/eegmmidb")
    segs, missing = realdata.load_segments(cfg["subjects"], cfg["runs"], root, download=not LOCAL)
    log.update(n_segments=len(segs), missing=missing)
    save_log()
    print(f"{len(segs)} segments, {len(missing)} missing", flush=True)
    code_dir = Path("/home/claude/fm_repos/brainomni") if LOCAL else Path("/kaggle/temp/BrainOmni")
    model, rep = bt.setup(code_dir, ckpt_dir=None if LOCAL else Path("/kaggle/temp/ckpt"), random_init=LOCAL)
    log["load_report"] = {k: v for k, v in rep.items() if k != "cfg"}
    lf = build_free_leadfield()
    pos = lf["pos"]
    C = pos.shape[0]
    masks = [("scattered", m) for m in cfg["masks"]["scattered"]] + [("patch", m) for m in cfg["masks"]["patch"]]
    covs = realdata.loso_covariances(segs)
    rows = []
    for n, g in enumerate(segs):
        mi = n % len(masks)
        mk, H = masks[mi]
        V = [c for c in range(C) if c not in H]
        R = probe.reref_matrix(C, V)
        Xr = probe.reref_visible(g["X"], V)
        tH = Xr[H]
        preds = {"spline": probe.spline_W(pos, V, H) @ Xr[V],
                 "best-linear": probe.linear_W(R @ covs[g["subject"]] @ R.T, V, H) @ Xr[V],
                 "brainomni": bt.infill(model, g["X"], pos, V, H, leaky=False),
                 "brainomni_leaky": bt.infill(model, g["X"], pos, V, H, leaky=True)}
        row = dict(seg=g["name"], subject=g["subject"], run=g["run"], mask=mi, mask_kind=mk)
        for m, P in preds.items():
            row[f"shape2_{m}"] = shape_err_w(P, tH, 512)
            row[f"shape5_{m}"] = shape_err_w(P, tH, 1280)
        for m in ("spline", "best-linear"):
            row[f"nmse_visref_{m}"] = float(((preds[m] - tH) ** 2).sum() / (tH ** 2).sum())
        rows.append(row)
        if (n + 1) % 10 == 0 or n + 1 == len(segs):
            (OUT / "rows.json").write_text(json.dumps(rows))
            log["done"] = n + 1
            save_log()
            print(f"{n + 1}/{len(segs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    log["ok"] = True
except Exception:
    log["ok"] = False
    log["error"] = traceback.format_exc()
    print(log["error"], flush=True)
log["total_min"] = (time.time() - t0) / 60
save_log()
