"""BrainOmni BrainTokenizer (NeurIPS 2025) as a masked-electrode infiller.

The tokenizer was pretrained to rebuild ALL channels from 75% of them: encoder("inverse solution")
on visible channels -> latent neurons -> decoder("forward solution") queried with each target
electrode's position embedding. Output is each channel's waveform z-scored per 2 s window (shape).
Its own preprocessing: average reference + one global std over all EEG channels (relative
amplitudes kept), positions centred and scaled; 256 Hz, 0.1-96 Hz band-pass, 2 s windows.

Leak-proof use (our protocol): reference and std from the VISIBLE electrodes only; the encoder
sees only visible electrodes; hidden electrodes are queried by position.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = "https://github.com/OpenTSLab/BrainOmni"


def _stub_deepspeed():
    """vq.py imports deepspeed.comm only for multi-GPU codebook syncing; inference never needs it."""
    if "deepspeed" in sys.modules:
        return
    comm = types.ModuleType("deepspeed.comm")
    comm.is_initialized = lambda: False
    comm.get_world_size = lambda: 1
    comm.broadcast = lambda *a, **k: None
    comm.all_reduce = lambda *a, **k: None
    comm.ReduceOp = types.SimpleNamespace(SUM="sum")
    ds = types.ModuleType("deepspeed")
    ds.comm = comm
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.comm"] = comm


def setup(code_dir: Path, ckpt_dir: Path | None = None, random_init: bool = False):
    """Clone the repo (if needed), import BrainTokenizer, load weights. Returns (model, load_report)."""
    import torch
    code_dir = Path(code_dir)
    if not (code_dir / "braintokenizer").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(code_dir)], check=True)
    _stub_deepspeed()
    sys.path.insert(0, str(code_dir))
    from braintokenizer.model import BrainTokenizer
    if random_init:      # local plumbing test only
        cfg = dict(window_length=512, n_filters=32, ratios=[8, 4, 2], kernel_size=5, last_kernel_size=5,
                   n_dim=256, n_neuro=16, n_head=4, dropout=0.0, codebook_dim=256, codebook_size=512,
                   num_quantizers=4, rotation_trick=True, quantize_optimize_method="ema")
        model = BrainTokenizer(**cfg)
        report = dict(random_init=True, cfg=cfg)
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download("OpenTSLab/BrainOmni", "braintokenizer/model_cfg.json", local_dir=ckpt_dir)
        w_path = hf_hub_download("OpenTSLab/BrainOmni", "braintokenizer/BrainTokenizer.pt", local_dir=ckpt_dir)
        cfg = json.loads(Path(cfg_path).read_text())
        model = BrainTokenizer(**cfg)
        sd = torch.load(w_path, map_location="cpu", weights_only=True)
        res = model.load_state_dict(sd, strict=False)
        report = dict(cfg=cfg, missing=list(res.missing_keys), unexpected=list(res.unexpected_keys),
                      n_params_loaded=len(sd))
    model = model.float().eval()
    return model, report


def norm_pos(pos):
    """BrainOmni's normalize_pos for EEG: centre, scale by sqrt(3 * mean squared radius)."""
    p = pos - pos.mean(0, keepdims=True)
    return p / np.sqrt(3 * np.mean(np.sum(p ** 2, axis=1)))


def infill(model, X, pos_all, V, H, leaky=False):
    """X: (64, T) volts at 256 Hz (band-passed). Returns hidden waveforms (|H|, T), z-scored per 2 s window.

    leaky=False: visible-only reference and std (leak-proof).
    leaky=True : BrainOmni's own pipeline order — reference and std over ALL 64 channels (hidden included).
    """
    import torch
    if leaky:
        Y = X - X.mean(0, keepdims=True)
        Y = Y / (Y.std() + 1e-12)
        Yv = Y[V]
    else:
        Yv = X[V] - X[V].mean(0, keepdims=True)
        Yv = Yv / (Yv.std() + 1e-12)
    p6 = np.concatenate([norm_pos(pos_all), np.zeros_like(pos_all)], 1).astype(np.float32)
    with torch.no_grad():
        x = torch.tensor(Yv[None], dtype=torch.float32)
        se = model.sensor_embed(torch.tensor(p6[None]), torch.zeros(1, len(pos_all), dtype=torch.long))
        xu = model.unfold(x)                                   # 1 C N L
        feat = model.encoder(xu, se[:, V])
        feat, _, _ = model.quantizer(feat)
        rec = model.decoder(feat, se[:, H])                    # 1 |H| N L
    rec = rec[0].numpy()
    return rec.reshape(rec.shape[0], -1)[:, :X.shape[1]]
