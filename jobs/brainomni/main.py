"""Probe 3: BrainOmni BrainTokenizer under the leak-proof physics probe (Kaggle CPU; no GPU quota).

Same recordings design as probe 1 (real vs patterns, k = 1, 2, 4), same masks. Per (recording, mask):
shape errors per 2 s window (BrainOmni's window) for BrainOmni (leak-proof), BrainOmni (its own
all-channel reference, 'leaky'), the linear smoother family and the RAP-MUSIC oracle — all on the
same recordings, so the H1 band and H2 reference are measured in-run.
Writes $JOB_OUT/rows.json and log.json every 25 units.  LOCAL_TEST=1: random weights, tiny run.
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("JOB_OUT", "out_brainomni")); OUT.mkdir(parents=True, exist_ok=True)
LOCAL = os.environ.get("LOCAL_TEST") == "1"
HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "config.json").read_text())
if LOCAL:
    cfg.update(n_trials={"1": 2, "2": 1, "4": 1})
log = dict(config=cfg, local_test=LOCAL)
save_log = lambda: (OUT / "log.json").write_text(json.dumps(log, indent=1, default=str))
W = 512                                                     # 2 s at 256 Hz


def shape_err_w(P, T):
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
    from physprobe import synth, probe, brainomni_tools as bt
    from physprobe.music import music_predict
    mne.set_log_level("ERROR")
    code_dir = Path("/home/claude/fm_repos/brainomni") if LOCAL else Path("/kaggle/temp/BrainOmni")
    model, rep = bt.setup(code_dir, ckpt_dir=None if LOCAL else Path("/kaggle/temp/ckpt"), random_init=LOCAL)
    log["load_report"] = rep
    log["cpu_count"] = os.cpu_count()
    save_log()
    print("load report:", {k: (v if k != "cfg" else "...") for k, v in rep.items()}, flush=True)

    lf = synth.build_free_leadfield()
    L, pos, names = lf["L"], lf["pos"], list(lf["info"].ch_names)
    C, M = L.shape[0], L.shape[1]
    S1 = probe.single_source_cov(L)
    rng = np.random.default_rng(cfg["seed"])
    masks = [("scattered", m) for m in cfg["masks"]["scattered"]] + [("patch", m) for m in cfg["masks"]["patch"]]
    fam = []
    for _, H in masks:
        V = [c for c in range(C) if c not in H]
        R = probe.reref_matrix(C, V)
        fam.append(dict(V=V, H=H, Lr=np.einsum("cd,dmi->cmi", R, L), **probe.linear_family(pos, S1, V, H)))
    units = []
    for k in cfg["ks"]:
        nk = cfg["n_trials"][str(k)]
        loc_stream = np.concatenate([rng.permutation(M) for _ in range(k * nk // M + 1)])
        for kind in cfg["kinds"]:
            for i in range(nk):
                locs = loc_stream[i * k:(i + 1) * k] if kind == "real" else None
                units.append(dict(name=f"k{k}_{kind}_{i:03d}", k=k, kind=kind, i=i,
                                  X=probe.make_trial(lf, k, kind, cfg["seconds"], rng, S1, locs=locs)))
    order = rng.permutation(len(units))
    log["n_units"] = len(units)
    rows = []
    for n, ui in enumerate(order):
        u = units[ui]
        mi = u["i"] % len(masks)
        f, mk = fam[mi], masks[mi][0]
        V, H = f["V"], f["H"]
        Xf = mne.filter.filter_data(u["X"], synth.FS, None, 96.0, verbose="ERROR")   # BrainOmni low-pass
        Xr = probe.reref_visible(Xf, V)
        tH = Xr[H]
        t1 = time.time()
        pb = bt.infill(model, Xf, pos, V, H, leaky=False)
        pl = bt.infill(model, Xf, pos, V, H, leaky=True)
        tb = time.time() - t1
        row = dict(trial=u["name"], k=u["k"], kind=u["kind"], mask=mi, mask_kind=mk,
                   shape_brainomni=shape_err_w(pb, tH), shape_brainomni_leaky=shape_err_w(pl, tH),
                   shape_oracle=shape_err_w(music_predict(f["Lr"], V, H, Xr[V], u["k"]), tH), sec_model=tb)
        for wn in ("spline", "idw", "best-linear", "linear-smooth"):
            row[f"shape_{wn}"] = shape_err_w(f[wn] @ Xr[V], tH)
        rows.append(row)
        if (n + 1) % 25 == 0 or n + 1 == len(order):
            (OUT / "rows.json").write_text(json.dumps(rows))
            log["done"] = n + 1
            log["minutes"] = (time.time() - t0) / 60
            save_log()
            print(f"{n + 1}/{len(order)} units, {(time.time() - t0) / 60:.1f} min, model {tb:.2f}s/unit", flush=True)
    log["ok"] = True
except Exception:
    log["ok"] = False
    log["error"] = traceback.format_exc()
    print(log["error"], flush=True)
log["total_min"] = (time.time() - t0) / 60
save_log()
