"""Probe 4, ZUNA part (GPU): real EEGMMIDB segments, three input conditions, vs splines.

(i)  released : all-channel average reference, TRUE values left in hidden channels (the leak)
(ii) leakproof: visible-average reference, hidden channels zero-filled
(iii) splinefill: visible-average reference, hidden channels filled with the spline prediction
Same segments and masks as jobs/real_brainomni (same config). One ZUNA call per GPU over all files.
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("JOB_OUT", "out_real_zuna")); OUT.mkdir(parents=True, exist_ok=True)
MOCK = os.environ.get("MOCK") == "1"
cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
if MOCK:
    cfg.update(subjects=[21, 22], runs=[4, 8])
log = dict(config=cfg, mock=MOCK)
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
    if not MOCK:
        subprocess.run(f"{sys.executable} -m pip install -q zuna mne", shell=True, check=True)
    import mne
    from physprobe import probe, realdata
    from physprobe.synth import build_free_leadfield
    mne.set_log_level("ERROR")
    root = Path("/home/claude/which-information/data/eegmmidb_v2/eeg2") if MOCK else Path("/kaggle/temp/eegmmidb")
    segs, missing = realdata.load_segments(cfg["subjects"], cfg["runs"], root, download=not MOCK)
    log.update(n_segments=len(segs), missing=missing)
    lf = build_free_leadfield()
    pos, info, names = lf["pos"], lf["info"], list(lf["info"].ch_names)
    C = pos.shape[0]
    masks = [("scattered", m) for m in cfg["masks"]["scattered"]] + [("patch", m) for m in cfg["masks"]["patch"]]
    work = Path("/tmp/real_zuna") if not MOCK else OUT / "work"
    fin = work / "in"; fin.mkdir(parents=True, exist_ok=True)
    files, plan = [], []
    for n, g in enumerate(segs):
        mi = n % len(masks)
        mk, H = masks[mi]
        V = [c for c in range(C) if c not in H]
        Xall = g["X"] - g["X"].mean(0, keepdims=True)                    # all-channel reference
        Xr = probe.reref_visible(g["X"], V)
        Ws = probe.spline_W(pos, V, H)
        inputs = {"released": Xall, "leakproof": probe.leakproof_input(g["X"], H)}
        sf = Xr.copy(); sf[H] = Ws @ Xr[V]
        inputs["splinefill"] = sf
        inf = info.copy(); inf["bads"] = [names[h] for h in H]
        for cond, Y in inputs.items():
            f = fin / f"{g['name']}_{cond}_raw.fif"
            mne.io.RawArray(Y, inf, verbose="ERROR").save(f, overwrite=True, verbose="ERROR")
            files.append(f)
        plan.append(dict(g=g, mi=mi, mk=mk, H=H, V=V, Xall=Xall, Xr=Xr, Ws=Ws))
    log["n_files"] = len(files)
    save_log()
    t1 = time.time()
    if MOCK:
        recs = {}
        for f in files:
            raw = mne.io.read_raw_fif(f, preload=True, verbose="ERROR"); raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            recs[f.name] = (raw.get_data(), raw.ch_names)
    else:
        from physprobe.zuna_tools import run_parallel
        recs = run_parallel([files[0::2], files[1::2]], work / "zuna", seed=4040, steps=25, n_gpus=2, log_dir=OUT)
    log["zuna_min"] = (time.time() - t1) / 60
    rows = []
    for p in plan:
        g, H, V = p["g"], p["H"], p["V"]
        hidden = [names[h] for h in H]
        tr_vis = probe.highpass(p["Xr"])[H]            # ZUNA high-passes its input at 0.5 Hz (no-op-ish here)
        tr_all = probe.highpass(p["Xall"])[H]
        sp_vis = p["Ws"] @ probe.highpass(p["Xr"])[V]
        sp_all = p["Ws"] @ probe.highpass(p["Xall"])[V]
        row = dict(seg=g["name"], subject=g["subject"], run=g["run"], mask=p["mi"], mask_kind=p["mk"])
        for cond in ("released", "leakproof", "splinefill"):
            data, ch = recs[f"{g['name']}_{cond}_raw.fif"]
            Z = data[[ch.index(h) for h in hidden]]
            T = min(Z.shape[1], tr_vis.shape[1])
            ref = tr_all if cond == "released" else tr_vis
            row[f"shape5_zuna_{cond}"] = shape_err_w(Z[:, :T], ref[:, :T], 1280)
            row[f"shape2_zuna_{cond}"] = shape_err_w(Z[:, :T], ref[:, :T], 512)
            if cond in ("released", "splinefill"):
                row[f"nmse_zuna_{cond}"] = float(((Z[:, :T] - ref[:, :T]) ** 2).sum() / (ref[:, :T] ** 2).sum())
        row["shape5_spline"] = shape_err_w(sp_vis, tr_vis, 1280)
        row["shape2_spline"] = shape_err_w(sp_vis, tr_vis, 512)
        row["nmse_spline_visref"] = float(((sp_vis - tr_vis) ** 2).sum() / (tr_vis ** 2).sum())
        row["nmse_spline_allref"] = float(((sp_all - tr_all) ** 2).sum() / (tr_all ** 2).sum())
        rows.append(row)
    (OUT / "rows.json").write_text(json.dumps(rows))
    log["ok"] = True
except Exception:
    log["ok"] = False
    log["error"] = traceback.format_exc()
    print(log["error"], flush=True)
log["total_min"] = (time.time() - t0) / 60
save_log()
