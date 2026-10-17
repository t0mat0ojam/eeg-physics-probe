"""Probe experiment: ZUNA vs linear references on real / patterns / surrogate recordings.

Every (trial, mask, copy) is one .fif whose info['bads'] holds that mask's hidden electrodes.
Files are processed in chunks, each chunk split over the GPUs (one ZUNA process per GPU);
rows.json and log.json are rewritten after every chunk, so a time-out keeps finished work.
MOCK=1 replaces ZUNA by spline interpolation (local test of the plumbing).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def run(cfg: dict, out: Path, mock: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    log = dict(config=cfg, mock=mock, chunks=[])
    save_log = lambda: (out / "log.json").write_text(json.dumps(log, indent=1, default=str))
    t0 = time.time()
    try:
        if not mock:
            subprocess.run(f"{sys.executable} -m pip install -q zuna mne", shell=True, check=True)
        import mne
        from . import synth, probe
        mne.set_log_level("ERROR")
        lf = synth.build_free_leadfield()
        L, pos, info = lf["L"], lf["pos"], lf["info"]
        C = L.shape[0]
        names = list(info.ch_names)
        rng = np.random.default_rng(cfg["seed"])
        S1 = probe.single_source_cov(L)
        tot = probe.total_cov(S1)
        lmin = L.reshape(C, -1)
        mn_cov = lmin @ lmin.T
        mn_cov = mn_cov / np.trace(mn_cov) + 1e-2 * np.eye(C) / C

        masks = [("scattered", m) for m in cfg["masks"]["scattered"]] + [("patch", m) for m in cfg["masks"]["patch"]]
        W = []
        for _, H in masks:
            V = [c for c in range(C) if c not in H]
            W.append(dict(V=V, **probe.linear_family(pos, S1, V, H)))
        trials = []
        M = L.shape[1]
        for k in cfg["ks"]:
            # stratified source locations: walk through shuffled copies of the whole grid
            loc_stream = np.concatenate([rng.permutation(M) for _ in range(k * cfg["n_trials"] // M + 1)])
            for kind in cfg["kinds"]:
                for i in range(cfg["n_trials"]):
                    locs = loc_stream[i * k:(i + 1) * k] if kind == "real" else None
                    X = probe.make_trial(lf, k, kind, cfg["seconds"], rng, S1, locs=locs).astype(np.float32)
                    trials.append(dict(name=f"k{k}_{kind}_{i:03d}", i=i, k=k, kind=kind, X=X, hp=probe.highpass(X.astype(float))))
        mpt = cfg.get("masks_per_trial", len(masks))       # each trial gets mpt masks, cycled
        units = [(t, (trials[t]["i"] * mpt + j) % len(masks)) for t in range(len(trials)) for j in range(mpt)]   # same masks for every kind
        rng.shuffle(units)                                     # mix conditions within every chunk
        work = Path(os.environ.get("PROBE_WORK", "/tmp/probe_work" if not mock else str(out / "work")))
        n_gpus = cfg.get("n_gpus", 1)
        per_chunk = max(1, cfg["files_per_gpu_per_chunk"] * n_gpus // cfg["copies"])
        rows = []
        log["n_units"] = len(units)
        log["setup_min"] = (time.time() - t0) / 60
        save_log()
        for ci in range(0, len(units), per_chunk):
            t1 = time.time()
            chunk = units[ci:ci + per_chunk]
            fin = work / "fif_in"
            shutil.rmtree(fin, ignore_errors=True)
            fin.mkdir(parents=True)
            files = []
            for (t, mi) in chunk:
                inf = info.copy()
                H = masks[mi][1]
                inf["bads"] = [names[h] for h in H]
                Xin = probe.leakproof_input(trials[t]["X"].astype(float), H)
                for c in range(cfg["copies"]):
                    f = fin / f"{trials[t]['name']}_m{mi}_c{c}_raw.fif"
                    mne.io.RawArray(Xin, inf, verbose="ERROR").save(f, verbose="ERROR")
                    files.append(f)
            if mock:
                recs = {}
                for f in files:
                    raw = mne.io.read_raw_fif(f, preload=True, verbose="ERROR")
                    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
                    recs[f.name] = (raw.get_data(), raw.ch_names)
            else:
                from .zuna_tools import run_parallel
                groups = [files[g::n_gpus] for g in range(n_gpus)]
                recs = run_parallel(groups, work / "zuna", seed=cfg["seed"] * 100 + ci, steps=cfg["steps"], n_gpus=n_gpus)
            for (t, mi) in chunk:
                tr, (mkind, H), Wm = trials[t], masks[mi], W[mi]
                hidden = [names[h] for h in H]
                truth = probe.highpass(probe.reref_visible(tr["X"].astype(float), Wm["V"]))   # visible-ref domain
                preds = []
                for c in range(cfg["copies"]):
                    data, ch = recs[f"{tr['name']}_m{mi}_c{c}_raw.fif"]
                    preds.append(data[[ch.index(h) for h in hidden]])     # ZUNA output, already high-passed
                T = min([truth.shape[1]] + [p.shape[1] for p in preds])
                tH = truth[H, :T]
                zm = np.mean([p[:, :T] / (p[:, :T].std() + 1e-30) for p in preds], 0)
                row = dict(trial=tr["name"], k=tr["k"], kind=tr["kind"], mask=mi, mask_kind=mkind,
                           shape_zuna=probe.shape_err(zm, tH), shape_zuna_single=probe.shape_err(preds[0][:, :T], tH))
                for wn in ("spline", "idw", "best-linear", "linear-smooth"):
                    row[f"shape_{wn}"] = probe.shape_err(Wm[wn] @ truth[Wm["V"], :T], tH)
                rows.append(row)
            (out / "rows.json").write_text(json.dumps(rows))
            log["chunks"].append(dict(start=ci, units=len(chunk), files=len(files), minutes=(time.time() - t1) / 60))
            save_log()
            print(f"chunk {ci // per_chunk + 1}/{-(-len(units) // per_chunk)}: {len(files)} files in "
                  f"{(time.time() - t1) / 60:.1f} min", flush=True)
            shutil.rmtree(work / "zuna", ignore_errors=True)
        log["ok"] = True
    except Exception:
        log["ok"] = False
        log["error"] = traceback.format_exc()
        print(log["error"], flush=True)
    log["total_min"] = (time.time() - t0) / 60
    save_log()
    return log
