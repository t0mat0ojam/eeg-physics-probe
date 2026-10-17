"""Probe 1: does ZUNA use head physics when filling in hidden electrodes?

For each number of sources k and each mask, real and surrogate recordings are infilled by ZUNA
(posterior mean of `copies` samples) and by linear references (splines, minimum-norm, best
linear). Per-trial squared error and hidden-channel power are saved after every mask, so a
time-out keeps everything finished so far. Settings: jobs/probe1/config.json.
MOCK=1 replaces ZUNA by spline interpolation (local test of the plumbing).
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("JOB_OUT", "out_probe1"))
OUT.mkdir(parents=True, exist_ok=True)
MOCK = os.environ.get("MOCK") == "1"
HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / "config.json").read_text())
log = dict(config=cfg, mock=MOCK, progress=[])


def save_log():
    (OUT / "log.json").write_text(json.dumps(log, indent=1, default=str))


t0 = time.time()
try:
    if not MOCK:
        subprocess.run(f"{sys.executable} -m pip install -q zuna mne", shell=True, check=True)
    import mne
    from physprobe import synth, probe
    mne.set_log_level("ERROR")
    lf = synth.build_free_leadfield()
    L, pos, info = lf["L"], lf["pos"], lf["info"]
    C = L.shape[0]
    names = list(info.ch_names)
    rng = np.random.default_rng(cfg["seed"])

    # trials: same for every mask, so per-mask results are paired
    trials, meta = {}, []
    sig_cov = {}
    for k in cfg["ks"]:
        sig_cov[k] = probe.trial_cov_samples(L, k, cfg["cov_samples"], rng)
        for kind in ("real", "surrogate"):
            for i in range(cfg["n_trials"]):
                X, noise_var = probe.make_trial(lf, k, kind, cfg["seconds"], rng, sig_cov[k])
                name = f"k{k}_{kind}_{i:03d}"
                trials[name] = X
                meta.append(dict(name=name, k=k, kind=kind))
    work = Path("/tmp/probe1") if not MOCK else OUT / "work"
    (work / "in").mkdir(parents=True, exist_ok=True)
    fifs = []
    for name, X in trials.items():
        for c in range(cfg["copies"]):
            fifs.append(synth.save_fif(X, work / "in" / f"{name}c{c}_raw.fif", info))
    log["n_files_per_call"] = len(fifs)
    log["setup_min"] = (time.time() - t0) / 60
    save_log()

    # linear references need the total covariance (signal + noise) per k
    noise_frac = 10 ** (-20 / 10)
    lmin = L.reshape(C, -1)
    mn_cov = lmin @ lmin.T
    mn_cov = mn_cov / np.trace(mn_cov) + 1e-2 * np.eye(C) / C

    truth_hp = {n: probe.highpass(X) for n, X in trials.items()}
    all_masks = [("scattered", H) for H in cfg["masks"]["scattered"]] + \
                [("patch", H) for H in cfg["masks"]["patch"]]
    rows = []
    for mi, (mkind, H) in enumerate(all_masks):
        t1 = time.time()
        V = [c for c in range(C) if c not in H]
        hidden = [names[h] for h in H]
        if MOCK:
            recs = {}
            Ws = probe.spline_W(pos, V, H)
            for f in fifs:
                X = trials[f.name.rsplit("c", 1)[0]]
                Y = X.copy()
                Y[H] = Ws @ X[V]
                recs[f.name] = (Y, names)
        else:
            from physprobe.zuna_tools import reconstruct_batch
            recs = reconstruct_batch(fifs, hidden, work / "zuna", seed=100 + mi,
                                     steps=cfg["steps"], gpu="0")
        W = {"spline": probe.spline_W(pos, V, H), "min-norm": probe.linear_W(mn_cov, V, H)}
        for m in meta:
            n, k = m["name"], m["k"]
            tot = sig_cov[k] / np.trace(sig_cov[k]) + noise_frac * np.eye(C) / C
            W["best-linear"] = probe.linear_W(tot, V, H)
            truth = truth_hp[n]
            preds = []
            for c in range(cfg["copies"]):
                data, ch = recs[f"{n}c{c}_raw.fif"]
                T = min(truth.shape[1], data.shape[1])
                preds.append(probe.highpass(data[:, :T])[[ch.index(h) for h in hidden]])
            T = min(p.shape[1] for p in preds)
            tH = truth[H, :T]
            zmean = np.mean([p[:, :T] for p in preds], 0)
            row = dict(mask=mi, mask_kind=mkind, k=k, kind=m["kind"], trial=n,
                       power=float((tH ** 2).sum()),
                       err_zuna=float(((tH - zmean) ** 2).sum()),
                       err_zuna_single=float(((tH - preds[0][:, :T]) ** 2).sum()))
            for wn, Wm in W.items():
                row[f"err_{wn}"] = float(((tH - Wm @ truth[V, :T]) ** 2).sum())
            rows.append(row)
        (OUT / "rows.json").write_text(json.dumps(rows))
        log["progress"].append(dict(mask=mi, kind=mkind, minutes=(time.time() - t1) / 60))
        save_log()
        print(f"mask {mi + 1}/{len(all_masks)} ({mkind}) done in {(time.time() - t1) / 60:.1f} min", flush=True)
    log["ok"] = True
except Exception:
    log["ok"] = False
    log["error"] = traceback.format_exc()
    print(log["error"], flush=True)
log["total_min"] = (time.time() - t0) / 60
save_log()
