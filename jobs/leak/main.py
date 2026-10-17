"""Leak test: does ZUNA's infill of a hidden electrode depend on the values left in that electrode?

ZUNA z-scores every channel per 5 s segment with that channel's OWN mean/std, then maps the model
output for masked cells back with the same mean/std. If the hidden channel still holds data, its
scale reaches the answer. Same recordings, hidden channels filled 5 ways:
  true | x10 | x0.1 | noise (white, std = median visible std) | zeros
Leak => output std of hidden channels tracks the fill (x10 -> ~10x louder).
Also checks the two-GPU port fix (files split over 2 processes).
"""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get("JOB_OUT", "out_leak")); OUT.mkdir(parents=True, exist_ok=True)
MOCK = os.environ.get("MOCK") == "1"
res = dict(rows=[], mock=MOCK)
t0 = time.time()
try:
    if not MOCK:
        subprocess.run(f"{sys.executable} -m pip install -q zuna mne", shell=True, check=True)
    import mne
    from physprobe import synth, probe
    mne.set_log_level("ERROR")
    lf = synth.build_free_leadfield(); info = lf["info"]; names = list(info.ch_names); C = len(names)
    rng = np.random.default_rng(3)
    S1 = probe.single_source_cov(lf["L"])
    hidden = ["FC3", "C2", "CP5", "F1", "AF4", "P3", "PO8", "FT8"]
    H = [names.index(h) for h in hidden]; V = [c for c in range(C) if c not in H]
    work = Path("/tmp/leak") if not MOCK else OUT / "work"; fin = work / "in"; fin.mkdir(parents=True, exist_ok=True)
    recs, files = {}, []
    for r in range(6):
        k = 1 if r < 2 else 4
        X = probe.make_trial(lf, k, "real", 10, rng, S1)
        recs[r] = X
        vis_sd = np.median(X[V].std(1))
        fills = dict(true=X[H], x10=10 * X[H], x01=0.1 * X[H],
                     noise=rng.normal(size=X[H].shape) * vis_sd, zeros=np.zeros_like(X[H]))
        for cond, F in fills.items():
            Y = X.copy(); Y[H] = F
            inf = info.copy(); inf["bads"] = hidden
            f = fin / f"r{r}_{cond}_raw.fif"
            mne.io.RawArray(Y, inf, verbose="ERROR").save(f, verbose="ERROR"); files.append(f)
    t1 = time.time()
    if MOCK:
        out = {}
        for f in files:
            raw = mne.io.read_raw_fif(f, preload=True, verbose="ERROR"); raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            out[f.name] = (raw.get_data(), raw.ch_names)
    else:
        from physprobe.zuna_tools import run_parallel
        out = run_parallel([files[0::2], files[1::2]], work / "zuna", seed=5, steps=25, n_gpus=2, log_dir=OUT)
    res["zuna_minutes"] = (time.time() - t1) / 60
    res["files"] = len(files)
    for f in files:
        r, cond = f.name.split("_")[0][1:], f.name.split("_")[1]
        truth = probe.highpass(recs[int(r)])
        data, ch = out[f.name]
        T = min(truth.shape[1], data.shape[1])
        for h in hidden:
            y, t = data[ch.index(h), :T], truth[names.index(h), :T]
            res["rows"].append(dict(rec=int(r), cond=cond, ch=h, std_out=float(y.std()), std_true=float(t.std()),
                                    corr=float(np.corrcoef(y, t)[0, 1]) if y.std() > 0 else 0.0,
                                    nmse=float(((y - t) ** 2).sum() / (t ** 2).sum())))
        vis = [ch.index(names[v]) for v in V]
        res.setdefault("visible_nmse", []).append(float(((data[vis, :T] - truth[V, :T]) ** 2).sum() / (truth[V, :T] ** 2).sum()))
    res["ok"] = True
except Exception:
    res["ok"] = False; res["error"] = traceback.format_exc(); print(res["error"], flush=True)
res["total_min"] = (time.time() - t0) / 60
(OUT / "result.json").write_text(json.dumps(res, indent=1))
summ = {}
for row in res["rows"]:
    s = summ.setdefault(row["cond"], dict(ratio=[], corr=[], nmse=[]))
    s["ratio"].append(row["std_out"] / row["std_true"]); s["corr"].append(row["corr"]); s["nmse"].append(row["nmse"])
print(json.dumps({c: {k: float(np.median(v)) for k, v in s.items()} for c, s in summ.items()}, indent=1))
