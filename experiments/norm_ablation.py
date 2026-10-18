"""Exploratory 1: does per-channel normalisation (ZUNA-style) remove the ability to use head physics?

Two identical small infill networks, trained only on physically possible recordings:
  G : global scaling (one scalar per segment) -> relative amplitudes across electrodes preserved
  Z : per-channel z-scoring of every visible channel, targets z-scored per hidden channel (as ZUNA)
Input per time point: visible values x_V(t) + segment context (top-4 principal components of the
visible segment in the same normalised domain, scaled by singular values). Output: hidden values.
Evaluation: shape error (1 - r per hidden channel, averaged) on held-out real and patterns
recordings, vs splines and an oracle (RAP-MUSIC with the true lead field).

  python experiments/norm_ablation.py [n_train] [n_test]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from physprobe import synth, probe  # noqa: E402
from pilot_shape import music_predict  # noqa: E402

torch.set_num_threads(2)
import os
SECS, NPC = float(os.environ.get("NA_SECS", "2.0")), 4


def shape_err_seg(P, T):
    p = P - P.mean(1, keepdims=True); t = T - T.mean(1, keepdims=True)
    r = (p * t).sum(1) / np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1) + 1e-300)
    return float(np.mean(1 - r))


def features(XV, mode):
    """Normalise the visible segment and build per-time inputs (T, C_V + NPC*C_V + NPC)."""
    XV = XV - XV.mean(1, keepdims=True)
    if mode == "G":
        s = np.sqrt((XV ** 2).mean()) + 1e-30
        Y = XV / s
    else:
        sd = XV.std(1, keepdims=True) + 1e-30
        Y = XV / sd
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    U = U[:, :NPC]
    U = U * np.sign(U[np.abs(U).argmax(0), np.arange(NPC)])           # fix sign of each PC
    ctx = np.concatenate([(U * S[:NPC] / np.sqrt(Y.shape[1])).T.ravel(), S[:NPC] / np.sqrt(Y.shape[1])])
    return Y, np.concatenate([Y.T, np.repeat(ctx[None], Y.shape[1], 0)], 1)


def targets(XH, XV, mode):
    XH = XH - XH.mean(1, keepdims=True)
    if mode == "G":
        XV = XV - XV.mean(1, keepdims=True)
        return XH / (np.sqrt((XV ** 2).mean()) + 1e-30)
    return XH / (XH.std(1, keepdims=True) + 1e-30)


class Net(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d_out))

    def forward(self, x):
        return self.f(x)


def make_set(lf, S1, H, V, n, kinds_ks, rng):
    out = []
    M = lf["L"].shape[1]
    for kind, k in kinds_ks:
        locs_stream = np.concatenate([rng.permutation(M) for _ in range(k * n // M + 1)])
        for i in range(n):
            if kind == "mix":
                kind_i = "real" if i % 2 == 0 else "patterns"
            else:
                kind_i = kind
            locs = locs_stream[i * k:(i + 1) * k] if kind_i == "real" else None
            X = probe.make_trial(lf, k, kind_i, SECS, rng, S1, locs=locs)
            Xr = probe.reref_visible(X, V)
            out.append(dict(kind=kind, k=k, XV=Xr[V].astype(np.float32), XH=Xr[H].astype(np.float32)))
    return out


def train(trainset, mode, epochs=25, per_seg=64, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for s in trainset:
        _, F = features(s["XV"], mode)
        Tg = targets(s["XH"], s["XV"], mode).T
        idx = rng.choice(F.shape[0], per_seg, replace=False)
        Xs.append(F[idx]); Ys.append(Tg[idx])
    X = torch.tensor(np.concatenate(Xs), dtype=torch.float32)
    Y = torch.tensor(np.concatenate(Ys), dtype=torch.float32)
    net = Net(X.shape[1], Y.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for b in range(0, n, 1024):
            j = perm[b:b + 1024]
            loss = ((net(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        sched.step()
    return net, tot / n


def predict(net, XV, mode):
    _, F = features(XV, mode)
    with torch.no_grad():
        return net(torch.tensor(F, dtype=torch.float32)).numpy().T


def main(n_train=6000, n_test=300, train_kind="real"):
    t0 = time.time()
    lf = synth.build_free_leadfield()
    L, pos = lf["L"], lf["pos"]
    C = L.shape[0]
    S1 = probe.single_source_cov(L)
    cfg = json.loads((ROOT / "jobs/probe1/config.json").read_text())
    results = {}
    for mname, H in (("patch", cfg["masks"]["patch"][0]), ("scattered", cfg["masks"]["scattered"][0])):
        V = [c for c in range(C) if c not in H]
        Ws = probe.spline_W(pos, V, H)
        R = probe.reref_matrix(C, V)
        Lr = np.einsum("cd,dmi->cmi", R, L)
        rng = np.random.default_rng(7 if mname == "patch" else 8)
        trainset = make_set(lf, S1, H, V, n_train // 3, [(train_kind, 1), (train_kind, 2), (train_kind, 4)], rng)
        nets = {}
        for mode in ("G", "Z"):
            nets[mode], loss = train(trainset, mode)
            print(f"[{mname}] trained {mode}: final loss {loss:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        testset = make_set(lf, S1, H, V, n_test, [(kd, k) for k in (1, 2, 4) for kd in ("real", "patterns")],
                           np.random.default_rng(1000 + len(mname)))
        rows = []
        for s in testset:
            e = dict(kind=s["kind"], k=s["k"],
                     spline=shape_err_seg(Ws @ s["XV"], s["XH"]),
                     G=shape_err_seg(predict(nets["G"], s["XV"], "G"), s["XH"]),
                     Z=shape_err_seg(predict(nets["Z"], s["XV"], "Z"), s["XH"]),
                     oracle=shape_err_seg(music_predict(Lr, V, H, s["XV"], s["k"]), s["XH"]))
            rows.append(e)
        res = {}
        for k in (1, 2, 4):
            Rr = [r for r in rows if r["k"] == k and r["kind"] == "real"]
            Pp = [r for r in rows if r["k"] == k and r["kind"] == "patterns"]
            ent = {}
            for m in ("spline", "G", "Z", "oracle"):
                ent[m] = dict(real=float(np.mean([r[m] for r in Rr])), patterns=float(np.mean([r[m] for r in Pp])))
                dR = np.array([r[m] - r["spline"] for r in Rr]); dP = np.array([r[m] - r["spline"] for r in Pp])
                ent[m]["score"] = float(dP.mean() - dR.mean())
                ent[m]["score_se"] = float(np.sqrt(dP.var() / len(dP) + dR.var() / len(dR)))
            res[f"k={k}"] = ent
            print(f"[{mname}] k={k}: " + "  ".join(
                f"{m}: real {v['real']:.3f} patt {v['patterns']:.3f} score {v['score']:+.3f}±{v['score_se']:.3f}"
                for m, v in ent.items()), flush=True)
        results[mname] = res
    name = "norm_ablation.json" if train_kind == "real" else f"norm_ablation_train_{train_kind}.json"
    if SECS != 2.0:
        name = name.replace(".json", f"_{int(SECS)}s.json")
    (ROOT / "results" / name).write_text(json.dumps(dict(n_train=n_train, n_test=n_test, secs=SECS,
                                                         train_kind=train_kind, results=results), indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 6000, int(a[1]) if len(a) > 1 else 300, a[2] if len(a) > 2 else "real")
