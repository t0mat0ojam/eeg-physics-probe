"""RAP-MUSIC oracle: scan the true (re-referenced) lead field for k dipoles, then least squares."""
import numpy as np


def music_predict(Lr, V, H, XV, k):
    """RAP-MUSIC with the (re-referenced) lead field Lr (C, M, 3); returns hidden prediction."""
    U, s, _ = np.linalg.svd(XV, full_matrices=False)
    Us = U[:, :k]
    LV, LH = Lr[V], Lr[H]
    M = Lr.shape[1]
    chosen = []                      # list of (loc, orientation-free columns)
    A = np.zeros((len(V), 0))
    for _ in range(k):
        if A.shape[1]:
            Pp = np.eye(len(V)) - A @ np.linalg.pinv(A)
        else:
            Pp = np.eye(len(V))
        Usp = Pp @ Us
        Usp, _ = np.linalg.qr(Usp)
        best, bj = -1, None
        for j in range(M):
            G = Pp @ LV[:, j]
            Q, _ = np.linalg.qr(G)
            val = np.linalg.svd(Q.T @ Usp, compute_uv=False)[0]
            if val > best:
                best, bj = val, j
        chosen.append(bj)
        A = np.concatenate([A, LV[:, bj]], 1)
    S = np.linalg.lstsq(A, XV, rcond=None)[0]                  # (3k, T)
    AH = np.concatenate([LH[:, j] for j in chosen], 1)
    return AH @ S
