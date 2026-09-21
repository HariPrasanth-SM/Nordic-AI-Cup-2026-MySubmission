"""
survivor.escape_model - learned P(death) for a predator encounter (the strategy document's "escapeability predictor").

Training data are REAL flee episodes logged by the controller in full runs (`flee_rows`, layout ROW below). The model is a small
ridge-regularised logistic regression (numpy only): transparent weights, so the fit itself tells us which features carry the risk.

  python -m survivor.lab fit-escape runs/lab1        # fits from the lab's logged episodes -> runs/lab1/escape_model.json
  SURVIVOR_ESCAPE_MODEL=runs/lab1/escape_model.json  # picked up by Controller when params.use_escape_model > 0.5
"""
import json
import math
import os

FEATURES = ["e_frac", "speed", "d0", "npred", "nb", "wall_d", "age", "noticed"]
ROW = FEATURES + ["dmin", "dur", "died"]          # layout of one logged flee episode


def fit(rows, l2=1e-2, iters=80):
    """rows: list of lists in ROW layout. Returns a JSON-able model dict."""
    import numpy as np
    X = np.array([r[:len(FEATURES)] for r in rows], float)
    y = np.array([r[len(ROW) - 1] for r in rows], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        W = p * (1 - p) + 1e-6
        H = Z.T @ (Z * W[:, None]) + l2 * np.eye(len(w))
        g = Z.T @ (p - y) + l2 * w
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum()))
    bins = []
    for lo, hi in ((0, .1), (.1, .2), (.2, .35), (.35, .5), (.5, .7), (.7, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 5:
            bins.append({"pred": [lo, hi], "n": int(m.sum()), "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    return {"features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "n": int(len(y)), "base_rate": float(y.mean()),
            "auc": auc, "acc": float(((p > 0.5) == pos).mean()), "calibration": bins}


class EscapeModel:
    def __init__(self, d):
        self.d = d
        self.mu, self.sd, self.w = d["mu"], d["sd"], d["w"]

    def p_die(self, e_frac, speed, d0, npred, nb, wall_d, age, noticed):
        x = (e_frac, speed, d0, npred, nb, wall_d, age, noticed)
        z = self.w[0] + sum(self.w[i + 1] * (x[i] - self.mu[i]) / self.sd[i] for i in range(len(x)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return EscapeModel(json.load(f))
    return None


def save(model, path):
    with open(path, "w") as f:
        json.dump(model, f, indent=1)
