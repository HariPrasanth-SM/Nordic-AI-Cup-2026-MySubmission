"""
survivor.value_model - the META-REWARD model: dense per-agent survival credit.

Meta reward: an agent earns +2 for every 5 s window it survives. The controller logs one row per agent per window
(features at the window start, y = 1 if the agent was still alive 5 s later). Fitting y gives a HAZARD MODEL:
    p_surv5(state) = P(agent survives the next 5 s)          V(state) = expected meta reward over the next 60 s = 2 * sum_k p^k
Uses: (1) birth gate: a birth must not lower the parent's modelled 5-s survival by more than `vm_drop` (learned escape reserve,
replaces the fixed 0.2*maxE+margin rule); (2) diagnostics: which state variables drive agent mortality; (3) the lab reports per-arm agent hazards,
a far less noisy readout than one final score per run.
Note the score itself is colony time (+dt per step): rewarding agent-seconds directly would favour big, fragile colonies, so the meta reward is
used for gating/evaluation, never as the optimisation target.
"""
import json
import math
import os

FEATURES = ["e_frac", "speed", "age", "hear", "nb", "n", "since_thr", "maxE"]
ROW = FEATURES + ["survived"]


def fit(rows, l2=1e-2, iters=80):
    import numpy as np
    X = np.array([r[:len(FEATURES)] for r in rows], float)
    y = np.array([r[len(FEATURES)] for r in rows], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    w[0] = math.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
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
    dead = y == 0
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    nd, ns = dead.sum(), (~dead).sum()
    auc = float((ranks[~dead].sum() - ns * (ns + 1) / 2) / max(1, nd * ns))            # P(survivor ranked above a dying agent)
    bins = []
    for lo, hi in ((0, .9), (.9, .95), (.95, .98), (.98, .99), (.99, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            bins.append({"pred": [lo, hi], "n": int(m.sum()), "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    return {"features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "n": int(len(y)), "survive5": float(y.mean()),
            "auc": auc, "calibration": bins}


class ValueModel:
    def __init__(self, d):
        self.d, self.mu, self.sd, self.w = d, d["mu"], d["sd"], d["w"]

    def p_surv(self, e_frac, speed, age, hear, nb, n, since_thr, maxE):
        x = (e_frac, speed, age, hear, nb, n, since_thr, maxE)
        z = self.w[0] + sum(self.w[i + 1] * (x[i] - self.mu[i]) / self.sd[i] for i in range(len(x)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def value(self, *feats, horizon=12):
        p = self.p_surv(*feats)
        return 2.0 * sum(p ** k for k in range(1, horizon + 1))


def load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return ValueModel(json.load(f))
    return None


def save(model, path):
    with open(path, "w") as f:
        json.dump(model, f, indent=1)
