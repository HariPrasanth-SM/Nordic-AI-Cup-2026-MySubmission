"""Multi-prototype DINO reference model.

The reference bank holds many crops per class. Nearest-neighbour similarity to the whole bank is
dominated by the single closest crop; K spherical-k-means prototypes per class (default 8) describe
the modes of the class (viewpoints, scales, shadows) and are calibrated: for every class we keep the
distribution of "member -> nearest own prototype" similarity, so a detection can be judged as
"typical / borderline / not close to ANY mode of its class" (a false positive or a wrong label).

Pure numpy; the encoder is not needed here.
"""
import json
import numpy as np


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def spherical_kmeans(x, k, seed=0, iterations=40):
    """Cosine k-means with k-means++ seeding. Returns unit-norm centres (<= k) and member assignment."""
    x = _normalize(np.asarray(x, np.float64)); n = len(x); k = max(1, min(int(k), n)); rng = np.random.default_rng(seed)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):   # k-means++: choose far points (cosine distance = 1 - sim)
        d = 1. - np.max(x @ np.array(centers).T, axis=1); d = np.maximum(d, 0.); s = d.sum()
        centers.append(x[rng.choice(n, p=d / s)] if s > 1e-12 else x[rng.integers(n)])
    centers = np.array(centers); assign = np.zeros(n, int)
    for it in range(iterations):
        new = np.argmax(x @ centers.T, axis=1)
        if it and (new == assign).all(): break
        assign = new
        for j in range(len(centers)):
            m = assign == j
            if m.any(): centers[j] = _normalize(x[m].sum(axis=0, keepdims=True))[0]
            else:   # re-seed an empty cluster with the worst-fitting point
                worst = int(np.argmin(np.max(x @ centers.T, axis=1))); centers[j] = x[worst]
    keep = np.unique(np.argmax(x @ centers.T, axis=1))
    return centers[keep].astype(np.float32), np.argmax(x @ centers[keep].T, axis=1)


def build_prototypes(features, labels, names, k=8, seed=0, quantile=.05, min_spread=.02):
    """features (N,D) unit vectors; labels -1 = background, 0..C-1 classes. Returns a dict for save()/ProtoScorer."""
    features = np.asarray(features, np.float32); labels = np.asarray(labels, int)
    centers = []; plabels = []; tau = np.zeros(len(names) + 1); spread = np.full(len(names) + 1, min_spread); counts = np.zeros(len(names) + 1, int)
    for cls in range(-1, len(names)):
        which = labels == cls
        if not which.any(): raise ValueError(f'No reference features for {"background" if cls < 0 else names[cls]}')
        c, _ = spherical_kmeans(features[which], k, seed + cls + 1)
        centers.append(c); plabels += [cls] * len(c); counts[cls + 1] = int(which.sum())
        sim = np.max(features[which] @ c.T, axis=1)   # member -> nearest own prototype
        tau[cls + 1] = float(np.quantile(sim, quantile)); spread[cls + 1] = float(max(np.median(sim) - tau[cls + 1], min_spread))
    return {'centers': np.concatenate(centers).astype(np.float32), 'labels': np.array(plabels, int), 'tau': tau, 'spread': spread,
            'counts': counts, 'meta': {'classes': list(names), 'k': int(k), 'quantile': quantile, 'dim': int(features.shape[1])}}


def save(path, proto, extra_meta=None):
    meta = dict(proto['meta'], **(extra_meta or {}))
    np.savez_compressed(path, centers=proto['centers'], labels=proto['labels'], tau=proto['tau'], spread=proto['spread'],
                        counts=proto['counts'], metadata=json.dumps(meta))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {'centers': z['centers'].astype(np.float32), 'labels': z['labels'].astype(int), 'tau': z['tau'].astype(float),
                'spread': z['spread'].astype(float), 'counts': z['counts'].astype(int), 'meta': json.loads(str(z['metadata'].item()))}


class ProtoScorer:
    """Scores unit-norm embeddings against every prototype. Index 0 of per-class arrays is background."""
    def __init__(self, proto, names):
        if list(proto['meta']['classes']) != list(names): raise ValueError('Prototype file was built for different classes')
        self.p = proto; self.n = len(names); self.centers = proto['centers']; self.labels = proto['labels']
        self.tau = proto['tau']; self.spread = proto['spread']; self.dim = self.centers.shape[1]
        if not np.isfinite(self.centers).all() or not np.allclose(np.linalg.norm(self.centers, axis=1), 1, atol=.01):
            raise ValueError('Prototype centres must be finite and normalized')

    def class_max(self, embeddings):
        """(B, n+1) best similarity to any prototype of each class; column 0 is background."""
        sims = np.asarray(embeddings, np.float32) @ self.centers.T; out = np.full((len(sims), self.n + 1), -1., np.float32)
        for cls in range(-1, self.n):
            m = self.labels == cls
            if m.any(): out[:, cls + 1] = sims[:, m].max(axis=1)
        return out

    def summarize(self, row, cls):
        """JSON-safe evidence for a detection labelled `cls` given its class_max row."""
        z = (row - self.tau) / self.spread                      # 0 = weakest 5% of genuine members, >0 typical
        own = int(cls) + 1; others = np.delete(np.arange(1, self.n + 1), int(cls))
        best = int(np.argmax(row[1:]))
        return {'proto_class': [round(float(v), 4) for v in row[1:]], 'proto_bg': round(float(row[0]), 4),
                'proto_own': round(float(row[own]), 4), 'proto_z_own': round(float(z[own]), 3),
                'proto_z_bg': round(float(z[0]), 3), 'proto_best': best, 'proto_z_best': round(float(z[1:].max()), 3),
                'proto_best_other': round(float(row[others].max()), 4)}
