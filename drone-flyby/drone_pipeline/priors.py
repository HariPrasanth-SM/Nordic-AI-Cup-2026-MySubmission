"""Object size prior in SOURCE pixels.

Flight altitude is constant, so a class keeps (nearly) the same size in source pixels: e.g. a helicopter that
is ~240 px in an L2 view (native) is ~120 px in an L1 view (2x downsample) and ~240 px in the source frame.
Detections are compared with per-class statistics measured on the training labels
(scripts/build_size_prior.py). The result is a LOW-VARIANCE multiplier in [floor, 1] per class: an atypical
size lowers a class' evidence, it never removes it.
"""
import json
import math
from pathlib import Path
import numpy as np


class SizePrior:
    def __init__(self, stats, names, floor=.6, size_sigma_min=.15, aspect_sigma_min=.25, sigma_scale=1.5, aspect_weight=.5):
        self.names = list(names); self.floor = floor; self.aspect_weight = aspect_weight
        self.mu = np.full((len(names), 2), np.nan); self.sd = np.ones((len(names), 2))
        for i, name in enumerate(names):
            s = stats.get(name)
            if not s or s.get('n', 0) < 5: continue
            self.mu[i] = (s['log_size_mean'], s['log_aspect_mean'])
            self.sd[i] = (max(s['log_size_std'] * sigma_scale, size_sigma_min), max(s['log_aspect_std'] * sigma_scale, aspect_sigma_min))

    @classmethod
    def load(cls, path, names, **kw):
        data = json.loads(Path(path).read_text())
        if list(data.get('classes', [])) != list(names): raise ValueError('Size prior was built for different classes')
        return cls(data['stats'], names, **kw)

    def factors(self, w, h):
        """Per-class multiplier in [floor, 1] for a complete box of source size w x h (classes without statistics: 1)."""
        w = max(float(w), 1.); h = max(float(h), 1.)
        f = np.array([math.log(math.sqrt(w * h)), math.log(w / h)])
        z = (f[None, :] - self.mu) / self.sd; z2 = z[:, 0] ** 2 + self.aspect_weight * z[:, 1] ** 2
        out = self.floor + (1. - self.floor) * np.exp(-.5 * np.minimum(z2, 16.))
        return np.where(np.isnan(self.mu[:, 0]), 1., out)


def class_stats(sizes):
    """sizes: array (N, 2) of source-pixel (w, h) for one class -> JSON-safe robust statistics."""
    s = np.asarray(sizes, float); s = s[(s[:, 0] > 1) & (s[:, 1] > 1)]
    if len(s) == 0: return {'n': 0}
    ls = np.log(np.sqrt(s[:, 0] * s[:, 1])); la = np.log(s[:, 0] / s[:, 1])
    def robust(x): return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
    return {'n': int(len(s)), 'log_size_mean': float(np.median(ls)), 'log_size_std': robust(ls),
            'log_aspect_mean': float(np.median(la)), 'log_aspect_std': robust(la),
            'size_px_p05': float(np.quantile(np.sqrt(s[:, 0] * s[:, 1]), .05)), 'size_px_median': float(np.exp(np.median(ls))),
            'size_px_p95': float(np.quantile(np.sqrt(s[:, 0] * s[:, 1]), .95))}
