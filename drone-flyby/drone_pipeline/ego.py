"""Constant ego-motion prior: the drone flies a straight line at constant speed, so static scene
points shift by the same source-pixel vector `v` every frame.

Registration between crops of different zoom levels fails often; when it does the tracker used to
assume "no motion" and lost every object outside the current view. Once `v` is locked (robust
median of past registrations, corrected by what matched detections actually did), a failed or
outlying registration is replaced by `H = translate(v * frames_elapsed)` and coasting stays accurate.
"""
import math
from collections import deque, Counter
import numpy as np
from .motion import Motion


class EgoPrior:
    def __init__(self, cfg):
        self.cfg = cfg; self.samples = deque(maxlen=cfg.window); self.stats = Counter()

    # -- estimate --------------------------------------------------------------------
    def _array(self):
        try: return np.array(list(self.samples), float).reshape(-1, 2)
        except Exception: return np.empty((0, 2))

    @property
    def velocity(self):
        s = self._array()
        return np.median(s, axis=0) if len(s) else None

    @property
    def sigma(self):
        s = self._array()
        if len(s) < 2: return float('inf')
        return float(max(1.4826 * np.median(np.abs(s - np.median(s, axis=0)), axis=0)))

    @property
    def locked(self):
        return self.cfg.enabled and len(self.samples) >= self.cfg.min_samples and self.sigma <= self.cfg.max_sigma_px

    # -- use -------------------------------------------------------------------------
    def prior_motion(self, gap, reason='ego_prior'):
        v = self.velocity; H = np.eye(3); H[:2, 2] = v * gap
        sigma = max(self.cfg.fallback_sigma_px, self.sigma) * math.sqrt(gap)
        return Motion(H, bool(self.cfg.trust_fallback), float(sigma),
                      {'reason': reason, 'velocity': v.tolist(), 'trusted': bool(self.cfg.trust_fallback),
                       'sigma_px': float(sigma), 'H': H.tolist()})

    def refine(self, motion, gap):
        """Return (motion to use, audit). Replaces failed or outlying registrations once locked."""
        gap = max(1, int(gap)); info = {'locked': bool(self.locked), 'samples': len(self.samples), 'action': 'none'}
        if not self.locked:
            return motion, info
        v = self.velocity; info.update(velocity=v.tolist(), sigma_px=self.sigma)
        if not motion.trusted:
            self.stats['fallback'] += 1; info['action'] = 'ego_prior'
            return self.prior_motion(gap), info
        if motion.details.get('reason') == 'registered':
            deviation = float(np.linalg.norm(motion.H[:2, 2] / gap - v))
            info['registration_deviation_px_per_frame'] = deviation
            if deviation > self.cfg.reject_outlier_px:
                self.stats['outlier'] += 1; info['action'] = 'registration_outlier'
                return self.prior_motion(gap, 'ego_prior_outlier'), info
            info['action'] = 'registered'
        return motion, info

    def update(self, motion, gap, innovations):
        """Learn from the motion the tracker just used and from how matched detections deviated from it."""
        if not self.cfg.enabled or not motion.trusted: return
        gap = max(1, int(gap)); reason = motion.details.get('reason'); H = motion.H
        if reason not in ('registered', 'ego_prior', 'ego_prior_outlier'): return
        if reason == 'registered':   # only near-pure translations describe the flight
            if abs(np.linalg.norm(H[:2, 0]) - 1) > .02 or abs(math.degrees(math.atan2(H[1, 0], H[0, 0]))) > 1.5: return
        correction = np.median(np.array(innovations, float), axis=0) if len(innovations) else None
        if reason != 'registered' and correction is None: return
        total = H[:2, 2] + (correction if correction is not None else 0.)
        sample = total / gap
        if np.isfinite(sample).all() and np.linalg.norm(sample) < self.cfg.max_speed_px_per_frame:
            self.samples.append(sample); self.stats['samples'] += 1
