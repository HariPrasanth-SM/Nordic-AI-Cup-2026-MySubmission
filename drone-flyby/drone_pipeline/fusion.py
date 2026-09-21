"""Causal multi-frame fusion: decides what is sent for the CURRENT frame.

Only frames up to the current one exist when the response is due (the organizer sends the next frame after
our answer, and a frame that was never requested cannot be answered), so the "window" is the past
`publish.window_ms` (2700 ms = 8 frames). Frame ids, not list positions, measure time, so skipped frames age
evidence correctly.

Per identity (tracker track):
* every observation in the window votes for a class (noisy-OR). One observation's evidence is
  YOLO confidence x DINO agreement (bank nearest-neighbour AND K-prototype match) x size prior (source pixels);
* a constant-velocity line through the window (ego-compensated) penalizes objects that jump;
* an unconfirmed single sighting is published at a reduced score;
* MISS-IN-VIEW anomaly: an object that was seen, is now inside the current (L1/L2) view, large enough to be
  detectable, and is NOT detected, loses score (x miss_factor per miss); a barely-seen track (<= anomaly_max_hits
  sightings) is removed after anomaly_misses such misses, an established one after drop_misses;
* an unobserved object is published at its dead-reckoned box only when confirmed and with enough evidence.
Unmatched detections are published as low-weight orphans only above orphan_min_conf.
"""
import logging
import math
from collections import deque
from pathlib import Path
import numpy as np
from dtos import OBJECT_CLASSES
from .geometry import nms

N = len(OBJECT_CLASSES)
log = logging.getLogger(__name__)


def _sigmoid(x): return 1. / (1. + math.exp(-max(-30., min(30., x))))


def _finite(*xs): return all(x is not None and math.isfinite(float(x)) for x in xs)


class Fusion:
    def __init__(self, cfg):
        self.cfg = cfg.publish; self.hist = {}; self.state = {}; self.last_frame = None; self.size = None
        c = self.cfg
        if c.size_prior:
            from .priors import SizePrior
            path = Path(c.size_prior)
            if path.is_file():
                self.size = SizePrior.load(path, OBJECT_CLASSES, floor=c.size_floor, sigma_scale=c.size_sigma_scale, aspect_weight=c.size_aspect_weight)
            else: log.warning('publish.size_prior %s not found: size prior disabled (scripts/build_size_prior.py)', path)

    # -- evidence ---------------------------------------------------------------------
    def _row(self, frame, level, d):
        c = self.cfg; ex = getattr(d, 'extra', None) or {}; agree = None; p = None; pagree = None; pz = None
        if ex.get('status') == 'verified':
            t, bg, o = (ex.get(k) for k in ('target_similarity', 'background_similarity', 'best_other_similarity'))
            if _finite(t, bg, o): agree = _sigmoid((min(t - bg, t - o) - c.dino_margin0) / c.dino_scale)
            sims = ex.get('class_similarities')
            if sims and len(sims) == N and all(math.isfinite(v) for v in sims):
                z = np.array(sims, float) / c.dino_tau; z -= z.max(); p = np.exp(z); p /= p.sum()
            pz = ex.get('proto_z_own')
            if _finite(pz, ex.get('proto_own'), ex.get('proto_bg')):   # K-prototype evidence replaces the single-neighbour class distribution
                pagree = min(_sigmoid((pz - c.proto_z0) / c.proto_scale), _sigmoid((ex['proto_own'] - ex['proto_bg']) / c.proto_bg_scale))
                pc = ex.get('proto_class')
                if pc and len(pc) == N and all(math.isfinite(v) for v in pc):
                    z = np.array(pc, float) / c.dino_tau; z -= z.max(); p = np.exp(z); p /= p.sum()
            else: pz = None
        b = d.box
        return {'frame': int(frame), 'level': int(level), 'cls': int(d.cls), 'score': float(d.score), 'partial': bool(d.partial),
                'agree': agree, 'pagree': pagree, 'proto_z': pz, 'p': p, 'center': (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)),
                'size': (float(b[2] - b[0]), float(b[3] - b[1]))}

    def _evidence(self, row):
        c = self.cfg
        s = row['score'] * (c.partial_weight if row['partial'] else c.full_weight)
        s *= c.dino_missing if row['agree'] is None else c.dino_floor + (1 - c.dino_floor) * row['agree']
        if row['pagree'] is not None: s *= c.proto_floor + (1 - c.proto_floor) * row['pagree']
        v = np.zeros(N)
        if row['p'] is not None and c.dino_class_weight > 0:
            v[row['cls']] = s * (1 - c.dino_class_weight); v += s * c.dino_class_weight * row['p']
        else: v[row['cls']] = s
        if self.size is not None and not row['partial']: v = v * self.size.factors(*row['size'])   # size compatibility per class
        return np.clip(v, 0., .98)

    def _stability(self, full_rows, ego):
        """Constant-velocity residual, in object sizes, of ego-compensated centres (1.0 = perfectly on the line)."""
        if ego is None or not ego.locked or len(full_rows) < 2: return 1., None
        v = ego.velocity; fr = np.array([x['frame'] for x in full_rows], float)
        g = np.array([x['center'] for x in full_rows], float) - v[None, :] * fr[:, None]
        size = float(np.mean([math.sqrt(max(x['size'][0] * x['size'][1], 1.)) for x in full_rows]))
        if len(full_rows) >= 3 and np.ptp(fr) >= 2:
            A = np.c_[np.ones(len(fr)), fr - fr.mean()]; res = g - A @ np.linalg.lstsq(A, g, rcond=None)[0]
        else: res = g - g.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) / size
        return math.exp(-.5 * (rms / self.cfg.stab_sigma) ** 2), rms

    # -- miss-in-view -----------------------------------------------------------------
    def _observable(self, t, r, level, rel, detector_ok):
        """Weight (0 = not a meaningful miss) of NOT detecting track `t` in the current view."""
        c = self.cfg
        if not detector_ok or level < 1 or rel > c.miss_max_uncertainty: return 0.
        x1, y1, x2, y2 = (float(v) for v in t.box); w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: return 0.
        rx1, ry1, rx2, ry2 = r.view.source_region_xyxy
        inter = max(0., min(x2, rx2) - max(x1, rx1)) * max(0., min(y2, ry2) - max(y1, ry1))
        if inter / (w * h) < c.miss_visible_frac: return 0.
        scale = (rx2 - rx1) / float(r.view.width)                # source px per view px
        if min(w, h) / scale < c.miss_min_view_px: return 0.
        st = self.state.get(t.id); weight = 1. if level >= 2 else c.miss_l1_weight
        if st is not None and level < st['min_level']: weight *= c.miss_cross_level_weight   # only ever seen at a finer level
        return weight

    # -- publication ------------------------------------------------------------------
    def process(self, r, detections, tracks, motion, now, dt, baseline, detector_info, ego=None, detector_ok=True, **_):
        c = self.cfg; frame = r.frame; level = r.view.resolution_level
        window = max(1, round(c.window_ms / max(1, r.frame_interval_ms)))
        skipped = 0 if self.last_frame is None else max(0, frame - self.last_frame - 1); self.last_frame = frame
        if detector_info.get('skipped'): detector_ok = False
        alive = {t.id for t in tracks}
        self.hist = {k: v for k, v in self.hist.items() if k in alive}; self.state = {k: v for k, v in self.state.items() if k in alive}
        used = set()
        for t in tracks:
            if t.updated and t.last_obs is not None:
                used.add(id(t.last_obs)); row = self._row(frame, level, t.last_obs)
                self.hist.setdefault(t.id, deque(maxlen=64)).append(row)
                st = self.state.setdefault(t.id, {'full_hits': 0, 'miss': 0., 'consec': 0., 'min_level': 9})
                st['consec'] = 0.; st['min_level'] = min(st['min_level'], level); st['full_hits'] += 0 if row['partial'] else 1
        items = []; decisions = []
        for t in tracks:
            hist = self.hist.get(t.id)
            if not hist: continue
            st = self.state[t.id]
            size = np.maximum(t.box[2:] - t.box[:2], 1.)
            rel = float(np.max(np.sqrt(np.maximum(np.diag(t.P)[:2], 0.)) / size))
            fresh = bool(t.updated and t.last_obs is not None)
            if c.miss_enabled and not fresh:
                weight = self._observable(t, r, level, rel, detector_ok)
                if weight > 0: st['miss'] += weight; st['consec'] += weight
            removed = None
            if c.miss_enabled:
                if st['full_hits'] <= c.anomaly_max_hits and st['consec'] >= c.anomaly_misses: removed = 'anomaly_seen_once_missed_in_view'
                elif st['consec'] >= c.drop_misses: removed = 'dropped_repeated_misses_in_view'
            recent = [x for x in hist if frame - x['frame'] < window]
            if not recent:
                if frame - hist[-1]['frame'] > c.coast_max_frames: continue
                recent = list(hist)[-2:]
            miss = np.ones(N)
            for x in recent: miss *= 1 - self._evidence(x)
            E = 1 - miss; order = np.argsort(-E); c1, c2 = int(order[0]), int(order[1])
            full = [x for x in recent if not x['partial']]
            age = frame - (full[-1]['frame'] if full else recent[-1]['frame'])
            stab, rms = self._stability(full, ego)
            recency = math.exp(-age / c.recency_tau)
            loc = 1. if age == 0 else math.exp(-c.loc_gain * rel)
            confirmed = st['full_hits'] >= c.confirm_hits
            checks = [x for x in (recent[-1]['agree'], recent[-1]['pagree']) if x is not None]
            strength = min(checks) if checks else 0.
            conf_factor = 1. if confirmed else c.single_sight_factor + (1 - c.single_sight_factor) * strength ** 2
            miss_pen = c.miss_factor ** st['consec'] if c.miss_enabled else 1.
            factor = stab * recency * loc * (1. if full else c.partial_only_factor) * conf_factor * miss_pen
            box = t.last_obs.box.copy() if fresh and not t.last_obs.partial else np.array(t.box, float)
            agrees = [x['agree'] for x in recent if x['agree'] is not None]
            source = 'fresh' if fresh else 'propagated'
            feats = {'n_obs': len(recent), 'n_full': len(full), 'full_hits': st['full_hits'], 'consec_miss': float(st['consec']),
                     'age_frames': int(age), 'stability': float(stab), 'stability_rms_sizes': None if rms is None else float(rms),
                     'recency': float(recency), 'loc': float(loc), 'confirmed': bool(confirmed), 'conf_factor': float(conf_factor),
                     'miss_penalty': float(miss_pen), 'dino_agree': float(np.mean(agrees)) if agrees else None,
                     'proto_z': recent[-1]['proto_z'], 'size_factor': None if self.size is None or recent[-1]['partial'] else float(self.size.factors(*recent[-1]['size'])[c1]),
                     'purity': float(E[c1] / max(E.sum(), 1e-9)), 'evidence': float(E[c1])}
            score = float(E[c1] * factor)
            skip = removed
            if skip is None and not fresh and (not confirmed or E[c1] < c.propagate_min_evidence): skip = 'unconfirmed_or_weak_unobserved'
            if skip is not None:
                decisions.append({'track_id': int(t.id), 'class': OBJECT_CLASSES[c1], 'score': score, 'source': source, 'published': False, 'removed_because': skip, **feats}); continue
            items.append({'track_id': int(t.id), 'box': box, 'cls': c1, 'score': score, 'fresh': fresh, 'source': source, 'features': feats})
            if E[c2] >= c.hedge_min and c2 != c1:
                items.append({'track_id': int(t.id), 'box': box.copy(), 'cls': c2, 'score': float(E[c2] * factor * c.hedge_scale),
                              'fresh': fresh, 'source': 'hedge', 'features': dict(feats, evidence=float(E[c2]))})
            decisions.append({'track_id': int(t.id), 'class': OBJECT_CLASSES[c1], 'score': score, 'source': source, 'published': True, **feats})
        for d in detections:   # detections no identity absorbed (below birth confidence, or gated out)
            if id(d) in used or d.partial or d.score < c.orphan_min_conf: continue
            ev = self._evidence(self._row(frame, level, d)); k = int(ev.argmax())
            items.append({'track_id': None, 'box': np.array(d.box, float), 'cls': k, 'score': float(ev[k] * c.orphan_factor),
                          'fresh': True, 'source': 'orphan', 'features': {'evidence': float(ev[k])}})
        items = [i for i in items if i['score'] >= c.min_publish]
        items = nms(items, c.nms_iou, box=lambda e: e['box'], label=lambda e: e['cls'], score=lambda e: e['score'])[:c.max_out]
        info = {'mode': 'fusion', 'window_frames': window, 'skipped_frames_before': skipped, 'ego_locked': bool(ego is not None and ego.locked),
                'size_prior': self.size is not None, 'detector_ok': bool(detector_ok), 'published': len(items),
                'removed': [d for d in decisions if not d['published']], 'decisions': decisions}
        return items, info, None


class Passthrough:
    """publish.mode = tracker: send exactly what the tracker exports."""
    def process(self, r, detections, tracks, motion, now, dt, baseline, detector_info, **_):
        return baseline, {'mode': 'tracker'}, None


def make_publisher(cfg):
    if cfg.publish.mode == 'fusion': return Fusion(cfg)
    if cfg.publish.mode == 'tracker': return Passthrough()
    from .precision_gate import PrecisionGate   # legacy behaviour, controlled by PRECISION_* env vars
    return PrecisionGate(cfg)
