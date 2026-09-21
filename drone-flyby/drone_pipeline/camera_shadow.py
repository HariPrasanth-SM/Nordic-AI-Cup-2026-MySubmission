"""Shadow of the organizer's *real* camera, and the only gate camera commands pass.

Why this exists: the organizer validates a command against its own camera, which can be
one command AHEAD of the view it sends us (the frame was rendered before our previous
command was applied). Gating against `request.view` alone therefore lets through moves
that are legal from the view but illegal from the real camera - exactly the
"ignored camera request ... from L2 (x, y)" errors, where "from" is our own last command.

The shadow replays our commands (a refusal is announced in `camera_command_feedback`),
classifies each request as lag0/lag1/desync, and every command is projected into the set
that is legal from ALL plausible real cameras. Holding (no command) is always legal.
"""
import math
import threading
from collections import Counter
from dataclasses import dataclass
from dtos import (ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS, FULL_FRAME_CENTER,
                  RequestedViewDto)
from utils import center_bounds_for_level


@dataclass(frozen=True)
class Cam:
    level: int
    x: int
    y: int

    def dump(self): return {'resolution_level': self.level, 'center_x': self.x, 'center_y': self.y}


def view_cam(request):
    v = request.view
    return Cam(int(v.resolution_level), int(v.center_x), int(v.center_y))


def level_hint(constraints):
    """Level whose official (delta, allowed levels) pair the constraints match, else None."""
    allowed = tuple(sorted(constraints.allowed_resolution_levels))
    hits = [l for l in (0, 1, 2) if tuple(sorted(ALLOWED_RESOLUTION_LEVELS[l])) == allowed
            and abs(MAXIMUM_CENTER_DELTA_PIXELS[l] - float(constraints.maximum_center_delta)) < 1e-6]
    return hits[0] if len(hits) == 1 else None


def _limits(cams, constraints, margin):
    cap = float(constraints.maximum_center_delta)
    return [max(0., min(MAXIMUM_CENTER_DELTA_PIXELS[c.level], cap) - margin) for c in cams]


def _bounds(constraints, level):
    lo_x, hi_x, lo_y, hi_y = center_bounds_for_level(level)
    b = constraints.bounds_for_level(level)
    if b is None or level not in constraints.allowed_resolution_levels: return None
    return (max(lo_x, b.minimum_center_x), min(hi_x, b.maximum_center_x),
            max(lo_y, b.minimum_center_y), min(hi_y, b.maximum_center_y))


def is_legal(cams, constraints, level, x, y, margin=.01):
    """Exact check of an integer command from every candidate camera."""
    if any(level not in ALLOWED_RESOLUTION_LEVELS[c.level] for c in cams): return False
    b = _bounds(constraints, level)
    if b is None or not (b[0] <= x <= b[1] and b[2] <= y <= b[3]): return False
    if level == 0:
        return (x, y) == FULL_FRAME_CENTER and constraints.full_view_reset_exempt_from_delta
    return all(math.hypot(x - c.x, y - c.y) <= r + 1e-7 for c, r in zip(cams, _limits(cams, constraints, margin)))


def _dykstra(target, rect, disks, iterations=200):
    """Nearest point to `target` in rect ∩ disks (all convex)."""
    def proj_rect(p): return (min(max(p[0], rect[0]), rect[1]), min(max(p[1], rect[2]), rect[3]))
    def proj_disk(p, c, r):
        dx, dy = p[0] - c[0], p[1] - c[1]; d = math.hypot(dx, dy)
        return p if d <= r else (c[0] + dx * r / d, c[1] + dy * r / d)
    projections = [proj_rect] + [(lambda p, c=c, r=r: proj_disk(p, c, r)) for c, r in disks]
    x = target; increments = [(0., 0.)] * len(projections)
    for _ in range(iterations):
        for i, proj in enumerate(projections):
            y = (x[0] + increments[i][0], x[1] + increments[i][1]); z = proj(y)
            increments[i] = (y[0] - z[0], y[1] - z[1]); x = z
    return x


def project(cams, constraints, level, target, margin=.01):
    """Integer (x, y) at `level` legal from every camera in `cams`, nearest to `target`; else None."""
    if not cams or any(level not in ALLOWED_RESOLUTION_LEVELS[c.level] for c in cams): return None
    b = _bounds(constraints, level)
    if b is None: return None
    if level == 0:
        return FULL_FRAME_CENTER if is_legal(cams, constraints, 0, *FULL_FRAME_CENTER, margin) else None
    tx, ty = float(target[0]), float(target[1])
    for shrink in (0., .5, 1.5, 3.):   # small extra shrink survives integer rounding on thin lenses
        radii = [max(0., r - shrink) for r in _limits(cams, constraints, margin)]
        p = _dykstra((tx, ty), b, [((c.x, c.y), r) for c, r in zip(cams, radii)])
        base = (round(p[0]), round(p[1]))
        options = [(base[0] + i, base[1] + j) for i in range(-2, 3) for j in range(-2, 3)]
        options = [o for o in options if is_legal(cams, constraints, level, o[0], o[1], margin)]
        if options:
            x, y = min(options, key=lambda o: math.hypot(o[0] - tx, o[1] - ty))
            return int(x), int(y)
    return None


class CameraShadow:
    def __init__(self, cfg=None):
        self.trust_after = getattr(cfg, 'shadow_trust_after', 3)
        self.margin = getattr(cfg, 'safety_margin_px', .01)
        self.est = self.prev = None; self.pending = None
        self.abandoned = set(); self.streak = 0; self.mode = 'init'
        self.stats = Counter(); self.lock = threading.RLock()

    # -- per request ---------------------------------------------------------------
    def observe(self, request):
        """Call once per new request, before planning. Returns a JSON-safe audit dict."""
        with self.lock:
            view = view_cam(request); fb = request.camera_command_feedback
            hint = level_hint(request.camera_constraints)
            if self.est is None:
                self.est = self.prev = view; self.mode = 'init'
            else:
                self.prev = self.est; pending, self.pending = self.pending, None
                if pending is not None and pending[0] not in self.abandoned:
                    if fb is not None and fb.frame == pending[1]: self.stats['rejected'] += 1
                    else: self.est = pending[2]
                if view == self.est and view != self.prev: self.mode = 'lag0'; self.streak = 0
                elif view == self.prev and view != self.est: self.mode = 'lag1'; self.streak += 1
                elif view == self.est: self.mode = 'same'
                else:
                    self.mode = 'desync'; self.stats['desync'] += 1
                    self.est = self.prev = view; self.streak = 0
            self.stats[self.mode] += 1
            if hint is not None and hint not in (self.est.level, view.level): self.stats['hint_mismatch'] += 1
            return {'mode': self.mode, 'view': view.dump(), 'estimated_camera': self.est.dump(),
                    'hint_level': hint, 'lag1_streak': self.streak, 'stats': dict(self.stats)}

    def candidates(self, request):
        """Cameras the next command must be legal from (estimate first)."""
        with self.lock:
            view = view_cam(request); hint = level_hint(request.camera_constraints)
            if self.est is None: return [view]
            hint_ok = hint is None or hint in (self.est.level, view.level)
            trusted = self.streak >= self.trust_after and hint_ok
            return [self.est] if trusted or view == self.est else [self.est, view]

    def commit(self, request, cam):
        with self.lock:
            if request.request_id not in self.abandoned:
                self.pending = None if cam is None else (request.request_id, request.frame, cam)

    def abandon(self, request_id):
        """The response for this request was never delivered; do not assume it was applied."""
        with self.lock:
            self.abandoned.add(request_id)
            if len(self.abandoned) > 256: self.abandoned = set(list(self.abandoned)[-128:])
            if self.pending is not None and self.pending[0] == request_id: self.pending = None

    # -- the gate ------------------------------------------------------------------
    def guard(self, request, command):
        """Return (legal RequestedViewDto | None, audit). Also records the command as pending."""
        with self.lock:
            cams = self.candidates(request); c = request.camera_constraints
            audit = {'candidates': [x.dump() for x in cams], 'proposed': None if command is None else command.model_dump(),
                     'limits_px': _limits(cams, c, self.margin), 'action': 'hold'}
            if command is None:
                self.commit(request, None); return None, audit
            xy = project(cams, c, command.resolution_level, (command.center_x, command.center_y), self.margin)
            if xy is None:
                audit['action'] = 'hold_no_legal_point'; self.commit(request, None); return None, audit
            final = RequestedViewDto(resolution_level=int(command.resolution_level), center_x=int(xy[0]), center_y=int(xy[1]))
            audit.update(final=final.model_dump(), action='unchanged' if final == command else 'corrected',
                         distances_px=[math.hypot(final.center_x - x.x, final.center_y - x.y) for x in cams])
            self.commit(request, Cam(final.resolution_level, final.center_x, final.center_y))
            return final, audit
