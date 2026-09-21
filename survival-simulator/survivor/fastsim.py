"""
Optional speed patches for the LOCAL simulator (training / analysis only).

The validation/evaluation server runs the official, unpatched simulator - nothing in here
is used by server.py. The patches keep the simulator semantics identical (same RNG stream,
same physics); they only remove redundant numpy/shapely work:
  * per-chunk cached edge / obstacle arrays (static after env creation)
  * one shapely polygon per observe() call + vectorised contains_xy
  * vectorised rel_dir / visibility-polygon sorting

`verify()` runs the original and the patched simulator side by side and compares every
observation, so you can prove equivalence on your own machine before trusting a training run.
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon

from src.elements import creature as _creature
from src.elements import environment as _environment
from src.elements.creature import Creature
from src.elements.environment import Environment

_ORIG = {}


class _EdgeList(list):
    """list of edges + cached (E,2,2) array"""
    __slots__ = ("arr",)


class _ObsList(list):
    """list of obstacles + cached bound arrays"""
    __slots__ = ("x0", "y0", "x1", "y1")


def _prune(x, y, radius, edges_arr):
    """keep only edges whose segment is within `radius` of (x, y) - farther edges can never be hit."""
    if edges_arr.shape[0] < 12:
        return edges_arr
    x1, y1 = edges_arr[:, 0, 0], edges_arr[:, 0, 1]
    vx, vy = edges_arr[:, 1, 0] - x1, edges_arr[:, 1, 1] - y1
    vv = vx * vx + vy * vy
    t = np.clip(((x - x1) * vx + (y - y1) * vy) / np.where(vv > 0, vv, 1.0), 0.0, 1.0)
    cx, cy = x1 + t * vx - x, y1 + t * vy - y
    return edges_arr[(cx * cx + cy * cy) <= (radius * 1.0001) ** 2]


def _compute_visibility_arr(x, y, direction, cone_angle, vision_radius, edges_arr, epsilon=1e-3):
    edges_arr = _prune(x, y, vision_radius, edges_arr)
    rays_list = []
    if edges_arr.size:
        corners = edges_arr.reshape(-1, 2)
        dx = corners[:, 0] - x
        dy = corners[:, 1] - y
        mask = dx * dx + dy * dy <= vision_radius ** 2
        dx, dy = dx[mask], dy[mask]
        if dx.size > 0:
            base = np.arctan2(dy, dx)
            all_angles = base[:, None] + np.array([-epsilon, 0.0, epsilon])
            rel = (all_angles - direction + np.pi) % (2 * np.pi) - np.pi
            m = np.abs(rel) <= cone_angle / 2
            rays_list.append(all_angles[m])
    rays_list.append(np.array([direction - cone_angle / 2, direction + cone_angle / 2,
                               direction - cone_angle / 4, direction + cone_angle / 4, direction]))
    rays = np.unique(np.concatenate(rays_list))
    rdx, rdy = np.cos(rays), np.sin(rays)

    if edges_arr.size == 0:
        px = x + rdx * vision_radius
        py = y + rdy * vision_radius
        pts = np.column_stack((px, py))
        ang = (np.arctan2(py - y, px - x) - direction + np.pi) % (2 * np.pi) - np.pi
        order = np.argsort(ang, kind="stable")
        return [tuple(p) for p in pts[order]], []

    x1, y1 = edges_arr[:, 0, 0], edges_arr[:, 0, 1]
    x2, y2 = edges_arr[:, 1, 0], edges_arr[:, 1, 1]
    vx, vy = x2 - x1, y2 - y1
    det = -rdx[:, None] * vy[None, :] + rdy[:, None] * vx[None, :]
    ok = np.abs(det) >= 1e-8
    det_safe = np.where(ok, det, 1.0)
    t = np.where(ok, (-vy[None, :] * (x1[None, :] - x) + vx[None, :] * (y1[None, :] - y)) / det_safe, np.inf)
    u = np.where(ok, (-rdy[:, None] * (x1[None, :] - x) + rdx[:, None] * (y1[None, :] - y)) / det_safe, np.inf)
    valid = (t >= 0) & (u >= 0) & (u <= 1)
    t[~valid] = np.inf
    min_idx = np.argmin(t, axis=1)
    min_t = t[np.arange(len(rays)), min_idx]
    min_t = np.minimum(min_t, vision_radius)
    px = x + rdx * min_t
    py = y + rdy * min_t
    hit = min_t < vision_radius
    hit_edges = [tuple(map(tuple, edges_arr[j])) for j in min_idx[hit]]
    ang = (np.arctan2(py - y, px - x) - direction + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(ang, kind="stable")
    pts = np.column_stack((px, py))[order]
    return [tuple(p) for p in pts], hit_edges


def _update_vision(self, edges):
    arr = getattr(edges, "arr", None)
    if arr is None:
        edges = list(edges) if edges is not None else []
        arr = np.array(edges, dtype=float) if edges else np.zeros((0, 2, 2))
    self._vision_poly, observed = _compute_visibility_arr(
        self.x, self.y, self.direction, self.cone_angle, self.vision_radius, arr)
    return self._vision_poly, observed


def _observe(self, agents=None, fruits=None, trees=None, obstacles=None, edges=None, predators=None):
    observations = []
    cos_dir, sin_dir = np.cos(-self.direction), np.sin(-self.direction)
    half_cone = self.cone_angle / 2.0
    if edges is None and obstacles:
        edges = [e for obs in obstacles for e in obs.edges]
    vision_poly, hit_edges = self.update_vision(edges)
    shape_holder = []
    sx, sy = self.x, self.y
    hear, vis = self.hearing_radius, self.vision_radius

    def process(obj_list, tag, include_direction=False, include_id=False):
        if not obj_list:
            return
        n = len(obj_list)
        xs = np.fromiter((o.x for o in obj_list), float, n)
        ys = np.fromiter((o.y for o in obj_list), float, n)
        dx, dy = xs - sx, ys - sy
        distances = np.hypot(dx, dy)
        angles = np.arctan2(dy, dx) - self.direction
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        nearby = distances <= hear
        visible = (~nearby) & (distances <= vis) & (np.abs(angles) <= half_cone)
        idxs = np.where(nearby | visible)[0]
        if len(idxs) == 0:
            return
        inside_far = {}
        far = np.where(visible)[0]
        if len(far):
            if not shape_holder:
                shape_holder.append(Polygon([(sx, sy)] + vision_poly))
            flags = shapely.contains_xy(shape_holder[0], xs[far], ys[far])
            inside_far = dict(zip(far.tolist(), flags.tolist()))
        if include_direction:
            dirs = np.fromiter((o.direction for o in obj_list), float, n)
            rel = ((np.arctan2(sy - ys, sx - xs) - dirs + np.pi) % (2 * np.pi)) - np.pi
        # original order: all "nearby" first, then far ones
        for idx in np.where(nearby)[0]:
            o = {"type": tag, "distance": float(distances[idx]), "angle": float(angles[idx])}
            if include_direction:
                o["rel_dir"] = float(rel[idx])
            if include_id:
                o["id"] = obj_list[idx].agent_id
            observations.append(o)
        for idx in far:
            if inside_far[int(idx)]:
                o = {"type": tag, "distance": float(distances[idx]), "angle": float(angles[idx])}
                if include_direction:
                    o["rel_dir"] = float(rel[idx])
                if include_id:
                    o["id"] = obj_list[idx].agent_id
                observations.append(o)

    if fruits:
        process(list(fruits), "Fruit")
    if agents:
        process([a for a in agents if a is not self], "Agent", True, True)
    if predators:
        process([p for p in predators if p is not self], "Predator", True)
    if trees:
        process(list(trees), "Tree")
    for (esx, esy), (eex, eey) in hit_edges:
        dxs, dys, dxe, dye = esx - sx, esy - sy, eex - sx, eey - sy
        observations.append({"type": "Edge", "coords": (
            (dxs * cos_dir - dys * sin_dir, dxs * sin_dir + dys * cos_dir),
            (dxe * cos_dir - dye * sin_dir, dxe * sin_dir + dye * cos_dir))})
    return observations


def _get_local_edges(self, creature):
    cache = self.__dict__.setdefault("_edge_cache", {})
    key = self.to_chunk(creature.x, creature.y)
    got = cache.get(key)
    if got is None:
        s = set()
        for ch in self._get_neighboring_chunks(creature.x, creature.y):
            s.update(self.grid_edges.get(ch, []))
        got = _EdgeList(s)
        got.arr = np.array(list(s), dtype=float) if s else np.zeros((0, 2, 2))
        cache[key] = got
    return got


def _get_local_obstacles(self, creature):
    cache = self.__dict__.setdefault("_obs_cache", {})
    key = self.to_chunk(creature.x, creature.y)
    got = cache.get(key)
    if got is None:
        s = set()
        for ch in self._get_neighboring_chunks(creature.x, creature.y):
            s.update(self.grid_obstacles.get(ch, []))
        got = _ObsList(s)
        got.x0 = np.array([o.x for o in s], float)
        got.y0 = np.array([o.y for o in s], float)
        got.x1 = got.x0 + np.array([o.width for o in s], float)
        got.y1 = got.y0 + np.array([o.height for o in s], float)
        cache[key] = got
    return got


def _get_local_objects(self, creature):
    la = set(); lf = set(); lt = set(); lp = set()
    for ch in self._get_neighboring_chunks(creature.x, creature.y):
        la.update(self.grid_agents.get(ch, []))
        lf.update(self.grid_fruits.get(ch, []))
        lt.update(self.grid_trees.get(ch, []))
        lp.update(self.grid_predators.get(ch, []))
    if creature in la:
        la.remove(creature)
    elif creature in lp:
        lp.remove(creature)
    return la, lf, lt, _get_local_obstacles(self, creature), lp, _get_local_edges(self, creature)


def _in_obstacle(self, point, radius, obstacles=None):
    if obstacles is None:
        return _ORIG["in_obstacle"](self, point, radius, obstacles)
    x0 = getattr(obstacles, "x0", None)
    if x0 is None:
        return _ORIG["in_obstacle"](self, point, radius, obstacles)
    if x0.size == 0:
        return False
    px, py = point
    return bool(np.any((x0 - radius < px) & (px < obstacles.x1 + radius) &
                       (obstacles.y0 - radius < py) & (py < obstacles.y1 + radius)))


def apply():
    if _ORIG:
        return
    _ORIG["observe"] = Creature.observe
    _ORIG["update_vision"] = Creature.update_vision
    _ORIG["get_local_objects"] = Environment._get_local_objects
    _ORIG["get_local_edges"] = Environment._get_local_edges
    _ORIG["get_local_obstacles"] = Environment._get_local_obstacles
    _ORIG["in_obstacle"] = Environment._in_obstacle
    Creature.observe = _observe
    Creature.update_vision = _update_vision
    Environment._get_local_objects = _get_local_objects
    Environment._get_local_edges = _get_local_edges
    Environment._get_local_obstacles = _get_local_obstacles
    Environment._in_obstacle = _in_obstacle


def revert():
    if not _ORIG:
        return
    Creature.observe = _ORIG["observe"]
    Creature.update_vision = _ORIG["update_vision"]
    Environment._get_local_objects = _ORIG["get_local_objects"]
    Environment._get_local_edges = _ORIG["get_local_edges"]
    Environment._get_local_obstacles = _ORIG["get_local_obstacles"]
    Environment._in_obstacle = _ORIG["in_obstacle"]
    _ORIG.clear()


def _canon(o):
    if o["type"] == "Edge":
        (a, b), (c, d) = o["coords"]
        return ("Edge", round(float(a), 5), round(float(b), 5), round(float(c), 5), round(float(d), 5))
    return (o["type"], round(o["distance"], 5), round(o["angle"], 5), round(o.get("rel_dir", 0.0), 5), o.get("id", -1))


def verify(seed=3, ticks=500, extra_agents=10, verbose=True):
    """Run original vs patched simulator with the same scripted actions, compare all observations."""
    import random, time
    from src.core import SimulationCore

    class A:  # minimal action
        def __init__(s, d, m, t): s.move_distance, s.move_direction, s.turn_angle, s.spawn_agent = d, m, t, False

    def run(patched):
        revert()
        if patched:
            apply()
        sim = SimulationCore(seed=seed)
        for _ in range(extra_agents):
            sim.env.spawn_agent()
        rng = random.Random(5)
        actions, out = [], []
        t0 = time.time()
        for _ in range(ticks):
            st = sim.step(actions)
            out.append([(o["agent_id"], sorted(_canon(x) for x in o["observations"])) for o in st["observations"] if o])
            actions = [(a.agent_id, A(rng.uniform(0, 12), rng.uniform(-3, 3), rng.uniform(-1, 1))) for a in sim.env.agents]
            for a in sim.env.agents:
                a.energy = 400
        revert()
        return out, time.time() - t0

    ref, t_ref = run(False)
    fast, t_fast = run(True)
    bad = 0
    for k, (r, f) in enumerate(zip(ref, fast)):
        if r != f:
            bad += 1
            if verbose and bad <= 3:
                print(f"  mismatch at tick {k}")
    ok = bad == 0
    if verbose:
        print(f"fastsim.verify: {'IDENTICAL' if ok else f'{bad} tick(s) differ'} over {ticks} ticks; "
              f"original {t_ref:.1f}s, patched {t_fast:.1f}s (x{t_ref / max(t_fast, 1e-9):.2f})")
    return ok, t_ref / max(t_fast, 1e-9)


if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sys.path.insert(0, ".")
    verify()
