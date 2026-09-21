"""
survivor.micro - controlled micro-experiments to TEST hypotheses about behaviours before building them into the policy.

  python -m survivor.micro list
  python -m survivor.micro run evade   --trials 8 --workers 15            # tactics arena (arms = behaviours)
  python -m survivor.micro run juvenile --trials 10 --workers 15          # why do newborns starve?
  python -m survivor.micro run decoy|doomed|cover ...                      # oracle / tactical hypotheses
  python -m survivor.micro run ablate   --seeds 12 --workers 15           # FULL runs with causal knobs (predators off, food floor)
  python -m survivor.micro run traits   --seeds 12 --workers 15           # FULL runs with FIXED trait presets (no breeding confound)
  python -m survivor.micro report runs/micro/evade_xxx                     # regenerate the report

Every experiment = hypothesis + scenario grid + ARMS (behaviours) + a primary metric, all compared PAIRED (same scenario, same seed).
Outputs (in the run dir):  report.md (quantitative + qualitative), summary.json (machine readable), results.jsonl.gz,
qual/*.png + narratives (exemplar trials: worst / median / best), so that a result can be *understood*, not just scored.

Reading rule used everywhere: an arm is only called BETTER / WORSE when the 95% bootstrap CI of the paired difference excludes 0
AND the mean effect exceeds the experiment's minimum effect. Otherwise INCONCLUSIVE (add trials or drop the idea).
"""
import argparse
import gzip
import json
import math
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from multiprocessing import Pool

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor.policy import Controller, Params, wrap, transform_predators  # noqa: E402

PI = math.pi
FLEE = ("flee", "flee_sprint")
STATE_COL = {"flee": "#e53935", "flee_sprint": "#d500f9", "forage": "#fb8c00", "camp": "#43a047", "seek_tree": "#1e88e5",
             "explore": "#9e9e9e", "unstick": "#8e24aa", "senescent": "#795548", "?": "#000000"}
DEATH_COST = float(os.environ.get("MICRO_DEATH_COST", "300"))   # energy one death is "worth" when trading kills against energy spent (MICRO_EXPERIMENTS.md s.4)


# ============================================================================================== statistics
def paired_stats(x, base, n_boot=3000, seed=1):
    """paired difference x - base: mean, bootstrap 95% CI, median, worst quartile, wins/losses, exact sign test, effect size"""
    d = [a - b for a, b in zip(x, base)]
    n = len(d)
    if n == 0:
        return {}
    mean = sum(d) / n
    sd = st.pstdev(d) if n > 1 else 0.0
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    wins, losses = sum(1 for v in d if v > 1e-12), sum(1 for v in d if v < -1e-12)
    m, k = wins + losses, min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    sd_s = sorted(d)
    return {"n": n, "mean": mean, "lo": lo, "hi": hi, "median": st.median(d), "q25": sd_s[int(0.25 * (n - 1))],
            "wins": wins, "losses": losses, "p": p, "dz": (mean / sd) if sd > 0 else 0.0}


def verdict(s, min_effect):
    if not s:
        return "n/a"
    if s["lo"] > 0 and s["mean"] >= min_effect:
        return "BETTER"
    if s["hi"] < 0 and s["mean"] <= -min_effect:
        return "WORSE"
    if s["lo"] > 0:
        return "better (small)"
    if s["hi"] < 0:
        return "worse (small)"
    return "inconclusive"


def fit_logistic(X, y, iters=60, l2=1e-2):
    """tiny numpy-free IRLS logistic regression with ridge; returns weights (bias first)"""
    import numpy as np
    X = np.asarray(X, float)
    y = np.asarray(y, float)
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
        if np.abs(step).max() < 1e-6:
            break
    p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
    # AUC by rank statistic
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum())
    return w, mu, sd, float(auc), float(((p > 0.5) == pos).mean())


# ============================================================================================== arena engine
_SIMS = {}


def _sim(seed):
    from src.core import SimulationCore
    from survivor import fastsim
    fastsim.apply()
    ms = 1 + seed % 3
    if ms not in _SIMS:
        s = SimulationCore(seed=ms)
        env = s.env
        env._spawn_pred_orig = env.spawn_predator
        env.spawn_predator = lambda *a, **k: None          # no natural predators / trees in an arena: only what the scenario places
        env.spawn_tree = lambda *a, **k: None
        _SIMS[ms] = s
    return _SIMS[ms]


def _clear_caches(env):
    env.__dict__.pop("_edge_cache", None)
    env.__dict__.pop("_obs_cache", None)


def _remove_boxes(env, boxes):
    for b in boxes:
        if b in env.obstacles:
            env.obstacles.remove(b)
    env.edges = set()
    for obs in env.obstacles:
        env.edges.update([((obs.x, obs.y), (obs.x + obs.width, obs.y)), ((obs.x + obs.width, obs.y), (obs.x + obs.width, obs.y + obs.height)),
                          ((obs.x, obs.y + obs.height), (obs.x + obs.width, obs.y + obs.height)), ((obs.x, obs.y), (obs.x, obs.y + obs.height))])
    env._update_obstacle_grid()
    env._update_edge_grid()
    _clear_caches(env)


class _Act:
    __slots__ = ("move_distance", "move_direction", "turn_angle", "spawn_agent")

    def __init__(self, d):
        self.move_distance, self.move_direction = d["move_distance"], d["move_direction"]
        self.turn_angle, self.spawn_agent = d["turn_angle"], d["spawn_agent"]


class World:
    def __init__(self, env, spec, cx, cy, tags):
        self.env, self.spec, self.cx, self.cy, self.tags = env, spec, cx, cy, tags


def run_arena(spec, arm, seed, trace=True):
    """one controlled trial. spec: agents/predators/tree/fruits/boxes/seconds/force_spawn. Returns raw record."""
    from src.elements.tree import Tree
    sim = _sim(seed)
    env = sim.env
    env.time = 0.0
    env.score = 0.0
    env._next_agent_id = 0
    env.rng.seed(seed * 131 + 5)
    env.agents.clear(); env.agents_dict.clear(); env.predators.clear(); env.trees.clear(); env.fruits.clear()
    getattr(env, "fruits_dict", {}).clear()
    rng = random.Random(seed * 7919 + 13)
    for _ in range(300):
        cx, cy = rng.uniform(450, 1150), rng.uniform(350, 850)
        if env._is_position_free(cx - 70, cy - 70, 140, 140):
            break
    boxes = [env.spawn_obstacle(cx + b["dx"], cy + b["dy"], b["w"], b["h"]) for b in spec.get("boxes", [])]
    _clear_caches(env)
    if spec.get("tree") is not None:
        tr = Tree(cx, cy)
        tr.age = spec["tree"].get("age", 30)
        tr.radius = 15
        env.trees.append(tr)
    for et in spec.get("extra_trees", []):                     # other trees so that "leave and explore" has a realistic payoff
        for _ in range(20):
            ang_, d_ = rng.uniform(0, 2 * PI), rng.uniform(*et["dist"])
            tx, ty = cx + d_ * math.cos(ang_), cy + d_ * math.sin(ang_)
            if 40 < tx < 1560 and 40 < ty < 1160 and env._is_position_free(tx - 10, ty - 10, 20, 20):
                t2 = Tree(tx, ty)
                t2.age = et.get("age", 30)
                t2.radius = 15
                env.trees.append(t2)
                break
    env._update_tree_grid()
    fpt = spec.get("fruits_per_tree", 0)
    if fpt:
        for tr_ in list(env.trees):
            for _ in range(fpt):
                a_, r_ = rng.uniform(0, 2 * PI), rng.uniform(12, 55)
                fx, fy = tr_.x + r_ * math.cos(a_), tr_.y + r_ * math.sin(a_)
                if env._is_position_free(fx, fy, 5, 5):
                    fr = env.spawn_fruit(fx, fy)
                    if fr:
                        age_s = rng.uniform(0, 45)                 # a mature unvisited tree holds ~5 fruits of mixed age
                        fr.age = 2.0 * age_s
                        fr.energy = min(60.0, 20.0 + fr.age)
    for f in spec.get("fruits", []):
        x, y = cx + f["dx"], cy + f["dy"]
        if env._is_position_free(x, y, 5, 5):
            fr = env.spawn_fruit(x, y)
            if fr:
                fr.energy = f["E"]
                fr.age = max(0.0, f["E"] - 20.0)
    tags = defaultdict(list)
    maxE = {}
    for a in spec["agents"]:
        ag = None
        for _ in range(30):
            ax, ay = cx + a["dx"] + rng.uniform(-6, 6), cy + a["dy"] + rng.uniform(-6, 6)
            if env._is_position_free(ax, ay, 20, 20):
                ag = env.spawn_agent(x=ax, y=ay)
                break
        if ag is None:
            continue
        ag.max_energy = a.get("maxE", 350.0)
        ag.energy = min(a["E"], ag.max_energy)
        ag.speed = a.get("speed", 10.0)
        ag.sprint_speed = a.get("sprint", max(20.0, ag.speed + 2.0))
        ag.age = a.get("age", 30.0)
        ag.max_age = 500.0
        tags[a["tag"]].append(ag.agent_id)
        maxE[ag.agent_id] = ag.max_energy
    env.agents_dict = {a.agent_id: a for a in env.agents}
    init_ids = [a.agent_id for a in env.agents]
    for p in spec.get("predators", []):
        ang = p.get("bearing", rng.uniform(0, 2 * PI))
        d = p["dist"] if not isinstance(p["dist"], (list, tuple)) else rng.uniform(*p["dist"])
        pr = env._spawn_pred_orig(x=cx + d * math.cos(ang), y=cy + d * math.sin(ang))
        if pr is None:
            continue
        pr.resting = False
        pr.energy = p.get("E", 150.0)
        pr.direction = (ang + PI + rng.uniform(-0.6, 0.6)) if p.get("aimed", True) else rng.uniform(-PI, PI)
    world = World(env, spec, cx, cy, dict(tags))
    arm.start(seed, spec, world)
    t_off = spec.get("t_off", 0.0)
    force = {f["tag"]: f["at"] for f in spec.get("force_spawn", [])}
    forced_done = set()

    fruit_acc = {"eaten": 0, "rot": 0, "eaten_E": 0.0}
    _orig_rm = env.remove_fruit

    def _rm(fruit):
        if fruit in env.fruits:
            if fruit.age > 100:
                fruit_acc["rot"] += 1
            else:
                fruit_acc["eaten"] += 1
                fruit_acc["eaten_E"] += fruit.energy
        _orig_rm(fruit)
    env.remove_fruit = _rm
    gained = defaultdict(float)
    known = {a.agent_id: (a.x, a.y, a.energy) for a in env.agents}
    E_start = {a.agent_id: a.energy for a in env.agents}
    spent = defaultdict(float)
    prevE = {a.agent_id: a.energy for a in env.agents}
    dead, born, first_flee, events = [], [], {}, []
    sprint_ticks = 0
    tr_a = defaultdict(lambda: {"x": [], "y": [], "E": [], "s": [], "d": []})
    tr_p = [{"x": [], "y": []} for _ in env.predators]
    tr_t = []
    last_state = {}
    actions = []
    tick = 0
    seconds = spec["seconds"]
    while env.time < seconds:
        st_ = sim.step(actions)
        tick += 1
        cur = {a.agent_id: a for a in env.agents}
        for aid, (x, y, e) in list(known.items()):
            if aid not in cur:
                cause = "predator" if e > 3.0 else "starved"
                dead.append((round(env.time, 1), aid, cause, round(e, 1), x, y))
                events.append((env.time, f"agent {aid} dies ({cause}) with energy {e:.0f} at ({x:.0f},{y:.0f})"))
                del known[aid]
        for aid, a in cur.items():
            if aid not in known:
                if aid not in init_ids:
                    born.append(aid)
                    maxE[aid] = a.max_energy
                    E_start[aid] = a.energy
                    events.append((env.time, f"child {aid} born with energy {a.energy:.0f} at ({a.x:.0f},{a.y:.0f})"))
                prevE[aid] = a.energy
            known[aid] = (a.x, a.y, a.energy)
            if a.energy < prevE.get(aid, a.energy):
                spent[aid] += prevE[aid] - a.energy
            elif a.energy > prevE.get(aid, a.energy) + 1.0:
                gained[aid] += a.energy - prevE[aid]
            prevE[aid] = a.energy
        if not cur:
            break
        prs = list(env.predators)
        obs = [o for o in st_["observations"] if o]
        step = {"game_status": "ok", "score": 0, "sim_time": st_["sim_time"] + t_off, "n_agents": len(obs), "agent_status": obs}
        acts = arm.act(step, world)
        amap = {a["agent_id"]: a for a in acts}
        for tag, tt in force.items():
            for aid in tags.get(tag, []):
                if aid in amap and env.time >= tt and (tag, aid) not in forced_done and cur[aid].energy > 101:
                    amap[aid]["spawn_agent"] = True
                    forced_done.add((tag, aid))
        for aid, a in cur.items():
            s_ = arm.state(aid)
            if s_ in FLEE and aid not in first_flee and prs:
                dd = min(math.hypot(p.x - a.x, p.y - a.y) for p in prs)
                first_flee[aid] = (round(env.time, 1), round(dd), round(a.energy), round(a.speed, 1))
            if s_ != last_state.get(aid) and trace:
                dd = min((math.hypot(p.x - a.x, p.y - a.y) for p in prs), default=-1)
                events.append((env.time, f"agent {aid}: {last_state.get(aid, 'start')} -> {s_} (energy {a.energy:.0f}, nearest predator {dd:.0f}u)"))
            last_state[aid] = s_
            am = amap.get(aid)
            if am and am["move_distance"] > a.speed + 0.5:
                sprint_ticks += 1
        if trace and tick % 2 == 0:
            tr_t.append(round(env.time, 1))
            for aid, a in cur.items():
                r = tr_a[aid]
                r["x"].append(round(a.x)); r["y"].append(round(a.y)); r["E"].append(round(a.energy)); r["s"].append(last_state.get(aid, "?"))
                r["d"].append(round(min((math.hypot(p.x - a.x, p.y - a.y) for p in prs), default=-1)))
            for i, p in enumerate(prs):
                if i < len(tr_p):
                    tr_p[i]["x"].append(round(p.x)); tr_p[i]["y"].append(round(p.y))
        actions = [(a["agent_id"], _Act(a)) for a in amap.values()]
    alive = {a.agent_id: a for a in env.agents}
    # obstacle rectangles near the arena (for plots / wall-pinning tag)
    rects = [(o.x, o.y, o.width, o.height) for o in env.obstacles if abs(o.x - cx) < 450 and abs(o.y - cy) < 450]
    raw = {"ctrl_stats": dict(getattr(getattr(arm, "ctrl", None), "stats", {}) or {}), "init": init_ids, "born": born, "dead": dead, "alive": {k: (round(v.energy, 1), round(v.max_energy)) for k, v in alive.items()},
           "E_start": {k: round(v, 1) for k, v in E_start.items()}, "maxE": {k: round(v) for k, v in maxE.items()},
           "spent": {k: round(v, 1) for k, v in spent.items()}, "first_flee": first_flee, "sprint_ticks": sprint_ticks,
           "tags": dict(tags), "center": (round(cx), round(cy)), "rects": rects, "seconds": seconds,
           "gained": {k: round(v, 1) for k, v in gained.items()}, "fruit": dict(fruit_acc),
           "predator_energy": [round(p.energy) for p in env.predators], "predator_resting": [bool(p.resting) for p in env.predators]}
    if trace:
        events.sort()
        raw["trace"] = {"t": tr_t, "agents": {str(k): v for k, v in tr_a.items()}, "preds": tr_p, "events": [(round(t, 1), s) for t, s in events[:60]]}
    env.remove_fruit = _orig_rm
    _remove_boxes(env, boxes)
    return raw


# ============================================================================================== arms (behaviours under test)
class ParamArm:
    """the standard controller with optional parameter overrides (arms that differ only in parameters)"""

    def __init__(self, name, cfg=None, over=None, allow_spawn=False, **kw):
        self.name, self.cfg, self.over, self.allow_spawn, self.kw = name, cfg, over or {}, allow_spawn, kw

    def start(self, seed, spec, world):
        p = Params.load(self.cfg) if self.cfg and os.path.exists(self.cfg) else Params()
        for k, v in self.over.items():
            setattr(p, k, float(v))
        if not self.allow_spawn:                       # reproduction is off in arenas unless the scenario forces it
            p.spawn_thr = p.spawn_thr_late = 1e9
            p.dbed_min_energy = p.dbed_min_late = 1e9
            p.cap_early = p.cap_late = 0.0
            p.dbed_extra = p.sen_extra = 0.0
            p.n_min = 0
        self.ctrl = Controller(p, seed=seed)
        self.p = p

    def act(self, step, world):
        return self.ctrl.act(step)

    def state(self, aid):
        return self.ctrl.last_state.get(aid, "?")


def _nearest_pred(o):
    pv = [x for x in o["observations"] if x["type"] == "Predator"]
    return min(pv, key=lambda z: z["distance"]) if pv else None


class StrafeArm(ParamArm):
    """hypothesis: a pursuer with a clipped turn rate (0.3 rad/tick) is beaten by TANGENTIAL movement at close range, not radial flight.
    mix=0 -> radial (the current behaviour), mix=1 -> pure tangential; only inside strafe_d units, always facing the predator."""

    def act(self, step, world):
        acts = self.ctrl.act(step)
        obs = {o["agent_id"]: o for o in step["agent_status"]}
        mix, sd = self.kw.get("mix", 0.5), self.kw.get("strafe_d", 90.0)
        for a in acts:
            if self.ctrl.last_state.get(a["agent_id"]) not in FLEE:
                continue
            q = _nearest_pred(obs[a["agent_id"]])
            if not q or q["distance"] > sd:
                continue
            d, ang, rel = q["distance"], q["angle"], q["rel_dir"]
            ph = wrap(ang + PI - rel)
            wx, wy = math.cos(ph), math.sin(ph)
            cross = wx * d * math.sin(ang) - wy * d * math.cos(ang)
            perp = ph + (PI / 2 if cross >= 0 else -PI / 2)
            away = wrap(ang + PI)
            vx = (1 - mix) * math.cos(away) + mix * math.cos(perp)
            vy = (1 - mix) * math.sin(away) + mix * math.sin(perp)
            a["move_direction"] = math.atan2(vy, vx)
            a["turn_angle"] = ang
        return acts


def _seg_hit(p, q, a, b):
    """segments p-q and a-b intersect?"""
    def cr(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3, d4 = cr(p, q, a), cr(p, q, b), cr(a, b, p), cr(a, b, q)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


class CoverArm(ParamArm):
    """hypothesis: putting a visible obstacle between agent and predator (line of sight blocked) beats running in the open.
    Uses only what the agent sees (edges in its cone); falls back to the base flee when no cover is visible."""

    def act(self, step, world):
        acts = self.ctrl.act(step)
        obs = {o["agent_id"]: o for o in step["agent_status"]}
        L = self.kw.get("reach", 110.0)
        for a in acts:
            if self.ctrl.last_state.get(a["agent_id"]) not in FLEE:
                continue
            o = obs[a["agent_id"]]
            q = _nearest_pred(o)
            edges = [e["coords"] for e in o["observations"] if e["type"] == "Edge"]
            if not q or not edges:
                continue
            P = (q["distance"] * math.cos(q["angle"]), q["distance"] * math.sin(q["angle"]))
            best = None
            for k in range(24):
                th = -PI + k * (2 * PI / 24)
                T = (L * math.cos(th), L * math.sin(th))
                if math.hypot(T[0] - P[0], T[1] - P[1]) < 45:
                    continue
                if any(_seg_hit((0, 0), T, e[0], e[1]) for e in edges):
                    continue                                   # cannot walk through the obstacle
                if not any(_seg_hit(P, T, e[0], e[1]) for e in edges):
                    continue                                   # predator would still see us there
                dist_away = math.hypot(T[0] - P[0], T[1] - P[1])
                if best is None or dist_away > best[0]:
                    best = (dist_away, th)
            if best is not None:
                a["move_direction"] = best[1]
                a["turn_angle"] = q["angle"]
        return acts


class DecoyOracleArm(ParamArm):
    """ORACLE upper bound (uses ground-truth positions): the strong agent flanks the predator to become its nearest target and leads it
    away from the weak agents, then disengages. If this does not help even with perfect information, the idea is dead."""

    def act(self, step, world):
        acts = self.ctrl.act(step)
        env = world.env
        strong = world.tags.get("strong", [])
        weak = [a for a in env.agents if a.agent_id in world.tags.get("weak", [])]
        prs = list(env.predators)
        if not prs or not strong or not weak:
            return acts
        pr = min(prs, key=lambda p: min(math.hypot(p.x - w.x, p.y - w.y) for w in weak))
        wx = sum(w.x for w in weak) / len(weak)
        wy = sum(w.y for w in weak) / len(weak)
        for a in acts:
            if a["agent_id"] not in strong:
                continue
            dag = env.agents_dict.get(a["agent_id"])
            if dag is None:
                continue
            dp = math.hypot(pr.x - dag.x, pr.y - dag.y)
            d_w = min(math.hypot(pr.x - w.x, pr.y - w.y) for w in weak)
            ux, uy = dag.x - wx, dag.y - wy
            n = math.hypot(ux, uy) or 1.0
            ux, uy = ux / n, uy / n                                  # direction away from the weak group
            if dp > d_w - 15 and dp > 70:                            # not yet the nearest target: close in from the flank
                tx, ty = pr.x + ux * 80 - dag.x, pr.y + uy * 80 - dag.y
            else:                                                    # nearest target: lead the predator away, keep >60 units
                ax, ay = dag.x - pr.x, dag.y - pr.y
                m = math.hypot(ax, ay) or 1.0
                tx, ty = 0.6 * ux + 0.4 * ax / m, 0.6 * uy + 0.4 * ay / m
            v = math.atan2(ty, tx)
            a["move_direction"] = wrap(v - dag.direction)
            a["move_distance"] = min(dag.speed, dag.speed)
            a["turn_angle"] = wrap(math.atan2(pr.y - dag.y, pr.x - dag.x) - dag.direction)
        return acts


class DoomedSpawnArm(ParamArm):
    """hypothesis: an agent that cannot escape converts its remaining energy into a child before contact.
    kw: dist = predator distance that triggers the spawn, multi = keep spawning while energy > 101."""

    def start(self, seed, spec, world):
        super().start(seed, spec, world)
        self.done = set()

    def act(self, step, world):
        acts = self.ctrl.act(step)
        obs = {o["agent_id"]: o for o in step["agent_status"]}
        for a in acts:
            aid = a["agent_id"]
            if aid not in world.tags.get("doomed", []):
                continue
            q = _nearest_pred(obs[aid])
            if q and q["distance"] < self.kw.get("dist", 130.0) and obs[aid]["energy"] > 101 and (self.kw.get("multi") or aid not in self.done):
                a["spawn_agent"] = True
                self.done.add(aid)
        return acts



class SharedAlarmArm(ParamArm):
    """hypothesis: agents that share predator sightings with neighbours (via frame transform) react earlier and die less."""

    def act(self, step, world):
        st2, n = transform_predators(step, self.kw.get("radius", 120.0))
        self.injected = getattr(self, "injected", 0) + n
        return self.ctrl.act(st2)


def make_arm(d):
    kind = d.get("kind", "param")
    cls = {"param": ParamArm, "strafe": StrafeArm, "cover": CoverArm, "decoy": DecoyOracleArm, "doomed": DoomedSpawnArm, "alarm": SharedAlarmArm}[kind]
    return cls(d["name"], cfg=d.get("cfg"), over=d.get("over"), allow_spawn=d.get("allow_spawn", False), **d.get("kw", {}))


# ============================================================================================== experiments
def _agents(n, E, speed, tag="a", maxE=350.0, ring=22.0, age=30.0):
    return [{"tag": tag, "dx": ring * math.cos(2 * PI * i / max(n, 1)), "dy": ring * math.sin(2 * PI * i / max(n, 1)),
             "E": E, "speed": speed, "maxE": maxE, "age": age} for i in range(n)]


class Exp:
    name = ""
    title = ""
    hypothesis = ""
    source = ""
    kind = "arena"
    primary = "value"
    min_effect = 0.05
    baseline = "base"
    seconds = 20.0
    higher_is_better = True

    def arms(self, cfg):
        raise NotImplementedError

    def cells(self):
        raise NotImplementedError

    def spec(self, cell, k):
        raise NotImplementedError

    def metrics(self, raw, cell, spec):
        raise NotImplementedError


def _tags_common(raw, spec):
    tags = []
    rects = raw["rects"]

    def near_wall(x, y):
        if x < 60 or y < 60 or x > 1540 or y > 1140:
            return True
        return any(rx - 25 < x < rx + rw + 25 and ry - 25 < y < ry + rh + 25 for rx, ry, rw, rh in rects)
    for (t, aid, cause, e, x, y) in raw["dead"]:
        if cause == "predator" and near_wall(x, y):
            tags.append("died_near_wall_or_obstacle")
    if any(v[1] < 90 for v in raw["first_flee"].values()):
        tags.append("first_reacted_inside_90u")
    if any(v[2] < 0.2 * raw["maxE"].get(aid, 350) for aid, v in raw["first_flee"].items()):
        tags.append("sprint_locked_at_first_contact")
    if any(c == "starved" for (_, _, c, _, _, _) in raw["dead"]):
        tags.append("starved_during_trial")
    if not raw["dead"] and raw["first_flee"]:
        tags.append("all_escaped")
    if raw["sprint_ticks"] > 0:
        tags.append("used_sprint")
    return tags


class Evade(Exp):
    name = "evade"
    title = "Predator evasion tactics (finite energy, real obstacles, energy priced)"
    hypothesis = ("H-E1: strafing (tangential motion) beats radial flight against a turn-limited pursuer. H-E2: sprinting is worth its energy only above an "
                  "energy/speed threshold. H-E3: the flee-optimizer's 'sprint everywhere' solution is not better once energy is real.")
    source = "senior review A/E, high-score doc s.1; our flee optimizer (trial kill 0.196 -> 0.091 but no full-run gain)"
    primary = "value"
    min_effect = 0.04
    baseline = "base"
    seconds = 20.0

    def arms(self, cfg):
        a = [{"name": "base", "cfg": cfg},
             {"name": "strafe50", "kind": "strafe", "cfg": cfg, "kw": {"mix": 0.5, "strafe_d": 100}},
             {"name": "strafe100", "kind": "strafe", "cfg": cfg, "kw": {"mix": 1.0, "strafe_d": 100}},
             {"name": "shared_alarm", "kind": "alarm", "cfg": cfg, "kw": {"radius": 120}},
             {"name": "nosprint", "cfg": cfg, "over": {"sprint_below": 0}},
             {"name": "sprint_always", "cfg": cfg, "over": {"sprint_below": 40, "sprint_min_frac": 0.05, "flee_cap": 24}}]
        best = os.path.join(ROOT, "runs", "flee1", "best.json")
        if os.path.exists(best):
            a.append({"name": "flee_opt", "cfg": best})
        return a

    def cells(self):
        return [{"E": E, "traits": T, "npred": P, "aimed": A} for E in (0.2, 0.6) for T in ("s10/sp20", "s16/sp20", "s10/sp32", "s16/sp32")
                for P in (1, 2) for A in (True, False)]

    def spec(self, c, k):
        sp_, spr_ = c["traits"].replace("s", "", 1).split("/sp")
        ag = _agents(3, c["E"] * 350.0, float(sp_), "a")
        for a in ag:
            a["sprint"] = float(spr_)
        return {"seconds": self.seconds, "tree": {"age": 30}, "t_off": 900.0,
                "agents": ag,
                "predators": [{"dist": (120, 300), "aimed": c["aimed"], "E": 150} for _ in range(c["npred"])]}

    def metrics(self, raw, c, spec):
        n = len(raw["init"])
        surv = sum(1 for a in raw["init"] if a in raw["alive"])
        spent = sum(raw["spent"].get(a, 0.0) for a in raw["init"]) / max(n, 1)
        ff = [v[1] for v in raw["first_flee"].values()]
        return {"value": surv / n - spent / DEATH_COST, "alive_frac": surv / n, "kill_frac": 1 - surv / n, "energy_spent": spent,
                "detect_dist": (sum(ff) / len(ff)) if ff else float("nan"), "sprint_ticks": raw["sprint_ticks"]}, _tags_common(raw, spec)


class Cover(Evade):
    name = "cover"
    title = "Obstacle cover vs open flight"
    hypothesis = "H-C1: a visible obstacle between agent and predator (line of sight blocked) lowers kill probability versus running in the open (senior review E)."
    source = "senior review E"

    def arms(self, cfg):
        return [{"name": "base", "cfg": cfg}, {"name": "cover", "kind": "cover", "cfg": cfg, "kw": {"reach": 110}},
                {"name": "strafe50", "kind": "strafe", "cfg": cfg, "kw": {"mix": 0.5, "strafe_d": 100}}]

    def cells(self):
        return [{"E": E, "traits": T, "npred": 1, "aimed": True, "box": bx} for E in (0.2, 0.6) for T in ("s10/sp20", "s16/sp20") for bx in (0, 1)]

    def spec(self, c, k):
        sp = super().spec(c, k)
        sp["predators"] = [{"dist": (140, 260), "aimed": True, "E": 150, "bearing": 0.0}]     # predator comes from +x
        if c["box"]:
            sp["boxes"] = [{"dx": 50, "dy": -35, "w": 45, "h": 70}]                          # a box between camp and predator, offset
        return sp


class Decoy(Exp):
    name = "decoy"
    title = "Reusable decoy (oracle upper bound)"
    hypothesis = ("H-D1: a strong agent (fast, energetic) that becomes the predator's nearest target and leads it away raises colony survival vs everyone "
                  "fleeing on their own. ORACLE version: if it fails with perfect information it is dead.")
    source = "high-score behaviours doc s.2"
    primary = "value"
    min_effect = 0.05
    seconds = 25.0

    def arms(self, cfg):
        return [{"name": "base", "cfg": cfg}, {"name": "decoy_oracle", "kind": "decoy", "cfg": cfg}]

    def cells(self):
        return [{"npred": P, "strongE": sE} for P in (1, 2) for sE in (150, 300)]

    def spec(self, c, k):
        ag = _agents(2, 90.0, 10.0, "weak", ring=16) + [{"tag": "strong", "dx": 30, "dy": 0, "E": c["strongE"], "speed": 17.0, "maxE": 350.0}]
        return {"seconds": self.seconds, "tree": {"age": 30}, "t_off": 900.0, "agents": ag,
                "predators": [{"dist": (150, 260), "aimed": True, "E": 150} for _ in range(c["npred"])]}

    def metrics(self, raw, c, spec):
        wk, sg = raw["tags"].get("weak", []), raw["tags"].get("strong", [])
        ws = sum(1 for a in wk if a in raw["alive"]) / max(1, len(wk))
        ss = sum(1 for a in sg if a in raw["alive"]) / max(1, len(sg))
        spent = sum(raw["spent"].get(a, 0.0) for a in raw["init"]) / max(1, len(raw["init"]))
        surv = (ws * len(wk) + ss * len(sg)) / max(1, len(raw["init"]))
        return {"value": surv - spent / DEATH_COST, "weak_alive": ws, "strong_alive": ss, "energy_spent": spent}, _tags_common(raw, spec)


class Doomed(Exp):
    name = "doomed"
    title = "Spawn before an unavoidable death"
    hypothesis = ("H-S1: an agent that cannot escape should convert its energy into a child before contact (senior review / high-score doc s.2). "
                  "Measured as agents alive 25 s later and the predation penalty paid.")
    source = "high-score behaviours doc s.2 (spawn-before-death test)"
    primary = "alive_count"
    min_effect = 0.2
    seconds = 25.0

    def arms(self, cfg):
        return [{"name": "base", "cfg": cfg},
                {"name": "spawn_130", "kind": "doomed", "cfg": cfg, "kw": {"dist": 130}},
                {"name": "spawn_70", "kind": "doomed", "cfg": cfg, "kw": {"dist": 70}},
                {"name": "spawn_130_multi", "kind": "doomed", "cfg": cfg, "kw": {"dist": 130, "multi": True}}]

    def cells(self):
        return [{"E": E, "d": d} for E in (120, 250, 320) for d in (70, 110)]

    def spec(self, c, k):
        return {"seconds": self.seconds, "tree": None, "t_off": 900.0,
                "agents": _agents(1, c["E"], 8.0, "doomed", ring=0.0),
                "predators": [{"dist": c["d"] + 20, "aimed": True, "E": 200}]}

    def metrics(self, raw, c, spec):
        alive = len(raw["alive"])
        pen = sum(e for (_, aid, cause, e, _, _) in raw["dead"] if cause == "predator") / 100.0
        return {"alive_count": float(alive), "children_born": float(len(raw["born"])), "predation_penalty": pen,
                "parent_alive": float(any(a in raw["alive"] for a in raw["init"]))}, _tags_common(raw, spec)


class Juvenile(Exp):
    name = "juvenile"
    title = "Why do newborns starve? (child-rearing arena)"
    hypothesis = ("(arena has 5 other mature trees at 110-320 units, like the real map at t~300) "
                  "H-J1: child survival depends mostly on how much ripe food is at the birth site and how many mouths share it; "
                  "H-J2: a newborn that eats immediately (ignores ripeness for its first seconds) survives more often. "
                  "Motivation: 36% of all agents die before age 50 s, 26% by starvation.")
    source = "our lineage analysis (deaths.csv); demography R<1"
    primary = "child_alive"
    min_effect = 0.08
    seconds = 45.0

    def arms(self, cfg):
        return [{"name": "base", "cfg": cfg}, {"name": "newborn_eager", "cfg": cfg, "over": {"newborn_eager_s": 25}},
                {"name": "ripe_off", "cfg": cfg, "over": {"ripe_age": 0, "ripe_unknown": 0}}]

    def cells(self):
        return [{"F": F, "K": K} for F in (0, 3, 8) for K in (0, 2, 4)]

    def spec(self, c, k):
        fr = [{"dx": 32 * math.cos(2 * PI * i / max(1, c["F"]) + 0.4), "dy": 32 * math.sin(2 * PI * i / max(1, c["F"]) + 0.4), "E": 60.0}
              for i in range(c["F"])]
        ag = [{"tag": "parent", "dx": 4, "dy": 0, "E": 260.0, "speed": 10.0, "maxE": 350.0, "age": 55.0}] + \
             _agents(c["K"], 140.0, 10.0, "mate", ring=26)
        return {"seconds": self.seconds, "tree": {"age": 30}, "fruits": fr, "agents": ag, "t_off": 0.0, "predators": [],
                "extra_trees": [{"dist": (110, 320), "age": 30} for _ in range(5)],
                "force_spawn": [{"tag": "parent", "at": 0.3}]}

    def metrics(self, raw, c, spec):
        ch = raw["born"]
        child = ch[0] if ch else None
        if child is None:
            return {"child_alive": float("nan"), "child_energy": float("nan"), "parent_alive": float("nan"), "mates_dE": float("nan")}, ["no_child_born"]
        mates = raw["tags"].get("mate", [])
        dE = [raw["alive"][m][0] - raw["E_start"][m] for m in mates if m in raw["alive"]]
        tags = []
        if child not in raw["alive"]:
            tags.append("child_starved")
        elif raw["alive"][child][0] < 75:
            tags.append("child_lost_energy")
        else:
            tags.append("child_fed")
        par = raw["tags"]["parent"][0]
        if par not in raw["alive"]:
            tags.append("parent_died")
        return {"child_alive": 1.0 if child in raw["alive"] else 0.0, "child_energy": raw["alive"][child][0] if child in raw["alive"] else 0.0,
                "parent_alive": 1.0 if par in raw["alive"] else 0.0, "mates_dE": (sum(dE) / len(dE)) if dE else float("nan")}, tags


class Harvest(Exp):
    name = "harvest"
    title = "Food harvesting efficiency (net energy income per agent-second)"
    hypothesis = ("H-F1: colony income is limited by HOW fruit is harvested, not by supply: in the ablation runs 35-42% of all fruit rotted uneaten and eaten fruit "
                  "averaged 43-45 of a possible 60 energy. H-F2: stricter ripeness discipline raises energy per fruit. H-F3: patrolling between trees "
                  "(dead-reckoned tree map) raises net income versus waiting at one tree.")
    source = "ablation deep-dive (fruit economy); our energy-budget analysis"
    primary = "net_rate"
    min_effect = 0.3
    seconds = 150.0

    def arms(self, cfg):
        R = {"rest_frac": 0.7}
        return [{"name": "base", "cfg": cfg},
                {"name": "rest", "cfg": cfg, "over": R},
                {"name": "rest+ripe", "cfg": cfg, "over": {**R, "ripe_age": 20, "ripe_unknown": 14}},
                {"name": "rest+patrol", "cfg": cfg, "over": {**R, "patrol": 1, "patrol_wait": 6, "patrol_ban": 25}},
                {"name": "rest+patrol+ripe", "cfg": cfg, "over": {**R, "patrol": 1, "patrol_wait": 6, "patrol_ban": 25, "ripe_age": 20, "ripe_unknown": 14}},
                {"name": "rest+scan_off", "cfg": cfg, "over": {**R, "camp_scan": 0.0, "vig_k": 0.0}}]

    def cells(self):
        # few agents, many trees (the real colony: ~8 agents, 20-40 mature trees) plus denser cases
        return [{"K": K, "trees": T} for K in (1, 2, 4) for T in (6, 12)]

    def spec(self, c, k):
        rng = (60, 220) if c["trees"] == 6 else (90, 340)
        return {"seconds": self.seconds, "tree": {"age": 30}, "t_off": 0.0, "predators": [],
                "extra_trees": [{"dist": rng, "age": 30} for _ in range(c["trees"])], "fruits_per_tree": 5,
                "agents": _agents(c["K"], 200.0, 10.0, "a", maxE=400.0, ring=18.0)}

    def metrics(self, raw, c, spec):
        K = len(raw["init"])
        T = raw["seconds"]
        E_end = sum(raw["alive"][a][0] for a in raw["init"] if a in raw["alive"])
        E0 = sum(raw["E_start"][a] for a in raw["init"])
        gross = sum(raw["gained"].get(a, 0.0) for a in raw["init"])
        spent = sum(raw["spent"].get(a, 0.0) for a in raw["init"])
        fr = raw["fruit"]
        eaten = max(1, fr["eaten"])
        tags = []
        if any(a not in raw["alive"] for a in raw["init"]):
            tags.append("agent_starved")
        if fr["rot"] > fr["eaten"]:
            tags.append("more_rot_than_eaten")
        dead_t = {aid: t_ for (t_, aid, _c, _e, _x, _y) in raw["dead"]}
        alive_s = sum(min(T, dead_t.get(a, T)) for a in raw["init"])
        return {"net_rate": (gross - spent) / max(alive_s, 1.0), "gross_rate": gross / max(alive_s, 1.0), "spend_rate": spent / max(alive_s, 1.0),
                "net_income": (E_end - E0) / K / T, "survival_s": alive_s / K, "energy_per_fruit": fr["eaten_E"] / eaten, "rot_fraction": fr["rot"] / max(1, fr["rot"] + fr["eaten"]),
                "fruits_eaten": float(fr["eaten"]), "patrol_leaves": float(raw["ctrl_stats"].get("patrol_leave", 0)),
                "camp_share": raw["ctrl_stats"].get("camp", 0) / max(1, raw["ctrl_stats"].get("agent_ticks", 1)), "alive_frac": sum(1 for a in raw["init"] if a in raw["alive"]) / K}, tags


ARENA = {e.name: e for e in (Evade(), Cover(), Decoy(), Doomed(), Juvenile(), Harvest())}


# ---------------------------------------------------------------------------------------------- macro (full-run) experiments
class Macro:
    kind = "macro"
    primary = "score"
    min_effect = 60.0
    baseline = "base"
    name = ""
    title = ""
    hypothesis = ""
    source = ""


class Ablate(Macro):
    name = "ablate"
    title = "What limits survival? causal ablations of the FULL simulator"
    hypothesis = ("H-A1: extinction is predator-driven (senior review s.2; our first analysis). H-A2: it is food/economy-driven. "
                  "Ablations: predators off, tree floor (food does not collapse), both. If 'no_predators' still goes extinct, predators are NOT the limit.")
    source = "senior review s.4 (diagnostic simulators)"

    def arms(self, cfg, horizon):
        return [{"name": "base", "knobs": {}}, {"name": "no_predators", "knobs": {"no_predators": True}},
                {"name": "tree_floor45", "knobs": {"tree_floor": 45}},
                {"name": "no_pred+floor45", "knobs": {"no_predators": True, "tree_floor": 45}}]


class Traits(Macro):
    name = "traits"
    title = "Trait presets with breeding switched OFF (fixed traits, full runs)"
    hypothesis = ("H-T1: speed and low max_energy help survival (senior review C warns lower max_energy is a trade-off). Lineage data is confounded by "
                  "birth time; fixed-trait colonies in identical worlds remove the confound.")
    source = "senior review C; lineage stratification"

    def arms(self, cfg, horizon):
        F = lambda **t: {"traits": t, "fixed_traits": True}     # noqa: E731
        return [{"name": "base", "knobs": {}},
                {"name": "fix_s10_E500", "knobs": F(speed=10, sprint=20, maxE=500)},
                {"name": "fix_s16_E500", "knobs": F(speed=16, sprint=22, maxE=500)},
                {"name": "fix_s10_E250", "knobs": F(speed=10, sprint=20, maxE=250)},
                {"name": "fix_s16_E250", "knobs": F(speed=16, sprint=22, maxE=250)},
                {"name": "fix_s20_E200_h75", "knobs": F(speed=20, sprint=30, maxE=200, hearing=75)}]


class Econ(Macro):
    name = "econ"
    title = "Config comparison in FULL runs (optionally with predators off / food floor to isolate the economy)"
    hypothesis = ("H-F4: an improved food/energy economy (patrol, ripeness, senescence) extends survival. Run with --no-predators to expose the food-limited regime "
                  "(the ablation showed extinction at ~1568 s there) and without to see the combined effect.")
    source = "ablation deep-dive"
    cfgs = []
    baseline = "base"

    def arms(self, cfg, horizon):
        return [{"name": os.path.splitext(os.path.basename(c))[0], "cfg": c, "knobs": {}} for c in self.cfgs]


MACRO = {e.name: e for e in (Ablate(), Traits(), Econ())}


# ============================================================================================== running
def _arena_job(job):
    exp_name, arm_d, ci, k, seed = job
    exp = ARENA[exp_name]
    cell = exp.cells()[ci]
    spec = exp.spec(cell, k)
    raw = run_arena(spec, make_arm(arm_d), seed, trace=True)
    m, tags = exp.metrics(raw, cell, spec)
    return {"arm": arm_d["name"], "ci": ci, "k": k, "seed": seed, "m": m, "tags": tags, "raw": raw}


def _macro_job(job):
    from survivor.runner import run_episode
    arm, cfg, seed, horizon = job
    cfg = arm.get("cfg", cfg)
    pd = asdict(Params.load(cfg)) if cfg and os.path.exists(cfg) else asdict(Params())
    r = run_episode(seed, pd, horizon=horizon, keep_paths=False, sample_every=100, knobs=arm["knobs"])
    for k_ in ("heat", "paths", "snaps", "obstacles"):
        r.pop(k_, None)
    return {"arm": arm["name"], "seed": seed, "res": r}


def run_experiment(name, cfg, trials, seeds, workers, horizon, out, arms_filter=None, cfgs=None, extra_knobs=None):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if name in ARENA:
        exp = ARENA[name]
        arms = exp.arms(cfg)
        if arms_filter:
            arms = [a for a in arms if a["name"] in arms_filter or a["name"] == exp.baseline]
        jobs = [(name, a, ci, k, 1000 + 97 * ci + k) for a in arms for ci in range(len(exp.cells())) for k in range(trials)]
        print(f"{name}: {len(arms)} arms x {len(exp.cells())} cells x {trials} trials = {len(jobs)} trials", flush=True)
        if workers > 1:
            with Pool(workers) as pool:
                res = pool.map(_arena_job, jobs, chunksize=2)
        else:
            res = [_arena_job(j) for j in jobs]
        meta = {"name": name, "cfg": cfg, "trials": trials, "arms": [a["name"] for a in arms], "wall_s": time.time() - t0}
    else:
        exp = MACRO[name]
        if cfgs:
            exp.cfgs = list(cfgs)
            exp.baseline = os.path.splitext(os.path.basename(cfgs[0]))[0]
        arms = exp.arms(cfg, horizon)
        for a in arms:
            a["knobs"] = {**a.get("knobs", {}), **(extra_knobs or {})}
        if arms_filter:
            arms = [a for a in arms if a["name"] in arms_filter or a["name"] == exp.baseline]
        jobs = [(a, cfg, s, horizon) for a in arms for s in range(1, seeds + 1)]
        print(f"{name}: {len(arms)} arms x {seeds} seeds, horizon {horizon:.0f}s = {len(jobs)} full runs", flush=True)
        if workers > 1:
            with Pool(workers) as pool:
                res = pool.map(_macro_job, jobs, chunksize=1)
        else:
            res = [_macro_job(j) for j in jobs]
        meta = {"name": name, "cfg": cfg, "seeds": seeds, "horizon": horizon, "arms": [a["name"] for a in arms], "wall_s": time.time() - t0, "knobs": extra_knobs or {}, "baseline": getattr(exp, "baseline", "base")}
    with gzip.open(os.path.join(out, "results.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)
    report(out)


def load(out):
    with gzip.open(os.path.join(out, "results.jsonl.gz"), "rt") as f:
        lines = [json.loads(x) for x in f]
    return lines[0]["meta"], lines[1:]


# ============================================================================================== reporting
def _fmt(x, nd=3):
    return "nan" if x != x else f"{x:.{nd}f}"


def _plot_trial(rec, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw = rec["raw"]
    tr = raw["trace"]
    cx, cy = raw["center"]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for (x, y, w, h) in raw["rects"]:
        ax.add_patch(plt.Rectangle((x, y), w, h, color="0.75"))
    ax.scatter([cx], [cy], marker="^", c="darkgreen", s=90, label="tree")
    for aid, r in tr["agents"].items():
        for i in range(1, len(r["x"])):
            ax.plot(r["x"][i - 1:i + 1], r["y"][i - 1:i + 1], color=STATE_COL.get(r["s"][i], "k"), lw=1.6, alpha=0.9)
        if r["x"]:
            ax.scatter([r["x"][0]], [r["y"][0]], c="k", s=16, zorder=5)
            ax.annotate(f"a{aid}", (r["x"][0], r["y"][0]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for i, p in enumerate(tr["preds"]):
        ax.plot(p["x"], p["y"], color="red", lw=1, ls="--")
        if p["x"]:
            ax.scatter([p["x"][0]], [p["y"][0]], marker="X", c="red", s=60)
    for (t, aid, cause, e, x, y) in raw["dead"]:
        ax.scatter([x], [y], marker="x", c="black", s=110, zorder=6)
        ax.annotate(f"t={t}s", (x, y), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax.set_xlim(cx - 380, cx + 380)
    ax.set_ylim(cy + 380, cy - 380)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.text(0.01, 0.01, "path colour = policy state (red flee, magenta sprint, green camp, orange forage, blue seek)", transform=ax.transAxes, fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close()


def _narrative(rec, exp):
    raw = rec["raw"]
    lines = [f"- arm **{rec['arm']}**, cell {exp.cells()[rec['ci']]}, seed {rec['seed']}: " +
             ", ".join(f"{k}={_fmt(v, 2)}" for k, v in rec["m"].items()) + f"; tags: {', '.join(rec['tags']) or '-'}"]
    for t, s in raw["trace"]["events"][:22]:
        lines.append(f"    t={t:5.1f}s  {s}")
    return "\n".join(lines)


def report_arena(exp, meta, res, out):
    L = []
    w = L.append
    arms = meta["arms"]
    base = exp.baseline
    key = lambda r: (r["ci"], r["k"])                                            # noqa: E731
    by = {a: {key(r): r for r in res if r["arm"] == a} for a in arms}
    keys = sorted(by[base].keys())
    w(f"# Micro-experiment `{exp.name}`: {exp.title}\n")
    w(f"**Hypothesis.** {exp.hypothesis}\n")
    w(f"**Source.** {exp.source}. **Design.** {len(exp.cells())} scenario cells x {meta['trials']} paired trials per arm, {exp.seconds:.0f} s each, "
      f"config `{meta['cfg']}`. Primary metric `{exp.primary}` (higher is better). Value prices energy: 1 death = {DEATH_COST:.0f} energy.\n")
    # ------------------------------------------------ quantitative
    w("## 1. Quantitative result (paired against the baseline arm)")
    w("| arm | mean " + exp.primary + " | paired Δ vs base [95% CI] | median Δ | worst-quartile Δ | wins/losses | sign-test p | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    summ = {}
    for a in arms:
        x = [by[a][k]["m"][exp.primary] for k in keys if k in by[a]]
        b = [by[base][k]["m"][exp.primary] for k in keys if k in by[a]]
        ok = [(u, v) for u, v in zip(x, b) if u == u and v == v]
        if not ok:
            continue
        x, b = zip(*ok)
        s = paired_stats(list(x), list(b))
        v = "(baseline)" if a == base else verdict(s, exp.min_effect)
        summ[a] = {"mean": sum(x) / len(x), "stats": s, "verdict": v}
        w(f"| {a} | {_fmt(sum(x) / len(x))} | {_fmt(s['mean'])} [{_fmt(s['lo'])}, {_fmt(s['hi'])}] | {_fmt(s['median'])} | {_fmt(s['q25'])} | {s['wins']}/{s['losses']} | {s['p']:.3f} | **{v}** |")
    # secondary metrics
    sec = [m for m in next(iter(by[base].values()))["m"].keys() if m != exp.primary]
    w("\n**Secondary metrics (arm means)**\n")
    w("| arm | " + " | ".join(sec) + " |")
    w("|---|" + "---|" * len(sec))
    for a in arms:
        vals = []
        for m in sec:
            xs = [r["m"][m] for r in by[a].values() if r["m"][m] == r["m"][m]]
            vals.append(_fmt(sum(xs) / len(xs), 3) if xs else "nan")
        w(f"| {a} | " + " | ".join(vals) + " |")
    # ------------------------------------------------ by scenario cell
    w("\n## 2. Where does it matter? mean primary metric by scenario cell")
    cells = exp.cells()
    w("| cell | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for ci, c in enumerate(cells):
        row = []
        for a in arms:
            xs = [r["m"][exp.primary] for (cc, _), r in by[a].items() if cc == ci and r["m"][exp.primary] == r["m"][exp.primary]]
            row.append(_fmt(sum(xs) / len(xs), 2) if xs else "nan")
        w(f"| {c} | " + " | ".join(row) + " |")
    # factor-wise paired deltas for every non-baseline arm (interactions show up here)
    cand = [a for a in arms if a != base and a in summ]
    if cand:
        w("\n**Paired Δ vs base by factor level (where does each arm help or hurt?)**\n")
        w("| factor=level | " + " | ".join(cand) + " |")
        w("|---|" + "---|" * len(cand))
        for fac in cells[0].keys():
            for lv in sorted({str(c[fac]) for c in cells}):
                row = []
                for a in cand:
                    d = []
                    for (ci, k), r in by[a].items():
                        if str(cells[ci][fac]) == lv and (ci, k) in by[base]:
                            u, v = r["m"][exp.primary], by[base][(ci, k)]["m"][exp.primary]
                            if u == u and v == v:
                                d.append(u - v)
                    row.append(f"{sum(d) / len(d):+.3f}" if d else "")
                w(f"| {fac}={lv} | " + " | ".join(row) + " |")
    # ------------------------------------------------ failure modes
    w("\n## 3. Failure modes / behaviours observed (share of trials carrying each tag)")
    alltags = sorted({t for r in res for t in r["tags"]})
    w("| tag | " + " | ".join(arms) + " |")
    w("|---|" + "---|" * len(arms))
    for tg in alltags:
        row = []
        for a in arms:
            rs = list(by[a].values())
            row.append(f"{100 * sum(1 for r in rs if tg in r['tags']) / len(rs):.0f}%")
        w(f"| {tg} | " + " | ".join(row) + " |")
    # ------------------------------------------------ escapeability model (evade / cover)
    if exp.name in ("evade", "cover"):
        rows_X, rows_y = [], []
        for r in res:
            raw = r["raw"]
            c = cells[r["ci"]]
            for aid in raw["init"]:
                ff = raw["first_flee"].get(str(aid), raw["first_flee"].get(aid))
                d0 = ff[1] if ff else 300
                sp_, spr_ = c["traits"].replace("s", "", 1).split("/sp")
                rows_X.append([c["E"], float(sp_), float(spr_), c["npred"], 1 if c["aimed"] else 0, d0])
                rows_y.append(0 if aid in raw["alive"] else 1)
        if sum(rows_y) >= 8 and sum(rows_y) < len(rows_y) - 8:
            wts, mu, sd, auc, acc = fit_logistic(rows_X, rows_y)
            w("\n## 4. Escapeability model (logistic regression on all agent-trials, all arms pooled)")
            w(f"P(die) from [energy fraction, speed, sprint, n_predators, aimed, distance at first reaction]; AUC {auc:.2f}, accuracy {acc:.2f}, {len(rows_y)} agent-trials, {sum(rows_y)} deaths.")
            names = ["bias", "energy_frac", "speed", "sprint", "n_pred", "aimed", "react_dist"]
            w("| feature | standardized weight | reading |")
            w("|---|---|---|")
            for nm, wt in zip(names, wts):
                w(f"| {nm} | {wt:+.2f} | {'raises' if wt > 0 else 'lowers'} death probability |")
    # ------------------------------------------------ qualitative
    w("\n## 5. Qualitative: exemplar trials (worst / median / best relative to baseline) with automatic narratives")
    os.makedirs(os.path.join(out, "qual"), exist_ok=True)
    narr = []
    for a in arms:
        pairs = []
        for k in keys:
            if k in by[a]:
                r = by[a][k]
                v = r["m"][exp.primary]
                bv = by[base][k]["m"][exp.primary]
                if v == v and bv == bv:
                    pairs.append(((v - bv) if a != base else v, r))
        if not pairs:
            continue
        pairs.sort(key=lambda z: z[0])
        picks = [("worst", pairs[0][1]), ("median", pairs[len(pairs) // 2][1]), ("best", pairs[-1][1])]
        w(f"\n### arm `{a}`")
        for lab, r in picks:
            fn = f"{a}_{lab}_c{r['ci']}_k{r['k']}.png"
            try:
                _plot_trial(r, os.path.join(out, "qual", fn), f"{exp.name}/{a} {lab}: cell {cells[r['ci']]} seed {r['seed']}")
                w(f"- {lab}: `qual/{fn}`")
            except Exception as ex:                                                   # plotting must never kill the report
                w(f"- {lab}: (plot failed: {ex})")
            narr.append(f"### {a} / {lab}\n" + _narrative(r, exp))
    open(os.path.join(out, "narratives.md"), "w").write("\n\n".join(narr) + "\n")
    w("\nFull event narratives for the exemplar trials: `narratives.md`.")
    # ------------------------------------------------ decision
    w("\n## 6. Decision")
    for a in arms:
        if a == base or a not in summ:
            continue
        s = summ[a]
        w(f"- `{a}`: {s['verdict']} (mean Δ {s['stats']['mean']:+.3f}, CI [{s['stats']['lo']:+.3f}, {s['stats']['hi']:+.3f}], n={s['stats']['n']}). "
          + ("Candidate to port into the policy, then run the transfer check (full-run compare on >=12 paired seeds)." if s["verdict"] == "BETTER"
             else "Do not port." if s["verdict"] in ("WORSE", "worse (small)") else "Not established; add trials or refine the arm."))
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump({"experiment": exp.name, "meta": meta, "arms": {a: {"mean": v["mean"], "delta": v["stats"], "verdict": v["verdict"]} for a, v in summ.items()}},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("\n".join(L))


def _demography(r):
    Bl = [b for b in r["births_list"] if len(b) >= 13]
    if not Bl or not any(len(x) > 17 for x in r["deaths_list"]):
        return []
    dage = {x[17]: x[2] for x in r["deaths_list"] if len(x) > 17}
    dtime = {x[17]: x[0] for x in r["deaths_list"] if len(x) > 17}
    ok, allc = Counter(), Counter()
    for b in Bl:
        allc[b[12]] += 1
        if b[11] not in dage or dage[b[11]] >= 40:
            ok[b[12]] += 1
    born = {b[11]: b[0] for b in Bl}
    for i, a in dage.items():
        born.setdefault(i, max(0.0, dtime[i] - a))
    return [(bt, allc.get(i, 0), ok.get(i, 0), 1 if (i in dage and dage[i] < 40) else 0) for i, bt in born.items()]


def report_macro(exp, meta, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = []
    w = L.append
    arms = meta["arms"]
    base = meta.get("baseline") or exp.baseline
    by = {a: {r["seed"]: r["res"] for r in res if r["arm"] == a} for a in arms}
    seeds = sorted(by[base].keys())
    H = meta["horizon"]
    w(f"# Full-run experiment `{exp.name}`: {exp.title}\n")
    w(f"**Hypothesis.** {exp.hypothesis}\n")
    w(f"**Source.** {exp.source}. **Design.** {len(seeds)} paired seeds x {len(arms)} arms, horizon {H:.0f} s, config `{meta['cfg']}`, extra knobs {meta.get('knobs') or 'none'}. Primary metric: score.\n")
    w("## 1. Quantitative (paired by seed)")
    w("| arm | mean score | median | mean survival s | seeds reaching horizon | paired Δscore vs base [95% CI] | worst-quartile Δ | wins/losses | p | verdict |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    summ = {}
    for a in arms:
        sc = [by[a][s]["score"] for s in seeds if s in by[a]]
        bs = [by[base][s]["score"] for s in seeds if s in by[a]]
        stt = paired_stats(sc, bs)
        v = "(baseline)" if a == base else verdict(stt, exp.min_effect)
        surv = [by[a][s]["sim_time"] for s in seeds if s in by[a]]
        full = sum(1 for s in seeds if s in by[a] and by[a][s]["survived_full"])
        summ[a] = {"mean": sum(sc) / len(sc), "stats": stt, "verdict": v, "surv": sum(surv) / len(surv)}
        w(f"| {a} | {sum(sc) / len(sc):.0f} | {st.median(sc):.0f} | {sum(surv) / len(surv):.0f} | {full}/{len(seeds)} | {stt['mean']:+.0f} [{stt['lo']:+.0f}, {stt['hi']:+.0f}] | {stt['q25']:+.0f} | {stt['wins']}/{stt['losses']} | {stt['p']:.3f} | **{v}** |")
    w("\n## 2. Mechanism: causes of death, demography and traits per arm")
    w("| arm | deaths: predator / young-starved / old-age | births | R (0-300 s) | R (300-800 s) | reach age 40 s (born 300-800) | mean speed@t=600 | mean maxE@t=600 |")
    w("|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = list(by[a].values())
        d = {k: sum(r["deaths"][k] for r in rs) for k in ("predator", "starved_young", "starved_old")}
        tot = max(1, sum(d.values()))
        dem = [x for r in rs for x in _demography(r)]

        def Rw(lo, hi):
            G = [x for x in dem if lo <= x[0] < hi]
            return (sum(x[2] for x in G) / len(G)) if len(G) >= 5 else float("nan")
        G2 = [x for x in dem if 300 <= x[0] < 800]
        reach = (100 * (1 - sum(x[3] for x in G2) / len(G2))) if len(G2) >= 5 else float("nan")

        def at600(key):
            v = []
            for r in rs:
                ts = r["ts"]
                if ts["t"] and ts["t"][-1] >= 600:
                    i = min(range(len(ts["t"])), key=lambda j: abs(ts["t"][j] - 600))
                    v.append(ts[key][i])
            return sum(v) / len(v) if v else float("nan")
        w(f"| {a} | {100 * d['predator'] / tot:.0f}% / {100 * d['starved_young'] / tot:.0f}% / {100 * d['starved_old'] / tot:.0f}% | {sum(r['births'] for r in rs) / len(rs):.0f} | "
          f"{_fmt(Rw(0, 300), 2)} | {_fmt(Rw(300, 800), 2)} | {_fmt(reach, 0)}% | {_fmt(at600('mean_speed'), 1)} | {_fmt(at600('mean_maxE'), 0)} |")
    w("- R = children that reach 40 s per newborn of the cohort; R < 1 means the cohort does not replace itself.")
    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    for a in arms:
        ts_ = sorted(by[a][s]["sim_time"] for s in seeds if s in by[a])
        xs = [0] + ts_ + [H]
        n = len(ts_)
        ys = [1.0] + [1 - (i + 1) / n for i in range(n)] + [1 - 1.0 if all(t < H - 1 for t in ts_) else sum(1 for t in ts_ if t >= H - 1) / n]
        ax[0].step(xs, ys, where="post", label=a)
        # mean population
        grid = list(range(100, int(H) + 1, 100))
        pm = []
        for g in grid:
            v = []
            for s in seeds:
                ts = by[a][s]["ts"]
                if ts["t"] and ts["t"][-1] >= g:
                    v.append(ts["n"][min(range(len(ts["t"])), key=lambda j: abs(ts["t"][j] - g))])
                else:
                    v.append(0)
            pm.append(sum(v) / len(v))
        ax[1].plot(grid, pm, label=a)
        em = []
        for g in grid:
            v = []
            for s in seeds:
                ts = by[a][s]["ts"]
                if ts["t"] and ts["t"][-1] >= g:
                    v.append(ts["mean_E"][min(range(len(ts["t"])), key=lambda j: abs(ts["t"][j] - g))])
            em.append(sum(v) / len(v) if v else float("nan"))
        ax[2].plot(grid, em, label=a)
    for a_, t_ in zip(ax, ["fraction of seeds still alive", "mean population (extinct = 0)", "mean energy of survivors"]):
        a_.set_title(t_)
        a_.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "curves.png"), dpi=100)
    plt.close()
    w("\nPlot: `curves.png` (survival curve, population, energy).")
    # ---- auto-interpretation
    w("\n## 3. Reading")
    if exp.name == "ablate":
        b = summ[base]["surv"]
        g = {k: summ[k]["surv"] for k in ("no_predators", "tree_floor45", "no_pred+floor45") if k in summ}
        w("- mean survival: base %.0f s; " % b + "; ".join(f"{k} {v:.0f} s ({v - b:+.0f})" for k, v in g.items()) + ".")
        w("- If 'no_predators' barely helps, extinction is NOT predator-limited: the demography/economy (R < 1) is the binding constraint. "
          "If 'tree_floor45' barely helps, food is not the limit either; if only the combination reaches the horizon, both constraints must be lifted together.")
        if len(g) == 3:
            gp, gf, gb = g["no_predators"] - b, g["tree_floor45"] - b, g["no_pred+floor45"] - b
            w(f"- interaction: gain(both) {gb:+.0f} vs sum of single gains {gp + gf:+.0f} -> "
              f"{'synergy (constraints compound)' if gb > gp + gf + 50 else 'roughly additive' if abs(gb - gp - gf) <= 50 else 'sub-additive (one constraint masks the other)'}.")
    else:
        w("- Compare arms on BOTH survival and the mechanism table: a trait that lowers predator deaths but raises young starvation (energy cost of speed) shows as "
          "'predator' share down and 'young-starved' share up with no net score change.")
    w("- These are diagnostic runs (the local simulator is modified); they identify bottlenecks, they are not competition policies.")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump({"experiment": exp.name, "meta": meta, "arms": {a: {"mean": v["mean"], "surv": v["surv"], "delta": v["stats"], "verdict": v["verdict"]} for a, v in summ.items()}},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("\n".join(L))


def report(out):
    meta, res = load(out)
    name = meta["name"]
    if name in ARENA:
        report_arena(ARENA[name], meta, res, out)
    else:
        report_macro(MACRO[name], meta, res, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--config", default=os.path.join(ROOT, "configs", "v5a_ripe_boot.json"))
    r.add_argument("--trials", type=int, default=6, help="paired trials per scenario cell (arena experiments)")
    r.add_argument("--seeds", type=int, default=12, help="paired seeds (full-run experiments)")
    r.add_argument("--horizon", type=float, default=3000.0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--arms", nargs="*", default=None)
    r.add_argument("--configs", nargs="*", default=None, help="econ: configs to compare (first = baseline)")
    r.add_argument("--no-predators", action="store_true", help="full-run experiments: switch predators off")
    r.add_argument("--tree-floor", type=int, default=0, help="full-run experiments: keep at least N trees alive")
    r.add_argument("--out", default=None)
    p = sub.add_parser("report")
    p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "list":
        for e in list(ARENA.values()) + list(MACRO.values()):
            print(f"{e.name:<9} [{e.kind:<5}] {e.title}\n           {e.hypothesis}\n")
        return
    if a.cmd == "report":
        report(a.out)
        return
    out = a.out or os.path.join(ROOT, "runs", "micro", f"{a.name}_{time.strftime('%m%d_%H%M%S')}")
    kn = {}
    if a.no_predators:
        kn["no_predators"] = True
    if a.tree_floor:
        kn["tree_floor"] = a.tree_floor
    run_experiment(a.name, a.config, a.trials, a.seeds, a.workers, a.horizon, out, a.arms, cfgs=a.configs, extra_knobs=kn)


if __name__ == "__main__":
    main()
