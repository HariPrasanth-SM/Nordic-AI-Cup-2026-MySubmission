"""
survivor.policy - mechanics-aware hive-mind controller for the Survival Simulator.

Pure python (no numpy needed at inference), <~0.1 ms per agent. Every behavioural threshold
lives in `Params`, so the optimiser (optimize.py) and the config files drive everything.

Facts about the simulator this controller is built on (all verified in src/):
  * move_direction is RELATIVE to the current heading (the README says absolute - it is wrong)
  * action order inside a tick is  move (pre-turn heading) -> turn -> spawn
  * score ~= survival seconds (+1/s); fruit only +0.02..0.06; being eaten costs energy/100
  * a predator only charges an agent when it is BEHIND it (|bearing| > 90 deg) or closer than 90 units
    -> facing the predator while backing away suppresses long-range charges
  * predators see with a 60-degree cone (rel_dir tells us whether the predator is looking at us)
  * walking costs 0.05/unit, sprinting 0.5/unit, standing still only 0.1/tick, turning <=0.5/tick
  * no agent lives past ~150 s (aging drain) -> reproduction is lifespan renewal, and a dying agent
    should turn its remaining energy into a child ("death-bed spawn")
  * trees (=food) collapse over time (spawn chance halves every 300 s) -> grow early, contract late
"""
import json
import math
import os
import random
from dataclasses import dataclass, fields, asdict

from survivor.colony import ColonyManager
from survivor import escape_model as _esc
from survivor import value_model as _val
from survivor import pursuit as _pur
import numpy as np

PI = math.pi
TWO_PI = 2 * PI
BIOME_PEN = {"river": 0.3, "swamp": 0.5, "desert": 0.8}
BAD_BIOMES = ("river", "desert")


def wrap(a):
    return (a + PI) % TWO_PI - PI


@dataclass
class Params:
    # ---- predator response ------------------------------------------------------------------
    pred_react_dist: float = 240.0    # ignore predators farther than this
    pred_sprint_dist: float = 100.0   # sprint when a noticing predator is closer than this
    pred_notice_margin: float = 0.35  # rad added to the predator's 30deg half-cone -> "it sees me"
    pred_side_w: float = 0.6          # 0 = flee straight away, 1 = sidestep out of its path (unnoticed predators)
    pred_cpa: float = 110.0           # unnoticing predator is a threat if its straight path passes closer than this
    sprint_below: float = 18.0        # only sprint when the walking speed is below this (sprint speed trait is ~20)
    shared_alarm: float = 0.0         # >0.5: agents share predator sightings with neighbours (frame transform, no global map)
    alarm_radius: float = 120.0
    flee_cap: float = 17.0            # never flee faster than this (predators walk 11 / sprint 15; extra speed just burns energy)
    pred_face: float = 1.0            # >0.5: keep facing the predator while fleeing (blocks long-range charges)
    flee_hold: float = 10.0           # ticks to keep fleeing after the last sighting
    sprint_min_frac: float = 0.22     # never start a sprint below this energy fraction (env caps at 20%)
    # ---- foraging ---------------------------------------------------------------------------
    fruit_max_dist: float = 320.0
    ripe_age: float = 15.0            # fruit energy grows 20->60 in its first 20 s: wait for it (0 = eat at once)
    ripe_unknown: float = 8.0         # unknown-age fruit (first seen while moving) must be OBSERVED this long before eating
    urgent_abs: float = 45.0          # absolute energy below which any fruit is eaten at once (waiting 20 s costs ~20 energy)
    urgent_frac: float = 0.30         # (legacy, unused)
    spawn_quiet_s: float = 0.0        # >0: normal/death-bed births wait until no predator was seen anywhere for this long (newborns cannot sprint)
    doom_spawn: float = 0.0           # >0.5: an agent a noticing predator is about to catch converts its energy into a child first
    doom_dist: float = 120.0
    doom_min_e: float = 110.0
    doom_frac: float = 0.5            # only when energy is below this fraction of max (a strong agent can still run)
    colony_mgr: float = 0.0           # >0.5: colony-level, age-balanced, staggered births replace per-agent spawn rules (survivor.colony)
    young_age_s: float = 30.0         # 'young' cohort = agents younger than this
    young_frac: float = 0.30          # births wait while the young cohort exceeds this share of the target population
    birth_gap_s: float = 0.0          # minimum seconds between births once t >= gap_after_s (0 = off)
    gap_after_s: float = 300.0
    reserve_margin: float = 30.0      # a parent must keep >= 0.2*max_energy + this after paying the 100-energy birth cost
    max_births_tick: float = 1.0
    n_lifeboat: float = 0.0           # isolated, energetic agents excluded from breeding (colony redundancy)
    lifeboat_hold_s: float = 20.0
    w_survivor: float = 0.0           # parent ranking bonus per past predator escape
    age_bonus: float = 0.5            # parent ranking bonus per second of age above 40 (closest to dying breeds first)
    use_escape_model: float = 0.0     # >0.5: use the learned P(death) (survivor.escape_model) for the terminal spawn
    doom_p: float = 0.6
    emap: float = 0.0                 # remember walls/edges (dead reckoned) ...
    flee_ray: float = 0.0             # ... and choose the flee direction by ray-casting over them (obstacle-aware escape)
    emap_age: float = 60.0
    late_t: float = 1500.0            # survival mode: after this time agents rest at a lower energy fraction (no costly exploring)
    late_rest_frac: float = 0.45
    urgent_floor: float = 0.0         # >0.5: below the sprint floor (0.2*max_energy) + urgent_margin, eat any fruit at once (restore escape readiness)
    urgent_margin: float = 40.0
    pred_face_dist: float = 0.0       # face the predator only while it is farther than this (inside 90 it charges regardless of facing: then look where you run)
    late_patience: float = 0.0        # >0: camp patience after late_t (fortress: wait at a tree instead of exploring)
    kite: float = 0.0                 # >0.5: do not sprint from a predator that is measurably tired/resting if we can outwalk it
    tired_v: float = 12.5             # predator speed estimate (units/tick) below which it is tired (walk cap 11) - fresh predators move 15
    vm_gate: float = 0.0              # >0.5: a birth may not lower the parent's modelled 5-s survival by more than vm_drop, nor below vm_min
    vm_drop: float = 0.01
    vm_min: float = 0.90
    mpc_flee: float = 0.0             # >0.5: noticed predators inside mpc_range are handled by the model-based planner (survivor.pursuit)
    mpc_range: float = 200.0
    mpc_h: float = 24.0               # planning horizon in ticks (0.1 s each)
    mpc_terminal: float = 0.0         # >0.5: when NO planned escape survives the horizon the agent is 'doomed': it converts energy into children first
    rest_frac: float = 1.01           # energy fraction above which an agent just rests (no leaving trees, no exploring); 1.01 = off
    senescence: float = 0.0           # >0.5: agents that detect aging drain stop moving/eating and turn their energy into children
    sen_fruit_r: float = 30.0         # a senescent agent still grabs a ripe fruit this close if it lets it spawn once more
    sen_extra: float = 6.0            # senescent spawns may exceed the population cap by this many agents
    patrol: float = 0.0               # >0.5: keep a dead-reckoned tree map and walk to the tree with the most accumulated fruit
    patrol_wait: float = 6.0          # seconds without gain at a tree before patrolling on
    patrol_ban: float = 25.0          # seconds before a tree we just left becomes eligible again (fruit needs ~25 s to ripen)
    patrol_max: float = 350.0         # never patrol farther than this
    tmap_age: float = 110.0           # forget trees not seen for this long (trees live ~57 s after maturing)
    newborn_eager_s: float = 0.0      # agents younger than this eat any fruit at once (newborns cannot afford to wait)
    eat_full_frac: float = 0.95       # above this energy fraction only grab fruit that is very close
    eat_full_dist: float = 35.0
    camp_radius: float = 12.0         # stand still (and scan) this close to a tree
    camp_patience: float = 30.0       # seconds camping without eating before the tree is written off
    camp_scan: float = 0.05           # rad/tick scan rotation while camping (CMA switched it off)
    vig_k: float = 0.1                # extra scan rad/tick per 1000 s (predators multiply over time)
    patience_burn: float = 2.0        # energy/s assumed while camping: patience shrinks as energy nears the sprint floor
    crowd_r: float = 70.0
    crowd_max: float = 2.0            # a tree is 'full' when this many lower-id agents are already at it
    tree_ban_r: float = 70.0
    tree_ban_s: float = 60.0
    # ---- exploration ------------------------------------------------------------------------
    explore_spin: float = 0.8         # fraction of the vision cone rotated per tick while travelling (0 = look ahead only)
    explore_wander: float = 0.06      # rad/tick random walk of the travel direction
    wall_margin: float = 50.0
    wall_gain: float = 2.2
    spread_w: float = 0.8             # repulsion from neighbours while exploring
    # ---- reproduction / population ----------------------------------------------------------
    spawn_food_min: float = 1.0       # normal spawns need this many fruits in view (child must find food at once)
    spawn_food_r: float = 90.0
    emergency_min_e: float = 160.0    # emergency spawn (colony nearly extinct) still leaves the parent alive
    spawn_thr: float = 180.0          # normal spawn when energy above this (early game) ...
    spawn_thr_late: float = 320.0     # ... rising linearly to this at t = spawn_ramp (births get expensive in the famine)
    spawn_ramp: float = 450.0
    dbed_age: float = 52.0            # "death-bed": spawn once age exceeds this (needs dbed_min_energy)
    dbed_min_energy: float = 120.0
    dbed_min_late: float = 200.0
    dbed_extra: float = 6.0           # death-bed spawns may exceed the cap by this many agents
    cap_early: float = 40.0
    cap_late: float = 7.0
    cap_tau: float = 350.0
    n_min: float = 3.0                # emergency: at/below this population spawn at any energy > 100
    select_q: float = 0.55            # only agents above this trait-score quantile reproduce normally
    w_energy: float = -0.5            # NEGATIVE: prefer a LOW max_energy (sprint floor = max_energy/5)
    w_vision: float = 0.25
    w_hearing: float = 0.4
    w_cone: float = 0.2
    w_speed: float = 1.0              # speed 10->20 cuts predator kills ~10x for weak agents (see predtest)
    w_sprint: float = 0.3

    @staticmethod
    def load(path_or_dict=None):
        p = Params()
        if path_or_dict is None:
            return p
        d = path_or_dict
        if isinstance(d, str):
            with open(d) as f:
                d = json.load(f)
        d = d.get("params", d)
        names = {f.name for f in fields(Params)}
        for k, v in d.items():
            if k in names:
                setattr(p, k, float(v))
        return p

    def save(self, path, **meta):
        with open(path, "w") as f:
            json.dump({"params": asdict(self), **meta}, f, indent=2)


# name: (low, high, log_scale) - bounds used by the optimiser
SPACE = {
    "pred_react_dist": (150, 300, False), "pred_sprint_dist": (50, 140, False),
    "pred_notice_margin": (0.0, 0.8, False), "pred_side_w": (0.0, 1.0, False),
    "pred_cpa": (40, 180, False), "pred_face": (0.0, 1.0, False), "flee_hold": (0, 40, False),
    "sprint_min_frac": (0.2, 0.5, False),
    "fruit_max_dist": (120, 400, False), "ripe_age": (0, 30, False), "ripe_unknown": (0, 25, False), "urgent_abs": (10, 110, False),
    "flee_cap": (12, 24, False), "mpc_flee": (0, 1, False), "mpc_range": (100, 300, False), "mpc_h": (8, 30, False), "mpc_terminal": (0, 1, False), "late_patience": (0, 120, False), "kite": (0, 1, False), "tired_v": (11, 14.5, False), "vm_gate": (0, 1, False), "vm_drop": (0, 0.05, False), "vm_min": (0.7, 0.99, False), "pred_face_dist": (0, 300, False), "urgent_floor": (0, 1, False), "urgent_margin": (0, 120, False), "colony_mgr": (0, 1, False), "young_age_s": (15, 60, False), "young_frac": (0.1, 0.8, False), "birth_gap_s": (0, 30, False), "gap_after_s": (0, 1500, False), "reserve_margin": (0, 120, False), "max_births_tick": (1, 3, False), "n_lifeboat": (0, 4, False), "lifeboat_hold_s": (5, 60, False), "w_survivor": (0, 3, False), "age_bonus": (0, 2, False), "use_escape_model": (0, 1, False), "doom_p": (0.3, 0.95, False), "emap": (0, 1, False), "flee_ray": (0, 1, False), "emap_age": (20, 200, False), "late_t": (600, 3000, False), "late_rest_frac": (0.2, 1.0, False), "spawn_quiet_s": (0, 60, False), "doom_spawn": (0, 1, False), "doom_dist": (50, 200, False), "doom_min_e": (102, 250, False), "doom_frac": (0.2, 1.0, False), "eat_full_frac": (0.6, 1.0, False), "tree_ban_s": (10, 150, False), "rest_frac": (0.3, 1.01, False), "senescence": (0, 1, False), "sen_fruit_r": (10, 80, False), "sen_extra": (0, 10, False), "patrol": (0, 1, False), "patrol_wait": (2, 25, False), "patrol_ban": (10, 60, False), "patrol_max": (100, 500, False), "tmap_age": (60, 200, False), "shared_alarm": (0, 1, False), "alarm_radius": (40, 200, False), "newborn_eager_s": (0, 60, False), "sprint_below": (10, 30, False), "spawn_food_min": (0, 4, False), "emergency_min_e": (101, 250, False),
    "eat_full_dist": (5, 90, False),
    "camp_radius": (8, 45, False), "camp_patience": (10, 70, False), "camp_scan": (0.0, 1.0, False), "vig_k": (0.0, 1.0, False), "patience_burn": (0.5, 5.0, False),
    "crowd_max": (1, 5, False), "tree_ban_s": (20, 150, False),
    "explore_spin": (0.0, 1.0, False), "explore_wander": (0.0, 0.25, False),
    "wall_margin": (25, 90, False), "spread_w": (0.0, 2.0, False),
    "spawn_thr": (110, 380, False), "spawn_thr_late": (150, 450, False), "spawn_ramp": (100, 1200, False),
    "dbed_min_late": (105, 320, False), "dbed_age": (35, 78, False), "dbed_min_energy": (102, 220, False),
    "dbed_extra": (0, 8, False), "cap_early": (8, 45, False), "cap_late": (3, 16, False),
    "cap_tau": (250, 2000, False), "n_min": (2, 6, False), "select_q": (0.0, 0.7, False),
    "w_energy": (-1.0, 1.0, False), "w_vision": (0.0, 1.0, False), "w_hearing": (0.0, 1.0, False),
    "w_cone": (0.0, 1.0, False), "w_speed": (0.0, 1.0, False), "w_sprint": (0.0, 1.0, False),
}

# default subset that the optimiser is allowed to move (the others keep the config value)
DEFAULT_FREE = ["ripe_age", "ripe_unknown", "urgent_abs", "flee_cap", "pred_sprint_dist", "pred_side_w",
                "camp_patience", "patience_burn", "camp_scan", "explore_spin", "spawn_thr", "spawn_thr_late",
                "spawn_ramp", "spawn_food_min", "dbed_min_energy", "cap_early", "cap_late", "cap_tau", "select_q",
                "w_energy", "w_speed"]


def vec_to_params(vec01, base, free):
    """vec in [0,1]^k -> Params (only `free` names change)"""
    p = Params(**asdict(base))
    for x, name in zip(vec01, free):
        lo, hi, _ = SPACE[name]
        x = min(1.0, max(0.0, float(x)))
        setattr(p, name, lo + x * (hi - lo))
    return p


def params_to_vec(p, free):
    out = []
    for name in free:
        lo, hi, _ = SPACE[name]
        out.append((getattr(p, name) - lo) / (hi - lo))
    return out


def _new_mem(rng, t):
    return {
        "x": 0.0, "y": 0.0, "h": 0.0,               # dead-reckoned pose in the agent's own frame
        "trv": rng.uniform(-PI, PI),                # travel direction relative to heading (explore)
        "side": rng.choice((-1, 1)),
        "camp_t0": None, "gain_t": t, "E_prev": None,
        "flee_until": -1.0, "flee_dir": 0.0,
        "stuck": 0, "d_prev": None, "unstick_until": -1.0, "unstick_dir": 0.0,
        "bans": [], "born": t, "fl": [], "still": 0, "tmap": [], "goto": None, "emap": {}, "escapes": 0, "nb": 0, "p_die": None, "sel_ok": True, "food_near": 99, "threat_now": False, "sen": None, "last_cost": 0.1, "last_spawn": False,
    }


def transform_predators(step, radius=120.0):
    """HIVE-MIND SHARED ALARM. If agent B can perceive neighbour A, and A perceives a predator that B does not, express that predator
    in B's own frame (distance, angle, rel_dir) using the mutual observation (A's range/bearing seen from B, plus the rel_dir heading cue)
    and inject it into B's observations. No global map or absolute coordinates needed.
    Frame algebra: A.heading in B frame = aAB + pi - relAB ; predator in B = pos(A in B) + R(hA_B) * predator in A."""
    by_id = {o["agent_id"]: o for o in step["agent_status"]}
    injected = 0
    new_status = []
    for B in step["agent_status"]:
        own = [x for x in B["observations"] if x["type"] == "Predator"]
        extra = []
        for nb in B["observations"]:
            if nb["type"] != "Agent" or nb["distance"] > radius:
                continue
            A = by_id.get(nb.get("id"))
            if A is None:
                continue
            aAB, dAB, rAB = nb["angle"], nb["distance"], nb["rel_dir"]
            hA = aAB + PI - rAB
            ax, ay = dAB * math.cos(aAB), dAB * math.sin(aAB)
            c, s_ = math.cos(hA), math.sin(hA)
            for pa in A["observations"]:
                if pa["type"] != "Predator":
                    continue
                px, py = pa["distance"] * math.cos(pa["angle"]), pa["distance"] * math.sin(pa["angle"])
                bx, by_ = ax + c * px - s_ * py, ay + s_ * px + c * py
                d = math.hypot(bx, by_)
                if any(abs(o["distance"] - d) < 40 and abs(wrap(o["angle"] - math.atan2(by_, bx))) < 0.5 for o in own + extra):
                    continue                                              # B already sees (or was already told about) this predator
                ph = wrap(pa["angle"] + PI - pa["rel_dir"]) + hA          # predator heading in B frame
                rel = wrap(math.atan2(-by_, -bx) - ph)
                extra.append({"type": "Predator", "distance": d, "angle": math.atan2(by_, bx), "rel_dir": rel, "shared": True})
                injected += 1
        if extra:
            B = dict(B)
            B["observations"] = list(B["observations"]) + extra
        new_status.append(B)
    st2 = dict(step)
    st2["agent_status"] = new_status
    return st2, injected


class Controller:
    """Stateful controller. `act(step)` takes the /predict payload (dict) and returns the action dicts."""

    def __init__(self, params=None, seed=0, escape_model=None):
        self.p = params if isinstance(params, Params) else Params.load(params)
        self.escape = escape_model
        self.value = None
        if self.p.vm_gate > 0.5:
            self.value = _val.load(os.environ.get("SURVIVOR_VALUE_MODEL") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "value_model.json"))
        if self.escape is None and self.p.use_escape_model > 0.5:
            self.escape = _esc.load(os.environ.get("SURVIVOR_ESCAPE_MODEL") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "escape_model.json"))
        self.seed = seed
        self.episode = -1
        self.t_last = -1.0
        self.mem = {}
        self.stats = {}
        self.reset()

    # ------------------------------------------------------------------ bookkeeping
    def reset(self):
        self.episode += 1
        self.t_last = -1.0
        self.mem = {}
        self.stats = {k: 0 for k in (
            "ticks", "agent_ticks", "flee", "flee_sprint", "forage", "camp", "seek_tree", "explore",
            "unstick", "spawn_normal", "spawn_dbed", "spawn_emergency", "bans", "threat_sightings")}
        self.pop_trace = []
        self.last_state = {}
        self.last_pred_t = -1e9
        self._cur = "explore"
        self.agg = {}          # state -> {n, dur, dE, died}
        self.camp_b = {}       # biome -> [episodes, seconds, energy gained]
        self.flee_tab = {}     # (energy bucket, speed bucket) -> [episodes, died]
        self.events = []       # ring of recent notable events (deaths)
        self.flee_rows = []    # logged flee episodes (escape-model training data)
        self.val_rows = []     # meta-reward windows: features at window start + survived 5 s?
        self.val_pend = {}
        self.val_t = -1e9
        self.colony = ColonyManager(self)

    _STATES = ("flee", "flee_sprint", "forage", "camp", "seek_tree", "explore", "unstick", "senescent")

    def _bump(self, k, v=1):
        self.stats[k] = self.stats.get(k, 0) + v
        if k in self._STATES:
            self._cur = k

    def pop_cap(self, t):
        p = self.p
        return p.cap_late + (p.cap_early - p.cap_late) * math.exp(-t / max(p.cap_tau, 1.0))

    def trait_score(self, a):
        p = self.p
        w = (p.w_energy, p.w_vision, p.w_hearing, p.w_cone, p.w_speed, p.w_sprint)
        # movement is capped at sprint_speed first, so the usable walking speed is min(speed, sprint_speed)
        v = (a["max_energy"] / 1000.0, a["vision_range"] / 400.0, a["hearing_radius"] / 100.0,
             a["vision_angle"] / (PI / 2), min(a["speed"], a["sprint_speed"]) / 20.0, a["sprint_speed"] / 40.0)
        s = sum(abs(x) for x in w) or 1.0
        return sum(x * y for x, y in zip(w, v)) / s

    # ------------------------------------------------------------------ main entry
    def act(self, step):
        if step.get("game_status", "ok") != "ok":
            return []
        t = float(step.get("sim_time", 0.0))
        agents = step.get("agent_status") or []
        if t < self.t_last - 1e-6 or (t <= 0.05 and self.t_last > 1.0):
            self.reset()                              # new simulation (evaluation runs three in a row)
        self.t_last = t
        if not agents:
            return []
        for a_ in agents:                                # colony-wide "a predator was seen" clock (used to time births)
            if any(o["type"] == "Predator" for o in a_["observations"]):
                self.last_pred_t = t
                break
        if self.p.shared_alarm > 0.5 and len(agents) > 1:
            step, _n = transform_predators(step, self.p.alarm_radius)
            agents = step["agent_status"]
        alive = {a["agent_id"] for a in agents}
        for k in [k for k in self.mem if k not in alive]:
            mm = self.mem.pop(k)
            self._close_ep(mm, t, mm.get("E_prev") or 0.0, "died")
            if len(self.events) < 400:
                ep = mm.get("ep")
                self.events.append([round(t, 1), "gone", self.last_state.get(k, "?"), round(mm.get("E_prev") or 0),
                                    round(mm.get("age", 0)), mm.get("biome", "?"), round(mm.get("ME", 0)),
                                    None if mm.get("thr_last") is None else round(t - mm["thr_last"], 1)])
            self.last_state.pop(k, None)

        n = len(agents)
        cap = self.pop_cap(t)
        scores = [self.trait_score(a) for a in agents]
        # strict rank-based breeding: only the top (1 - select_q) fraction by trait score may reproduce normally,
        # so a rare fast mutant takes over the gene pool within a few generations (ties broken by lowest id)
        if n > 4:
            k = max(2, int(round((1.0 - self.p.select_q) * n)))
            order = sorted(range(n), key=lambda i: (-scores[i], agents[i]["agent_id"]))
            elite = set(order[:k])
        else:
            elite = set(range(n))
        n_eff = n
        out = []
        self._bump("ticks")
        for i, a in enumerate(agents):
            act = self._act_one(a, t, n_eff, cap, i in elite)
            if act["spawn_agent"]:
                n_eff += 1
            out.append(act)
        self.colony.after(step, agents, out, t)
        self._val_tick(agents, t)
        self._bump("agent_ticks", n)
        if int(t * 10) % 100 == 0:
            self.pop_trace.append((round(t, 1), n))
        return out

    # ------------------------------------------------------------------ helpers
    def _steer(self, phi, edges, mates, m, margin):
        """bend travel direction `phi` (relative to heading) away from walls (visible edges) and neighbours"""
        p = self.p
        ux, uy = math.cos(phi), math.sin(phi)
        rx = ry = 0.0
        for e in edges:
            (x1, y1), (x2, y2) = e
            vx, vy = x2 - x1, y2 - y1
            vv = vx * vx + vy * vy
            if vv <= 0:
                continue
            tt = -(x1 * vx + y1 * vy) / vv
            tt = 0.0 if tt < 0 else (1.0 if tt > 1 else tt)
            qx, qy = x1 + tt * vx, y1 + tt * vy
            dq = math.hypot(qx, qy)
            if dq < margin and dq > 1e-6 and (qx * ux + qy * uy) > -0.3 * dq:
                k = (margin - dq) / margin
                rx -= qx / dq * k
                ry -= qy / dq * k
        gain = p.wall_gain
        if mates and p.spread_w > 0:
            for (d, ang, _mid) in mates:
                if d < 90 and d > 1e-6:
                    k = p.spread_w * (90 - d) / 90 / gain
                    rx -= math.cos(ang) * k
                    ry -= math.sin(ang) * k
        if rx == 0.0 and ry == 0.0:
            return phi
        vx, vy = ux + gain * rx, uy + gain * ry
        if math.hypot(vx, vy) < 0.2:
            return wrap(phi + m["side"] * PI / 2)
        return math.atan2(vy, vx)

    def _flee_dir(self, thr, m):
        """direction (relative to heading) to run from threat `thr` = (d, ang, rel, noticed)"""
        d, ang, rel, noticed = thr
        away = wrap(ang + PI)
        side_w = self.p.pred_side_w * (1.0 if not noticed else 0.25)
        if d < 95:
            side_w = 0.0
        if side_w <= 0:
            return away
        ph = wrap(ang + PI - rel)                     # predator heading in my frame
        wx, wy = math.cos(ph), math.sin(ph)
        px, py = d * math.cos(ang), d * math.sin(ang)
        cross = wx * py - wy * px
        perp = ph + (PI / 2 if cross >= 0 else -PI / 2)
        vx = (1 - side_w) * math.cos(away) + side_w * math.cos(perp)
        vy = (1 - side_w) * math.sin(away) + side_w * math.sin(perp)
        return math.atan2(vy, vx)

    def _want_spawn(self, a, m, t, n_eff, cap, sel_ok, threatened, food_near=99):
        p = self.p
        if p.colony_mgr > 0.5:
            return False                                    # births are decided at colony level (survivor.colony)
        E, age = a["energy"], a["age"]
        sen = p.senescence > 0.5 and m.get("sen") is not None
        doom = (p.doom_spawn > 0.5 and threatened and m.get("thr_now") is not None and m["thr_now"] < p.doom_dist
                and p.doom_min_e <= E < p.doom_frac * a["max_energy"] and n_eff < cap + p.dbed_extra + p.sen_extra)
        if E <= 101.0 or (threatened and not sen and not doom):
            return False
        if doom:
            self._bump("spawn_doom")
            return True
        if sen and E > 102.0 and n_eff < cap + p.dbed_extra + p.sen_extra:
            self._bump("spawn_sen")
            return True
        if n_eff <= p.n_min and E >= p.emergency_min_e:
            self._bump("spawn_emergency")
            return True
        if p.spawn_quiet_s > 0 and t - self.last_pred_t < p.spawn_quiet_s:
            return False                                   # a predator was seen recently: newborns (energy 75, no sprint) would be born into danger
        r = min(1.0, t / max(p.spawn_ramp, 1.0))
        thr = min(p.spawn_thr + (p.spawn_thr_late - p.spawn_thr) * r, 0.85 * a["max_energy"])
        dmin = min(p.dbed_min_energy + (p.dbed_min_late - p.dbed_min_energy) * r, 0.8 * a["max_energy"])
        if age >= p.dbed_age and E >= dmin and n_eff < cap + p.dbed_extra and sel_ok:
            self._bump("spawn_dbed")
            return True
        if E >= thr and n_eff < cap and sel_ok and food_near >= p.spawn_food_min:
            self._bump("spawn_normal")
            return True
        return False

    # ------------------------------------------------------------------ per agent
    def _act_one(self, a, t, n_eff, cap, sel_ok):
        p = self.p
        aid = a["agent_id"]
        m = self.mem.get(aid)
        if m is None:
            rng = random.Random(hash((self.seed, self.episode, aid)) & 0xFFFFFFFF)
            m = _new_mem(rng, t)
            m["rng"] = rng
            self.mem[aid] = m
        rng = m["rng"]
        E, ME = a["energy"], a["max_energy"]
        sp, spr = a["speed"], a["sprint_speed"]
        biome = a["biome"]
        # aging drain (0.01*age per tick, only after max_age) shows up as energy loss our own actions cannot explain
        if p.senescence > 0.5 and m["sen"] is None and m["E_prev"] is not None and not m["last_spawn"]:
            if (m["E_prev"] - E) - m["last_cost"] > 0.3:
                m["sen"] = t
        if m["E_prev"] is not None and E - m["E_prev"] > 3.0:
            m["gain_t"] = t
        m["E_prev"] = E

        fruits, trees, mates, preds, edges = [], [], [], [], []
        for o in a["observations"]:
            ty = o["type"]
            if ty == "Fruit":
                fruits.append((o["distance"], o["angle"]))
            elif ty == "Tree":
                trees.append((o["distance"], o["angle"]))
            elif ty == "Agent":
                mates.append((o["distance"], o["angle"], o.get("id", -1)))
            elif ty == "Predator":
                preds.append((o["distance"], o["angle"], o["rel_dir"]))
            elif ty == "Edge":
                edges.append(o["coords"])

        fruits = self._ledger(a, m, t, fruits)

        # ---------------- 1. predator threats
        threats = []
        for (d, ang, rel) in preds:
            if d > p.pred_react_dist:
                continue
            noticed = d < 65.0 or (abs(rel) < 0.5236 + p.pred_notice_margin and d < 265.0)
            if not noticed:
                ph = wrap(ang + PI - rel)
                wx, wy = math.cos(ph), math.sin(ph)
                px, py = d * math.cos(ang), d * math.sin(ang)
                closing = wx * (-px) + wy * (-py) > 0
                if not (closing and abs(wx * py - wy * px) < p.pred_cpa):
                    continue
            threats.append((d, ang, rel, noticed))

        if preds:
            m["pv_t"] = t
            m["pv_d"] = min(x[0] for x in preds)
            pr = min(preds, key=lambda x: x[0])                      # nearest predator: estimate its speed from its dead-reckoned world track
            wpx, wpy = m["x"] + pr[0] * math.cos(m["h"] + pr[1]), m["y"] + pr[0] * math.sin(m["h"] + pr[1])
            last = m.get("pw")
            if last is not None and 0.0 < t - last[0] < 0.25 and math.hypot(wpx - last[1], wpy - last[2]) < 40.0:
                v = math.hypot(wpx - last[1], wpy - last[2]) / max(1.0, (t - last[0]) / 0.1)
                m["pv"] = v if m.get("pv") is None else 0.6 * m["pv"] + 0.4 * v
                if math.hypot(wpx - last[1], wpy - last[2]) >= 3.0:
                    m["pdir"] = wrap(math.atan2(wpy - last[2], wpx - last[1]) - m["h"])      # predator heading in our local frame
            else:
                m["pv"] = None
                m["pdir"] = None
            m["pw"] = (t, wpx, wpy)
        m["thr_now"] = min((x[0] for x in threats if x[3]), default=None)
        m["nb"] = sum(1 for (md, _a, _i) in mates if md < 80.0)
        m["sel_ok"] = sel_ok
        m["p_die"] = None
        m["doomed"] = False
        if p.emap > 0.5:
            self._emap_update(m, t, edges)
        move_d = 0.0
        move_dir = 0.0
        turn = 0.0
        threatened = False
        if threats:
            threatened = True
            if m.get("thr_last", -99.0) < t - 5.0:
                m["thr_first"] = t
                m["thr_dmin"] = 1e9
            m["thr_last"] = t
            m["thr_dmin"] = min(m["thr_dmin"], min(x[0] for x in threats))
            self._bump("threat_sightings")
            thr = min(threats, key=lambda x: x[0])
            if self.escape is not None and p.use_escape_model > 0.5:
                m["p_die"] = self.escape.p_die(E / ME, sp, thr[0], len(threats), m["nb"], self._edge_min_dist(edges), a["age"], 1.0 if thr[3] else 0.0)
            pl = None
            if p.mpc_flee > 0.5 and thr[3] and thr[0] < p.mpc_range:
                pl = self._mpc_plan(m, preds, edges, sp, spr, E, ME)
            if pl is not None:                                     # model-based emergency control (survivor.pursuit)
                phi = pl.theta
                sprint = pl.sprint
                move_d = spr if sprint else sp
                turn = self._mpc_turn if pl.face_pred else phi
                m["doomed"] = pl.best_captured
                self._bump("mpc_doomed" if pl.best_captured else "mpc")
                self._bump("flee_sprint" if sprint else "flee")
            else:
                phi = self._flee_dir(thr, m)
                if p.flee_ray > 0.5 and p.emap > 0.5:
                    phi = self._ray_dir(m, phi, thr)
                phi = self._steer(phi, edges, None, m, p.wall_margin)
                sprint = thr[3] and thr[0] < p.pred_sprint_dist and E >= p.sprint_min_frac * ME and sp < p.sprint_below
                if sprint and p.kite > 0.5 and m.get("pv") is not None and m["pv"] <= p.tired_v and sp >= 12.0:
                    sprint = False                                   # the predator is exhausted (walk cap 11) or resting: outwalk it, keep the energy
                    self._bump("kite_nosprint")
                move_d = spr if sprint else min(sp, p.flee_cap)
                face = p.pred_face > 0.5 and thr[0] >= p.pred_face_dist
                turn = thr[1] if face else phi
                self._bump("flee_sprint" if sprint else "flee")
            move_dir = phi
            m["flee_until"] = t + p.flee_hold * 0.1
            m["flee_dir"] = wrap(phi - turn)
        elif t < m["flee_until"]:
            threatened = True
            move_d, move_dir, turn = min(sp, p.flee_cap), m["flee_dir"], 0.0
            m["flee_dir"] = wrap(m["flee_dir"])
            self._bump("flee")
        else:
            # ---------------- 2. unstick maneuver
            if t < m["unstick_until"]:
                move_d, move_dir, turn = sp, m["unstick_dir"], 0.0
                self._bump("unstick")
            else:
                move_d, move_dir, turn = self._forage(a, m, t, fruits, trees, mates, edges, rng)

        self.last_state[aid] = self._cur
        self._track(m, a, t, preds, threats, edges, mates)
        if preds and m.get("ep"):
            m["ep"][4] = min(m["ep"][4], min(x[0] for x in preds))

        # ---------------- 3. reproduction (spawn happens after move/turn inside the tick)
        food_near = sum(1 for f in fruits if f[0] < p.spawn_food_r)
        m["food_near"] = food_near
        m["threat_now"] = threatened
        spawn = self._want_spawn(a, m, t, n_eff, cap, sel_ok, threatened, food_near)

        # ---------------- 4. sanitise + dead reckoning
        cap_d = spr
        if E < ME / 5.0 and move_d > sp:
            cap_d = sp
        move_d = max(0.0, min(move_d, cap_d))
        pen = BIOME_PEN.get(biome, 1.0)
        ang_m = m["h"] + move_dir
        m["x"] += move_d * pen * math.cos(ang_m)
        m["y"] += move_d * pen * math.sin(ang_m)
        m["h"] += turn
        m["still"] = m["still"] + 1 if move_d <= 0.0 else 0
        m["last_cost"] = 0.1 + min(PI, abs(turn)) / TWO_PI + 0.05 * min(move_d, sp) + (0.5 * (move_d - sp) if move_d > sp else 0.0)
        m["last_spawn"] = bool(spawn)
        return {"agent_id": aid, "move_distance": float(move_d), "move_direction": float(move_dir),
                "turn_angle": float(turn), "spawn_agent": bool(spawn)}

    # ------------------------------------------------------------------ non-threat behaviour
    def _forage(self, a, m, t, fruits, trees, mates, edges, rng):
        p = self.p
        E, ME, sp = a["energy"], a["max_energy"], a["speed"]
        biome = a["biome"]
        full = E >= p.eat_full_frac * ME
        if p.patrol > 0.5:
            self._tmap_update(m, t, trees, a)
        if p.senescence > 0.5 and m.get("sen") is not None:
            # dying anyway: do not compete with young agents for fruit and do not burn energy walking
            self._bump("senescent")
            if fruits and E < 100.0:
                cs = [(d, ang) for (d, ang, oage, known) in fruits if d <= p.sen_fruit_r and oage >= min(p.ripe_age, 10.0)]
                if cs:
                    d, ang = min(cs)
                    return min(d, sp), ang, ang
            return 0.0, 0.0, 0.0

        # fruit first. Colony-wide ripeness discipline: fruit energy grows 20 -> 60 in its first 20 s, so everyone
        # waits until a fruit has been OBSERVED long enough (exact age if we saw it spawn, lower bound otherwise)
        waiting = False
        if fruits:
            urgent = E < p.urgent_abs or a["age"] < p.newborn_eager_s or (p.urgent_floor > 0.5 and E < 0.2 * ME + p.urgent_margin)
            cands = []
            for (d, ang, oage, known) in fruits:
                if d > p.fruit_max_dist:
                    continue
                if full and d > p.eat_full_dist:
                    continue
                if oage < (p.ripe_age if known else p.ripe_unknown) and not urgent:
                    waiting = True
                    continue
                cands.append((d, ang))
            if cands:
                d, ang = min(cands)
                self._bump("forage")
                self._progress(m, t, d)
                return min(d, sp), ang, ang

        # nearest non-banned tree that is not already full (agents with lower ids have priority at a tree)
        m["bans"] = [b for b in m["bans"] if b[2] > t]
        my_id = a["agent_id"]
        mw = [(m["x"] + md * math.cos(m["h"] + ma), m["y"] + md * math.sin(m["h"] + ma), mid) for (md, ma, mid) in mates]
        tree = None
        crowded_near = None
        cr2 = p.crowd_r ** 2
        for (d, ang) in sorted(trees):
            tx = m["x"] + d * math.cos(m["h"] + ang)
            ty = m["y"] + d * math.sin(m["h"] + ang)
            if any((tx - bx) ** 2 + (ty - by) ** 2 < p.tree_ban_r ** 2 for bx, by, _ in m["bans"]):
                continue
            lower = sum(1 for (mx, my, mid) in mw if mid < my_id and (mx - tx) ** 2 + (my - ty) ** 2 < cr2)
            if lower >= p.crowd_max:
                if crowded_near is None:
                    crowded_near = (d, ang)
                continue
            tree = (d, ang, tx, ty)
            break
        if tree is None and crowded_near is not None and crowded_near[0] < 90 and m["camp_t0"] is None and m.get("leave_t", -9) < t - 3:
            m["trv"] = wrap(crowded_near[1] + PI + m["rng"].uniform(-0.7, 0.7))   # walk away from the full tree
            m["leave_t"] = t

        if tree is not None and biome != "river":
            d, ang, tx, ty = tree
            m["goto"] = None
            if d > p.camp_radius:
                m["camp_t0"] = None
                self._bump("seek_tree")
                self._progress(m, t, d)
                return min(d - p.camp_radius * 0.5, sp), ang, ang
            # camping at the tree
            if m["camp_t0"] is None:
                m["camp_t0"] = t
                m["gain_t"] = max(m["gain_t"], t)
            if E >= self._rest_frac(t) * ME:
                m["gain_t"] = max(m["gain_t"], t)          # comfortable: no reason to leave (exploring costs ~11 energy/s)
            crowd = sum(1 for (md, _, _mid) in mates if md < p.crowd_r)
            base_pat = p.late_patience if (p.late_patience > 0 and t >= p.late_t) else p.camp_patience
            patience = base_pat / (1.0 + 0.6 * max(0, crowd - p.crowd_max + 1))
            # hungry agents give up sooner: leave while there is still energy to sprint / travel
            patience = min(patience, max(14.0, (E - 0.3 * ME) / max(p.patience_burn, 0.1)))
            idle = t - max(m["camp_t0"], m["gain_t"])
            if idle <= patience and p.patrol > 0.5 and not waiting and idle > p.patrol_wait and self._patrol_leave(m, t, tx, ty, mw, my_id):
                m["camp_t0"] = None
                self._bump("bans")
                tree = None
            elif idle > patience:
                m["bans"].append((tx, ty, t + p.tree_ban_s))
                m["camp_t0"] = None
                self._bump("bans")
                m["trv"] = wrap(ang + PI + rng.uniform(-0.8, 0.8))
            else:
                self._bump("camp")
                return 0.0, 0.0, min(1.2, p.camp_scan + p.vig_k * t / 1000.0)
        else:
            m["camp_t0"] = None

        if m["goto"] is not None:                       # patrol: walk to a remembered tree that is not in view
            gx, gy, exp_t = m["goto"]
            dist = math.hypot(gx - m["x"], gy - m["y"])
            if t > exp_t or dist < 2 * p.camp_radius:
                m["goto"] = None
            elif tree is None:
                angg = wrap(math.atan2(gy - m["y"], gx - m["x"]) - m["h"])
                phi = self._steer(angg, edges, mates, m, p.wall_margin)
                self._bump("seek_tree")
                return min(sp, dist), phi, phi

        if waiting:                                    # unripe fruit in sight and no tree to camp at: hold still
            self._bump("camp")
            return 0.0, 0.0, min(1.2, p.camp_scan + p.vig_k * t / 1000.0)

        if E >= self._rest_frac(t) * ME:                       # comfortable and nothing to do: rest (1 energy/s) instead of exploring (11/s)
            self._bump("camp")
            return 0.0, 0.0, min(1.2, p.camp_scan + p.vig_k * t / 1000.0)

        # explore
        self._bump("explore")
        m["d_prev"] = None
        m["stuck"] = 0
        if biome in BAD_BIOMES:
            wander = 0.0
        else:
            wander = p.explore_wander
        m["trv"] = wrap(m["trv"] + rng.gauss(0.0, wander))
        phi = self._steer(m["trv"], edges, mates, m, p.wall_margin)
        spin = p.explore_spin * a["vision_angle"]
        m["trv"] = wrap(phi - spin)
        return sp, phi, spin

    def _close_ep(self, m, t, E, outcome):
        ep = m.get("ep")
        if not ep:
            return
        g, t0, E0, b0, dmin, ME0, sp0 = ep[:7]
        m["ep"] = None
        dur, dE = t - t0, E - E0
        A = self.agg.setdefault(g, {"n": 0, "dur": 0.0, "dE": 0.0, "died": 0})
        A["n"] += 1
        A["dur"] += dur
        A["dE"] += dE
        if outcome == "died":
            A["died"] += 1
        if g == "camp":
            cb = self.camp_b.setdefault(b0, [0, 0.0, 0.0])
            cb[0] += 1; cb[1] += dur; cb[2] += dE
        elif g == "flee":
            eb = "E<20%" if E0 < 0.2 * ME0 else ("E20-40%" if E0 < 0.4 * ME0 else "E>40%")
            sb = "spd<13" if sp0 < 13 else ("spd13-17" if sp0 < 17 else "spd>=17")
            F = self.flee_tab.setdefault(eb + "|" + sb, [0, 0])
            F[0] += 1
            if outcome == "died":
                F[1] += 1
            if len(self.flee_rows) < 8000 and len(ep) >= 13:
                self.flee_rows.append([round(E0 / max(ME0, 1.0), 3), round(sp0, 1), round(ep[7]), ep[8], ep[10], round(ep[11]), round(ep[9]),
                                       ep[12], round(dmin) if dmin < 1e8 else -1, round(dur, 1), 1 if outcome == "died" else 0])
            if outcome != "died":
                m["escapes"] = m.get("escapes", 0) + 1

    def _val_feats(self, a, m, t, n, e=None):
        E = a["energy"] if e is None else e
        return [round(E / a["max_energy"], 3), round(a["speed"], 1), round(a["age"]), round(a["hearing_radius"]), m.get("nb", 0), n,
                round(min(60.0, t - m.get("thr_last", -99.0))), round(a["max_energy"])]

    def _val_tick(self, agents, t):
        """meta-reward windows: +2 per 5 s survived. Row = features at window start + 1 if the agent is still alive 5 s later."""
        if t - self.val_t < 5.0:
            return
        alive = {a["agent_id"] for a in agents}
        for aid, f in self.val_pend.items():
            if len(self.val_rows) < 8000:
                self.val_rows.append(f + [1 if aid in alive else 0])
        self.val_pend = {}
        n = len(agents)
        for a in agents:
            m = self.mem.get(a["agent_id"])
            if m is not None:
                self.val_pend[a["agent_id"]] = self._val_feats(a, m, t, n)
        self.val_t = t

    def _new_ep(self, g, t, a, preds, threats, edges, mates):
        d0 = min((x[0] for x in preds), default=-1.0)
        nb = sum(1 for (md, _a, _i) in mates if md < 80.0)
        wall = self._edge_min_dist(edges) if g == "flee" else 300.0
        noticed = 1.0 if any(x[3] for x in threats) else 0.0
        return [g, t, a["energy"], a["biome"], 1e9, a["max_energy"], a["speed"], d0, len(threats), a["age"], nb, wall, noticed]

    def _track(self, m, a, t, preds, threats=(), edges=(), mates=()):
        cur = self._cur
        g = "flee" if cur in ("flee", "flee_sprint") else cur
        m["age"] = a["age"]; m["biome"] = a["biome"]; m["ME"] = a["max_energy"]
        ep = m.get("ep")
        if ep is None:
            m["ep"] = self._new_ep(g, t, a, preds, threats, edges, mates)
        elif ep[0] != g:
            self._close_ep(m, t, a["energy"], "end")
            m["ep"] = self._new_ep(g, t, a, preds, threats, edges, mates)

    def _mpc_plan(self, m, preds, edges, sp, spr, E, ME):
        """build the local-frame problem (we are at the origin, heading 0), advance predators one tick for the observation latency, plan"""
        p = self.p
        near = sorted(preds, key=lambda x: x[0])[:3]
        items = []
        for i, (d, ang, _rel) in enumerate(near):
            px, py = d * math.cos(ang), d * math.sin(ang)
            ph = m["pdir"] if (i == 0 and m.get("pdir") is not None) else math.atan2(-py, -px)      # heading estimate, else 'aimed at us'
            pv = m.get("pv") if i == 0 else None
            pE = 0.0 if (pv is not None and pv < 1.5) else (30.0 if (pv is not None and pv <= p.tired_v) else 130.0)   # asleep / tired (capped at 11) / fresh
            items.append((px, py, ph, pE))
        arr = np.array(items, float)
        z = np.zeros((1, 1))
        npx, npy, nph, npE = _pur.pred_step(arr[None, :, 0], arr[None, :, 1], arr[None, :, 2], arr[None, :, 3], z, z, z)
        self._mpc_turn = float(math.atan2(npy[0, 0], npx[0, 0]))
        preds_now = [(float(npx[0, k]), float(npy[0, k]), float(nph[0, k]), float(npE[0, k])) for k in range(len(items))]
        segs = [(x1, y1, x2, y2) for ((x1, y1), (x2, y2)) in edges]
        if p.emap > 0.5 and m.get("emap"):
            h, x0, y0 = m["h"], m["x"], m["y"]
            ch, sh = math.cos(h), math.sin(h)
            for (ax_, ay_, bx_, by_, _te) in m["emap"].values():
                d1x, d1y, d2x, d2y = ax_ - x0, ay_ - y0, bx_ - x0, by_ - y0
                segs.append((d1x * ch + d1y * sh, -d1x * sh + d1y * ch, d2x * ch + d2y * sh, -d2x * sh + d2y * ch))
        E_arr = None
        if segs:
            E_arr = np.array(segs, float)
            dd = np.minimum(np.hypot(E_arr[:, 0], E_arr[:, 1]), np.hypot(E_arr[:, 2], E_arr[:, 3]))
            E_arr = E_arr[np.argsort(dd)[:40]]
        return _pur.plan(sp, spr, E, ME, preds_now, E_arr, H=int(p.mpc_h))

    def _edge_min_dist(self, edges):
        best = 300.0
        for e in edges:
            (x1, y1), (x2, y2) = e
            vx, vy = x2 - x1, y2 - y1
            vv = vx * vx + vy * vy
            tt = 0.0 if vv <= 0 else max(0.0, min(1.0, -(x1 * vx + y1 * vy) / vv))
            d = math.hypot(x1 + tt * vx, y1 + tt * vy)
            if d < best:
                best = d
        return best

    def _emap_update(self, m, t, edges):
        """dead-reckoned memory of the walls/obstacle edges this agent has seen (the 'wall layer' of the ecological memory)"""
        h, x, y = m["h"], m["x"], m["y"]
        c, s_ = math.cos(h), math.sin(h)
        em = m["emap"]
        for e in edges:
            (x1, y1), (x2, y2) = e
            ax, ay = x + x1 * c - y1 * s_, y + x1 * s_ + y1 * c
            bx, by = x + x2 * c - y2 * s_, y + x2 * s_ + y2 * c
            em[(round(ax / 14), round(ay / 14), round(bx / 14), round(by / 14))] = (ax, ay, bx, by, t)
        if len(em) > 260:
            keep = sorted(em.items(), key=lambda kv: -kv[1][4])[:200]
            m["emap"] = dict(keep)
        elif int(t * 10) % 100 == 0:
            m["emap"] = {k: v for k, v in em.items() if t - v[4] <= self.p.emap_age}

    def _ray_dir(self, m, phi, thr):
        """obstacle-aware escape: among directions around `phi`, prefer long free paths (over remembered edges), never toward the predator"""
        em = m["emap"]
        if not em:
            return phi
        h, x, y = m["h"], m["x"], m["y"]
        ch, sh = math.cos(h), math.sin(h)
        segs = []
        for (ax, ay, bx, by, _te) in em.values():
            d1x, d1y, d2x, d2y = ax - x, ay - y, bx - x, by - y
            if min(abs(d1x), abs(d1y), abs(d2x), abs(d2y)) > 260 and math.hypot(d1x, d1y) > 300 and math.hypot(d2x, d2y) > 300:
                continue
            segs.append((d1x * ch + d1y * sh, -d1x * sh + d1y * ch, d2x * ch + d2y * sh, -d2x * sh + d2y * ch))
        away = wrap(thr[1] + PI)
        Lmax = 150.0
        best = None
        for k in range(-8, 9):
            th = phi + k * 0.30
            cx, cy = math.cos(th), math.sin(th)
            free = Lmax
            for (x1, y1, x2, y2) in segs:
                ex, ey = x2 - x1, y2 - y1
                den = cx * ey - cy * ex
                if abs(den) < 1e-9:
                    continue
                tt = (x1 * ey - y1 * ex) / den
                uu = (x1 * cy - y1 * cx) / den
                if 0 < tt < free and 0.0 <= uu <= 1.0:
                    free = tt
            score = math.cos(wrap(th - away)) + 1.5 * free / Lmax
            if free < 40.0:
                score -= 2.5
            if math.cos(wrap(th - thr[1])) > 0.3:
                score -= 2.0
            if best is None or score > best[0]:
                best = (score, th)
        return wrap(best[1])

    def summary(self):
        """compact, JSON-able state-tracking summary of the current episode"""
        out = {"states": {}, "camp_biome": {}, "flee": {}, "events": self.events[:200], "colony": self.colony.summary()}
        for g, A in self.agg.items():
            n = max(1, A["n"])
            out["states"][g] = {"episodes": A["n"], "mean_dur_s": round(A["dur"] / n, 1), "net_energy_per_s": round(A["dE"] / max(A["dur"], 1e-9), 2),
                                "ended_by_death": A["died"], "total_s": round(A["dur"])}
        for b, v in self.camp_b.items():
            out["camp_biome"][b] = {"episodes": v[0], "seconds": round(v[1]), "net_energy_per_s": round(v[2] / max(v[1], 1e-9), 2)}
        for k, v in sorted(self.flee_tab.items()):
            out["flee"][k] = {"episodes": v[0], "died": v[1], "death_rate": round(v[1] / max(1, v[0]), 3)}
        return out

    def _ledger(self, a, m, t, fruits):
        """Track when each fruit first appeared (only certain for a stationary agent that hears it spawn).
        Returns [(dist, angle, observed_age, age_is_exact)]."""
        x, y, h = m["x"], m["y"], m["h"]
        hear = a["hearing_radius"]
        certain = m["still"] >= 3
        fl = m["fl"]
        seen = set()
        out = []
        for (d, ang) in fruits:
            wx = x + d * math.cos(h + ang)
            wy = y + d * math.sin(h + ang)
            hit = None
            for i, e in enumerate(fl):
                if i not in seen and (e[0] - wx) ** 2 + (e[1] - wy) ** 2 < 196.0:
                    hit = i
                    break
            if hit is None:
                fl.append([wx, wy, t, t, bool(certain and d <= 0.92 * hear)])
                hit = len(fl) - 1
            e = fl[hit]
            seen.add(hit)
            e[3] = t
            out.append((d, ang, t - e[2], bool(e[4])))
        keep = []
        for i, e in enumerate(fl):
            if i in seen:
                keep.append(e)
            elif t - e[3] < 45.0 and math.hypot(e[0] - x, e[1] - y) > 0.8 * hear:
                keep.append(e)
        m["fl"] = keep
        return out

    def _rest_frac(self, t):
        p = self.p
        return p.rest_frac if t < p.late_t else min(p.rest_frac, p.late_rest_frac)

    def _tmap_update(self, m, t, trees, a):
        """dead-reckoned map of trees: [wx, wy, last_seen, last_visit]"""
        p = self.p
        x, y, h = m["x"], m["y"], m["h"]
        tm = m["tmap"]
        seen = set()
        for (d, ang) in trees:
            wx, wy = x + d * math.cos(h + ang), y + d * math.sin(h + ang)
            hit = None
            for i, e in enumerate(tm):
                if i not in seen and (e[0] - wx) ** 2 + (e[1] - wy) ** 2 < 900.0:
                    hit = i
                    break
            if hit is None:
                tm.append([wx, wy, t, -1e9])
                hit = len(tm) - 1
            e = tm[hit]
            e[0], e[1], e[2] = 0.7 * e[0] + 0.3 * wx, 0.7 * e[1] + 0.3 * wy, t
            seen.add(hit)
            if d <= p.camp_radius * 1.5:
                e[3] = t
        hear = a["hearing_radius"]
        keep = []
        for i, e in enumerate(tm):
            if i in seen:
                keep.append(e)
            elif t - e[2] <= p.tmap_age and math.hypot(e[0] - x, e[1] - y) >= 0.8 * hear:
                keep.append(e)                       # (a tree that should be audible but is not is gone)
        m["tmap"] = keep

    def _patrol_leave(self, m, t, cx, cy, mw, my_id):
        """leave the exhausted tree for the remembered tree with most accumulated fruit per distance"""
        p = self.p
        best = None
        for e in m["tmap"]:
            if (e[0] - cx) ** 2 + (e[1] - cy) ** 2 < 1600.0:
                continue
            if any((e[0] - bx) ** 2 + (e[1] - by) ** 2 < p.tree_ban_r ** 2 for bx, by, _ in m["bans"]):
                continue
            if any(mid < my_id and (mx - e[0]) ** 2 + (my - e[1]) ** 2 < 3600.0 for (mx, my, mid) in mw):
                continue
            dist = math.hypot(e[0] - m["x"], e[1] - m["y"])
            if dist > p.patrol_max:
                continue
            acc = min(50.0, t - e[3]) * 0.1 if e[3] > -1e8 else 2.0      # ~0.1 fruit/s accumulates (rots after 50 s)
            if acc < 1.0:
                continue
            sc = acc / (dist + 40.0)
            if best is None or sc > best[0]:
                best = (sc, e[0], e[1])
        if best is None:
            return False
        m["bans"].append((cx, cy, t + p.patrol_ban))
        m["goto"] = (best[1], best[2], t + 30.0)
        self._bump("patrol_leave")
        return True

    def _progress(self, m, t, d):
        """detect being stuck on a wall while approaching a target"""
        dp = m["d_prev"]
        if dp is not None and d > dp - 0.5:
            m["stuck"] += 1
            if m["stuck"] > 14:
                m["stuck"] = 0
                m["unstick_until"] = t + 1.0
                m["unstick_dir"] = wrap(m["rng"].choice((-1, 1)) * PI / 2 + m["rng"].uniform(-0.5, 0.5))
        else:
            m["stuck"] = max(0, m["stuck"] - 1)
        m["d_prev"] = d
