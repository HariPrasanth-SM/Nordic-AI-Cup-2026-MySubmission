"""
survivor.runner - headless local evaluation with ground-truth diagnostics.

  python -m survivor.runner check                                   # self tests (semantics, fastsim, policy)
  python -m survivor.runner eval --params configs/v1_default.json --seeds 1 2 3 4 5 --out runs/e1
  python -m survivor.runner watch --params configs/v1_default.json --seed 1   # pygame window

`eval` drives src.core.SimulationCore directly (no HTTP) with exactly the dict payload the server gets,
and writes one JSON per seed (time series, death causes, score decomposition, paths, snapshots).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor.policy import Controller, Params  # noqa: E402


class _Act:
    __slots__ = ("move_distance", "move_direction", "turn_angle", "spawn_agent")

    def __init__(self, d):
        self.move_distance = d["move_distance"]
        self.move_direction = d["move_direction"]
        self.turn_angle = d["turn_angle"]
        self.spawn_agent = d["spawn_agent"]


def _params(p):
    if isinstance(p, Params):
        return p
    return Params.load(p)


def run_episode(seed, params, horizon=3000.0, fast=True, sample_every=50, snap_times=(), keep_paths=True,
                controller_factory=None, verbose=False, knobs=None):
    """Run one full simulation in-process. Returns a JSON-serialisable result dict."""
    from src.core import SimulationCore
    if fast:
        from survivor import fastsim
        fastsim.apply()
    t_wall0 = time.perf_counter()
    sim = SimulationCore(seed=seed)
    env = sim.env
    ctrl = controller_factory(seed) if controller_factory else Controller(_params(params), seed=seed)
    knobs = knobs or {}
    # ---- causal ablation knobs (diagnosis only, never used by the server) ---------------------------------
    if knobs.get("no_predators"):
        env.spawn_predator = lambda *a, **k: None
        env.predators.clear()
    tree_floor = int(knobs.get("tree_floor", 0))
    tr = knobs.get("traits")
    if tr:
        _map = {"speed": "speed", "sprint": "sprint_speed", "maxE": "max_energy", "hearing": "hearing_radius",
                "vision": "vision_radius", "cone": "cone_angle"}
        for ag_ in env.agents:
            for k_, v_ in tr.items():
                setattr(ag_, _map[k_], float(v_))
            ag_.energy = min(ag_.energy, ag_.max_energy)
        if knobs.get("fixed_traits", True):
            _orig_sp = env.spawn_agent

            def _fixed_spawn(x=None, y=None, parent=None):
                ag2 = _orig_sp(x, y, parent)
                if parent is not None and ag2 is not None:
                    for a_ in _map.values():
                        setattr(ag2, a_, getattr(parent, a_))
                return ag2
            env.spawn_agent = _fixed_spawn

    ev = {"eh": [0, 0, 0, 0, 0], "deaths": [], "births": [], "eaten": 0, "eaten_energy": 0.0, "rotted": 0, "pred_pen": 0.0}
    deaths_by = {"predator": 0, "starved_young": 0, "starved_old": 0}
    births_total = [0]

    orig_kill = env.kill_agent

    def _ctx(agent):
        """ground-truth context of an agent (used for death / birth diagnostics)"""
        td = min((math.hypot(tr.x - agent.x, tr.y - agent.y) for tr in env.trees), default=-1.0)
        nb = sum(1 for o in env.agents if o is not agent and (o.x - agent.x) ** 2 + (o.y - agent.y) ** 2 < 6400)
        pd_ = min((math.hypot(p_.x - agent.x, p_.y - agent.y) for p_ in env.predators), default=-1.0)
        return td, nb, pd_

    def kill(agent):
        if agent in env.agents:
            if agent.energy <= 0:
                cause = "starved_old" if agent.age > agent.max_age else "starved_young"
            else:
                cause = "predator"
                ev["pred_pen"] += agent.energy / 100.0
            deaths_by[cause] += 1
            mm = getattr(ctrl, "mem", {}).get(agent.agent_id, {})
            td, nb, pd_ = _ctx(agent)
            thr_ago = (env.time - mm["thr_first"]) if mm.get("thr_first") is not None and env.time - mm.get("thr_last", -99) < 3.0 else None
            pv_ago = (env.time - mm["pv_t"]) if mm.get("pv_t") is not None else None
            bm = env.biome_map[min(max(int(agent.x), 0), env.width - 1), min(max(int(agent.y), 0), env.height - 1)].type
            ev["deaths"].append([round(env.time, 1), cause, round(agent.age, 1), round(agent.energy, 1),
                                 round(agent.max_age, 1), round(agent.x), round(agent.y),
                                 getattr(ctrl, "last_state", {}).get(agent.agent_id, "?"),
                                 None if thr_ago is None else round(thr_ago, 1),
                                 None if not mm.get("thr_dmin") or mm.get("thr_dmin") > 1e8 else round(mm["thr_dmin"]),
                                 round(td), nb, round(agent.speed, 1), round(agent.max_energy), bm,
                                 None if pv_ago is None else round(pv_ago, 1), round(pd_), agent.agent_id])
        orig_kill(agent)
    env.kill_agent = kill

    orig_rm = env.remove_fruit

    def rm(fruit):
        if fruit in env.fruits:
            if fruit.age > 100:
                ev["rotted"] += 1
            else:
                ev["eaten"] += 1
                ev["eaten_energy"] += fruit.energy
                ev["eh"][min(4, int((fruit.energy - 20) // 10))] += 1
                # who ate it? nearest agent within reach, its policy state, urgency, and whether it was 'certain' (still) 
                best = None
                for o in env.agents:
                    dd = (o.x - fruit.x) ** 2 + (o.y - fruit.y) ** 2
                    if best is None or dd < best[0]:
                        best = (dd, o)
                if best is not None and best[0] < 30 ** 2:
                    o = best[1]
                    mm = getattr(ctrl, "mem", {}).get(o.agent_id, {})
                    young = fruit.energy < 30
                    key = ("young" if young else "old") + "|" + getattr(ctrl, "last_state", {}).get(o.agent_id, "?") + \
                          "|" + ("hungry" if o.energy < 0.3 * o.max_energy else "ok") + "|" + ("still" if mm.get("still", 0) >= 3 else "moving")
                    ev.setdefault("eater", {})
                    ev["eater"][key] = ev["eater"].get(key, 0) + 1
        orig_rm(fruit)
    env.remove_fruit = rm

    orig_spawn = env.spawn_agent

    def spawn(x=None, y=None, parent=None):
        ag = orig_spawn(x, y, parent)
        if parent is not None and ag is not None:
            births_total[0] += 1
            td, nb, _pd = _ctx(ag)
            ev["births"].append([round(env.time, 1), round(ag.speed, 2), round(ag.sprint_speed, 2), round(ag.max_energy),
                                 round(ag.hearing_radius, 1), round(ag.vision_radius, 1), round(ag.cone_angle, 3),
                                 round(parent.energy), round(parent.age), round(td), nb, ag.agent_id, parent.agent_id])
        return ag
    env.spawn_agent = spawn

    ts = {k: [] for k in ("t", "n", "mean_E", "mean_age", "trees", "mature", "fruits", "preds", "births", "eaten",
                          "d_pred", "d_starve_young", "d_starve_old", "mean_speed", "mean_vision", "mean_hear",
                          "mean_cone", "mean_maxE", "score", "ready")}
    paths, snaps = {}, []
    heat = [[0] * 30 for _ in range(40)]
    snap_pending = sorted(snap_times)
    obstacles = [[o.x, o.y, o.width, o.height] for o in env.obstacles]

    # energy ledger by policy state: [ticks, walk, sprint, turn, living, aging, spawn, income]
    ledger = {}
    prev = {}          # agent_id -> (E, state, costs tuple) from the previous tick
    pol_time = 0.0
    n_ticks = 0
    n_agent_ticks = 0
    peak = 0
    pop_sum = 0
    actions = []
    last_print = 0.0
    state = None
    while True:
        state = sim.step(actions)
        n_ticks += 1
        if tree_floor and len(env.trees) < tree_floor:
            for _ in range(min(3, tree_floor - len(env.trees))):
                env.spawn_tree()
        n = state["num_agents"]
        t = state["sim_time"]
        peak = max(peak, n)
        pop_sum += n
        if n == 0 or t > horizon:
            break
        obs_list = [o for o in state["observations"] if o is not None]
        step = {"game_status": "ok", "score": state["score"], "sim_time": t, "n_agents": n, "agent_status": obs_list}
        p0 = time.perf_counter()
        acts = ctrl.act(step)
        pol_time += time.perf_counter() - p0
        n_agent_ticks += len(obs_list)
        actions = [(a["agent_id"], _Act(a)) for a in acts]
        # ---- energy ledger: income of the PREVIOUS action = dE + its costs
        seen_now = {}
        for o in obs_list:
            aid = o["agent_id"]
            pv = prev.get(aid)
            if pv is not None:
                E0, st0, cs = pv
                lg = ledger.setdefault(st0, [0.0] * 8)
                lg[7] += (o["energy"] - E0) + sum(cs)
            seen_now[aid] = o
        prev = {}
        for o, a in zip(obs_list, acts):
            aid = o["agent_id"]
            ag = env.agents_dict.get(aid)
            if ag is None:
                continue
            E, sp_ = o["energy"], o["speed"]
            d = max(0.0, min(a["move_distance"], o["sprint_speed"]))
            if E < o["max_energy"] / 5.0 and d > sp_:
                d = sp_
            walk = 0.05 * min(d, sp_)
            spr = (d - sp_) * 0.5 if d > sp_ else 0.0
            trn = min(math.pi, abs(a["turn_angle"])) / (2 * math.pi)
            liv = 0.1
            age_c = 0.01 * ag.age if ag.age > ag.max_age else 0.0
            sp_c = 100.0 if (a["spawn_agent"] and E > 100) else 0.0
            stn = ctrl.last_state.get(aid, "?") if hasattr(ctrl, "last_state") else "?"
            lg = ledger.setdefault(stn, [0.0] * 8)
            lg[0] += 1; lg[1] += walk; lg[2] += spr; lg[3] += trn; lg[4] += liv; lg[5] += age_c; lg[6] += sp_c
            prev[aid] = (E, stn, (walk, spr, trn, liv, age_c, sp_c))

        if n_ticks % sample_every == 0:
            ag = env.agents
            k = max(1, len(ag))
            ts["t"].append(round(t, 1)); ts["n"].append(n)
            ts["mean_E"].append(round(sum(a.energy for a in ag) / k, 1))
            ts["mean_age"].append(round(sum(a.age for a in ag) / k, 1))
            ts["trees"].append(len(env.trees)); ts["mature"].append(sum(1 for tr in env.trees if tr.age >= 20))
            ts["fruits"].append(len(env.fruits)); ts["preds"].append(len(env.predators))
            ts["births"].append(births_total[0]); ts["eaten"].append(ev["eaten"])
            ts["d_pred"].append(deaths_by["predator"]); ts["d_starve_young"].append(deaths_by["starved_young"])
            ts["d_starve_old"].append(deaths_by["starved_old"])
            ts["mean_speed"].append(round(sum(a.speed for a in ag) / k, 2))
            ts["mean_vision"].append(round(sum(a.vision_radius for a in ag) / k, 1))
            ts["mean_hear"].append(round(sum(a.hearing_radius for a in ag) / k, 1))
            ts["mean_cone"].append(round(sum(a.cone_angle for a in ag) / k, 3))
            ts["mean_maxE"].append(round(sum(a.max_energy for a in ag) / k, 1))
            ts["score"].append(round(state["score"], 2))
            # escape-ready: energy above the sprint floor (max_energy/5) plus ~4 s of sprint burst (about 40 energy)
            ts["ready"].append(round(sum(1 for a in ag if a.energy >= a.max_energy / 5 + 40) / k, 3))
        if n_ticks % 10 == 0:
            for a in env.agents:
                heat[min(39, int(a.x // 40))][min(29, int(a.y // 40))] += 1
            if keep_paths and n_ticks % 20 == 0:
                for a in env.agents:
                    if len(paths) < 400 or a.agent_id in paths:
                        paths.setdefault(a.agent_id, []).append(
                            [round(t, 1), round(a.x), round(a.y), ctrl.last_state.get(a.agent_id, "?")[:2]])
        if snap_pending and t >= snap_pending[0]:
            snap_pending.pop(0)
            snaps.append({
                "t": round(t, 1),
                "agents": [[round(a.x), round(a.y), round(a.energy), round(a.age), round(a.direction, 2)] for a in env.agents],
                "preds": [[round(p_.x), round(p_.y), int(p_.resting)] for p_ in env.predators],
                "trees": [[round(tr.x), round(tr.y), round(tr.age)] for tr in env.trees],
                "fruits": [[round(f.x), round(f.y)] for f in env.fruits],
            })
        if verbose and t - last_print >= 100:
            last_print = t
            print(f"  seed {seed} t={t:6.0f} n={n:3d} score={state['score']:8.1f} trees={len(env.trees)} preds={len(env.predators)}", flush=True)

    score = float(state["score"])
    sim_time = float(env.time)
    wall = time.perf_counter() - t_wall0
    res = {
        "seed": seed, "score": score, "sim_time": sim_time, "horizon": horizon,
        "survived_full": bool(n > 0 and sim_time > horizon - 1e-6),
        "alive_end": int(state["num_agents"]),
        "score_parts": {"survival": sim_time, "fruit": ev["eaten_energy"] / 1000.0, "predation_penalty": -ev["pred_pen"]},
        "peak_pop": peak, "mean_pop": pop_sum / max(1, n_ticks),
        "births": births_total[0], "deaths": deaths_by, "fruits_eaten": ev["eaten"], "eaten_hist": ev["eh"], "eater": ev.get("eater", {}), "fruits_rotted": ev["rotted"],
        "policy_stats": ctrl.stats, "track": (ctrl.summary() if hasattr(ctrl, "summary") else {}),
        "timing": {"wall_s": wall, "policy_ms_per_tick": 1000 * pol_time / max(1, n_ticks),
                   "policy_us_per_agent_tick": 1e6 * pol_time / max(1, n_agent_ticks),
                   "sim_ms_per_tick": 1000 * (wall - pol_time) / max(1, n_ticks)},
        "ledger": {k: [round(x, 1) for x in v] for k, v in ledger.items()},
        "ts": ts, "deaths_list": ev["deaths"], "births_list": ev["births"][:6000],
        "heat": heat, "obstacles": obstacles, "snaps": snaps, "paths": paths if keep_paths else {},
    }
    res["flee_rows"] = getattr(ctrl, "flee_rows", [])[:3000]
    res["val_rows"] = getattr(ctrl, "val_rows", [])[:8000]
    return res


def _worker(job):
    seed, params_dict, horizon, fast, snaps, out_dir, verbose = job[:7]
    knobs = job[7] if len(job) > 7 else None
    res = run_episode(seed, params_dict, horizon=horizon, fast=fast, snap_times=snaps, verbose=verbose, knobs=knobs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"seed_{seed}.json"), "w") as f:
            json.dump(res, f)
    return res


def evaluate(params, seeds, horizon=3000.0, workers=1, fast=True, out_dir=None, snaps=(), verbose=False):
    from dataclasses import asdict
    pd = asdict(_params(params))
    jobs = [(s, pd, horizon, fast, tuple(snaps), out_dir, verbose) for s in seeds]
    if workers <= 1:
        return [_worker(j) for j in jobs]
    with Pool(workers) as pool:
        return list(pool.imap(_worker, jobs))


def compare(configs, seeds, horizon, workers, out_root, fast=True):
    """evaluate several parameter files on the SAME seeds in one pool and print a comparison table"""
    from dataclasses import asdict
    import statistics as st
    names = [os.path.splitext(os.path.basename(c))[0] for c in configs]
    jobs, tags = [], []
    for c, n in zip(configs, names):
        pd = asdict(_params(c))
        od = os.path.join(out_root, n) if out_root else None
        for sd in seeds:
            jobs.append((sd, pd, horizon, fast, (100, 400, 900, 1500, 2100, 2800), od, False))
            tags.append(n)
    if workers <= 1:
        res = [_worker(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            res = pool.map(_worker, jobs, chunksize=1)
    print(f"\n{'config':<26}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}{'full':>6}{'pred%':>7}{'young%':>8}{'old%':>6}{'births':>8}{'maxE':>6}{'spd':>6}")
    rows = {}
    for n in names:
        rs = [r for r, t in zip(res, tags) if t == n]
        sc = [r["score"] for r in rs]
        d = {k: sum(r["deaths"][k] for r in rs) for k in ("predator", "starved_young", "starved_old")}
        tot = max(1, sum(d.values()))
        # final-third traits of the survivors' timelines
        sp = [r["ts"]["mean_speed"][len(r["ts"]["mean_speed"]) // 2] for r in rs if r["ts"]["mean_speed"]]
        me = [r["ts"]["mean_maxE"][len(r["ts"]["mean_maxE"]) // 2] for r in rs if r["ts"]["mean_maxE"]]
        rows[n] = sc
        print(f"{n:<26}{st.mean(sc):>8.1f}{st.median(sc):>8.1f}{min(sc):>8.1f}{max(sc):>8.1f}{sum(r['survived_full'] for r in rs):>6}"
              f"{100 * d['predator'] / tot:>7.0f}{100 * d['starved_young'] / tot:>8.0f}{100 * d['starved_old'] / tot:>6.0f}"
              f"{st.mean(r['births'] for r in rs):>8.0f}{(st.mean(me) if me else 0):>6.0f}{(st.mean(sp) if sp else 0):>6.1f}")
    print("\nper-seed scores:")
    for n in names:
        print(f"  {n:<24}" + " ".join(f"{x:6.0f}" for x in rows[n]))
    # paired statistics against the first config (same seeds = common random numbers)
    from survivor.micro import paired_stats, verdict
    print(f"\nPAIRED vs {names[0]} (same seeds): mean delta [95% bootstrap CI] | median | worst-quartile | wins/losses | sign-test p | verdict (min effect 60)")
    for n in names[1:]:
        s_ = paired_stats(rows[n], rows[names[0]])
        print(f"  {n:<24}{s_['mean']:+8.1f} [{s_['lo']:+.0f}, {s_['hi']:+.0f}] | {s_['median']:+.0f} | {s_['q25']:+.0f} | {s_['wins']}/{s_['losses']} | p={s_['p']:.3f} | {verdict(s_, 60.0)}")
    return res


def summarize(results):
    import statistics as st
    sc = [r["score"] for r in results]
    return {
        "n_seeds": len(sc), "mean_score": st.mean(sc), "min_score": min(sc), "max_score": max(sc),
        "std": st.pstdev(sc) if len(sc) > 1 else 0.0,
        "full_survival": sum(r["survived_full"] for r in results),
        "mean_sim_time": st.mean(r["sim_time"] for r in results),
    }


# ---------------------------------------------------------------------------------------- self tests
def check():
    import random
    from src.core import SimulationCore
    ok = True

    def report(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    print("1) action semantics (README says absolute, code says relative)")
    sim = SimulationCore(seed=7)
    env = sim.env
    env.agents.clear(); env.agents_dict.clear()
    ag = env.spawn_agent(x=800, y=600)
    ag.direction = 1.0
    x0, y0 = ag.x, ag.y
    env.agent_step(ag.agent_id, move_distance=5.0, move_direction=math.pi / 2, turn_angle=0.0)
    moved = math.atan2(ag.y - y0, ag.x - x0)
    want = 1.0 + math.pi / 2
    report("move_direction is relative to heading", abs(((moved - want + math.pi) % (2 * math.pi)) - math.pi) < 1e-6)
    ag.direction = 0.0
    x0, y0 = ag.x, ag.y
    env.agent_step(ag.agent_id, move_distance=5.0, move_direction=0.0, turn_angle=math.pi / 2)
    report("move uses the PRE-turn heading, then turn applies",
           ag.x - x0 > 0.5 and abs(ag.y - y0) < 1e-6 and abs(ag.direction - math.pi / 2) < 1e-6)

    print("2) fastsim equivalence")
    from survivor import fastsim
    good, speedup = fastsim.verify(ticks=250, extra_agents=8, verbose=False)
    fastsim.revert()
    report("patched simulator observations identical", good, f"(x{speedup:.2f} faster)")

    print("3) policy contract + fruit-approach sanity")
    res = run_episode(1, Params(), horizon=40.0, fast=True, keep_paths=False)
    report("40 s episode runs, agents alive", res["alive_end"] > 0, f"score={res['score']:.1f} n_end={res['alive_end']}")
    c = Controller(Params())
    step = {"game_status": "ok", "score": 0, "sim_time": 1.0, "n_agents": 1, "agent_status": [{
        "agent_id": 0, "energy": 100, "biome": "forest", "age": 1, "speed": 10, "sprint_speed": 20,
        "hearing_radius": 50, "vision_angle": 1.05, "vision_range": 200, "max_energy": 500,
        "observations": [{"type": "Fruit", "distance": 80.0, "angle": 0.7}]}]}
    a = c.act(step)[0]
    report("comfortable agent does NOT rush a fruit of unknown age (ripeness discipline)", a["move_distance"] == 0.0)
    step["agent_status"][0]["energy"] = 30
    a = c.act(step)[0]
    report("starving agent goes for the fruit at once (relative angle)", abs(a["move_direction"] - 0.7) < 1e-9 and a["move_distance"] > 0)
    step["agent_status"][0]["energy"] = 100
    step["agent_status"][0]["energy"] = 300
    step["agent_status"][0]["observations"] = [{"type": "Predator", "distance": 60.0, "angle": 0.3, "rel_dir": 0.1}]
    a = c.act(step)[0]
    away = abs(((a["move_direction"] - (0.3 + math.pi)) + math.pi) % (2 * math.pi) - math.pi)
    report("flees away from a close predator, sprinting and facing it",
           away < 0.2 and a["move_distance"] > 10 and abs(a["turn_angle"] - 0.3) < 1e-9)
    c.act({"game_status": "ok", "score": 0, "sim_time": 0.0, "n_agents": 0, "agent_status": []})
    report("episode reset when sim_time restarts", c.episode >= 1)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


STATE_COL = {"flee": (255, 60, 60), "flee_sprint": (255, 0, 160), "forage": (255, 170, 0), "camp": (60, 200, 90),
             "seek_tree": (80, 140, 255), "explore": (190, 190, 190), "unstick": (170, 80, 220), "senescent": (140, 100, 70), "?": (255, 255, 255)}


def watch(params, seed, speed=2, record=None, video=None, every=10, headless=False, follow=None, horizon=3000.0, fast=False):
    """
    Live view of a run with the controller in charge.
      keys: SPACE pause | RIGHT step (paused) | UP/DOWN speed x2 | TAB follow next agent | F follow oldest |
            V toggle vision polygons | S screenshot | ESC quit
    Overlays: agent colour = policy state, bar = energy (red ring = below the 20% sprint floor), red X = predator with
    its 90-unit charge circle, side panel with live totals; a followed agent gets a yellow ring + detail panel.
    --headless --record DIR [--video out.mp4]: no window, frames every `every` ticks (mp4 needs `pip install imageio imageio-ffmpeg`).
    """
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    else:
        os.environ.pop("SDL_VIDEODRIVER", None)
    import pygame
    from src.core import SimulationCore
    if fast:
        from survivor import fastsim
        fastsim.apply()
    sim = SimulationCore(seed=seed)
    env = sim.env
    ctrl = Controller(_params(params), seed=seed)
    pygame.init()
    MAPW, PANEL = 1120, 340
    MAPH = int(MAPW * env.height / env.width)
    screen = pygame.display.set_mode((MAPW + PANEL, MAPH))
    pygame.display.set_caption(f"survivor watch - seed {seed}")
    mapsurf = pygame.Surface((MAPW, MAPH))
    zoom = MAPW / env.width
    font = pygame.font.SysFont("dejavusansmono,consolas,monospace", 14)
    small = pygame.font.SysFont("dejavusansmono,consolas,monospace", 12)
    if record:
        os.makedirs(record, exist_ok=True)
    frames = []
    counters = {"births": 0, "d_pred": 0, "d_young": 0, "d_old": 0}
    orig_kill = env.kill_agent

    def kill(agent):
        if agent in env.agents:
            if agent.energy <= 0:
                counters["d_old" if agent.age > agent.max_age else "d_young"] += 1
            else:
                counters["d_pred"] += 1
        orig_kill(agent)
    env.kill_agent = kill
    orig_spawn = env.spawn_agent

    def spawn(x=None, y=None, parent=None):
        ag = orig_spawn(x, y, parent)
        if parent is not None and ag is not None:
            counters["births"] += 1
        return ag
    env.spawn_agent = spawn

    clock = pygame.time.Clock()
    actions, paused, running, tick = [], False, True, 0
    show_vis = False
    followed = follow
    step_once = False
    last_state = None
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_RIGHT:
                    step_once = True
                elif e.key == pygame.K_UP:
                    speed = min(64, speed * 2)
                elif e.key == pygame.K_DOWN:
                    speed = max(1, speed // 2)
                elif e.key == pygame.K_v:
                    show_vis = not show_vis
                elif e.key == pygame.K_TAB and env.agents:
                    ids = sorted(a.agent_id for a in env.agents)
                    followed = ids[(ids.index(followed) + 1) % len(ids)] if followed in ids else ids[0]
                elif e.key == pygame.K_f and env.agents:
                    followed = max(env.agents, key=lambda a: a.age).agent_id
                elif e.key == pygame.K_s:
                    pygame.image.save(screen, f"watch_{seed}_{tick:06d}.png")
        if not paused or step_once:
            for _ in range(1 if step_once else speed):
                state = sim.step(actions)
                tick += 1
                last_state = state
                if state["num_agents"] == 0 or state["sim_time"] > horizon:
                    running = False
                    break
                obs = [o for o in state["observations"] if o]
                acts = ctrl.act({"game_status": "ok", "score": state["score"], "sim_time": state["sim_time"],
                                 "n_agents": state["num_agents"], "agent_status": obs})
                actions = [(a["agent_id"], _Act(a)) for a in acts]
            step_once = False
        # ---- draw
        env.draw(mapsurf)
        ag_by_id = {a.agent_id: a for a in env.agents}
        for a in env.agents:
            st = ctrl.last_state.get(a.agent_id, "?")
            col = STATE_COL.get(st, (255, 255, 255))
            cx, cy = int(a.x * zoom), int(a.y * zoom)
            pygame.draw.circle(mapsurf, col, (cx, cy), 6)
            pygame.draw.circle(mapsurf, (0, 0, 0), (cx, cy), 6, 1)
            frac = max(0.0, min(1.0, a.energy / a.max_energy))
            pygame.draw.rect(mapsurf, (40, 40, 40), (cx - 8, cy - 12, 16, 3))
            pygame.draw.rect(mapsurf, (int(255 * (1 - frac)), int(255 * frac), 0), (cx - 8, cy - 12, int(16 * frac), 3))
            if a.energy < a.max_energy / 5:
                pygame.draw.circle(mapsurf, (255, 0, 0), (cx, cy), 9, 1)      # cannot sprint
            if show_vis or a.agent_id == followed:
                poly = getattr(a, "_vision_poly", None)
                if poly:
                    pygame.draw.lines(mapsurf, (255, 255, 0) if a.agent_id == followed else (120, 120, 60), True,
                                      [(int(a.x * zoom), int(a.y * zoom))] + [(int(x * zoom), int(y * zoom)) for x, y in poly], 1)
        for pr in env.predators:
            px, py = int(pr.x * zoom), int(pr.y * zoom)
            pygame.draw.line(mapsurf, (255, 0, 0), (px - 6, py - 6), (px + 6, py + 6), 3)
            pygame.draw.line(mapsurf, (255, 0, 0), (px - 6, py + 6), (px + 6, py - 6), 3)
            pygame.draw.circle(mapsurf, (255, 120, 120), (px, py), int(90 * zoom), 1)
        if followed in ag_by_id:
            fa = ag_by_id[followed]
            pygame.draw.circle(mapsurf, (255, 255, 0), (int(fa.x * zoom), int(fa.y * zoom)), 13, 2)
        screen.fill((20, 20, 24))
        screen.blit(mapsurf, (0, 0))
        # ---- panel
        ag = env.agents
        k = max(1, len(ag))
        cnt = {}
        for a in ag:
            st = ctrl.last_state.get(a.agent_id, "?")
            cnt[st] = cnt.get(st, 0) + 1
        y = 8
        def line(txt, col=(230, 230, 230), f=font):
            nonlocal y
            screen.blit(f.render(txt, True, col), (MAPW + 10, y))
            y += 18
        sc = last_state["score"] if last_state else 0.0
        line(f"t={env.time:7.1f}s   score={sc:8.1f}", (255, 255, 120))
        line(f"agents {len(ag):3d}   predators {len(env.predators):2d}")
        line(f"trees {len(env.trees):3d} (mature {sum(1 for t_ in env.trees if t_.age >= 20)})  fruits {len(env.fruits)}")
        line(f"mean E {sum(a.energy for a in ag) / k:5.0f}   below-floor {sum(1 for a in ag if a.energy < a.max_energy / 5)}")
        line(f"mean speed {sum(a.speed for a in ag) / k:4.1f}  maxE {sum(a.max_energy for a in ag) / k:4.0f}")
        line(f"births {counters['births']}  deaths: pred {counters['d_pred']} young {counters['d_young']} old {counters['d_old']}")
        line(f"sim speed x{speed}{'  [PAUSED]' if paused else ''}", (150, 200, 255))
        y += 6
        for st, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            pygame.draw.circle(screen, STATE_COL.get(st, (255, 255, 255)), (MAPW + 16, y + 7), 5)
            screen.blit(font.render(f"{st:<12}{c:3d}", True, (220, 220, 220)), (MAPW + 28, y))
            y += 18
        y += 8
        if followed in ag_by_id:
            fa = ag_by_id[followed]
            mm = ctrl.mem.get(followed, {})
            line(f"FOLLOW agent {followed}", (255, 255, 0))
            line(f" state {ctrl.last_state.get(followed, '?')}   biome {env.biome_map[min(max(int(fa.x), 0), env.width - 1), min(max(int(fa.y), 0), env.height - 1)].type}", f=small)
            line(f" E {fa.energy:5.0f}/{fa.max_energy:4.0f}  age {fa.age:5.1f}/{fa.max_age:5.1f}", f=small)
            line(f" speed {fa.speed:4.1f} sprint {fa.sprint_speed:4.1f}", f=small)
            line(f" hear {fa.hearing_radius:4.0f} vis {fa.vision_radius:4.0f} cone {math.degrees(fa.cone_angle):3.0f}deg", f=small)
            o = (last_state["observations"][[x["agent_id"] for x in last_state["observations"] if x].index(followed)]
                 if last_state and followed in [x["agent_id"] for x in last_state["observations"] if x] else None)
            if o:
                c2 = {}
                for ob in o["observations"]:
                    c2[ob["type"]] = c2.get(ob["type"], 0) + 1
                line(" sees " + ",".join(f"{k[:3]}{v}" for k, v in c2.items()), f=small)
                pv = [ob for ob in o["observations"] if ob["type"] == "Predator"]
                if pv:
                    q = min(pv, key=lambda z: z["distance"])
                    line(f" nearest predator {q['distance']:.0f}u  bearing {math.degrees(q['angle']):+.0f}deg", (255, 120, 120), f=small)
            line(f" camp_t0 {mm.get('camp_t0')}  bans {len(mm.get('bans', []))}", f=small)
        elif ag:
            line("TAB = follow an agent, F = oldest", (150, 150, 150), f=small)
        pygame.display.flip()
        if record and tick % every == 0:
            pygame.image.save(screen, os.path.join(record, f"f_{tick:06d}.png"))
            frames.append(os.path.join(record, f"f_{tick:06d}.png"))
        if not headless:
            clock.tick(30)
    pygame.quit()
    if record and video and frames:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(video, [imageio.imread(f) for f in frames], fps=15)
            print("video written to", video)
        except Exception as ex:
            print("could not write video (pip install imageio imageio-ffmpeg):", ex, "- frames are in", record)
    print(f"finished at t={env.time:.1f}s score={last_state['score'] if last_state else 0:.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    e = sub.add_parser("eval")
    e.add_argument("--params", default=os.path.join(ROOT, "configs", "v1_default.json"))
    e.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    e.add_argument("--horizon", type=float, default=3000.0)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    e.add_argument("--out", default=None)
    e.add_argument("--no-fast", action="store_true")
    e.add_argument("--snaps", type=float, nargs="*", default=[100, 400, 900, 1500, 2100, 2800])
    c = sub.add_parser("compare")
    c.add_argument("--configs", nargs="+", required=True)
    c.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    c.add_argument("--horizon", type=float, default=3000.0)
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c.add_argument("--out", default=None)
    w = sub.add_parser("watch")
    w.add_argument("--params", default=os.path.join(ROOT, "configs", "v1_default.json"))
    w.add_argument("--seed", type=int, default=1)
    w.add_argument("--speed", type=int, default=2, help="sim ticks per frame")
    w.add_argument("--follow", type=int, default=None, help="agent id to follow")
    w.add_argument("--headless", action="store_true")
    w.add_argument("--record", default=None, help="directory for PNG frames")
    w.add_argument("--video", default=None, help="mp4 path (needs imageio-ffmpeg)")
    w.add_argument("--every", type=int, default=10, help="save a frame every N ticks when recording")
    w.add_argument("--horizon", type=float, default=3000.0)
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(0 if check() else 1)
    if a.cmd == "watch":
        watch(a.params, a.seed, a.speed, a.record, a.video, a.every, a.headless, a.follow, a.horizon)
        return
    if a.cmd == "compare":
        compare(a.configs, a.seeds, a.horizon, a.workers, a.out)
        return
    t0 = time.time()
    res = evaluate(a.params, a.seeds, a.horizon, a.workers, not a.no_fast, a.out, a.snaps, verbose=a.workers == 1)
    print(f"{'seed':>6} {'score':>9} {'sim_t':>8} {'alive':>5} {'peak':>5} {'births':>6} {'d_pred':>6} {'d_young':>7} {'d_old':>6} {'fruit':>6} {'wall_s':>7}")
    for r in res:
        d = r["deaths"]
        print(f"{r['seed']:>6} {r['score']:>9.1f} {r['sim_time']:>8.1f} {r['alive_end']:>5} {r['peak_pop']:>5} {r['births']:>6} "
              f"{d['predator']:>6} {d['starved_young']:>7} {d['starved_old']:>6} {r['fruits_eaten']:>6} {r['timing']['wall_s']:>7.0f}")
    s = summarize(res)
    print(f"MEAN {s['mean_score']:.1f}  MIN {s['min_score']:.1f}  MAX {s['max_score']:.1f}  full-survival {s['full_survival']}/{s['n_seeds']}  "
          f"(total wall {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
