"""
survivor.predtest - controlled predator-evasion trials (seconds per trial, no full run needed).

One (or a few) camping agent(s) at a tree, one hungry predator released at a random bearing/distance,
count how long the agents survive. Use it to tune the predator-response parameters quickly:

  python -m survivor.predtest --params configs/v1_default.json --trials 40 --set pred_face=0 pred_side_w=0.3
"""
import argparse, math, os, random, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dataclasses import asdict
from survivor.policy import Controller, Params
from survivor.runner import _Act


_SIMS = {}


def _get_sim(map_seed, fast=True):
    from src.core import SimulationCore
    if fast:
        from survivor import fastsim
        fastsim.apply()
    if map_seed not in _SIMS:
        _SIMS[map_seed] = SimulationCore(seed=map_seed)
    return _SIMS[map_seed]


def trial(params, seed, n_agents=1, seconds=40.0, dist=(150, 320), fast=True, mode="camp", energy=300.0, speed=None,
          n_pred=1, aimed=True, t_off=0.0):
    from src.elements.tree import Tree
    sim = _get_sim(1 + seed % 3, fast)
    env = sim.env
    env.time = 0.0; env.score = 0.0; env._next_agent_id = 0
    env.rng.seed(seed * 131 + 5)          # deterministic trials
    rng = random.Random(seed * 7919 + 1)
    env.agents.clear(); env.agents_dict.clear(); env.predators.clear(); env.trees.clear(); env.fruits.clear()
    while True:
        cx, cy = rng.uniform(500, 1100), rng.uniform(400, 800)
        if env._is_position_free(cx - 60, cy - 60, 120, 120):
            break
    if mode == "camp":
        tr = Tree(cx, cy); tr.age = 30; tr.radius = 15
        env.trees.append(tr)
    env._update_tree_grid(); env._update_fruit_grid()
    for i in range(n_agents):
        ag = env.spawn_agent(x=cx + rng.uniform(-25, 25), y=cy + rng.uniform(-25, 25))
        ag.energy = energy
        if speed:
            ag.speed = speed; ag.sprint_speed = max(20.0, speed + 2.0)   # sprint trait stays ~20 unless it mutates separately
    env.agent_observations = {}
    n_ok = 0
    for k in range(n_pred):
        ang = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(*dist)
        pr = env.spawn_predator(x=cx + d * math.cos(ang), y=cy + d * math.sin(ang))
        if pr is None:
            continue
        n_ok += 1
        pr.resting = False; pr.energy = 150
        # aimed: heading roughly at the camp; unaimed: random heading (tests detection + sidestepping)
        pr.direction = (ang + math.pi + rng.uniform(-0.6, 0.6)) if aimed else rng.uniform(-math.pi, math.pi)
    if n_ok == 0:
        return None
    env.agents_dict = {a.agent_id: a for a in env.agents}
    pp = Params(**asdict(params)); pp.spawn_thr = 1e9; pp.dbed_min_energy = 1e9; pp.n_min = 0
    ctrl = Controller(pp, seed=seed)
    actions = []
    first_death = None
    kills = 0
    spent = 0.0
    agent_ticks = 0
    while env.time < seconds:
        st = sim.step(actions)
        n = st["num_agents"]
        kills = n_agents - n
        if kills and first_death is None:
            first_death = env.time
        for a in env.agents:
            if a.energy < energy:
                spent += energy - a.energy          # energy burnt this tick (moving, sprinting, scanning, living)
            a.energy = max(a.energy, energy)        # remove starvation from the picture
        agent_ticks += n
        if n == 0:
            break
        obs = [o for o in st["observations"] if o]
        acts = ctrl.act({"game_status": "ok", "score": 0, "sim_time": st["sim_time"] + t_off, "n_agents": n, "agent_status": obs})
        actions = [(a["agent_id"], _Act(a)) for a in acts]
    return {"kills": kills, "first_death": first_death, "n": n_agents, "spent": spent, "agent_s": agent_ticks / 10.0}


def run(params, trials, n_agents, seconds, energy=300.0, speed=None):
    res = [trial(params, s, n_agents, seconds, energy=energy, speed=speed) for s in range(1, trials + 1)]
    res = [r for r in res if r]
    tot = sum(r["n"] for r in res)
    kills = sum(r["kills"] for r in res)
    return kills / max(1, tot), sum(1 for r in res if r["kills"] > 0) / max(1, len(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(ROOT, "configs", "v1_default.json"))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--agents", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--energy", type=float, default=300.0)
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    p = Params.load(a.params)
    for kv in a.set:
        k, v = kv.split("="); setattr(p, k, float(v))
    frac, any_ = run(p, a.trials, a.agents, a.seconds, a.energy, a.speed)
    print(f"kill fraction {frac:.2f}  (trials with >=1 death {any_:.2f})  set={a.set}")
