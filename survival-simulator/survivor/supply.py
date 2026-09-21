"""
survivor.supply - the food supply of the world with NO agents and NO predators: why 3000 s is out of reach.

  python -m survivor.supply --seeds 1 2 3 4 5 6          # ~1-2 min per seed on a normal core
Prints, per checkpoint: mature trees, fruits/s, and how many agents that supply can feed at your capture rate.
Defaults are the measured colony economics: 45 energy per eaten fruit, 6 energy per agent-second (movement 2-3 + living 1 + aging 2 + births 1.2).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def one(seed, horizon):
    from src.core import SimulationCore
    sim = SimulationCore(seed=seed)
    env = sim.env
    env.agents.clear(); env.agents_dict = {}
    env.spawn_predator = lambda *a, **k: None
    rows, n = [], 0
    while env.time < horizon:
        env.non_agent_step(0.1)
        n += 1
        if n % 1000 == 0:
            mature = [t for t in env.trees if t.age >= 20]

            def bio(t):
                return env.biome_map[min(max(int(t.x), 0), env.width - 1), min(max(int(t.y), 0), env.height - 1)]
            rows.append({"t": round(env.time), "mature": len(mature), "fruit_per_s": sum(bio(t).fruit_spawn_rate for t in mature),
                         "good": sum(bio(t).fruit_spawn_rate for t in mature if bio(t).type in ("forest", "grassland"))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--horizon", type=float, default=3000.0)
    ap.add_argument("--e-fruit", type=float, default=45.0)
    ap.add_argument("--e-agent", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    allr = []
    for s in a.seeds:
        t0 = time.time()
        allr.append(one(s, a.horizon))
        print(f"seed {s} done in {time.time() - t0:.0f}s", flush=True)
    if a.out:
        json.dump(allr, open(a.out, "w"))
    print(f"\n{'t':>5} | mature trees | fruits/s | fruits/s forest+grass | agents fed at 100% / 50% capture (E/fruit {a.e_fruit:.0f}, E/agent-s {a.e_agent:.0f})")
    for i in range(2, len(allr[0]), 3):
        m = sum(r[i]["mature"] for r in allr) / len(allr)
        f = sum(r[i]["fruit_per_s"] for r in allr) / len(allr)
        g = sum(r[i]["good"] for r in allr) / len(allr)
        print(f"{allr[0][i]['t']:>5} | {m:12.1f} | {f:8.2f} | {g:21.2f} | {f * a.e_fruit / a.e_agent:6.1f} / {f * a.e_fruit / a.e_agent * 0.5:5.1f}")


if __name__ == "__main__":
    main()
