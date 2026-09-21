"""
survivor.optimize_flee - tune the predator-response / detection parameters directly on controlled evasion trials.

Why: predators end every run (the last kills in every extinct run are predator kills), and a full 3000 s run is a very
noisy, slow signal for that. A trial is a few seconds: a camp of agents at a tree, 1-2 hungry predators released
around them (aimed at the camp or wandering randomly), scenarios over energy x walking speed. Fitness = mean kill
fraction + a small price for the energy burnt (sprinting, fleeing, scanning are not free).

  python -m survivor.optimize_flee --base configs/v5a_ripe_boot.json --out runs/flee1 --pop 24 --gens 20 --workers 15
  -> runs/flee1/best.json  (all other parameters unchanged). Then verify on full runs:
  python -m survivor.runner compare --configs configs/v5a_ripe_boot.json runs/flee1/best.json --seeds 1 2 3 4 5 6 7 8 9 10 11 12 --horizon 3000 --workers 15
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from survivor.policy import Params, SPACE, vec_to_params, params_to_vec  # noqa: E402
from survivor.optimize import CEM  # noqa: E402

FLEE_FREE = ["pred_react_dist", "pred_sprint_dist", "pred_notice_margin", "pred_side_w", "pred_cpa", "pred_face",
             "flee_hold", "sprint_min_frac", "flee_cap", "sprint_below", "camp_scan", "vig_k", "wall_margin"]

# scenario grid: (energy, walking speed, aimed at camp?, number of predators)
SCENARIOS = [(E, S, A, P) for E in (70, 200) for S in (10, 14, 18) for A in (True, False) for P in (1, 2)]


def _job(job):
    from survivor.predtest import trial
    pd, sc_idx, k, seconds, t_off = job
    E, S, A, P = SCENARIOS[sc_idx]
    p = Params(**pd)
    r = trial(p, seed=1000 + 37 * sc_idx + k, n_agents=3, seconds=seconds, energy=float(E), speed=float(S),
              n_pred=P, aimed=A, t_off=t_off, dist=(120, 300))
    if r is None:
        return sc_idx, 0.0, 0.0
    return sc_idx, r["kills"] / r["n"], r["spent"] / max(r["agent_s"], 1e-9)


def score(pool, cands, trials, seconds, t_off, lam):
    jobs, owner = [], []
    for ci, p in enumerate(cands):
        pd = asdict(p)
        for si in range(len(SCENARIOS)):
            for k in range(trials):
                jobs.append((pd, si, k, seconds, t_off))
                owner.append(ci)
    res = pool.map(_job, jobs, chunksize=4) if pool else [_job(j) for j in jobs]
    out = [[0.0, 0.0, 0] for _ in cands]
    per = [dict() for _ in cands]
    for ci, (si, kill, spent) in zip(owner, res):
        out[ci][0] += kill; out[ci][1] += spent; out[ci][2] += 1
        d = per[ci].setdefault(si, [0.0, 0])
        d[0] += kill; d[1] += 1
    fits = []
    for ci, (k_, s_, n_) in enumerate(out):
        kill = k_ / n_
        energy = s_ / n_
        fits.append((kill + lam * max(0.0, energy - 1.0), kill, energy, {SCENARIOS[si]: v[0] / v[1] for si, v in per[ci].items()}))
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "configs", "v5a_ripe_boot.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--free", nargs="*", default=FLEE_FREE)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--trials", type=int, default=2, help="trials per scenario per candidate (24 scenarios)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--t-off", type=float, default=900.0, help="controller clock offset (time-dependent vigilance)")
    ap.add_argument("--lam", type=float, default=0.01, help="price of 1 energy/s above the idle 1.0 (kill-fraction units)")
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = Params.load(a.base)
    x0 = [min(1, max(0, v)) for v in params_to_vec(base, a.free)]
    try:
        import cma
        es = cma.CMAEvolutionStrategy(x0, a.sigma, {"popsize": a.pop, "bounds": [0, 1], "seed": a.seed + 1, "verbose": -9})
        use_cma = True
    except ImportError:
        es = CEM(x0, a.sigma, a.pop, a.seed)
        use_cma = False
    print(f"optimizer {'CMA-ES' if use_cma else 'CEM'}  free={len(a.free)}  pop={a.pop}  scenarios={len(SCENARIOS)} x {a.trials} trials  workers={a.workers}", flush=True)
    pool = Pool(a.workers) if a.workers > 1 else None
    try:
        b = score(pool, [base], max(a.trials, 6), a.seconds, a.t_off, a.lam)[0]
        best = {"fit": b[0], "params": asdict(base)}
        print(f"baseline  fit={b[0]:.4f}  kill fraction={b[1]:.3f}  energy/agent-s={b[2]:.2f}", flush=True)
        base.save(os.path.join(a.out, "best.json"), fitness=b[0], kill=b[1])
        for g in range(a.gens):
            t0 = time.time()
            xs = es.ask()
            cands = [vec_to_params(x, base, a.free) for x in xs]
            res = score(pool, cands, a.trials, a.seconds, a.t_off, a.lam)
            fits = [r[0] for r in res]
            i = min(range(len(fits)), key=lambda j: fits[j])
            if fits[i] < best["fit"]:
                best = {"fit": fits[i], "params": asdict(cands[i])}
                cands[i].save(os.path.join(a.out, "best.json"), fitness=fits[i], kill=res[i][1], energy=res[i][2], gen=g)
            es.tell(xs, fits) if use_cma else es.tell(xs, [-f for f in fits])
            with open(os.path.join(a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps({"gen": g, "best": best["fit"], "gen_best": fits[i], "kill": res[i][1], "energy": res[i][2]}) + "\n")
            print(f"gen {g:2d}  gen_best={fits[i]:.4f} (kill {res[i][1]:.3f}, energy {res[i][2]:.2f})  best_so_far={best['fit']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        if pool:
            pool.close()
    # noise-free re-evaluation of baseline vs best with many more trials
    pool = Pool(a.workers) if a.workers > 1 else None
    final = score(pool, [base, Params(**best["params"])], max(a.trials * 4, 12), a.seconds, a.t_off, a.lam)
    if pool:
        pool.close()
    print(f"\nre-evaluated with {max(a.trials * 4, 12)} trials/scenario:")
    print(f"  baseline kill fraction {final[0][1]:.3f} energy {final[0][2]:.2f}   ->   best kill fraction {final[1][1]:.3f} energy {final[1][2]:.2f}")
    for sc in SCENARIOS:
        print(f"  E={sc[0]:>3} speed={sc[1]:>2} {'aimed  ' if sc[2] else 'wander '} preds={sc[3]}:  {final[0][3][sc]:.2f} -> {final[1][3][sc]:.2f}")
    diff = {k: round(v, 3) for k, v in best["params"].items() if abs(v - getattr(base, k)) > 1e-9}
    print("changed:", json.dumps(diff))


if __name__ == "__main__":
    main()
