"""
survivor.optimize_arena - tune policy parameters on a micro-experiment arena (seconds per candidate, common random numbers).

  python -m survivor.optimize_arena --exp harvest --base configs/v7a_economy.json --out runs/oa_harvest --pop 24 --gens 15 --trials 4 --workers 15

Fitness = mean of the arena's primary metric over all scenario cells x trials, with the SAME scenarios/seeds for every candidate.
After the search, the best candidate and the base are re-run on FRESH seeds and compared PAIRED (bootstrap CI): only that verdict counts.
The result is a config file; its real test is the transfer check (`micro run econ` and `runner compare` on many seeds).
"""
import argparse
import json
import os
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
from survivor import micro  # noqa: E402

ECON_FREE = ["ripe_age", "ripe_unknown", "urgent_abs", "camp_patience", "patience_burn", "camp_scan", "vig_k", "camp_radius",
             "patrol", "patrol_wait", "patrol_ban", "patrol_max", "fruit_max_dist", "eat_full_frac", "crowd_max", "tree_ban_s"]


def _job(job):
    exp_name, pd, ci, k, seed_off = job
    exp = micro.ARENA[exp_name]
    cell = exp.cells()[ci]
    spec = exp.spec(cell, k)
    arm = micro.ParamArm("cand", cfg=None, over=pd)
    raw = micro.run_arena(spec, arm, 1000 + 97 * ci + k + seed_off, trace=False)
    m, _tags = exp.metrics(raw, cell, spec)
    v = m[exp.primary]
    return ci, k, (v if v == v else 0.0)


def evaluate(pool, exp_name, cands, trials, seed_off):
    exp = micro.ARENA[exp_name]
    ncell = len(exp.cells())
    jobs, owner = [], []
    for i, p in enumerate(cands):
        pd = asdict(p)
        for ci in range(ncell):
            for k in range(trials):
                jobs.append((exp_name, pd, ci, k, seed_off))
                owner.append(i)
    res = pool.map(_job, jobs, chunksize=2) if pool else [_job(j) for j in jobs]
    vals = [dict() for _ in cands]
    for i, (ci, k, v) in zip(owner, res):
        vals[i][(ci, k)] = v
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="harvest")
    ap.add_argument("--base", default=os.path.join(ROOT, "configs", "v5a_ripe_boot.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--free", nargs="*", default=ECON_FREE)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--holdout-trials", type=int, default=12)
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
    exp = micro.ARENA[a.exp]
    print(f"arena optimizer {'CMA-ES' if use_cma else 'CEM'} exp={a.exp} primary={exp.primary} free={len(a.free)} pop={a.pop} "
          f"cells={len(exp.cells())} trials={a.trials} workers={a.workers}", flush=True)
    pool = Pool(a.workers) if a.workers > 1 else None
    best = None
    try:
        b0 = evaluate(pool, a.exp, [base], a.trials, 0)[0]
        f0 = sum(b0.values()) / len(b0)
        best = {"fit": f0, "params": asdict(base)}
        print(f"base fitness {f0:.4f}", flush=True)
        for g in range(a.gens):
            t0 = time.time()
            xs = es.ask()
            cands = [vec_to_params(x, base, a.free) for x in xs]
            vals = evaluate(pool, a.exp, cands, a.trials, 0)
            fits = [sum(v.values()) / len(v) for v in vals]
            i = max(range(len(fits)), key=lambda j: fits[j])
            if fits[i] > best["fit"]:
                best = {"fit": fits[i], "params": asdict(cands[i])}
                cands[i].save(os.path.join(a.out, "best.json"), fitness=fits[i], gen=g, exp=a.exp)
            es.tell(xs, [-f for f in fits]) if use_cma else es.tell(xs, fits)
            print(f"gen {g:2d} gen_best={fits[i]:.4f} best_so_far={best['fit']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        # hold-out verification on FRESH seeds, paired
        cand = Params(**best["params"])
        res = evaluate(pool, a.exp, [base, cand], a.holdout_trials, 100000)
    finally:
        if pool:
            pool.close()
    keys = sorted(res[0].keys())
    stats = micro.paired_stats([res[1][k] for k in keys], [res[0][k] for k in keys])
    v = micro.verdict(stats, exp.min_effect)
    print(f"\nHOLD-OUT ({len(keys)} paired trials, fresh seeds): base {sum(res[0].values()) / len(keys):.4f} -> best {sum(res[1].values()) / len(keys):.4f}; "
          f"paired delta {stats['mean']:+.4f} [{stats['lo']:+.4f}, {stats['hi']:+.4f}] wins/losses {stats['wins']}/{stats['losses']} p={stats['p']:.3f} -> {v}")
    cand.save(os.path.join(a.out, "best.json"), holdout_delta=stats["mean"], verdict=v, exp=a.exp)
    diff = {k: round(x, 3) for k, x in best["params"].items() if abs(x - getattr(base, k)) > 1e-9}
    print("changed vs base:", json.dumps(diff))
    print("next: transfer check ->  python -m survivor.micro run econ --configs <base> " + os.path.join(a.out, "best.json") + " --seeds 12 --no-predators")


if __name__ == "__main__":
    main()
