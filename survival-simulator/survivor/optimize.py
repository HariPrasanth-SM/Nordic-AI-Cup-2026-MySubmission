"""
survivor.optimize - black-box policy search (CMA-ES, or a built-in CEM fallback) over the controller
parameters, using full simulations as the fitness signal. This is the "learning" step of the solution.

  python -m survivor.optimize --base configs/v1_default.json --out runs/opt1 \
        --pop 12 --gens 6 --seeds 101 102 103 104 --horizon 3000 --workers 12

Fitness of a candidate = (1-alpha)*mean(score) + alpha*min(score) (alpha=0.25) over the training seeds (robustness matters:
the evaluation averages only 3 runs). Every generation is appended to <out>/log.jsonl and the best
parameter set so far is written to <out>/best.json; the run is resumable (--resume).
Validate the final candidate on HELD-OUT seeds:  python -m survivor.runner eval --params runs/opt1/best.json --seeds 201 202 203 204 205 206
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor.policy import Params, SPACE, DEFAULT_FREE, vec_to_params, params_to_vec  # noqa: E402
from survivor.runner import _worker  # noqa: E402


ALPHA = 0.25


def fitness(scores, alpha=None):
    alpha = ALPHA if alpha is None else alpha
    m = sum(scores) / len(scores)
    return (1 - alpha) * m + alpha * min(scores)


def evaluate_population(pool, cands, seeds, horizon):
    jobs, idx = [], []
    for ci, p in enumerate(cands):
        pd = asdict(p)
        for s in seeds:
            jobs.append((s, pd, horizon, True, (), None, False))
            idx.append(ci)
    # longest jobs first is unknown; imap keeps order, chunksize 1 keeps workers busy
    res = pool.map(_worker, jobs, chunksize=1) if pool else [_worker(j) for j in jobs]
    per = [[] for _ in cands]
    for ci, r in zip(idx, res):
        per[ci].append(r)
    return per


class CEM:
    """diagonal cross-entropy method in [0,1]^k (fallback when the `cma` package is missing)"""

    def __init__(self, x0, sigma0, pop, seed):
        self.mu = list(x0)
        self.sd = [sigma0] * len(x0)
        self.pop = pop
        self.rng = random.Random(seed)
        self.k = len(x0)
        self.best = None

    def ask(self):
        return [[min(1, max(0, self.rng.gauss(m, s))) for m, s in zip(self.mu, self.sd)] for _ in range(self.pop)]

    def tell(self, xs, fs):
        order = sorted(range(len(xs)), key=lambda i: -fs[i])
        el = [xs[i] for i in order[: max(2, len(xs) // 3)]]
        for j in range(self.k):
            col = [e[j] for e in el]
            m = sum(col) / len(col)
            v = sum((c - m) ** 2 for c in col) / len(col)
            self.mu[j] = 0.7 * self.mu[j] + 0.3 * m
            self.sd[j] = max(0.02, 0.7 * self.sd[j] + 0.3 * math.sqrt(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "configs", "v1_default.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--free", nargs="*", default=DEFAULT_FREE)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104])
    ap.add_argument("--horizon", type=float, default=3000.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.25, help="fitness = (1-alpha)*mean + alpha*min over seeds")
    a = ap.parse_args()
    global ALPHA
    ALPHA = a.alpha

    os.makedirs(a.out, exist_ok=True)
    for n in a.free:
        assert n in SPACE, f"unknown parameter {n}"
    base = Params.load(a.base)
    x0 = params_to_vec(base, a.free)
    x0 = [min(1, max(0, v)) for v in x0]

    try:
        import cma
        use_cma = True
    except ImportError:
        cma = None
        use_cma = False
    print(f"optimizer: {'CMA-ES' if use_cma else 'built-in CEM (pip install cma for CMA-ES)'}  free params={len(a.free)}  "
          f"pop={a.pop} seeds={a.seeds} horizon={a.horizon} workers={a.workers}", flush=True)

    state_path = os.path.join(a.out, "state.pkl")
    gen0, best = 0, {"fit": -1e9}
    if a.resume and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            es, gen0, best = pickle.load(f)
        print(f"resumed at generation {gen0}", flush=True)
    else:
        es = (cma.CMAEvolutionStrategy(x0, a.sigma, {"popsize": a.pop, "bounds": [0, 1], "seed": a.seed + 1,
                                                       "verbose": -9}) if use_cma else CEM(x0, a.sigma, a.pop, a.seed))
    pool = Pool(a.workers) if a.workers > 1 else None
    t_start = time.time()
    try:
        # generation "-1": the starting point itself (so we always know the baseline under identical seeds)
        if gen0 == 0 and not a.resume:
            per = evaluate_population(pool, [base], a.seeds, a.horizon)
            sc = [r["score"] for r in per[0]]
            f0 = fitness(sc)
            best = {"fit": f0, "params": asdict(base), "gen": -1, "scores": sc}
            base.save(os.path.join(a.out, "best.json"), fitness=f0, scores=sc, gen=-1)
            with open(os.path.join(a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps({"gen": -1, "best_fit": f0, "cands": [{"fit": f0, "scores": sc}]}) + "\n")
            print(f"baseline  fit={f0:.1f}  scores={[round(s) for s in sc]}", flush=True)

        for g in range(gen0, a.gens):
            if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
                print("time budget reached", flush=True)
                break
            t0 = time.time()
            xs = es.ask()
            cands = [vec_to_params(x, base, a.free) for x in xs]
            per = evaluate_population(pool, cands, a.seeds, a.horizon)
            fits, rows = [], []
            for x, p, rs in zip(xs, cands, per):
                sc = [r["score"] for r in rs]
                fi = fitness(sc)
                fits.append(fi)
                rows.append({"fit": fi, "scores": [round(s, 1) for s in sc],
                             "sim_t": [round(r["sim_time"]) for r in rs], "x": [round(v, 4) for v in x]})
                if fi > best["fit"]:
                    best = {"fit": fi, "params": asdict(p), "gen": g, "scores": sc}
                    p.save(os.path.join(a.out, "best.json"), fitness=fi, scores=sc, gen=g)
            if use_cma:
                es.tell(xs, [-f for f in fits])
            else:
                es.tell(xs, fits)
            with open(os.path.join(a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps({"gen": g, "best_fit": best["fit"], "gen_best": max(fits), "gen_mean": sum(fits) / len(fits),
                                    "wall_s": time.time() - t0, "cands": rows}) + "\n")
            with open(state_path, "wb") as f:
                pickle.dump((es, g + 1, best), f)
            print(f"gen {g:2d}  gen_best={max(fits):8.1f}  gen_mean={sum(fits) / len(fits):8.1f}  "
                  f"best_so_far={best['fit']:8.1f} (gen {best['gen']})  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        if pool:
            pool.close()
    print("best parameters written to", os.path.join(a.out, "best.json"))
    diff = {k: round(v, 3) for k, v in best["params"].items() if abs(v - getattr(base, k)) > 1e-9}
    print("changed vs base:", json.dumps(diff))


if __name__ == "__main__":
    main()
