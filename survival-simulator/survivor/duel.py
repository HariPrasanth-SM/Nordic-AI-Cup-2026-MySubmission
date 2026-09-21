"""
survivor.duel - the 2-minute arena GATE for encounter controllers.

  python -m survivor.duel --n 200 --workers 15 --flee-rows runs/lab1/flee_rows.csv
  python -m survivor.duel --arms heuristic mpc mpc_emap "mine:mpc_flee=1,mpc_h=32,mpc_range=250"

One agent + one aimed predator in the real simulator arena (real obstacles). Scenarios (first-sight distance, energy fraction, walking speed) are SAMPLED FROM THE
LOGGED ENCOUNTERS (`flee_rows.csv`; built-in marginals from lab1 if omitted), so the arena reproduces where deaths really happen. Metric: predator kills
(starvation in the finite arena is not counted), paired by scenario; sign test on discordant duels. History: it must be read with care - arena wins have failed
to transfer before (flee optimizer); large paired effects in the HARD regime are the ones worth a full run.
"""
import argparse
import csv
import os
import random
import sys
from math import comb

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ARMS = {
    "heuristic": {},
    "mpc": {"mpc_flee": 1},
    "mpc_emap": {"mpc_flee": 1, "emap": 1},
    "mpc_far": {"mpc_flee": 1, "mpc_range": 300},
}
D_BINS = [((25, 40), 3592), ((40, 60), 17532), ((60, 90), 13815), ((90, 120), 10050), ((120, 160), 8904), ((160, 240), 39918)]
E_BINS = [((0.02, 0.1), 13831), ((0.1, 0.2), 21965), ((0.2, 0.3), 14453), ((0.3, 0.45), 14505), ((0.45, 0.6), 10147), ((0.6, 1.0), 18880)]
S_BINS = [((8, 11), 32768), ((11, 13), 12588), ((13, 15), 18831), ((15, 17), 7482), ((17, 20), 22142)]


def _pick(rng, bins):
    r = rng.random() * sum(w for _, w in bins)
    acc = 0
    for (lo, hi), w in bins:
        acc += w
        if r <= acc:
            return rng.uniform(lo, hi)
    return rng.uniform(*bins[-1][0])


def scenarios(n, flee_rows, seed0):
    rng = random.Random(7)
    rows = []
    if flee_rows and os.path.exists(flee_rows):
        with open(flee_rows) as f:
            for r in csv.DictReader(f):
                rows.append((float(r["d0"]), float(r["e_frac"]), float(r["speed"])))
    out = []
    for i in range(n):
        if rows:
            d, e, sp = rng.choice(rows)
            out.append((min(max(d, 25.0), 240.0), min(max(e, 0.02), 1.0), min(max(sp, 8.0), 20.0), seed0 + i))
        else:
            out.append((_pick(rng, D_BINS), _pick(rng, E_BINS), _pick(rng, S_BINS), seed0 + i))
    return out


def _job(job):
    cfg, name, over, sc = job
    from survivor import micro as M
    D, Ef, sp, seed = sc
    ag = M._agents(1, Ef * 350.0, float(sp), "a")
    ag[0]["sprint"] = 20.0
    spec = {"seconds": 20.0, "tree": {"age": 30}, "t_off": 900.0, "agents": ag, "predators": [{"dist": (D, D), "aimed": True, "E": 150}]}
    raw = M.run_arena(spec, M.make_arm({"name": name, "cfg": cfg, "over": over}), seed, trace=False)
    return name, seed, any(d[2] == "predator" for d in raw["dead"]), raw["spent"].get(raw["init"][0], 0.0)


def gate(base, arms, n, workers, flee_rows=None, seed0=5000, pool=None, log=print):
    """run the duel gate; prints/logs a markdown table and returns it as a string"""
    sc = scenarios(n, flee_rows, seed0)
    jobs = [(base, nm, ov, s) for nm, ov in arms.items() for s in sc]
    if pool is not None:
        res = pool.map(_job, jobs, chunksize=4)
    elif workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pl:
            res = pl.map(_job, jobs, chunksize=4)
    else:
        res = [_job(j) for j in jobs]
    by = {nm: {} for nm in arms}
    for nm, seed, k, sp_ in res:
        by[nm][seed] = (k, sp_)
    names = list(arms)
    ref = names[0]
    lines = [f"{len(sc)} duels per arm from {'logged encounters ' + flee_rows if flee_rows else 'built-in lab1 marginals'}; base {os.path.basename(base)}"]

    def rate(nm, pred):
        ks = [s for s in sc if pred(s)]
        return (sum(by[nm][s[3]][0] for s in ks) / len(ks), len(ks)) if ks else (float("nan"), 0)
    regimes = [("all", lambda s: True), ("first sight < 90", lambda s: s[0] < 90), ("first sight >= 90", lambda s: s[0] >= 90),
               ("energy < 0.2", lambda s: s[1] < 0.2), ("energy >= 0.2", lambda s: s[1] >= 0.2)]
    lines.append("| arm | " + " | ".join(r[0] for r in regimes) + " | energy spent / duel | paired vs " + ref + " (fewer/more kills, sign-test p) |")
    lines.append("|---|" + "---|" * (len(regimes) + 2))
    for nm in names:
        cells = [f"{100 * rate(nm, f)[0]:.0f}% (n={rate(nm, f)[1]})" for _, f in regimes]
        en = sum(by[nm][s[3]][1] for s in sc) / len(sc)
        if nm == ref:
            pv = "-"
        else:
            w = sum(1 for s in sc if by[nm][s[3]][0] < by[ref][s[3]][0])
            l = sum(1 for s in sc if by[nm][s[3]][0] > by[ref][s[3]][0])
            m = w + l
            pp = min(1.0, 2 * sum(comb(m, i) for i in range(0, min(w, l) + 1)) / 2 ** m) if m else 1.0
            pv = f"{w}/{l}, p={pp:.4f}"
        lines.append(f"| {nm} | " + " | ".join(cells) + f" | {en:.0f} | {pv} |")
    for ln in lines:
        log(ln)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--base", default=os.path.join(ROOT, "configs", "v10_stack.json"))
    ap.add_argument("--flee-rows", default=None)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--arms", nargs="*", default=["heuristic", "mpc"])
    a = ap.parse_args()
    arms = {}
    for x in a.arms:
        if ":" in x:
            nm, kv = x.split(":", 1)
            arms[nm] = {k: float(v) for k, v in (p.split("=") for p in kv.split(","))}
        else:
            arms[x] = ARMS[x]
    gate(a.base, arms, a.n, a.workers, a.flee_rows, a.seed0)


if __name__ == "__main__":
    main()
