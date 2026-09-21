"""
survivor.pipeline - the complete training + selection pipeline (budgeted, cached, resumable, paired).

  python -m survivor.pipeline run --out runs/pipe1 --budget-hours 2 --workers 15            # the 2-hour job
  python -m survivor.pipeline run --out runs/pipe1 --budget-hours 1 --workers 15 --resume   # continue an interrupted run (adds budget)
  python -m survivor.pipeline status runs/pipe1                                              # progress from another terminal
  python -m survivor.pipeline report runs/pipe1                                              # rewrite PIPELINE_REPORT.md

WHAT IT DOES (every number below is measured on YOUR machine with paired seeds; nothing is adopted on a hunch)
  S0 preflight   self-tests (runner check) and a timing calibration: the cost of one full run decides all stage sizes.
  S1 screening   every candidate BEHAVIOUR (rest, senescence, ripeness, patrol, newborn feeding, quiet spawning, doom spawning,
                 shared alarm, colony size, scanning, sprint policy) is run as "baseline + feature" on the SAME seeds as the baseline.
                 Racing: clearly harmful features are dropped early; the rest get more seeds.
  S2 combination greedy forward selection: add features best-first, keep an addition only if it improves the current best config (paired).
  S3 tuning      CMA-ES (CEM fallback) over ~12 continuous parameters on the combined config. Fitness = paired score difference
                 vs the reference on a ROTATING slice of seeds (common random numbers, no single-seed overfitting).
  S4 final       baseline, the combined config and the tuned finalists run on FRESH seeds the search never saw; the winner must beat
                 the baseline in a paired comparison, otherwise the baseline is kept. Output: final_config.json + PIPELINE_REPORT.md.

Everything runs the real simulator (full 3000 s runs by default), in-process, exactly like `runner eval`.
Cache: <out>/cache.jsonl (config hash x seed) - re-running or resuming never repeats a run.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import statistics as st
from collections import OrderedDict
from dataclasses import asdict
from multiprocessing import Pool

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker: no oversubscription
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor.policy import Params, SPACE, vec_to_params, params_to_vec  # noqa: E402
from survivor.micro import paired_stats, verdict  # noqa: E402

# ------------------------------------------------------------------------------------------------ candidates
FEATURES = OrderedDict([
    ("rest",          ({"rest_frac": 0.7}, "comfortable agents rest instead of wandering (arena: net income -1.96 -> +0.55)")),
    ("senescence",    ({"senescence": 1.0}, "aging agents stop eating/moving and convert energy into children (aging = 37-46% of income)")),
    ("ripe20",        ({"ripe_age": 20.0, "ripe_unknown": 14.0}, "stricter ripeness (eaten fruit averages 43-45 of 60; ignoring ripeness is fatal for newborns)")),
    ("patrol",        ({"patrol": 1.0, "patrol_wait": 6.0, "patrol_ban": 25.0}, "walk to the tree with most accumulated fruit (35-42% of fruit rots)")),
    ("newborn_eager", ({"newborn_eager_s": 25.0}, "newborns eat at once (arena +0.07..+0.08, inconclusive twice)")),
    ("quiet_spawn",   ({"spawn_quiet_s": 20.0}, "no births within 20 s of a predator sighting (newborns cannot sprint)")),
    ("doom_spawn",    ({"doom_spawn": 1.0}, "an agent about to be caught converts its energy into a child (arena multi-spawn +0.27, inconclusive)")),
    ("alarm",         ({"shared_alarm": 1.0}, "share predator sightings between neighbours (null in arena and in 60 full seeds; kept as a control)")),
    ("colony_big",    ({"cap_late": 10.0, "cap_tau": 900.0, "spawn_thr_late": 250.0}, "bigger late colony (predators halve the colony; buffer)")),
    ("colony_small",  ({"cap_late": 5.0, "cap_tau": 500.0}, "smaller, better-fed late colony")),
    ("scan_more",     ({"camp_scan": 0.25, "vig_k": 0.25}, "more predator scanning while camping")),
    ("nosprint",      ({"sprint_below": 0.0}, "never sprint (arena price-dependent)")),
])

TUNE_BASE = ["cap_early", "cap_late", "cap_tau", "spawn_thr", "spawn_thr_late", "dbed_min_energy", "camp_patience",
             "ripe_age", "ripe_unknown", "select_q", "w_energy", "urgent_abs"]
TUNE_EXTRA = {"rest": "rest_frac", "quiet_spawn": "spawn_quiet_s", "senescence": "sen_extra", "patrol": "patrol_wait"}

SEEDS_A = list(range(5001, 5201))       # screening / combination / tuning
SEEDS_B = list(range(9001, 9201))       # FRESH seeds for the final decision
FRACS = OrderedDict([("calib", 0.05), ("screen", 0.25), ("combo", 0.15), ("tune", 0.30), ("final", 0.25)])


# ------------------------------------------------------------------------------------------------ infrastructure
def cfg_key(pd, horizon, knobs=None):
    d = {k: round(float(v), 6) for k, v in sorted(pd.items())}
    return hashlib.md5(json.dumps({"p": d, "h": horizon, "k": knobs or {}}, sort_keys=True).encode()).hexdigest()[:16]


def _job(job):
    key, pd, seed, horizon, knobs = job
    from survivor.runner import run_episode
    t0 = time.time()
    r = run_episode(seed, pd, horizon=horizon, keep_paths=False, sample_every=200, knobs=knobs or None)
    return key, seed, {"score": r["score"], "sim_t": r["sim_time"], "full": bool(r["survived_full"]), "deaths": r["deaths"],
                       "births": r["births"], "wall": time.time() - t0, "fruit": r["fruits_eaten"]}


class Cache:
    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line:
                    x = json.loads(line)
                    self.d[(x["k"], x["seed"])] = x["r"]

    def has(self, k, s):
        return (k, s) in self.d

    def get(self, k, s):
        return self.d.get((k, s))

    def put(self, k, s, r):
        self.d[(k, s)] = r
        with open(self.path, "a") as f:
            f.write(json.dumps({"k": k, "seed": s, "r": r}) + "\n")


class Engine:
    def __init__(self, out, workers, horizon, log):
        self.out, self.workers, self.horizon, self.log = out, workers, horizon, log
        self.cache = Cache(os.path.join(out, "cache.jsonl"))
        self.pool = Pool(workers) if workers > 1 else None

    def close(self):
        if self.pool:
            self.pool.close()

    def key(self, p):
        return cfg_key(asdict(p), self.horizon)

    def run_missing(self, cfgs, seeds, deadline=None):
        """run all (config, seed) pairs missing from the cache, interleaved by seed so a deadline leaves balanced data"""
        jobs = []
        for s in seeds:
            for p in cfgs.values():
                k = self.key(p)
                if not self.cache.has(k, s) and not any(j[0] == k and j[2] == s for j in jobs):
                    jobs.append((k, asdict(p), s, self.horizon, None))
        done = 0
        chunk = max(self.workers * 2, 2)
        for i in range(0, len(jobs), chunk):
            if deadline is not None and time.time() > deadline:
                break
            part = jobs[i:i + chunk]
            res = self.pool.map(_job, part, chunksize=1) if self.pool else [_job(j) for j in part]
            for k, s, r in res:
                self.cache.put(k, s, r)
            done += len(part)
        return done

    def scores(self, p, seeds):
        k = self.key(p)
        return {s: self.cache.get(k, s)["score"] for s in seeds if self.cache.has(k, s)}

    def results(self, p, seeds):
        k = self.key(p)
        return {s: self.cache.get(k, s) for s in seeds if self.cache.has(k, s)}

    def paired(self, pa, pb, seeds):
        a, b = self.scores(pa, seeds), self.scores(pb, seeds)
        ks = [s for s in seeds if s in a and s in b]
        if not ks:
            return None
        return paired_stats([a[s] for s in ks], [b[s] for s in ks])

    def mean_wall(self):
        w = [r["wall"] for r in self.cache.d.values()]
        return (sum(w) / len(w)) if w else 60.0


class Pipe:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        os.makedirs(self.out, exist_ok=True)
        self.state_path = os.path.join(self.out, "state.json")
        self.state = json.load(open(self.state_path)) if (a.resume and os.path.exists(self.state_path)) else {"stages": {}}
        self.t_start = time.time()
        self.end = self.t_start + a.budget_hours * 3600.0
        self.logf = open(os.path.join(self.out, "pipeline.log"), "a")
        self.base = Params.load(a.base)
        self.eng = Engine(self.out, a.workers, a.horizon, self.log)

    # ---------------------------------------------------------------- utilities
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - self.t_start) / 60:5.1f}m left {(self.end - time.time()) / 60:5.1f}m] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1, default=str)

    def deadline(self, stage):
        """absolute deadline for a stage = its share of the remaining budget"""
        names = list(FRACS.keys())
        i = names.index(stage)
        rest = sum(FRACS[n] for n in names[i:])
        return time.time() + max(0.0, (self.end - time.time())) * FRACS[stage] / rest

    def cfg(self, over):
        p = Params(**asdict(self.base))
        for k, v in over.items():
            setattr(p, k, float(v))
        return p

    def merge(self, names):
        over = {}
        for n in names:
            over.update(FEATURES[n][0])
        return over

    def cap_runs(self, stage, workers=None):
        """how many full runs fit into this stage on this machine"""
        c = max(self.eng.mean_wall(), 1.0)
        return int(FRACS[stage] * (self.end - self.t_start) * (workers or self.a.workers) / c)

    # ---------------------------------------------------------------- stages
    def s0_preflight(self):
        self.log("S0 preflight: self-tests")
        from survivor import runner
        if not runner.check():
            self.log("SELF-TEST FAILED - fix before training"); sys.exit(1)
        self.log("S0 calibration: baseline on the first seeds to measure cost per run")
        n = 2 if self.a.smoke else max(self.a.workers, 4)
        self.eng.run_missing({"base": self.base}, SEEDS_A[:n], self.deadline("calib"))
        c = self.eng.mean_wall()
        self.state["cost_per_run_s"] = c
        base_sc = list(self.eng.scores(self.base, SEEDS_A[:n]).values())
        self.state["baseline_calib"] = {"n": len(base_sc), "mean": (sum(base_sc) / len(base_sc)) if base_sc else None}
        self.log(f"S0 done: {c:.1f} s per full run; baseline mean on {len(base_sc)} seeds = {self.state['baseline_calib']['mean']:.0f}; "
                 f"budget fits ~{int((self.end - time.time()) * self.a.workers / c)} more runs")
        self.save()

    def s1_screen(self):
        feats = list(FEATURES.keys()) if not self.a.features else [f for f in self.a.features if f in FEATURES]
        smax = 3 if self.a.smoke else max(12, min(36, self.cap_runs("screen") // max(1, len(feats) + 1)))
        rnd = 3 if self.a.smoke else 6
        dl = self.deadline("screen")
        self.log(f"S1 screening {len(feats)} features, up to {smax} paired seeds each (rounds of {rnd}), deadline in {(dl - time.time()) / 60:.0f} min")
        alive = list(feats)
        cfgs = {f: self.cfg(FEATURES[f][0]) for f in feats}
        seeds_done = 0
        while seeds_done < smax and alive and time.time() < dl:
            seeds = SEEDS_A[seeds_done:seeds_done + rnd]
            todo = {"base": self.base, **{f: cfgs[f] for f in alive}}
            self.eng.run_missing(todo, seeds, dl)
            seeds_done += rnd
            used = SEEDS_A[:seeds_done]
            rows = []
            for f in list(alive):
                s = self.eng.paired(cfgs[f], self.base, used)
                if s and s["n"] >= 12 and s["hi"] < 0:
                    alive.remove(f)                                   # clearly harmful: stop spending seeds on it
                    self.log(f"  drop {f}: paired {s['mean']:+.0f} [{s['lo']:+.0f},{s['hi']:+.0f}] n={s['n']}")
                elif s:
                    rows.append((f, s))
            if rows:
                self.log("  after %d seeds: " % seeds_done + "  ".join(f"{f} {s['mean']:+.0f}" for f, s in sorted(rows, key=lambda x: -x[1]["mean"])[:8]))
        table = []
        for f in feats:
            s = self.eng.paired(cfgs[f], self.base, SEEDS_A[:max(seeds_done, 1)])
            table.append({"feature": f, "why": FEATURES[f][1], "n": s["n"] if s else 0, "mean": s["mean"] if s else None,
                          "lo": s["lo"] if s else None, "hi": s["hi"] if s else None, "wins": s["wins"] if s else 0,
                          "losses": s["losses"] if s else 0, "dropped": f not in alive})
        self.state["screen"] = table
        self.save()
        for r in table:
            if r["mean"] is not None:
                self.log(f"  {r['feature']:<14} n={r['n']:>2}  paired {r['mean']:+7.1f} [{r['lo']:+.0f},{r['hi']:+.0f}]  w/l {r['wins']}/{r['losses']}{'  (dropped)' if r['dropped'] else ''}")

    def s2_combine(self):
        table = [r for r in self.state.get("screen", []) if r["mean"] is not None and not r["dropped"]]
        ranked = [r["feature"] for r in sorted(table, key=lambda r: -r["mean"]) if r["mean"] > (0 if self.a.smoke else 10)]
        dl = self.deadline("combo")
        cmax = 3 if self.a.smoke else max(12, min(40, self.cap_runs("combo") // max(2, len(ranked))))
        self.log(f"S2 greedy combination over {ranked}, {cmax} paired seeds per step, deadline in {(dl - time.time()) / 60:.0f} min")
        chosen, cur = [], self.base
        seeds = SEEDS_A[:cmax]
        steps = []
        for f in ranked:
            if time.time() > dl:
                break
            cand = self.cfg(self.merge(chosen + [f]))
            self.eng.run_missing({"cur": cur, "cand": cand}, seeds, dl)
            s = self.eng.paired(cand, cur, seeds)
            if s is None:
                break
            keep = s["mean"] > (0 if self.a.smoke else 10) and s["lo"] > -150
            steps.append({"add": f, "n": s["n"], "mean": s["mean"], "lo": s["lo"], "hi": s["hi"], "kept": keep})
            self.log(f"  + {f:<14} vs current: {s['mean']:+7.1f} [{s['lo']:+.0f},{s['hi']:+.0f}] n={s['n']} -> {'KEEP' if keep else 'reject'}")
            if keep:
                chosen.append(f)
                cur = cand
        self.state["combo"] = {"chosen": chosen, "steps": steps, "over": self.merge(chosen)}
        self.save()
        s = self.eng.paired(cur, self.base, seeds)
        if s:
            self.log(f"S2 result: features {chosen or '(none)'}; combined vs baseline {s['mean']:+.1f} [{s['lo']:+.0f},{s['hi']:+.0f}] n={s['n']}")

    def s3_tune(self):
        chosen = self.state.get("combo", {}).get("chosen", [])
        ref = self.cfg(self.merge(chosen))
        free = list(TUNE_BASE) + [TUNE_EXTRA[f] for f in chosen if f in TUNE_EXTRA and TUNE_EXTRA[f] not in TUNE_BASE]
        free = [n for n in dict.fromkeys(free) if n in SPACE]
        dl = self.deadline("tune")
        pop = 4 if self.a.smoke else 12
        m = 2 if self.a.smoke else max(4, min(8, self.cap_runs("tune") // (6 * pop)))
        pool_n = 4 if self.a.smoke else 24
        gens_max = 1 if self.a.smoke else 40
        self.log(f"S3 tuning {len(free)} parameters, pop {pop}, {m} paired seeds per generation (rotating over {pool_n}), deadline in {(dl - time.time()) / 60:.0f} min")
        x0 = [min(1.0, max(0.0, v)) for v in params_to_vec(ref, free)]
        try:
            import cma
            es = cma.CMAEvolutionStrategy(x0, 0.10, {"popsize": pop, "bounds": [0, 1], "seed": 1, "verbose": -9})
            use_cma = True
        except ImportError:
            from survivor.optimize import CEM
            es = CEM(x0, 0.10, pop, 1)
            use_cma = False
        self.log(f"  optimizer: {'CMA-ES' if use_cma else 'CEM (pip install cma for CMA-ES)'}")
        hist, top = [], []
        for g in range(gens_max):
            if time.time() > dl:
                break
            sl = [SEEDS_A[(g * m + i) % pool_n] for i in range(m)]
            xs = es.ask()
            cands = [vec_to_params(x, ref, free) for x in xs]
            cfgs = {"ref": ref, **{f"c{i}": c for i, c in enumerate(cands)}}
            self.eng.run_missing(cfgs, sl, dl)
            refs = self.eng.scores(ref, sl)
            fits = []
            for c in cands:
                sc = self.eng.scores(c, sl)
                ks = [s for s in sl if s in sc and s in refs]
                fits.append(sum(sc[s] - refs[s] for s in ks) / len(ks) if ks else -1e9)
            if all(f < -1e8 for f in fits):
                break
            es.tell(xs, [-f for f in fits]) if use_cma else es.tell(xs, fits)
            bi = max(range(len(fits)), key=lambda i: fits[i])
            top.append((fits[bi], asdict(cands[bi])))
            top = sorted(top, key=lambda z: -z[0])[:6]
            hist.append({"gen": g, "best": fits[bi], "mean": sum(fits) / len(fits), "slice": sl})
            self.log(f"  gen {g:2d}: best paired {fits[bi]:+7.1f}  mean {sum(fits) / len(fits):+7.1f}  (vs reference on {len(sl)} seeds)")
            self.state["tune"] = {"free": free, "hist": hist}
            self.save()
        finalists = []
        try:
            mean_vec = list(es.mean) if use_cma else list(es.mu)
            finalists.append(("tuned_mean", asdict(vec_to_params(mean_vec, ref, free))))
        except Exception:
            pass
        for i, (f, pdict) in enumerate(top[:2]):
            finalists.append((f"tuned_best{i + 1}", pdict))
        self.state["tune"] = {"free": free, "hist": hist, "finalists": finalists, "ref_over": self.merge(chosen)}
        self.save()

    def s4_final(self):
        chosen = self.state.get("combo", {}).get("chosen", [])
        cands = OrderedDict([("baseline", self.base)])
        if chosen:
            cands["combined"] = self.cfg(self.merge(chosen))
        for name, pdict in self.state.get("tune", {}).get("finalists", []):
            cands[name] = Params(**pdict)
        dl = self.deadline("final")
        fmax = 3 if self.a.smoke else max(16, min(80, self.cap_runs("final") // max(1, len(cands))))
        rnd = 3 if self.a.smoke else 8
        self.log(f"S4 final: {list(cands)} on FRESH seeds, up to {fmax} per config (rounds of {rnd}), deadline in {(dl - time.time()) / 60:.0f} min")
        n = 0
        while n < fmax and time.time() < dl:
            self.eng.run_missing(cands, SEEDS_B[n:n + rnd], dl)
            n += rnd
            ss = self.eng.paired(cands.get("combined", self.base), self.base, SEEDS_B[:n]) if "combined" in cands else None
            if ss:
                self.log(f"  {n} fresh seeds: combined vs baseline {ss['mean']:+.0f} [{ss['lo']:+.0f},{ss['hi']:+.0f}]")
        # only seeds that every finalist completed count (paired, balanced)
        seeds = [s for s in SEEDS_B[:n] if all(self.eng.cache.has(self.eng.key(p), s) for p in cands.values())]
        rows = []
        for name, p in cands.items():
            res = self.eng.results(p, seeds)
            sc = [res[s]["score"] for s in seeds if s in res]
            if not sc:
                continue
            row = {"name": name, "n": len(sc), "mean": sum(sc) / len(sc), "median": st.median(sc), "sd": st.pstdev(sc) if len(sc) > 1 else 0.0,
                   "p1000": sum(1 for x in sc if x >= 1000) / len(sc), "p1500": sum(1 for x in sc if x >= 1500) / len(sc),
                   "max": max(sc), "min": min(sc), "mean_sim_t": sum(res[s]["sim_t"] for s in seeds) / len(sc),
                   "full": sum(1 for s in seeds if res[s]["full"])}
            if name != "baseline":
                ps = self.eng.paired(p, self.base, seeds)
                row.update({"d_mean": ps["mean"], "d_lo": ps["lo"], "d_hi": ps["hi"], "wins": ps["wins"], "losses": ps["losses"], "d_p": ps["p"]})
            rows.append(row)
        # decision: best mean among finalists whose paired CI does not show harm; baseline is the fallback
        best = "baseline"
        best_mean = next(r["mean"] for r in rows if r["name"] == "baseline")
        for r in rows:
            if r["name"] != "baseline" and r["mean"] > best_mean and r.get("d_lo", -1e9) > -60:
                best, best_mean = r["name"], r["mean"]
        win_p = cands[best]
        meta = {"pipeline": "survivor.pipeline", "winner": best, "fresh_seeds": len(seeds), "table": rows,
                "features": chosen if best != "baseline" else []}
        final_path = os.path.join(self.out, "final_config.json")
        win_p.save(final_path, **meta)
        self.state["final"] = {"rows": rows, "winner": best, "seeds": len(seeds), "path": final_path}
        self.save()
        self.log(f"S4 decision: winner = {best}  (fresh-seed mean {best_mean:.0f} vs baseline "
                 f"{next(r['mean'] for r in rows if r['name'] == 'baseline'):.0f}); written to {final_path}")

    # ---------------------------------------------------------------- run
    def run(self):
        self.log(f"pipeline start: budget {self.a.budget_hours:.2f} h, {self.a.workers} workers, horizon {self.a.horizon:.0f} s, base {self.a.base}"
                 + (" [SMOKE]" if self.a.smoke else ""))
        try:
            for name, fn in (("s0", self.s0_preflight), ("s1", self.s1_screen), ("s2", self.s2_combine), ("s3", self.s3_tune), ("s4", self.s4_final)):
                if self.state["stages"].get(name) == "done" and self.a.resume:
                    self.log(f"{name} already done (resume) - skipping")
                    continue
                t0 = time.time()
                fn()
                self.state["stages"][name] = "done"
                self.state.setdefault("timing", {})[name] = time.time() - t0
                self.save()
        finally:
            self.eng.close()
        write_report(self.out, self.state, self.a)
        self.log(f"DONE. Report: {os.path.join(self.out, 'PIPELINE_REPORT.md')}")
        self.log(f"Serve the result:  SURVIVOR_PARAMS={os.path.join(self.out, 'final_config.json')} python -m survivor.server")


# ------------------------------------------------------------------------------------------------ report
def write_report(out, state, a=None):
    L = []
    w = L.append
    w("# Pipeline report\n")
    tm = state.get("timing", {})
    w(f"Cost per full run on this machine: {state.get('cost_per_run_s', float('nan')):.1f} s. Stage times (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in tm.items()) + "\n")
    if state.get("baseline_calib"):
        w(f"Baseline calibration: mean score {state['baseline_calib']['mean']:.0f} on {state['baseline_calib']['n']} seeds.\n")
    if state.get("screen"):
        w("## S1 feature screening (paired vs baseline, same seeds; a difference below ~100 is within noise)")
        w("| feature | why | n | paired Δ [95% CI] | wins/losses | status |")
        w("|---|---|---|---|---|---|")
        for r in sorted(state["screen"], key=lambda r: -(r["mean"] if r["mean"] is not None else -1e9)):
            if r["mean"] is None:
                w(f"| {r['feature']} | {r['why']} | 0 | - | - | not run |")
            else:
                w(f"| {r['feature']} | {r['why']} | {r['n']} | {r['mean']:+.0f} [{r['lo']:+.0f}, {r['hi']:+.0f}] | {r['wins']}/{r['losses']} | {'dropped early' if r['dropped'] else 'kept for S2'} |")
    if state.get("combo"):
        c = state["combo"]
        w("\n## S2 greedy combination")
        w("| step | adds | n | Δ vs current [95% CI] | decision |")
        w("|---|---|---|---|---|")
        for i, s in enumerate(c["steps"]):
            w(f"| {i + 1} | {s['add']} | {s['n']} | {s['mean']:+.0f} [{s['lo']:+.0f}, {s['hi']:+.0f}] | {'KEEP' if s['kept'] else 'reject'} |")
        w(f"\nChosen features: **{', '.join(c['chosen']) or 'none'}**; overrides: `{json.dumps(c['over'])}`")
    if state.get("tune", {}).get("hist"):
        h = state["tune"]["hist"]
        w(f"\n## S3 tuning ({len(state['tune']['free'])} parameters: {', '.join(state['tune']['free'])})")
        w("| gen | best candidate Δ vs reference | population mean Δ |")
        w("|---|---|---|")
        for r in h:
            w(f"| {r['gen']} | {r['best']:+.0f} | {r['mean']:+.0f} |")
        w("(fitness is noisy: each generation uses a different slice of seeds; the final table below is what counts)")
    if state.get("final"):
        f = state["final"]
        w(f"\n## S4 final evaluation on {f['seeds']} FRESH seeds (never seen by the search)")
        w("| config | mean | median | sd | P(score>=1000) | P(score>=1500) | best | reached 3000 s | mean survival s | paired Δ vs baseline [95% CI] | wins/losses |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in f["rows"]:
            d = f"{r['d_mean']:+.0f} [{r['d_lo']:+.0f}, {r['d_hi']:+.0f}]" if "d_mean" in r else "-"
            wl = f"{r['wins']}/{r['losses']}" if "wins" in r else "-"
            w(f"| {r['name']} | {r['mean']:.0f} | {r['median']:.0f} | {r['sd']:.0f} | {100 * r['p1000']:.0f}% | {100 * r['p1500']:.0f}% | {r['max']:.0f} | {r['full']}/{r['n']} | {r['mean_sim_t']:.0f} | {d} | {wl} |")
        w(f"\n**Winner: `{f['winner']}`** -> `{f['path']}`")
        w("\nThe validation server is ONE draw (or the mean of a few) from the distribution in the P(score >= ...) columns; expect scatter of about one 'sd' around the mean.")
        w("\n## Submit\n```bash\nSURVIVOR_PARAMS=%s python -m survivor.server\npython -m survivor.analyze server logs/server_*.jsonl   # after the attempt\n```" % f["path"])
    open(os.path.join(out, "PIPELINE_REPORT.md"), "w").write("\n".join(L) + "\n")


def status(out):
    sp = os.path.join(out, "state.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        print("stages done:", [k for k, v in s.get("stages", {}).items() if v == "done"], " cost/run %.1fs" % s.get("cost_per_run_s", float("nan")))
        if s.get("combo"):
            print("chosen features:", s["combo"]["chosen"])
        if s.get("final"):
            print("winner:", s["final"]["winner"])
    lp = os.path.join(out, "pipeline.log")
    if os.path.exists(lp):
        print("".join(open(lp).readlines()[-12:]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--base", default=os.path.join(ROOT, "configs", "v5a_ripe_boot.json"))
    r.add_argument("--budget-hours", type=float, default=2.0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--horizon", type=float, default=3000.0)
    r.add_argument("--features", nargs="*", default=None, help="restrict screening to these features")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--smoke", action="store_true", help="tiny end-to-end test of the pipeline (few seeds, tiny sizes)")
    s = sub.add_parser("status")
    s.add_argument("out")
    p = sub.add_parser("report")
    p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "status":
        status(a.out)
    elif a.cmd == "report":
        write_report(a.out, json.load(open(os.path.join(a.out, "state.json"))))
        print("written", os.path.join(a.out, "PIPELINE_REPORT.md"))
    else:
        Pipe(a).run()


if __name__ == "__main__":
    main()
