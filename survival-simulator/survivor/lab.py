"""
survivor.lab - the 25-minute experiment lab for the hierarchical colony controller.

  python -m survivor.lab run --out runs/lab1 --budget-min 25 --workers 15          # ~13 arms x paired fresh seeds, live tables
  python -m survivor.lab report runs/lab1                                            # rewrite LAB_REPORT.md
  python -m survivor.lab fit-escape runs/lab1                                        # learn P(death | encounter) from logged flee episodes
  python -m survivor.lab run --out runs/lab2 --arms control colony_all --budget-min 15 --escape-model runs/lab1/escape_model.json

Each ARM = the pipeline-winner parameters + a set of switches of the new layers (colony-level births, senescence, lifeboats,
obstacle-aware flee, anti-clustering, survival-mode phase ...). All arms use the SAME seeds (paired). Seeds run in rounds over all
arms so that the deadline always leaves balanced data. Everything is cached (out/lab_cache.jsonl): re-running or resuming is free.

The report answers, per arm: how long the colony survives, whether it beats the control (paired CI), and WHY (deaths by cause,
demography R, cohort structure, escape-ready share, births by reason, denials by reason, flee death rate), plus the learned
escape model and diagnostic arms (predators off / food floor) that show which wall is still binding.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
import time
from collections import OrderedDict, Counter
from dataclasses import asdict
from multiprocessing import Pool

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor.policy import Params  # noqa: E402
from survivor.micro import paired_stats, verdict, _demography  # noqa: E402
from survivor.pipeline import cfg_key, Cache  # noqa: E402
from survivor import escape_model as EM  # noqa: E402
from survivor import value_model as VM  # noqa: E402

ALL = {"colony_mgr": 1, "senescence": 1, "young_frac": 0.5, "birth_gap_s": 8.0, "gap_after_s": 300.0, "reserve_margin": 20.0,
       "n_lifeboat": 2, "emap": 1, "flee_ray": 1, "crowd_max": 2, "spread_w": 1.5, "late_t": 1200.0, "late_rest_frac": 0.45}
# name -> (overrides on the base config, simulator knobs, max seeds or None, description)
ARMS = OrderedDict([
    ("control",     ({}, None, None, "pipeline winner with per-agent rules (the current best)")),
    ("mgr",         ({"colony_mgr": 1}, None, None, "colony-level, age-balanced births (young cohort <= 30% of target)")),
    ("mgr_y50",     ({"colony_mgr": 1, "young_frac": 0.5}, None, None, "same, young cohort <= 50%")),
    ("mgr_sen",     ({"colony_mgr": 1, "young_frac": 0.5, "senescence": 1}, None, None, "+ senescent agents convert energy into children")),
    ("mgr_stagger", ({"colony_mgr": 1, "young_frac": 0.5, "birth_gap_s": 8.0, "gap_after_s": 300.0}, None, None, "+ minimum 8 s between births after t=300 (no bursts)")),
    ("mgr_reserve", ({"colony_mgr": 1, "young_frac": 0.5, "reserve_margin": 40.0}, None, None, "+ parent keeps sprint reserve (0.2*maxE+40) after paying")),
    ("mgr_boat",    ({"colony_mgr": 1, "young_frac": 0.5, "n_lifeboat": 2}, None, None, "+ 2 isolated lifeboat agents excluded from breeding")),
    ("mgr_ray",     ({"colony_mgr": 1, "young_frac": 0.5, "emap": 1, "flee_ray": 1}, None, None, "+ obstacle-aware flee over remembered edges")),
    ("mgr_spread",  ({"colony_mgr": 1, "young_frac": 0.5, "crowd_max": 2, "spread_w": 1.5}, None, None, "+ anti-clustering (max 2 agents per tree, spread)")),
    ("phase",       ({"late_t": 1200.0, "late_rest_frac": 0.45}, None, None, "control + survival mode after t=1200 (rest earlier, no exploring)")),
    ("colony_all",  (ALL, None, None, "all layers on")),
    ("diag_nopred", (ALL, {"no_predators": True}, 16, "DIAGNOSTIC colony_all with predators OFF (is food the binding wall now?)")),
    ("diag_floor",  (ALL, {"tree_floor": 45}, 16, "DIAGNOSTIC colony_all with a food floor (are predators the binding wall now?)")),
])
CHECKPOINTS = (300, 600, 900, 1200, 1800)


def _at(ts, key, t):
    best = None
    for tt, v in zip(ts["t"], ts.get(key, [])):
        if tt <= t:
            best = v
        else:
            break
    return best


def _lab_job(job):
    key, pd, seed, horizon, knobs = job
    from survivor.runner import run_episode
    t0 = time.time()
    knobs = dict(knobs) if knobs else None
    if knobs and "value_model" in knobs:                       # a model fitted during this session: set for THIS job's controller
        os.environ["SURVIVOR_VALUE_MODEL"] = os.path.abspath(knobs.pop("value_model"))
    r = run_episode(seed, pd, horizon=horizon, keep_paths=False, sample_every=100, knobs=knobs or None)
    ts = r["ts"]
    dem = _demography(r)
    R, reach = {}, {}
    for lo, hi in ((0, 300), (300, 800), (800, 1300)):
        G = [x for x in dem if lo <= x[0] < hi]
        if len(G) >= 5:
            R[f"{lo}-{hi}"] = sum(x[2] for x in G) / len(G)
            reach[f"{lo}-{hi}"] = 1 - sum(x[3] for x in G) / len(G)
    bt = [b[0] for b in r["births_list"]]
    disp = None
    if bt:
        nb = max(1, int(max(r["sim_time"], 40) // 20))
        c = [0] * nb
        for x in bt:
            c[min(nb - 1, int(x // 20))] += 1
        mm = sum(c) / nb
        disp = (st.pvariance(c) / mm) if mm > 0 else None
    col = r["track"].get("colony", {})
    tl = col.get("timeline", [])

    def young_at(t):
        best = None
        for row in tl:
            if row[0] <= t:
                best = row
        return (best[2] / max(1, best[1])) if best else None
    deaths_age = Counter()
    for x in r["deaths_list"]:
        deaths_age[("young" if x[2] < 40 else "adult") + "_" + x[1]] += 1
    fl = r["flee_rows"]
    step = max(1, len(fl) // 300)
    tt, nn = ts["t"], ts["n"]
    agent_s = sum(nn[i] * (tt[i + 1] - tt[i]) for i in range(len(tt) - 1)) if len(tt) > 1 else 0.0
    vr = r.get("val_rows", [])
    vstep = max(1, len(vr) // 300)
    return key, seed, {
        "score": r["score"], "sim_t": r["sim_time"], "full": bool(r["survived_full"]), "deaths": r["deaths"], "births": r["births"],
        "wall": time.time() - t0, "fruit": r["fruits_eaten"],
        "pop": {t: _at(ts, "n", t) if ts["t"] and ts["t"][-1] >= t else 0 for t in CHECKPOINTS},
        "ready": {t: _at(ts, "ready", t) for t in CHECKPOINTS if ts["t"] and ts["t"][-1] >= t},
        "meanE": {t: _at(ts, "mean_E", t) for t in CHECKPOINTS if ts["t"] and ts["t"][-1] >= t},
        "young": {t: young_at(t) for t in CHECKPOINTS if ts["t"] and ts["t"][-1] >= t},
        "traits600": ({k: _at(ts, k, 600) for k in ("mean_speed", "mean_hear", "mean_vision", "mean_cone", "mean_maxE")} if ts["t"] and ts["t"][-1] >= 600 else {}),
        "pen": sum(x[3] for x in r["deaths_list"] if x[1] == "predator") / 100.0,
        "hz": {"agent_s": agent_s, "pred": r["deaths"].get("predator", 0), "starve": r["deaths"].get("starved_young", 0), "old": r["deaths"].get("starved_old", 0)},
        "val_n": len(vr), "val_dead": sum(1 for x in vr if x[-1] == 0), "val": vr[::vstep][:300],
        "R": R, "reach40": reach, "disp": disp, "c_births": col.get("births", {}), "c_deny": col.get("deny", {}),
        "deaths_age": dict(deaths_age), "flee": fl[::step][:300], "flee_n": len(fl), "flee_died": sum(x[-1] for x in fl),
        "policy": {k: v for k, v in r["policy_stats"].items() if k in ("flee", "flee_sprint", "camp", "forage", "seek_tree", "explore", "senescent", "agent_ticks")},
    }


class Lab:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.cache = Cache(os.path.join(a.out, "lab_cache.jsonl"))
        self.base = Params.load(a.base)
        if a.escape_model:                                   # must be set BEFORE the pool forks its workers
            os.environ["SURVIVOR_ESCAPE_MODEL"] = os.path.abspath(a.escape_model)
        self.pool = Pool(a.workers) if a.workers > 1 else None
        self.logf = open(os.path.join(a.out, "lab.log"), "a")
        self.t0 = time.time()
        self.arms = OrderedDict((n, v) for n, v in ARMS.items() if (not a.arms or n in a.arms or n == "control"))
        if a.escape_model:
            self.arms["escape_terminal"] = ({**ALL, "use_escape_model": 1, "doom_p": 0.6}, None, None, "colony_all + terminal spawn from the learned escape model")

    def log(self, m):
        line = f"[{time.strftime('%H:%M:%S')} +{(time.time() - self.t0) / 60:5.1f}m] {m}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def params(self, name):
        over = self.arms[name][0]
        p = Params(**asdict(self.base))
        for k, v in over.items():
            setattr(p, k, float(v))
        return p

    def key(self, name):
        return cfg_key(asdict(self.params(name)), self.a.horizon, self.arms[name][1])

    def run(self):
        a = self.a
        end = self.t0 + a.budget_min * 60.0
        seeds = list(range(a.seed0, a.seed0 + a.seeds))
        rnd = a.round_size
        self.log(f"lab start: {len(self.arms)} arms, up to {a.seeds} paired seeds (rounds of {rnd}), budget {a.budget_min} min, {a.workers} workers, horizon {a.horizon:.0f}s")
        for i in range(0, len(seeds), rnd):
            if time.time() > end:
                self.log("budget reached")
                break
            jobs = []
            for s in seeds[i:i + rnd]:
                for n, (over, knobs, maxs, _d) in self.arms.items():
                    if maxs is not None and seeds.index(s) >= maxs:
                        continue
                    k = self.key(n)
                    if not self.cache.has(k, s):
                        jobs.append((k, asdict(self.params(n)), s, a.horizon, knobs))
            uniq = {}
            for j in jobs:
                uniq[(j[0], j[2])] = j
            jobs = list(uniq.values())
            chunk = max(a.workers * 2, 2)
            for c in range(0, len(jobs), chunk):
                part = jobs[c:c + chunk]
                res = self.pool.map(_lab_job, part, chunksize=1) if self.pool else [_lab_job(j) for j in part]
                for k, s, r in res:
                    self.cache.put(k, s, r)
            self.table(seeds[:i + rnd])
        if self.pool:
            self.pool.close()
        write_report(a.out, self)
        fit_escape(a.out, self)
        self.log(f"DONE. Report: {os.path.join(a.out, 'LAB_REPORT.md')}")

    # ------------------------------------------------------------------ analysis helpers
    def res(self, name, seeds):
        k = self.key(name)
        return {s: self.cache.get(k, s) for s in seeds if self.cache.has(k, s)}

    def table(self, seeds):
        base = self.res("control", seeds)
        rows = []
        for n in self.arms:
            r = self.res(n, seeds)
            ks = [s for s in seeds if s in r and s in base]
            if not ks:
                continue
            sc = [r[s]["score"] for s in ks]
            ps = paired_stats(sc, [base[s]["score"] for s in ks]) if n != "control" else None
            rows.append((n, len(ks), sum(sc) / len(sc), sum(r[s]["sim_t"] for s in ks) / len(ks), ps))
        self.log(f"--- after {len(seeds)} seeds (mean score | mean survival s | paired delta vs control [95% CI]) ---")
        for n, k, m, sv, ps in sorted(rows, key=lambda x: -x[2]):
            d = f"{ps['mean']:+6.0f} [{ps['lo']:+.0f},{ps['hi']:+.0f}] w/l {ps['wins']}/{ps['losses']}" if ps else "(control)"
            self.log(f"   {n:<14} n={k:>2}  {m:7.0f} | {sv:6.0f} | {d}")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else float("nan")


def write_report(out, lab):
    a = lab.a
    seeds_all = sorted({s for (k, s) in lab.cache.d.keys()})
    L = []
    w = L.append
    w("# Colony lab report\n")
    w(f"Base config `{a.base}`; horizon {a.horizon:.0f} s; arms share seeds (paired). Control = the pipeline winner with per-agent rules.\n")
    base = lab.res("control", seeds_all)
    w("## 1. How long does the colony survive? (paired vs control)")
    w("| arm | what | n | mean score | median | sd | mean survival s | alive at 900 s | alive at 1500 s | reached 3000 s | P(score>=1000) | P(>=1500) | paired Δ [95% CI] | w/l |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    summ = {}
    for n in lab.arms:
        r = lab.res(n, seeds_all)
        ks = [s for s in seeds_all if s in r and s in base]
        if not ks:
            continue
        sc = [r[s]["score"] for s in ks]
        sv = [r[s]["sim_t"] for s in ks]
        ps = paired_stats(sc, [base[s]["score"] for s in ks]) if n != "control" else None
        summ[n] = (ks, r)
        d = f"{ps['mean']:+.0f} [{ps['lo']:+.0f}, {ps['hi']:+.0f}] ({verdict(ps, 60.0)})" if ps else "-"
        wl = f"{ps['wins']}/{ps['losses']}" if ps else "-"
        w(f"| {n} | {lab.arms[n][3]} | {len(ks)} | {sum(sc) / len(sc):.0f} | {st.median(sc):.0f} | {st.pstdev(sc) if len(sc) > 1 else 0:.0f} | {sum(sv) / len(sv):.0f} | "
          f"{100 * sum(1 for x in sv if x >= 900) / len(sv):.0f}% | {100 * sum(1 for x in sv if x >= 1500) / len(sv):.0f}% | {sum(1 for s in ks if r[s]['full'])}/{len(ks)} | "
          f"{100 * sum(1 for x in sc if x >= 1000) / len(sc):.0f}% | {100 * sum(1 for x in sc if x >= 1500) / len(sc):.0f}% | {d} | {wl} |")
    w("\n(differences below ~100 points are within noise; identical reruns differ by about that much)\n")
    w("## 2. Why? mechanism per arm (means over paired seeds)")
    w("| arm | deaths predator / young-starved / old-age | births | R born 0-300 | R 300-800 | reach 40 s (300-800) | birth burstiness (var/mean, Poisson=1) | pop @300/600/900/1200 | escape-ready % @300/600/900 | young share @300/600 | flee episodes -> death rate |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, (ks, r) in summ.items():
        d = Counter()
        for s in ks:
            d.update(r[s]["deaths"])
        tot = max(1, sum(d.values()))
        pop = "/".join(f"{_mean([r[s]['pop'].get(str(t), r[s]['pop'].get(t)) for s in ks]):.1f}" for t in (300, 600, 900, 1200))
        rd = "/".join(f"{100 * _mean([r[s]['ready'].get(str(t), r[s]['ready'].get(t)) for s in ks if r[s]['ready']]):.0f}" for t in (300, 600, 900))
        yg = "/".join(f"{100 * _mean([r[s]['young'].get(str(t), r[s]['young'].get(t)) for s in ks if r[s]['young']]):.0f}" for t in (300, 600))
        fn, fd = sum(r[s]["flee_n"] for s in ks), sum(r[s]["flee_died"] for s in ks)
        w(f"| {n} | {100 * d['predator'] / tot:.0f}% / {100 * d['starved_young'] / tot:.0f}% / {100 * d['starved_old'] / tot:.0f}% | {_mean([r[s]['births'] for s in ks]):.0f} | "
          f"{_mean([r[s]['R'].get('0-300') for s in ks]):.2f} | {_mean([r[s]['R'].get('300-800') for s in ks]):.2f} | {100 * _mean([r[s]['reach40'].get('300-800') for s in ks]):.0f}% | "
          f"{_mean([r[s]['disp'] for s in ks]):.1f} | {pop} | {rd} | {yg} | {fd}/{fn} ({100 * fd / max(1, fn):.0f}%) |")
    w("\n## 1a. Risk profile (survival probabilities and lower tail) and the kill-penalty component of the score")
    w("| arm | n | P(T>=900 s) | P(T>=1200 s) | P(T>=1500 s) | P(T>=1800 s) | CVaR20 (mean of worst 20% of scores) | paired Δ P(T>=1200) wins/losses vs control | kill penalty per run (score points) | paired Δ penalty [95% CI] |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for n, (ks, r) in summ.items():
        sv = [r[s]["sim_t"] for s in ks]
        sc = sorted(r[s]["score"] for s in ks)
        m20 = sc[:max(1, len(sc) // 5)]
        wl = "-"
        if n != "control":
            wins = sum(1 for s in ks if r[s]["sim_t"] >= 1200 and base[s]["sim_t"] < 1200)
            loss = sum(1 for s in ks if r[s]["sim_t"] < 1200 and base[s]["sim_t"] >= 1200)
            wl = f"{wins}/{loss}"
        pen = [r[s].get("pen") for s in ks]
        bpen = [base[s].get("pen") for s in ks]
        if all(x is not None for x in pen) and all(x is not None for x in bpen):
            dp = paired_stats(pen, bpen) if n != "control" else None
            ptxt = f"{sum(pen) / len(pen):.1f}"
            dtxt = f"{dp['mean']:+.1f} [{dp['lo']:+.1f}, {dp['hi']:+.1f}]" if dp else "-"
        else:
            ptxt, dtxt = "-", "-"
        w(f"| {n} | {len(ks)} | " + " | ".join(f"{100 * sum(1 for x in sv if x >= c) / len(sv):.0f}%" for c in (900, 1200, 1500, 1800)) +
          f" | {sum(m20) / len(m20):.0f} | {wl} | {ptxt} | {dtxt} |")
    w("\n## 1b. Dense meta-reward metrics (agent level: +2 per 5 s survived). Far less noisy than one final score per run")
    w("| arm | agent-seconds per run | predator deaths per 1000 agent-s | paired Δ predator hazard vs control [95% CI] (negative = safer) | starvation deaths per 1000 agent-s | agent 5-s survival | meta reward per agent-hour (max 1440) |")
    w("|---|---|---|---|---|---|---|")
    for n, (ks, r) in summ.items():
        hz = [r[s].get("hz") for s in ks]
        if not all(hz):
            continue
        As = sum(h["agent_s"] for h in hz)
        if As <= 0:
            continue
        ph = [1000.0 * r[s]["hz"]["pred"] / r[s]["hz"]["agent_s"] if r[s]["hz"]["agent_s"] > 0 else 0.0 for s in ks]
        bh = [1000.0 * base[s]["hz"]["pred"] / base[s]["hz"]["agent_s"] if base[s].get("hz") and base[s]["hz"]["agent_s"] > 0 else 0.0 for s in ks]
        d = paired_stats(ph, bh) if n != "control" else None
        vn, vd = sum(r[s].get("val_n", 0) for s in ks), sum(r[s].get("val_dead", 0) for s in ks)
        p5 = (1 - vd / vn) if vn else float("nan")
        dtxt = f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]" if d else "-"
        w(f"| {n} | {As / len(ks):.0f} | {1000.0 * sum(h['pred'] for h in hz) / As:.2f} | {dtxt} | "
          f"{1000.0 * sum(h['starve'] for h in hz) / As:.2f} | {100 * p5:.1f}% | {1440 * p5:.0f} |")
    w("\n## 2b. Encounter conditions and evolved traits (the escape model says energy fraction and first-sight distance carry the risk)")
    w("| arm | flee episodes | mean energy fraction at first sight | mean first-sight distance | share noticed | mean speed of fleeing agents | death rate | traits at 600 s: speed / hearing / vision / cone(rad) / maxE |")
    w("|---|---|---|---|---|---|---|---|")
    for n, (ks, r) in summ.items():
        rows = [x for s in ks for x in r[s].get("flee", [])]
        if not rows:
            continue
        tr = [r[s].get("traits600") for s in ks if r[s].get("traits600")]
        trs = " / ".join(f"{_mean([t.get(k) for t in tr]):.1f}" for k in ("mean_speed", "mean_hear", "mean_vision")) + " / " + \
              f"{_mean([t.get('mean_cone') for t in tr]):.2f} / {_mean([t.get('mean_maxE') for t in tr]):.0f}" if tr else "-"
        w(f"| {n} | {len(rows)} | {_mean([x[0] for x in rows]):.2f} | {_mean([x[2] for x in rows]):.0f} | {100 * _mean([x[7] for x in rows]):.0f}% | "
          f"{_mean([x[1] for x in rows]):.1f} | {100 * _mean([x[10] for x in rows]):.0f}% | {trs} |")
    w("\n## 3. What the colony layer decided (births by reason, denials by reason; totals over seeds)")
    w("| arm | births by reason | denials by reason |")
    w("|---|---|---|")
    for n, (ks, r) in summ.items():
        b, dn = Counter(), Counter()
        for s in ks:
            b.update(r[s]["c_births"])
            dn.update(r[s]["c_deny"])
        if b or dn:
            w(f"| {n} | {dict(b)} | {dict(dn)} |")
    w("\n## 4. Where do juveniles die? (share of all deaths, totals over seeds)")
    w("| arm | " + " | ".join(["young_predator", "young_starved_young", "young_starved_old", "adult_predator", "adult_starved_young", "adult_starved_old"]) + " |")
    w("|---|---|---|---|---|---|---|")
    for n, (ks, r) in summ.items():
        c = Counter()
        for s in ks:
            c.update(r[s]["deaths_age"])
        tot = max(1, sum(c.values()))
        w(f"| {n} | " + " | ".join(f"{100 * c[k] / tot:.0f}%" for k in ["young_predator", "young_starved_young", "young_starved_old", "adult_predator", "adult_starved_young", "adult_starved_old"]) + " |")
    if "diag_nopred" in summ or "diag_floor" in summ:
        w("\n## 5. Which wall is binding now? (diagnostic arms vs colony_all on the same seeds)")
        ca = summ.get("colony_all")
        for dn in ("diag_nopred", "diag_floor"):
            if dn in summ and ca:
                ks = [s for s in summ[dn][0] if s in ca[1]]
                if ks:
                    d = paired_stats([summ[dn][1][s]["score"] for s in ks], [ca[1][s]["score"] for s in ks])
                    w(f"- `{dn}` vs `colony_all`: mean survival {_mean([summ[dn][1][s]['sim_t'] for s in ks]):.0f} s vs {_mean([ca[1][s]['sim_t'] for s in ks]):.0f} s; "
                      f"score {d['mean']:+.0f} [{d['lo']:+.0f}, {d['hi']:+.0f}] (n={d['n']}).")
        w("- Predators off helps a lot -> predators are still the wall. Food floor helps a lot -> food is the wall. Both help little -> the colony layer itself (births/energy) is the limit.")
    ef = os.path.join(out, "escape_model.json")
    if os.path.exists(ef):
        m = json.load(open(ef))
        w(f"\n## 6. Learned escape model (logistic regression on {m['n']} logged flee episodes; death rate {100 * m['base_rate']:.0f}%; AUC {m['auc']:.2f})")
        w("| feature | standardized weight | reading |")
        w("|---|---|---|")
        for f, wt in zip(m["features"], m["w"][1:]):
            w(f"| {f} | {wt:+.2f} | {'raises' if wt > 0 else 'lowers'} the probability of dying |")
        w("\nCalibration (predicted vs observed): " + "; ".join(f"[{b['pred'][0]:.2f},{b['pred'][1]:.2f}) n={b['n']} pred {b['mean_pred']:.2f} obs {b['observed']:.2f}" for b in m["calibration"]))
    vf = os.path.join(out, "value_model.json")
    if os.path.exists(vf):
        m = json.load(open(vf))
        w(f"\n## 6b. Meta-reward hazard model ({m['n']} agent windows; 5-s survival {100 * m['survive5']:.1f}%; AUC {m['auc']:.2f})")
        w("| feature | standardized weight (positive = safer) |")
        w("|---|---|")
        for f, wt in zip(m["features"], m["w"][1:]):
            w(f"| {f} | {wt:+.2f} |")
    w("\n## 7. What to send back")
    w("This report (sections 1-6), the last 30 lines of `lab.log`, and anything surprising. `flee_rows.csv` and `lab_cache.jsonl` hold the raw per-run data.")
    # best arm (excluding diagnostics) whose paired CI does not show harm; control is the fallback
    best, best_mean = "control", None
    for n, (ks, r) in summ.items():
        if n.startswith("diag_"):
            continue
        sc = [r[s]["score"] for s in ks]
        mean = sum(sc) / len(sc)
        ok = n == "control" or paired_stats(sc, [base[s]["score"] for s in ks])["lo"] > -60
        if ok and (best_mean is None or mean > best_mean):
            best, best_mean = n, mean
    if summ:
        lab.params(best).save(os.path.join(out, "best_arm.json"), arm=best, note="highest-mean arm whose paired CI vs control shows no harm")
        w(f"\n**Best arm: `{best}`** (mean {best_mean:.0f}) saved to `{os.path.join(out, 'best_arm.json')}`. "
          f"Next: `python -m survivor.pipeline run --base {os.path.join(out, 'best_arm.json')} --out runs/pipe2 --budget-hours 1.5 --workers 15` to tune it, "
          f"or serve it: `SURVIVOR_PARAMS={os.path.join(out, 'best_arm.json')} python -m survivor.server`.")
    open(os.path.join(out, "LAB_REPORT.md"), "w").write("\n".join(L) + "\n")


def fit_escape(out, lab=None):
    cache = lab.cache if lab else Cache(os.path.join(out, "lab_cache.jsonl"))
    rows = []
    for (k, s), r in cache.d.items():
        rows.extend(r.get("flee", []))
    if len(rows) < 50:
        print(f"escape model: only {len(rows)} flee episodes logged - need more runs")
        return None
    import csv
    with open(os.path.join(out, "flee_rows.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(EM.ROW)
        cw.writerows(rows)
    m = EM.fit(rows)
    EM.save(m, os.path.join(out, "escape_model.json"))
    print(f"escape model: n={m['n']} death rate {100 * m['base_rate']:.0f}% AUC {m['auc']:.2f} -> {os.path.join(out, 'escape_model.json')}")
    for f, wt in zip(m["features"], m["w"][1:]):
        print(f"   {f:<8} {wt:+.2f}")
    if lab is not None:
        write_report(out, lab)
    return m


def fit_value(out, lab=None):
    cache = lab.cache if lab else Cache(os.path.join(out, "lab_cache.jsonl"))
    rows = []
    for (k, s), r in cache.d.items():
        rows.extend(r.get("val", []))
    if len(rows) < 200:
        print(f"value model: only {len(rows)} logged windows - need more runs")
        return None
    import csv
    with open(os.path.join(out, "value_rows.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(VM.ROW)
        cw.writerows(rows)
    m = VM.fit(rows)
    VM.save(m, os.path.join(out, "value_model.json"))
    print(f"value model: n={m['n']} windows, 5-s survival {100 * m['survive5']:.1f}%, AUC {m['auc']:.2f} -> {os.path.join(out, 'value_model.json')}")
    for f, wt in zip(m["features"], m["w"][1:]):
        print(f"   {f:<10} {wt:+.2f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--base", default=os.path.join(ROOT, "configs", "v9_pipe1_winner.json"))
    r.add_argument("--budget-min", type=float, default=25.0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--seeds", type=int, default=48)
    r.add_argument("--seed0", type=int, default=7001)
    r.add_argument("--round-size", type=int, default=8)
    r.add_argument("--horizon", type=float, default=3000.0)
    r.add_argument("--arms", nargs="*", default=None)
    r.add_argument("--escape-model", default=None, help="add the escape_terminal arm using this model")
    fv = sub.add_parser("fit-value")
    fv.add_argument("out")
    f = sub.add_parser("fit-escape")
    f.add_argument("out")
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("--base", default=os.path.join(ROOT, "configs", "v9_pipe1_winner.json"))
    p.add_argument("--horizon", type=float, default=3000.0)
    a = ap.parse_args()
    if a.cmd == "fit-value":
        fit_value(a.out)
    elif a.cmd == "fit-escape":
        fit_escape(a.out)
    elif a.cmd == "report":
        a.arms, a.escape_model, a.workers, a.seeds, a.seed0, a.round_size, a.budget_min = None, None, 1, 0, 7001, 8, 0
        a.out = a.out
        lab = Lab(a)
        write_report(a.out, lab)
        print("written", os.path.join(a.out, "LAB_REPORT.md"))
    else:
        Lab(a).run()


if __name__ == "__main__":
    main()
