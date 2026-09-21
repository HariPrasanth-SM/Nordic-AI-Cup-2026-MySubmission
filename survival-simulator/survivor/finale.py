"""
survivor.finale - the 35-minute final round, built from what labs 1 and 2 showed.

  python -m survivor.finale run --out runs/fin1 --budget-min 35 --workers 15 \
        --reuse runs/lab1/lab_cache.jsonl runs/lab2/lab_cache.jsonl --escape-model runs/lab2/escape_model.json

WHY (from your lab data)
  * The ~1800 in lab1 was the DIAGNOSTIC arm with predators switched off; real arms are 900-1015 and none beat control significantly
    (best +79 [-31,+191] on 32 seeds is what noise gives: expected best of 12 null arms ~ +70).
  * The escape model (AUC 0.8) says risk is driven by energy fraction and first-sight distance; predators are worth +848, food +280.
  * Predator-response parameters were never tuned in full runs.
  So: extend the arms that looked best (free: cached seeds are reused), screen new predator-side ideas one factor at a time,
  and then decide ONLY on a fresh seed range no earlier step ever touched.

STAGES (deadline-driven rounds of 8 seeds over all arms of a stage, so a deadline always leaves balanced data)
  A extend    lab1's best arms (mgr_reserve, phase, mgr_spread, mgr_stagger) + control to more paired seeds.
  B screen    new arms: escape-reserve eating, breeding for perception/low max_energy, stacked structural combos, and one-factor-at-a-time
              predator-response parameters (both directions). Racing drops clearly harmful arms after 16 seeds.
  C verify    control + the top-2 arms by SHRUNK paired effect (empirical-Bayes shrinkage vs winner's curse) + a stack of the best
              non-conflicting arms, on FRESH seeds (11001+). Winner = highest fresh mean whose paired CI shows no harm, else control.
Outputs: final_config.json, FINALE_REPORT.md, LAB_REPORT.md (mechanism tables for every arm), lab.log, lab_cache.jsonl.
"""
import argparse
import json
import os
import shutil
import statistics as st
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor import lab as L  # noqa: E402
from survivor.micro import paired_stats, verdict  # noqa: E402
from survivor.pipeline import cfg_key  # noqa: E402

NEW_DEFAULTS = {"urgent_floor": 0.0, "urgent_margin": 40.0}     # params added after lab1: omitted from the cache key at default so lab1/lab2 results stay reusable
TAU = 60.0                                                        # prior sd of a real effect (points) for shrinkage
SEEDS_MAIN = 7001                                                 # seeds used by labs 1 and 2 (arms A/B are compared here)
SEEDS_FRESH = 11001                                               # never used by any earlier selection (the pipeline's 9001-9080 favoured the current control)

RES = {"urgent_floor": 1.0, "urgent_margin": 40.0}
PERC = {"w_vision": 1.0, "w_cone": 1.0, "w_hearing": 0.8, "w_speed": 0.6}
CRS = {"colony_mgr": 1, "young_frac": 0.5, "reserve_margin": 40.0, "birth_gap_s": 8.0, "gap_after_s": 300.0}
PHASE = {"late_t": 1200.0, "late_rest_frac": 0.45}
SPREAD = {"crowd_max": 2, "spread_w": 1.5}

A_ARMS = ["control", "mgr_reserve", "phase", "mgr_spread", "mgr_stagger"]
HEAR = {"w_hearing": 2.0, "w_speed": 0.8, "w_vision": 0.3, "w_cone": 0.2}
B_ARMS = OrderedDict([
    # --- escape readiness / perception (the two strongest risk factors in the escape model: energy fraction, first-sight distance)
    ("reserve_eat", (RES, "below the sprint floor + 40 eat any fruit at once (escape-ready energy)")),
    ("hear_breed", (HEAR, "breed hard for hearing radius (omni detection: 50 -> up to 100), speed second")),
    ("perc_breed", (PERC, "breed for vision + cone + hearing")),
    ("lowE_breed", ({"w_energy": -1.0}, "breed harder for low max_energy (higher energy FRACTION at encounters)")),
    ("c_rsp", ({**CRS, **PHASE}, "colony births + reserve + staggering + survival-mode phase")),
    ("c_all_pr", ({**CRS, **PHASE, **SPREAD, **RES, **PERC}, "everything structural + reserve eating + perception breeding")),
    # --- predator-response parameters: NEVER tuned on full runs (one factor at a time, both directions)
    ("cpa0", ({"pred_cpa": 0.0}, "react only to predators that have noticed us (fewer wasted flees)")),
    ("cpa200", ({"pred_cpa": 200.0}, "also dodge unnoticed predators whose path passes within 200")),
    ("sprint60", ({"pred_sprint_dist": 60.0}, "sprint only when the predator is within 60")),
    ("sprint160", ({"pred_sprint_dist": 160.0}, "sprint from 160")),
    ("hold3", ({"flee_hold": 3.0}, "stop fleeing sooner (0.3 s)")),
    ("hold25", ({"flee_hold": 25.0}, "keep fleeing 2.5 s after the last sighting")),
    ("react150", ({"pred_react_dist": 150.0}, "ignore predators farther than 150")),
    ("react300", ({"pred_react_dist": 300.0}, "react from 300")),
    ("fleecap24", ({"flee_cap": 24.0}, "flee at up to speed 24 instead of 17")),
    ("side0", ({"pred_side_w": 0.0}, "flee straight away, no sidestep")),
    # --- energy waste never tuned: scanning, exploring, foraging radius
    ("scan_off", ({"camp_scan": 0.0, "vig_k": 0.0}, "no scanning rotation while camping")),
    ("explore_nospin", ({"explore_spin": 0.0}, "look ahead only while exploring (spin costs ~1.4 energy/s)")),
    ("fruit150", ({"fruit_max_dist": 150.0}, "forage only within 150 (less exposure)")),
    ("fruit450", ({"fruit_max_dist": 450.0}, "forage up to 450")),
])
FR = OrderedDict([("A", 0.05), ("B", 0.55), ("C", 0.40)])


class Finale:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        ns = SimpleNamespace(out=a.out, base=a.base, escape_model=a.escape_model, workers=a.workers, arms=None, horizon=a.horizon)
        self.lab = L.Lab(ns)
        arms = OrderedDict()
        for n in (A_ARMS[:2] if a.smoke else A_ARMS):
            arms[n] = L.ARMS[n]
        for n, (over, desc) in list(B_ARMS.items())[: (3 if a.smoke else None)]:
            arms[n] = (over, None, None, desc)
        if a.escape_model and not a.smoke:
            arms["mgr_terminal"] = ({**CRS, "use_escape_model": 1.0, "doom_p": 0.6}, None, None, "colony births + reserve + terminal spawn from the escape model")
        self.lab.arms = arms
        self.lab.key = self._key
        self.seeds_main, self.seeds_fresh = SEEDS_MAIN, SEEDS_FRESH
        self.t0 = time.time()
        self.end = self.t0 + a.budget_min * 60.0
        self.log_f = self.lab.logf
        self.preload(a.reuse or [])

    # ------------------------------------------------------------------ infra
    def log(self, m):
        self.lab.log(m)

    def _key(self, name):
        pd = asdict(self.lab.params(name))
        for k, dv in NEW_DEFAULTS.items():
            if abs(pd[k] - dv) < 1e-12:
                pd.pop(k)
        return cfg_key(pd, self.a.horizon, self.lab.arms[name][1])

    def preload(self, paths):
        n = 0
        for p in paths:
            if not os.path.exists(p):
                self.log(f"reuse: {p} not found (skipped)")
                continue
            for line in open(p):
                line = line.strip()
                if line:
                    x = json.loads(line)
                    k = (x["k"], x["seed"])
                    if k not in self.lab.cache.d:
                        self.lab.cache.d[k] = x["r"]
                        n += 1
        self.log(f"reuse: {n} cached runs loaded from {len(paths)} file(s)")

    def deadline(self, stage):
        names = list(FR.keys())
        i = names.index(stage)
        rest = sum(FR[n] for n in names[i:])
        return time.time() + max(0.0, self.end - time.time()) * FR[stage] / rest

    def pair(self, name, seeds):
        base = self.lab.res("control", seeds)
        r = self.lab.res(name, seeds)
        ks = [s for s in seeds if s in r and s in base]
        if not ks:
            return None
        return paired_stats([r[s]["score"] for s in ks], [base[s]["score"] for s in ks])

    def run_stage(self, names, seed_list, deadline, race=False):
        alive = list(names)
        done, rnd = 0, (2 if self.a.smoke else 8)
        while done < len(seed_list) and alive and time.time() < deadline:
            seeds = seed_list[done:done + rnd]
            jobs, seen = [], set()
            for s in seeds:
                for n in alive:
                    k = self.lab.key(n)
                    if (k, s) in seen or self.lab.cache.has(k, s):
                        continue
                    seen.add((k, s))
                    jobs.append((k, asdict(self.lab.params(n)), s, self.a.horizon, self.lab.arms[n][1]))
            chunk = max(self.a.workers * 2, 2)
            for c in range(0, len(jobs), chunk):
                if time.time() > deadline:
                    break
                part = jobs[c:c + chunk]
                res = self.lab.pool.map(L._lab_job, part, chunksize=1) if self.lab.pool else [L._lab_job(j) for j in part]
                for k, s, r in res:
                    self.lab.cache.put(k, s, r)
            done += rnd
            used = seed_list[:done]
            if race:
                for n in list(alive):
                    if n == "control":
                        continue
                    ps = self.pair(n, used)
                    if ps and ps["n"] >= (2 if self.a.smoke else 16) and ps["hi"] < 0:
                        alive.remove(n)
                        self.log(f"  drop {n}: {ps['mean']:+.0f} [{ps['lo']:+.0f},{ps['hi']:+.0f}] n={ps['n']}")
            self.lab.table(used)
        return alive

    def ranking(self, names):
        seeds = list(range(self.seeds_main, self.seeds_main + 300))
        out = []
        for n in names:
            if n == "control":
                continue
            ps = self.pair(n, seeds)
            if ps and ps["n"] >= (2 if self.a.smoke else self.a.min_n):
                se = max((ps["hi"] - ps["lo"]) / 3.92, 1e-6)
                shr = ps["mean"] * TAU ** 2 / (TAU ** 2 + se ** 2)
                out.append((shr, n, ps))
        return sorted(out, key=lambda z: -z[0])

    def build_stack(self, ranked):
        over, used = {}, []
        for shr, n, ps in ranked:
            if shr <= 0 or n in ("stack",):
                continue
            o = self.lab.arms[n][0]
            if any(k in over and abs(float(over[k]) - float(v)) > 1e-12 for k, v in o.items()):
                continue
            over.update(o)
            used.append(n)
            if len(used) >= 3:
                break
        return over, used

    # ------------------------------------------------------------------ run
    def run(self):
        a = self.a
        self.log(f"finale: {len(self.lab.arms)} arms, budget {a.budget_min} min, {a.workers} workers, horizon {a.horizon:.0f}s"
                 + (" [SMOKE]" if a.smoke else ""))
        nA = 2 if a.smoke else a.seeds_a
        nB = 4 if a.smoke else a.seeds_b
        seedsA = list(range(self.seeds_main, self.seeds_main + nA))
        seedsB = list(range(self.seeds_main, self.seeds_main + nB))
        if nA > 0:
            self.log("== Stage A: extend lab1's best arms (cached seeds reused) ==")
            self.run_stage([n for n in self.lab.arms if n in A_ARMS], seedsA, self.deadline("A"))
        else:
            self.log("== Stage A skipped: lab1 arms enter the ranking with their cached seeds ==")
        self.log("== Stage B: screen new predator-side / structural arms (racing) ==")
        self.run_stage(["control"] + [n for n in self.lab.arms if n not in A_ARMS], seedsB, self.deadline("B"), race=True)
        ranked = self.ranking(list(self.lab.arms))
        self.log("== ranking by shrunk paired effect (prior sd %.0f) ==" % TAU)
        for shr, n, ps in ranked[:10]:
            self.log(f"   {n:<14} shrunk {shr:+6.1f} | raw {ps['mean']:+6.1f} [{ps['lo']:+.0f},{ps['hi']:+.0f}] n={ps['n']}")
        top = [n for _, n, _ in ranked[:2]]
        over, used = self.build_stack(ranked)
        cands = ["control"] + top
        if len(used) >= 2 and all(set(over.items()) != set(self.lab.arms[n][0].items()) for n in top):
            self.lab.arms["stack"] = (over, None, None, "stack of " + " + ".join(used))
            cands.append("stack")
        self.log(f"== Stage C: FRESH-seed verification of {cands} ==")
        nC = 4 if a.smoke else a.seeds_c
        fresh = list(range(self.seeds_fresh, self.seeds_fresh + nC))
        self.run_stage(cands, fresh, self.deadline("C"))
        self.finish(cands, fresh, ranked, used)

    def finish(self, cands, fresh, ranked, used):
        rows = []
        base = self.lab.res("control", fresh)
        for n in cands:
            r = self.lab.res(n, fresh)
            ks = [s for s in fresh if s in r and s in base]
            if not ks:
                continue
            sc = [r[s]["score"] for s in ks]
            row = {"name": n, "n": len(ks), "mean": sum(sc) / len(sc), "median": st.median(sc), "sd": st.pstdev(sc) if len(sc) > 1 else 0.0,
                   "p1000": sum(1 for x in sc if x >= 1000) / len(sc), "p1500": sum(1 for x in sc if x >= 1500) / len(sc), "max": max(sc),
                   "surv": sum(r[s]["sim_t"] for s in ks) / len(ks)}
            if n != "control":
                ps = paired_stats(sc, [base[s]["score"] for s in ks])
                row.update({"d": ps["mean"], "lo": ps["lo"], "hi": ps["hi"], "w": ps["wins"], "l": ps["losses"]})
            rows.append(row)
        ctrl = next((r for r in rows if r["name"] == "control"), None)
        winner = "control"
        best = ctrl["mean"] if ctrl else -1e9
        for r in rows:
            if r["name"] != "control" and r["mean"] > max(best, (ctrl["mean"] if ctrl else 0) + self.a.min_gain) and r.get("lo", -1e9) > -60:
                winner, best = r["name"], r["mean"]
        over = self.lab.arms[winner][0]
        p = self.lab.params(winner)
        path = os.path.join(self.a.out, "final_config.json")
        p.save(path, winner=winner, table=rows, note="survivor.finale: chosen on FRESH seeds only")
        uses_model = float(over.get("use_escape_model", 0)) > 0.5
        if uses_model and self.a.escape_model:
            shutil.copy(self.a.escape_model, os.path.join(self.a.out, "escape_model.json"))
        self.log(f"DECISION: winner = {winner}  (fresh mean {best:.0f} vs control {ctrl['mean']:.0f})" if ctrl else f"DECISION: winner = {winner}")
        L_ = []
        w = L_.append
        w("# Finale report\n")
        w(f"Budget {self.a.budget_min} min; control = `{self.a.base}`; horizon {self.a.horizon:.0f} s. Stages A/B use seeds {self.seeds_main}+; stage C uses FRESH seeds {self.seeds_fresh}+ that no selection ever saw.\n")
        w("## Ranking after stages A and B (shrunk = paired effect pulled toward 0 by its standard error: protects against winner's curse)")
        w("| arm | what | n | raw paired Δ [95% CI] | shrunk Δ | w/l |")
        w("|---|---|---|---|---|---|")
        for shr, n, ps in ranked:
            w(f"| {n} | {self.lab.arms[n][3]} | {ps['n']} | {ps['mean']:+.0f} [{ps['lo']:+.0f}, {ps['hi']:+.0f}] | {shr:+.0f} | {ps['wins']}/{ps['losses']} |")
        w(f"\nStack candidate built from: {used or 'nothing (fewer than two non-conflicting positive arms)'}\n")
        w("## Stage C: fresh-seed verification (the numbers that decide)")
        w("| config | n | mean | median | sd | mean survival s | P(>=1000) | P(>=1500) | best | paired Δ vs control [95% CI] | w/l |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            d = f"{r['d']:+.0f} [{r['lo']:+.0f}, {r['hi']:+.0f}]" if "d" in r else "-"
            wl = f"{r['w']}/{r['l']}" if "w" in r else "-"
            w(f"| {r['name']} | {r['n']} | {r['mean']:.0f} | {r['median']:.0f} | {r['sd']:.0f} | {r['surv']:.0f} | {100 * r['p1000']:.0f}% | {100 * r['p1500']:.0f}% | {r['max']:.0f} | {d} | {wl} |")
        w(f"\n**Winner: `{winner}`** -> `{path}`  (rule: highest fresh mean that beats control by at least {self.a.min_gain:.0f} points and whose paired CI shows no harm, else control).")
        w("\nReading it: the validation server is ONE draw (the final evaluation averages 3); with sd ~230 a single draw scatters by about that much. "
          "A paired effect below ~100 needs 100+ seeds to confirm, so treat sub-100 differences as 'probably harmless, unproven'.")
        w("\n## Next\n```bash\n" + (f"SURVIVOR_ESCAPE_MODEL={os.path.join(self.a.out, 'escape_model.json')} " if uses_model else "") +
          f"SURVIVOR_PARAMS={path} python -m survivor.server\n"
          f"python -m survivor.runner eval --params {path} --seeds 401 402 403 404 405 406 407 408 409 410 --workers 15 --out runs/fin_eval   # local look\n"
          "python -m survivor.analyze local runs/fin_eval\n```")
        open(os.path.join(self.a.out, "FINALE_REPORT.md"), "w").write("\n".join(L_) + "\n")
        if self.lab.pool:
            self.lab.pool.close()
        L.write_report(self.a.out, self.lab)
        self.log(f"DONE. {os.path.join(self.a.out, 'FINALE_REPORT.md')}  (mechanism tables: LAB_REPORT.md)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--base", default=os.path.join(ROOT, "configs", "v9_pipe1_winner.json"))
    r.add_argument("--budget-min", type=float, default=35.0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--horizon", type=float, default=3000.0)
    r.add_argument("--reuse", nargs="*", default=None, help="lab_cache.jsonl files from earlier labs (same base config => free seeds)")
    r.add_argument("--escape-model", default=None)
    r.add_argument("--seeds-a", type=int, default=0, help="max paired seeds in stage A (0 = skip; lab1 arms rank with cached seeds)")
    r.add_argument("--seeds-b", type=int, default=24)
    r.add_argument("--seeds-c", type=int, default=48)
    r.add_argument("--min-n", type=int, default=20)
    r.add_argument("--min-gain", type=float, default=20.0, help="a candidate must beat control by this many points on the fresh seeds")
    r.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    Finale(a).run()


if __name__ == "__main__":
    main()
