"""
survivor.finale2 - round 2 of the final experiments, on top of the round-1 winner (`configs/v10_stack.json`).

  python -m survivor.finale2 run --out runs/fin2 --budget-min 35 --workers 15
  # optional second pass (10-15 min) with the meta-reward hazard model fitted in the first pass:
  python -m survivor.finale2 run --out runs/fin2b --budget-min 15 --workers 15 --value-model runs/fin2/value_model.json --reuse runs/fin2/lab_cache.jsonl

WHAT CHANGED SINCE ROUND 1 (all evidence-based)
  * round 1 found a real gain on FRESH seeds: stack 1021 vs control 877, paired +144 [+64,+227] (34 wins / 14 losses) -> it is the new control.
  * duel arenas (real controller vs one aimed predator): 86% of deaths of FAST agents begin with first sight inside the 90-unit charge zone;
    36% end stuck on an obstacle, 45% with exhausted energy; facing the predator is right (survival 0.51 facing vs 0.20 never facing).
  * the score is colony time (+dt/step), so a per-agent "2 reward per 5 s" would favour big fragile colonies. It is used as a DENSE MEASUREMENT instead:
    every run logs agent 5-s survival windows; the report gives per-arm predator hazard per 1000 agent-seconds (paired), a far less noisy readout than
    the final score, and a hazard model (value_model.json) that can gate births (arm `vm_gate`, second pass).
STAGES  B: screen ~11 arms on 24-32 paired seeds (racing)   C: FRESH seeds 13001+: control + top-2 by shrunk score + best-hazard arm + stack.
"""
import argparse
import os
import sys
from collections import OrderedDict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from survivor import finale as F  # noqa: E402
from survivor import lab as L  # noqa: E402
from survivor.micro import paired_stats  # noqa: E402

PERC = F.PERC
HEAR_ONLY = {"w_hearing": 6.0, "w_speed": 0.5, "w_vision": 0.2, "w_cone": 0.2}
ARMS2 = OrderedDict([
    ("react150", ({"pred_react_dist": 150.0}, "ignore predators farther than 150 (round 1: +99 [-33,+220] on the old base)")),
    ("cpa200", ({"pred_cpa": 200.0}, "also dodge unnoticed predators passing within 200 (round 1: +48)")),
    ("side0", ({"pred_side_w": 0.0}, "flee straight away, no sidestep (round 1: +46)")),
    ("perc_breed", (PERC, "breed for vision + cone + hearing (round 1: +41)")),
    ("hear_only", (HEAR_ONLY, "breed almost only for hearing radius (omni detection; 86% of fast-agent deaths start inside the charge zone)")),
    ("reserve_eat", (F.RES, "below the sprint floor + 40 eat any fruit at once (round 1: +60)")),
    ("meta_parent", ({"age_bonus": 1.0, "w_survivor": 1.0}, "meta-reward parent ranking: oldest survivors with past escapes breed first")),
    ("stagger", ({"birth_gap_s": 8.0, "gap_after_s": 300.0}, "minimum 8 s between births after t=300 (best juvenile survival in round 1)")),
    ("kite", ({"kite": 1.0}, "no sprint against a measurably tired/resting predator (null in duels; cheap control)")),
    ("fortress", ({"late_t": 900.0, "late_rest_frac": 0.30, "late_patience": 60.0, "cap_late": 4.0}, "late game: rest at 30% energy, wait 60 s at a tree, target colony of 4")),
])
FR2 = OrderedDict([("A", 0.0), ("B", 0.60), ("C", 0.40)])


class Finale2(F.Finale):
    ARMS = None            # set below (ARMS2); a subclass may replace it
    FRESH = 13001

    def __init__(self, a):
        from types import SimpleNamespace
        os.makedirs(a.out, exist_ok=True)
        ns = SimpleNamespace(out=a.out, base=a.base, escape_model=None, workers=a.workers, arms=None, horizon=a.horizon)
        self.a = a
        self.lab = L.Lab(ns)
        arms = OrderedDict([("control", ({}, None, None, "round-1 winner (stack) with per-agent tactics unchanged"))])
        for n, (over, desc) in list((self.ARMS or ARMS2).items())[: (3 if a.smoke else None)]:
            arms[n] = (over, None, None, desc)
        if a.value_model:
            arms["vm_gate"] = ({"vm_gate": 1.0, "vm_drop": 0.01, "vm_min": 0.90}, {"value_model": os.path.abspath(a.value_model)},
                               None, "births gated by the learned meta-reward hazard model (parent's 5-s survival may not drop > 1 pp)")
        self.lab.arms = arms
        self.lab.key = self._key
        import time
        self.seeds_main, self.seeds_fresh = 7001, self.FRESH
        self.t0 = time.time()
        self.end = self.t0 + a.budget_min * 60.0
        self.log_f = self.lab.logf
        self.preload(a.reuse or [])
        F.FR.clear()
        F.FR.update(FR2)

    def _key(self, name):
        # no new-parameter compatibility shim needed: this base is new. Knobs (value-model path) are part of the key.
        from dataclasses import asdict
        from survivor.pipeline import cfg_key
        return cfg_key(asdict(self.lab.params(name)), self.a.horizon, self.lab.arms[name][1])

    def hazard_pick(self, names):
        """arm with the best paired predator-hazard improvement (dense meta-reward signal) - nominated even if its score CI spans 0"""
        seeds = list(range(self.seeds_main, self.seeds_main + 300))
        base = self.lab.res("control", seeds)
        best = None
        for n in names:
            if n == "control":
                continue
            r = self.lab.res(n, seeds)
            ks = [s for s in seeds if s in r and s in base and r[s].get("hz") and base[s].get("hz") and r[s]["hz"]["agent_s"] > 0 and base[s]["hz"]["agent_s"] > 0]
            if len(ks) < (2 if self.a.smoke else self.a.min_n):
                continue
            x = [1000.0 * r[s]["hz"]["pred"] / r[s]["hz"]["agent_s"] for s in ks]
            b = [1000.0 * base[s]["hz"]["pred"] / base[s]["hz"]["agent_s"] for s in ks]
            ps = paired_stats(x, b)
            if best is None or ps["mean"] < best[0]:
                best = (ps["mean"], n, ps)
        return best

    def run(self):
        a = self.a
        self.log(f"finale2: {len(self.lab.arms)} arms on top of {os.path.basename(a.base)}, budget {a.budget_min} min, {a.workers} workers, horizon {a.horizon:.0f}s"
                 + (" [SMOKE]" if a.smoke else ""))
        nB = 4 if a.smoke else a.seeds_b
        seedsB = list(range(self.seeds_main, self.seeds_main + nB))
        self.log("== Stage B: screen arms on the round-1 winner (racing) ==")
        self.run_stage(list(self.lab.arms), seedsB, self.deadline("B"), race=True)
        ranked = self.ranking(list(self.lab.arms))
        self.log("== ranking by shrunk paired score effect (prior sd %.0f) ==" % F.TAU)
        for shr, n, ps in ranked[:12]:
            self.log(f"   {n:<12} shrunk {shr:+6.1f} | raw {ps['mean']:+6.1f} [{ps['lo']:+.0f},{ps['hi']:+.0f}] n={ps['n']}")
        hp = self.hazard_pick(list(self.lab.arms))
        if hp:
            self.log(f"== best paired predator-hazard arm (meta-reward signal): {hp[1]} {hp[0]:+.2f} per 1000 agent-s [{hp[2]['lo']:+.2f},{hp[2]['hi']:+.2f}] ==")
        top = [n for _, n, _ in ranked[:2]]
        if hp and hp[1] not in top and hp[0] < 0:
            top.append(hp[1])
        over, used = self.build_stack(ranked)
        cands = ["control"] + top
        if len(used) >= 2 and all(set(over.items()) != set(self.lab.arms[n][0].items()) for n in top):
            self.lab.arms["stack2"] = (over, None, None, "stack of " + " + ".join(used))
            cands.append("stack2")
        self.log(f"== Stage C: FRESH-seed verification of {cands} ==")
        nC = 4 if a.smoke else a.seeds_c
        fresh = list(range(self.seeds_fresh, self.seeds_fresh + nC))
        self.run_stage(cands, fresh, self.deadline("C"))
        self.finish(cands, fresh, ranked, used)
        L.fit_value(a.out, self.lab)
        L.write_report(a.out, self.lab)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--base", default=os.path.join(ROOT, "configs", "v10_stack.json"))
    r.add_argument("--budget-min", type=float, default=35.0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--horizon", type=float, default=3000.0)
    r.add_argument("--reuse", nargs="*", default=None)
    r.add_argument("--value-model", default=None, help="adds the vm_gate arm using this fitted hazard model")
    r.add_argument("--seeds-b", type=int, default=32)
    r.add_argument("--seeds-c", type=int, default=48)
    r.add_argument("--min-n", type=int, default=20)
    r.add_argument("--min-gain", type=float, default=20.0)
    r.add_argument("--escape-model", default=None)
    r.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    Finale2(a).run()


if __name__ == "__main__":
    main()
