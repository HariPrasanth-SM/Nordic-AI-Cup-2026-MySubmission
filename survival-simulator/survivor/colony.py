"""
survivor.colony - the colony layer of the hierarchical controller (the strategy document's population + energy managers).

Per-agent behaviour (camp / forage / flee ...) stays in policy.Controller. This layer sees the WHOLE colony every tick and:
  * measures colony state: population, age cohorts, energy, escape-ready fraction (energy >= sprint floor + ~4 s of sprint);
  * (params.colony_mgr > 0.5) decides births at COLONY level instead of per agent:
      - demand-driven: births only while population < target (the cap schedule) and the young cohort is not already large
        (age-balanced, no boom/bust bursts); optional minimum gap between births after `gap_after_s` (staggering);
      - parent chosen by ranking (surplus energy, age = closest to dying, trait rank, past escapes), never below the
        escape reserve (parent keeps >= 0.2*max_energy + reserve_margin after paying 100);
      - senescent agents (aging drain detected) always convert their energy into children;
      - lifeboats: up to n_lifeboat isolated, energetic agents are excluded from breeding while the colony is larger than that;
      - terminal spawn: an agent the escape model says will very probably die (P >= doom_p) spawns first;
  * logs every decision with a reason code, so the run explains itself (births by reason, denials by reason, cohorts over time).
"""
from collections import Counter

READY_MARGIN = 40.0


class ColonyManager:
    def __init__(self, ctrl):
        self.c = ctrl
        self.last_birth_t = -1e9
        self.last_sample = -1e9
        self.last_boat_t = -1e9
        self.boats = set()
        self.births = Counter()
        self.deny = Counter()
        self.tl = []            # (t, n, young, mid, old, ready_frac, mean_E, target, n_boats)
        self.events = []        # (t, reason, energy, age, n)

    # ------------------------------------------------------------------
    def after(self, step, agents, out, t):
        p = self.c.p
        n = len(agents)
        if n == 0:
            return
        ages = [a["age"] for a in agents]
        young = sum(1 for x in ages if x < p.young_age_s)
        old = sum(1 for x in ages if x >= 75.0)
        ready = sum(1 for a in agents if a["energy"] >= 0.2 * a["max_energy"] + READY_MARGIN)
        target = self.c.pop_cap(t)
        if p.n_lifeboat >= 1 and t - self.last_boat_t >= p.lifeboat_hold_s:
            self._pick_boats(agents, t, n)
        if p.colony_mgr > 0.5:
            self._approve(agents, out, t, n, young, target)
        if t - self.last_sample >= 10.0:
            self.last_sample = t
            self.tl.append((round(t), n, young, n - young - old, old, round(ready / n, 3), round(sum(a["energy"] for a in agents) / n, 1),
                            round(target, 1), len(self.boats)))

    def _pick_boats(self, agents, t, n):
        p = self.c.p
        self.last_boat_t = t
        L = int(min(p.n_lifeboat, max(0, 0.4 * n)))
        cands = []
        for a in agents:
            m = self.c.mem.get(a["agent_id"])
            if m and a["energy"] >= 0.2 * a["max_energy"] + READY_MARGIN and a["age"] >= 20.0:
                cands.append((m.get("nb", 0), -a["energy"], a["agent_id"]))
        cands.sort()
        self.boats = {c[2] for c in cands[:L]}

    def _approve(self, agents, out, t, n, young, target):
        p = self.c.p
        amap = {a["agent_id"]: a for a in out}
        mem = self.c.mem
        r = min(1.0, t / max(p.spawn_ramp, 1.0))
        normal, dbed, sen, doom, anyE = [], [], [], [], []
        for a in agents:
            aid = a["agent_id"]
            m = mem.get(aid)
            if m is None or aid not in amap:
                continue
            E, ME, age = a["energy"], a["max_energy"], a["age"]
            if E <= 102.0:
                continue
            thr = min(p.spawn_thr + (p.spawn_thr_late - p.spawn_thr) * r, 0.85 * ME)              # same schedules as the per-agent rules
            dmin = min(p.dbed_min_energy + (p.dbed_min_late - p.dbed_min_energy) * r, 0.8 * ME)
            reserve = 0.2 * ME + p.reserve_margin                                                  # escape reserve the parent keeps
            surplus = E - 100.0 - reserve
            score = surplus + p.age_bonus * max(0.0, age - 40.0) + p.w_survivor * 40.0 * min(m.get("escapes", 0), 5)
            if p.senescence > 0.5 and m.get("sen") is not None:
                sen.append((score + 1000.0, aid, E, age))
                continue
            threatened = m.get("threat_now")
            pdie = m.get("p_die")
            doomed_plan = p.mpc_terminal > 0.5 and threatened and m.get("doomed")
            if ((p.use_escape_model > 0.5 and threatened and pdie is not None and pdie >= p.doom_p) or doomed_plan) and E >= p.doom_min_e:
                doom.append((score, aid, E, age))
                continue
            if threatened:
                continue
            anyE.append((score, aid, E, age))
            if aid in self.boats and n > p.n_lifeboat + 1:
                continue
            if m.get("food_near", 99) < p.spawn_food_min:
                continue
            if p.spawn_quiet_s > 0 and t - self.c.last_pred_t < p.spawn_quiet_s:
                continue
            if not m.get("sel_ok", True) and n > 4:
                continue
            if p.vm_gate > 0.5 and self.c.value is not None and (E >= thr or age >= p.dbed_age):
                f0 = self.c._val_feats(a, m, t, n)
                f1 = self.c._val_feats(a, m, t, n, e=E - 100.0)
                p0, p1 = self.c.value.p_surv(*f0), self.c.value.p_surv(*f1)
                if p1 < p.vm_min or p0 - p1 > p.vm_drop:            # the birth would make the parent too likely to die within 5 s
                    self.deny["vm_gate"] += 1
                    continue
            if E >= thr and surplus >= 0:
                normal.append((score, aid, E, age, "normal"))
            elif age >= p.dbed_age and E >= dmin:                       # death-bed: about to age out, spend the energy on a child
                dbed.append((score, aid, E, age, "dbed"))

        def grant(cand, reason):
            score, aid, E, age = cand[:4]
            amap[aid]["spawn_agent"] = True
            self.births[reason] += 1
            if len(self.events) < 600:
                self.events.append((round(t, 1), reason, round(E), round(age), n))

        given = 0
        for cand in sorted(sen, reverse=True)[:int(p.max_births_tick)]:            # dying anyway: aging would burn the energy
            if n < target + p.sen_extra + given:
                grant(cand, "senescent"); given += 1
        for cand in sorted(doom, reverse=True)[:1]:
            if n < target + p.dbed_extra + p.sen_extra + given:
                grant(cand, "terminal"); given += 1
        if n <= p.n_min:                                                             # colony nearly extinct
            em = [c for c in anyE if c[2] >= p.emergency_min_e] or anyE
            if em and t - self.last_birth_t >= 1.0:
                grant(max(em), "emergency"); self.last_birth_t = t
            return
        deficit = target - n
        young_cap = p.young_frac * max(target, n)
        gap = p.birth_gap_s if t >= p.gap_after_s else 0.0
        pool = (normal if deficit >= 0.5 else []) + (dbed if deficit > -p.dbed_extra else [])
        if not pool:
            self.deny["no_eligible_parent" if not (normal or dbed) else "at_target"] += 1
        elif young >= young_cap + 0.5:
            self.deny["young_cohort_full"] += 1
        elif t - self.last_birth_t < gap:
            self.deny["gap"] += 1
        else:
            best = max(pool)
            grant(best, best[4])
            self.last_birth_t = t

    # ------------------------------------------------------------------
    def summary(self):
        tl = self.tl[::3]                      # every 30 s
        return {"births": dict(self.births), "deny": dict(self.deny), "timeline": tl[:120], "events": self.events[:200]}
