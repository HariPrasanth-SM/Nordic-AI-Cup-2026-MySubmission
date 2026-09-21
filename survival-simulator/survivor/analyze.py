"""
survivor.analyze - quantitative + qualitative analysis.

  python -m survivor.analyze local  runs/e1            # -> runs/e1/analysis/report.md + PNGs   (ground-truth diagnostics)
  python -m survivor.analyze server logs/server_*.jsonl # -> report of a validation/evaluation run as seen by the server

Paste report.md (it is compact and self-explanatory) plus the validation score(s) back into the chat.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _f(x, n=1):
    return f"{x:.{n}f}"


def load_runs(d):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "seed_*.json")))]
    if not runs:
        sys.exit(f"no seed_*.json in {d}")
    return runs


def _at(ts, key, t):
    """value of ts[key] at (nearest sample <=) time t"""
    best = None
    for tt, v in zip(ts["t"], ts[key]):
        if tt <= t:
            best = v
        else:
            break
    return best


def local_report(d):
    runs = load_runs(d)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    L = []
    w = L.append
    scores = [r["score"] for r in runs]
    w(f"# Local run analysis: `{d}`  ({len(runs)} seeds, horizon {runs[0]['horizon']:.0f}s)\n")
    w("## 1. Headline")
    w(f"- mean score **{st.mean(scores):.1f}**, min {min(scores):.1f}, max {max(scores):.1f}, "
      f"full-horizon survival in **{sum(r['survived_full'] for r in runs)}/{len(runs)}** seeds")
    w("- (evaluation = average of 3 preset-seed runs; a full 3000 s survival is worth ~3000 points, everything else is a few %)\n")
    w("| seed | score | sim_t | alive_end | peak | mean_pop | births | died: predator / starved_young / starved_old | fruit eaten | survival | fruit pts | predation pts | wall s |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        d_, p_ = r["deaths"], r["score_parts"]
        w(f"| {r['seed']} | {r['score']:.1f} | {r['sim_time']:.0f} | {r['alive_end']} | {r['peak_pop']} | {r['mean_pop']:.1f} | {r['births']} | "
          f"{d_['predator']} / {d_['starved_young']} / {d_['starved_old']} | {r['fruits_eaten']} | {p_['survival']:.0f} | {p_['fruit']:.1f} | {p_['predation_penalty']:.1f} | {r['timing']['wall_s']:.0f} |")

    # ---- timeline checkpoints
    w("\n## 2. Timeline (mean over seeds still running at that time)")
    w("| t | pop | mean energy | mean age | trees (mature) | fruits on map | predators | cum. predator deaths | cum. young starved | mean speed trait | mean hearing | mean vision | escape-ready % |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    horizon = runs[0]["horizon"]
    for t in [x for x in range(0, int(horizon) + 1, max(100, int(horizon // 10)))][1:]:
        rows = [(r, _at(r["ts"], "n", t)) for r in runs if r["sim_time"] >= t - 1]
        if not rows:
            continue
        def m(key):
            vals = [_at(r["ts"], key, t) for r, _ in rows if key in r["ts"]]
            vals = [v for v in vals if v is not None]
            return st.mean(vals) if vals else float("nan")
        w(f"| {t} | {m('n'):.1f} ({len(rows)}/{len(runs)} alive) | {m('mean_E'):.0f} | {m('mean_age'):.0f} | {m('trees'):.0f} ({m('mature'):.0f}) | "
          f"{m('fruits'):.0f} | {m('preds'):.1f} | {m('d_pred'):.0f} | {m('d_starve_young'):.0f} | {m('mean_speed'):.1f} | {m('mean_hear'):.0f} | {m('mean_vision'):.0f} | {100 * m('ready'):.0f} |")

    # ---- deaths
    w("\n## 3. Death analysis (all seeds)")
    allw = [(r["seed"], x) for r in runs for x in r["deaths_list"]]
    by = Counter(x[1] for _, x in allw)
    tot = max(1, sum(by.values()))
    w(f"- deaths: " + ", ".join(f"{k} {v} ({100 * v / tot:.0f}%)" for k, v in by.most_common()))
    for cause in ("predator", "starved_young", "starved_old"):
        xs = [x for _, x in allw if x[1] == cause]
        if xs:
            ages = [x[2] for x in xs]
            es = [x[3] for x in xs]
            extra = ""
            if cause == "predator":
                low = sum(1 for e in es if e < 100) / len(es)
                extra = f"; energy at death median {st.median(es):.0f} ({100 * low:.0f}% below 100 = could not sprint)"
            w(f"- {cause}: age median {st.median(ages):.0f}s (p10 {sorted(ages)[len(ages) // 10]:.0f}, p90 {sorted(ages)[9 * len(ages) // 10]:.0f}){extra}")
    # predator kill clustering
    clusters = 0
    kills = 0
    for r in runs:
        tk = sorted(x[0] for x in r["deaths_list"] if x[1] == "predator")
        kills += len(tk)
        clusters += sum(1 for a, b in zip(tk, tk[1:]) if b - a < 1.0)
    if kills:
        w(f"- predator kills arriving <1 s after another kill: {clusters}/{kills} ({100 * clusters / kills:.0f}%) -> "
          f"{'agents are bunched together (spread them out)' if clusters / kills > 0.25 else 'kills are mostly isolated'}")
    # predator kills over time
    if kills:
        hist = Counter(int(x[0] // 300) * 300 for _, x in allw if x[1] == "predator")
        w("- predator kills per 300 s window: " + ", ".join(f"{k}-{k + 300}: {v}" for k, v in sorted(hist.items())))


    # ---- death context (ground truth diagnostics)
    COLS = ["t", "cause", "age", "E", "max_age", "x", "y", "state", "thr_ago", "thr_dmin", "tree_d", "nb80", "spd", "maxE", "biome", "pv_ago", "pred_d", "id"]
    ext = [x for _, x in allw if len(x) >= len(COLS)]
    if ext:
        w("\n## 3b. How agents die (context at the moment of death)")
        def c(rows, name):
            return [x[COLS.index(name)] for x in rows]
        pr = [x for x in ext if x[1] == "predator"]
        if pr:
            below = sum(1 for x in pr if x[3] < x[13] / 5.0)
            seen = [x for x in pr if x[8] is not None]
            w(f"- predator victims ({len(pr)}): state {dict(Counter(c(pr, 'state')).most_common())}")
            w(f"  - energy below THEIR OWN sprint floor (max_energy/5): {100 * below / len(pr):.0f}%  (median energy {st.median(c(pr, 'E')):.0f}, median max_energy {st.median(c(pr, 'maxE')):.0f}, median speed {st.median(c(pr, 'spd')):.1f})")
            w(f"  - threat had been registered before death: {100 * len(seen) / len(pr):.0f}%; seconds from first sighting to death: median {st.median([x[8] for x in seen]) if seen else float('nan'):.1f}; closest approach seen: median {st.median([x[9] for x in seen if x[9] is not None]) if seen else float('nan'):.0f} units")
            w(f"  - biome {dict(Counter(c(pr, 'biome')).most_common())}; nearest tree median {st.median(c(pr, 'tree_d')):.0f}; neighbours within 80: mean {st.mean(c(pr, 'nb80')):.2f}")
        ys = [x for x in ext if x[1] == "starved_young"]
        if ys:
            w(f"- young-starved ({len(ys)}): state {dict(Counter(c(ys, 'state')).most_common())}; nearest tree median {st.median(c(ys, 'tree_d')):.0f}; "
              f"neighbours within 80 mean {st.mean(c(ys, 'nb80')):.2f}; biome {dict(Counter(c(ys, 'biome')).most_common(3))}; age median {st.median(c(ys, 'age')):.0f}s")
        B = [b for r in runs for b in r["births_list"] if len(b) >= 12]
        if B:
            w(f"- births ({len(B)}): parent energy median {st.median(b[7] for b in B):.0f}, parent age median {st.median(b[8] for b in B):.0f}s, "
              f"tree distance at birth median {st.median(b[9] for b in B):.0f}, neighbours at birth mean {st.mean(b[10] for b in B):.2f}")
            early = [b for b in B if b[0] < 300]
            if early:
                w(f"- births before t=300: {len(early)}; mean child maxE {st.mean(b[3] for b in early):.0f}, mean child speed {st.mean(b[1] for b in early):.1f}")
    # ---- raw exports for offline analysis
    import csv
    with open(os.path.join(out, "deaths.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["seed"] + COLS)
        for sd, x in allw:
            cw.writerow([sd] + list(x) + [""] * (len(COLS) - len(x)))
    with open(os.path.join(out, "timeseries.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        keys = list(runs[0]["ts"].keys())
        cw.writerow(["seed"] + keys)
        for r in runs:
            for i in range(len(r["ts"]["t"])):
                cw.writerow([r["seed"]] + [r["ts"][k][i] for k in keys])


    # ---- demography: is each birth cohort replacing itself? (needs ids: births_list[11]=child, [12]=parent)
    dem = []
    for r in runs:
        Bl = [b for b in r["births_list"] if len(b) >= 13]
        if not Bl or not any(len(x) > 17 for x in r["deaths_list"]):
            continue
        dage = {x[17]: x[2] for x in r["deaths_list"] if len(x) > 17}
        dtime = {x[17]: x[0] for x in r["deaths_list"] if len(x) > 17}
        kids_ok = Counter()
        kids_all = Counter()
        for b in Bl:
            cid, pid = b[11], b[12]
            kids_all[pid] += 1
            if cid not in dage or dage[cid] >= 40:
                kids_ok[pid] += 1
        born = {b[11]: b[0] for b in Bl}
        for aid_, a_ in dage.items():
            born.setdefault(aid_, max(0.0, dtime[aid_] - a_))
        for aid_, bt in born.items():
            dem.append((bt, kids_all.get(aid_, 0), kids_ok.get(aid_, 0), 1 if (aid_ in dage and dage[aid_] < 40) else 0))
    if dem:
        w("\n## 3c. Demography (per birth cohort)")
        w("| born in | agents | reach age 40 s | children per agent | children that reach 40 s per agent (R) |")
        w("|---|---|---|---|---|")
        for lo, hi in ((0, 150), (150, 300), (300, 500), (500, 800), (800, 1200), (1200, 3001)):
            G = [d for d in dem if lo <= d[0] < hi]
            if len(G) >= 5:
                w(f"| {lo}-{hi} s | {len(G)} | {100 * (1 - sum(d[3] for d in G) / len(G)):.0f}% | {sum(d[1] for d in G) / len(G):.2f} | **{sum(d[2] for d in G) / len(G):.2f}** |")
        w("- R < 1 means the cohort does not replace itself (the colony shrinks); R > 1 means it grows. Extinction = R stays below 1 while food/predators tighten.")

    # ---- extinction diagnosis
    dead_runs = [r for r in runs if not r["survived_full"]]
    if dead_runs:
        w("\n## 4. Runs that went extinct before the horizon")
        for r in dead_runs:
            last = [x for x in r["deaths_list"] if x[0] >= r["sim_time"] - 60]
            c = Counter(x[1] for x in last)
            ts = r["ts"]
            w(f"- seed {r['seed']}: extinct at t={r['sim_time']:.0f}. Last 60 s deaths: {dict(c)}. "
              f"State ~100 s before: pop {_at(ts, 'n', r['sim_time'] - 100)}, mean energy {_at(ts, 'mean_E', r['sim_time'] - 100)}, "
              f"trees {_at(ts, 'trees', r['sim_time'] - 100)} ({_at(ts, 'mature', r['sim_time'] - 100)} mature), predators {_at(ts, 'preds', r['sim_time'] - 100)}")

    # ---- behaviour
    w("\n## 5. Policy behaviour (share of agent-ticks) and reproduction")
    ps = Counter()
    for r in runs:
        ps.update(r["policy_stats"])
    at = max(1, ps["agent_ticks"])
    w("- states: " + ", ".join(f"{k} {100 * ps[k] / at:.1f}%" for k in ("flee", "flee_sprint", "forage", "seek_tree", "camp", "explore", "unstick")))
    w(f"- spawns: normal {ps['spawn_normal']}, death-bed {ps['spawn_dbed']}, emergency {ps['spawn_emergency']}; tree bans {ps['bans']}; predator sightings {ps['threat_sightings']}")
    w(f"- fruits eaten {sum(r['fruits_eaten'] for r in runs)}, rotted uneaten {sum(r['fruits_rotted'] for r in runs)}")
    tm = [r["timing"] for r in runs]
    w(f"- speed: policy {st.mean(x['policy_us_per_agent_tick'] for x in tm):.0f} us/agent-tick, simulator {st.mean(x['sim_ms_per_tick'] for x in tm):.1f} ms/tick")


    # ---- energy budget by state (analytic costs + residual income, summed over runs)
    agg = {}
    for r in runs:
        for k, v in r.get("ledger", {}).items():
            a_ = agg.setdefault(k, [0.0] * 8)
            for i_, x in enumerate(v):
                a_[i_] += x
    if agg:
        tt = sum(v[0] for v in agg.values()) or 1
        sec = tt / 10.0
        w("\n## 5b. Energy budget (per agent-second spent IN each state)")
        w("| state | time share | walk | sprint | turn | living | aging | births | **income** | net |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        lt = [0.0] * 8
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            t_ = v[0] / 10.0
            if t_ <= 0:
                continue
            row = [x / t_ for x in v[1:]]
            w(f"| {k} | {100 * v[0] / tt:.1f}% | " + " | ".join(f"{x:.2f}" for x in row[:6]) + f" | {row[6]:.2f} | {row[6] - sum(row[:6]):+.2f} |")
            for i_ in range(8):
                lt[i_] += v[i_]
        w(f"| **ALL** | 100% | " + " | ".join(f"{x / sec:.2f}" for x in lt[1:7]) + f" | {lt[7] / sec:.2f} | {(lt[7] - sum(lt[1:7])) / sec:+.2f} |")
        w(f"- per agent-second: movement {(lt[1] + lt[2] + lt[3]) / sec:.2f}, living {lt[4] / sec:.2f}, aging {lt[5] / sec:.2f}, births {lt[6] / sec:.2f}, income {lt[7] / sec:.2f}"
          f"  (birth share of income {100 * lt[6] / max(lt[7], 1e-9):.0f}%, aging {100 * lt[5] / max(lt[7], 1e-9):.0f}%)")
    # ---- fruit economy
    H = [sum(r.get("eaten_hist", [0] * 5)[i_] for r in runs) for i_ in range(5)]
    if sum(H):
        w("\n## 5c. Fruit economy")
        w(f"- fruit energy at the moment of eating [20-30,30-40,40-50,50-60,60]: {H} = {[round(100 * x / sum(H)) for x in H]}%  (a fruit ripens 20->60 in 20 s; rots at 50 s)")
        w(f"- average energy per fruit eaten: {1000 * sum(r['score_parts']['fruit'] for r in runs) / max(1, sum(r['fruits_eaten'] for r in runs)):.1f} (max 60)")
        EA = Counter()
        for r in runs:
            EA.update(r.get("eater", {}))
        if EA:
            tot_e = sum(EA.values())
            w("- who eats (young|state|hungry?|moving?): " + "; ".join(f"{k} {100 * v / tot_e:.0f}%" for k, v in EA.most_common(6)))
    # ---- controller state tracking
    tr = [r.get("track") for r in runs if r.get("track")]
    if tr:
        w("\n## 5d. State tracking (episodes recorded by the controller)")
        S_ = {}
        for t_ in tr:
            for g, v in t_.get("states", {}).items():
                a_ = S_.setdefault(g, [0, 0.0, 0.0, 0])
                a_[0] += v["episodes"]; a_[1] += v["total_s"]; a_[2] += v["net_energy_per_s"] * v["total_s"]; a_[3] += v["ended_by_death"]
        w("| state | episodes | mean duration s | net energy / s | episodes ended by death |")
        w("|---|---|---|---|---|")
        for g, a_ in sorted(S_.items(), key=lambda kv: -kv[1][1]):
            w(f"| {g} | {a_[0]} | {a_[1] / max(1, a_[0]):.1f} | {a_[2] / max(a_[1], 1e-9):+.2f} | {a_[3]} |")
        F_ = {}
        for t_ in tr:
            for k, v in t_.get("flee", {}).items():
                a_ = F_.setdefault(k, [0, 0]); a_[0] += v["episodes"]; a_[1] += v["died"]
        if F_:
            w("- flee episodes -> death rate by energy (fraction of max) | speed trait:  " +
              "; ".join(f"{k}: {v[1]}/{v[0]} ({100 * v[1] / max(1, v[0]):.0f}%)" for k, v in sorted(F_.items())))
        C_ = {}
        for t_ in tr:
            for b, v in t_.get("camp_biome", {}).items():
                a_ = C_.setdefault(b, [0, 0.0, 0.0]); a_[0] += v["episodes"]; a_[1] += v["seconds"]; a_[2] += v["net_energy_per_s"] * v["seconds"]
        if C_:
            w("- camping net energy/s by biome: " + "; ".join(f"{b} {a_[2] / max(a_[1], 1e-9):+.2f} ({a_[1]:.0f}s)" for b, a_ in sorted(C_.items(), key=lambda kv: -kv[1][1])))

    # ---- automatic findings
    w("\n## 6. Automatic findings")
    F = []
    if by.get("predator", 0) / tot > 0.35:
        F.append("Predators are a dominant killer -> improve evasion / breed speed / keep energy above the 20% sprint floor / spread agents.")
    if by.get("starved_young", 0) / tot > 0.35:
        F.append("Many agents starve before reaching old age -> population exceeds the food supply: lower cap_early/cap_late or raise spawn_thr.")
    if dead_runs:
        F.append(f"{len(dead_runs)} run(s) went extinct early: fix these first, one extinction costs far more than any fruit bonus.")
    fr, ro = sum(r["fruits_eaten"] for r in runs), sum(r["fruits_rotted"] for r in runs)
    if ro > 0.4 * max(1, fr):
        F.append("Lots of fruit rots uneaten -> agents are not where the food is (exploration / tree seeking too weak) or ripe_age too high.")
    if ps["explore"] / at > 0.25:
        F.append("Agents spend >25% of their time exploring -> food discovery is the bottleneck.")
    if ps["camp"] / at < 0.2 and by:
        F.append("Little camping -> energy is being burnt walking; check camp_patience and tree bans.")
    if sum(H) and H[0] / sum(H) > 0.4:
        F.append("Over 40% of fruit is eaten before it is 10 s old (20-30 energy instead of 60): ripeness discipline is failing -> raise ripe_age/ripe_unknown, lower urgent_abs.")
    if agg:
        _t = sum(v[6] for v in agg.values())
        _i = sum(v[7] for v in agg.values())
        if _i and _t / _i > 0.22:
            F.append("Births consume >22% of all energy income; if most children die young (see section 3) raise spawn_thr / spawn_food_min.")
    if not F:
        F.append("No obvious single failure mode; compare against validation score and look at the plots.")
    for f in F:
        w(f"- {f}")
    open(os.path.join(out, "report.md"), "w").write("\n".join(L) + "\n")

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    for r in runs:
        t = r["ts"]["t"]
        ax[0, 0].plot(t, r["ts"]["n"], label=f"seed {r['seed']}")
        ax[0, 1].plot(t, r["ts"]["mean_E"])
        ax[0, 2].plot(t, r["ts"]["preds"])
        ax[1, 0].plot(t, r["ts"]["mature"])
        ax[1, 1].plot(t, r["ts"]["mean_speed"])
        ax[1, 2].plot(t, r["ts"]["d_pred"], "-", label="predator")
        ax[1, 2].plot(t, r["ts"]["d_starve_young"], "--")
    for a_, ttl in zip(ax.ravel(), ["population", "mean energy (sprint floor = 100)", "predators alive", "mature trees",
                                     "mean speed trait (breeding)", "cumulative deaths: predator (solid) / young starved (dashed)"]):
        a_.set_title(ttl); a_.set_xlabel("sim time (s)"); a_.grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].axhline(100, color="r", ls=":")
    plt.tight_layout(); plt.savefig(os.path.join(out, "timeseries.png"), dpi=110); plt.close()

    # trait evolution
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    for r in runs:
        t = r["ts"]["t"]
        for a_, key, ttl in zip(ax, ["mean_speed", "mean_hear", "mean_vision", "mean_maxE"], ["speed", "hearing", "vision range", "max energy"]):
            a_.plot(t, r["ts"][key]); a_.set_title(ttl); a_.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "traits.png"), dpi=110); plt.close()

    # map snapshots + heatmap for the longest-lived / first seed
    r = max(runs, key=lambda x: x["sim_time"])
    snaps = r["snaps"]
    if snaps:
        k = len(snaps)
        fig, axs = plt.subplots(2, math.ceil((k + 1) / 2), figsize=(5 * math.ceil((k + 1) / 2), 8))
        axs = axs.ravel()
        for a_ in axs:
            a_.set_axis_off()
        for a_, s in zip(axs, snaps):
            a_.set_axis_on(); a_.set_xlim(0, 1600); a_.set_ylim(1200, 0); a_.set_aspect("equal")
            for ob in r["obstacles"]:
                a_.add_patch(plt.Rectangle((ob[0], ob[1]), ob[2], ob[3], color="0.7"))
            a_.scatter([x[0] for x in s["trees"]], [x[1] for x in s["trees"]], c=["darkgreen" if x[2] >= 20 else "lightgreen" for x in s["trees"]], s=40, marker="^")
            a_.scatter([x[0] for x in s["fruits"]], [x[1] for x in s["fruits"]], c="orange", s=8)
            if s["agents"]:
                sc = a_.scatter([x[0] for x in s["agents"]], [x[1] for x in s["agents"]], c=[x[2] for x in s["agents"]], cmap="viridis", vmin=0, vmax=300, s=25, edgecolor="k", linewidth=.3)
            a_.scatter([x[0] for x in s["preds"]], [x[1] for x in s["preds"]], c="red", marker="X", s=60)
            a_.set_title(f"seed {r['seed']} t={s['t']:.0f}: {len(s['agents'])} agents, {len(s['preds'])} preds", fontsize=9)
        plt.suptitle("snapshots: agents coloured by energy (0-300), red X predators, triangles trees (dark = mature), orange fruit")
        plt.tight_layout(); plt.savefig(os.path.join(out, "snapshots.png"), dpi=100); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    import numpy as np
    H = np.zeros((40, 30))
    for rr in runs:
        H += np.array(rr["heat"])
    ax[0].imshow(np.log1p(H.T), extent=[0, 1600, 1200, 0], cmap="magma"); ax[0].set_title("where agents spent time (log, all seeds)")
    st_col = {"fl": "red", "fo": "orange", "ca": "green", "se": "blue", "ex": "gray", "un": "purple"}
    for aid, p in list(r["paths"].items())[:25]:
        for a, b in zip(p, p[1:]):
            if abs(a[1] - b[1]) < 120 and abs(a[2] - b[2]) < 120:
                ax[1].plot([a[1], b[1]], [a[2], b[2]], color=st_col.get(a[3], "k"), lw=.7)
    ax[1].set_xlim(0, 1600); ax[1].set_ylim(1200, 0); ax[1].set_aspect("equal")
    ax[1].set_title(f"seed {r['seed']} sample paths: red flee, orange fruit, blue seek tree, green camp, gray explore, purple unstick")
    plt.tight_layout(); plt.savefig(os.path.join(out, "heat_paths.png"), dpi=100); plt.close()
    print("\n".join(L))
    print(f"\n[plots + raw csv in {out}: timeseries.png traits.png snapshots.png heat_paths.png deaths.csv timeseries.csv]")


def server_report(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ticks = [r for r in rows if "t" in r and "event" not in r]
    events = [r for r in rows if "event" in r]
    out_dir = os.path.join(os.path.dirname(paths[0]) or ".", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    L = []
    w = L.append
    w(f"# Server-side run analysis: {', '.join(os.path.basename(p) for p in paths)}\n")
    eps = sorted({r["ep"] for r in ticks})
    w("| episode | last sim_t | last score | agents at last row | last row status | log rows | gap med / p95 ms | body-wait med ms | our cpu med / p99 ms | payload KB |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    facts = []
    for e in eps:
        tk = [r for r in ticks if r["ep"] == e]
        last = tk[-1]
        gaps = sorted(r["gap_ms"] for r in tk if r.get("gap_ms") is not None)
        cpu = sorted(r.get("cpu_ms", r.get("comp_ms", 0.0)) for r in tk)
        body = sorted(r.get("body_ms", 0.0) for r in tk)
        kb = st.mean(r["bytes"] for r in tk) / 1024
        gm = gaps[len(gaps) // 2] if gaps else float("nan")
        g95 = gaps[int(.95 * len(gaps))] if gaps else float("nan")
        w(f"| {e} | {last['t']:.0f} | {last.get('score') or 0:.1f} | {last['n']} | {last.get('status')} | {len(tk)} | {gm:.0f} / {g95:.0f} | "
          f"{body[len(body) // 2]:.1f} | {cpu[len(cpu) // 2]:.2f} / {cpu[int(.99 * len(cpu))]:.2f} | {kb:.1f} |")
        facts.append((e, last, gm))
    w("\n(gap = time between consecutive requests as seen here = their simulator step + network + our work; body-wait = time this server waited for the request body to arrive)\n")
    w("## Findings")
    go = [e for e in events if e.get("event") == "game_over"]
    for e, last, gm in facts:
        g = [x for x in go if x.get("ep") == e]
        if g:
            lo = g[-1].get("last_ok")
            w(f"- episode {e}: server received game status '{g[-1]['status']}' after last ok state {('t=%.1f agents=%d score=%.1f' % tuple(lo)) if lo else 'n/a'} -> the game ended normally (extinction or end of time).")
        else:
            w(f"- episode {e}: **no game_over message was received**; last row t={last['t']:.0f}, agents={last['n']}. If agents were still alive this means the run was ended from the "
              f"other side (accumulated-wait cap / a timeout) or the server was stopped. Median gap {gm:.0f} ms x 30000 ticks = {30000 * gm / 1000:.0f} s.")
    for e, last, gm in facts:
        w(f"- episode {e}: at a {gm:.0f} ms median gap the 1200 s cap would allow ~{1200 / (gm / 1000) * 0.1 if gm == gm and gm > 0 else float('nan'):.0f} s of simulation (upper bound on the latency cost).")
    for g in go:
        tk_ = g.get("track") or {}
        if tk_:
            w(f"- episode {g.get('ep')} state tracking: " + "; ".join(f"{k}: {v['episodes']} ep, {v['net_energy_per_s']:+.2f} E/s" for k, v in sorted(tk_.get("states", {}).items(), key=lambda kv: -kv[1]["total_s"])))
            if tk_.get("flee"):
                w(f"  flee death rates: " + "; ".join(f"{k} {v['died']}/{v['episodes']}" for k, v in sorted(tk_["flee"].items())))
            ev_ = tk_.get("events") or []
            if ev_:
                w(f"  last deaths (t, state, energy, age, biome): " + "; ".join(f"{e_[0]} {e_[2]} E{e_[3]} a{e_[4]} {e_[5]}" for e_ in ev_[-8:]))
    sd = [e for e in events if e.get("event") == "shutdown"]
    if sd:
        w(f"- server requests {sd[-1]['requests']}, errors {sd[-1]['errors']}")
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for e in eps:
        tk = [r for r in ticks if r["ep"] == e]
        t = [r["t"] for r in tk]
        ax[0].plot(t, [r["n"] for r in tk], label=f"ep {e}")
        ax[1].plot(t, [r["E"] for r in tk])
        ax[2].plot(t, [r["spd"] for r in tk])
        ax[3].plot(t, [r.get("gap_ms") or 0 for r in tk], lw=.4)
    for a_, ttl in zip(ax, ["population", "mean energy", "mean speed trait", "gap between requests (ms)"]):
        a_.set_title(ttl); a_.grid(alpha=.3)
    ax[0].legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "server_curves.png"), dpi=110); plt.close()
    open(os.path.join(out_dir, "server_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[plot: {out_dir}/server_curves.png]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["local", "server"])
    ap.add_argument("path", nargs="+")
    a = ap.parse_args()
    if a.kind == "local":
        local_report(a.path[0])
    else:
        server_report(a.path)
