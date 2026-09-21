"""
survivor.flee_stats - where do predator encounters turn deadly? (from the flee episodes the lab logs)

  python -m survivor.flee_stats runs/lab1/flee_rows.csv [runs/fin2/flee_rows.csv ...]

Prints death rate by first-sight distance, energy fraction, walking speed, distance to the nearest wall/obstacle, neighbours and
whether the predator had noticed the agent, so we can see WHICH situations to fix (e.g. what share of deaths start inside the 90-unit charge zone).
"""
import csv
import sys


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
    return rows


def table(rows, key, edges, label):
    print(f"\n{label}\n| bin | episodes | deaths | death rate | share of all deaths |")
    print("|---|---|---|---|---|")
    tot = sum(r["died"] for r in rows) or 1
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = [r for r in rows if lo <= r[key] < hi]
        if g:
            d = sum(r["died"] for r in g)
            print(f"| {lo:g} - {hi:g} | {len(g)} | {int(d)} | {100 * d / len(g):.0f}% | {100 * d / tot:.0f}% |")


def main():
    rows = load(sys.argv[1:])
    if not rows:
        print("usage: python -m survivor.flee_stats flee_rows.csv ...")
        return
    print(f"{len(rows)} logged flee episodes; overall death rate {100 * sum(r['died'] for r in rows) / len(rows):.1f}%")
    table(rows, "d0", [0, 40, 60, 90, 120, 160, 250, 1e9], "first-sight distance d0 (charge zone = 90)")
    table(rows, "e_frac", [0, .1, .2, .3, .45, .6, 1.01], "energy fraction at first sight (sprint floor = 0.2)")
    table(rows, "speed", [0, 11, 13, 15, 17, 25], "walking speed (predator sprints at 15, walks at 11 when tired)")
    table(rows, "wall_d", [0, 25, 50, 100, 200, 1e9], "distance to nearest visible wall/obstacle edge")
    table(rows, "noticed", [0, 0.5, 1.5], "predator had noticed the agent (1) or not (0)")
    table(rows, "nb", [0, 1, 2, 4, 99], "neighbours within 80")
    inside = [r for r in rows if 0 <= r["d0"] < 90]
    print(f"\nepisodes whose first sight was INSIDE the charge zone: {100 * len(inside) / len(rows):.0f}% of episodes, "
          f"{100 * sum(r['died'] for r in inside) / max(1, sum(r['died'] for r in rows)):.0f}% of deaths")


if __name__ == "__main__":
    main()
