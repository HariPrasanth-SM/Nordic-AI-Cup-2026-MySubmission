"""
survivor.finale3 - round 3: the MODEL-BASED ENCOUNTER CONTROLLER (survivor.pursuit) and the terminal energy policy, on top of the round-2 winner.

  python -m survivor.finale3 run --out runs/fin3 --budget-min 40 --workers 15 --flee-rows runs/lab1/flee_rows.csv --escape-model runs/lab2/escape_model.json

STAGE 0  duel gate (2 min): heuristic vs MPC on duels sampled from the LOGGED encounter distribution (arena evidence; must show fewer kills before we trust the full runs).
STAGE B  ~10 arms x 24-32 paired seeds on `configs/v11_stagger.json` (racing).      STAGE C  FRESH seeds 15001+: control, top-2 by shrunk score, best-hazard arm, stack.

WHY these arms (evidence): duels show the planner cuts predator kills from 75% to ~52% when first sight is inside the charge zone at low energy, and from 35% to 26% on the
logged mixture (24 vs 9 discordant duels); 64% of real deaths start inside the charge zone, 61% at energy below the sprint floor. Terminal policy: a kill costs
(energy / 100) score points, so a doomed agent with energy converts it into children first (never burns energy on purpose: energy is what makes escapes possible).
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

from survivor import finale2 as F2  # noqa: E402
from survivor import lab as L  # noqa: E402
from survivor import duel as D  # noqa: E402

MPC = {"mpc_flee": 1.0}
SPEED = {"w_speed": 3.0, "w_sprint": 0.5, "w_hearing": 0.1, "w_vision": 0.1, "w_cone": 0.1}
ARMS3 = OrderedDict([
    ("mpc", (MPC, "model-based encounter controller (samples 96 escape plans over 2.4 s against the known predator rules; cheapest safe plan)")),
    ("mpc_term", ({**MPC, "mpc_terminal": 1.0}, "mpc + terminal policy: if NO plan survives the horizon the agent spawns first (kill costs energy/100 points)")),
    ("mpc_emap", ({**MPC, "emap": 1.0}, "mpc + remembered wall/obstacle edges in the planner (36% of fast-agent duel deaths end stuck on obstacles)")),
    ("mpc_far", ({**MPC, "mpc_range": 300.0}, "mpc engaged from 300 instead of 200 (earlier, cheaper avoidance)")),
    ("speed_breed", (SPEED, "breed for walking speed (death rate 29% below speed 11 vs 17% above 15; 35% of encounters involve speed < 11)")),
    ("food3", ({"spawn_food_min": 3.0}, "births only where >= 3 fruits are near (nursery: 33% of deaths are starvation of young agents)")),
    ("newborn_eager", ({"newborn_eager_s": 25.0}, "newborns eat at once (round 1: +155 alone on the old base)")),
    ("stagger12", ({"birth_gap_s": 12.0}, "births spaced >= 12 s (stagger 8 lowered starvation deaths 21% and raised energy at first sight 0.37 -> 0.44)")),
])


class Finale3(F2.Finale2):
    ARMS = ARMS3
    FRESH = 15001

    def __init__(self, a):
        super().__init__(a)
        if a.escape_model and not a.smoke:
            self.lab.arms["term_esc"] = ({"use_escape_model": 1.0, "doom_p": 0.85}, None, None, "terminal spawn when the learned P(death) >= 0.85 (escape model)")
            os.environ["SURVIVOR_ESCAPE_MODEL"] = os.path.abspath(a.escape_model)

    def run(self):
        a = self.a
        if a.duel_n > 0:
            self.log("== Stage 0: duel gate (arena evidence) ==")
            txt = D.gate(a.base, OrderedDict([("heuristic", {}), ("mpc", {"mpc_flee": 1.0}), ("mpc_emap", {"mpc_flee": 1.0, "emap": 1.0})]),
                         (12 if a.smoke else a.duel_n), a.workers, a.flee_rows, pool=self.lab.pool, log=self.log)
            open(os.path.join(a.out, "duel_gate.md"), "w").write(txt + "\n")
        super().run()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--base", default=os.path.join(ROOT, "configs", "v11_stagger.json"))
    r.add_argument("--budget-min", type=float, default=40.0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--horizon", type=float, default=3000.0)
    r.add_argument("--reuse", nargs="*", default=None)
    r.add_argument("--value-model", default=None)
    r.add_argument("--escape-model", default=None)
    r.add_argument("--flee-rows", default=None)
    r.add_argument("--duel-n", type=int, default=160)
    r.add_argument("--seeds-b", type=int, default=32)
    r.add_argument("--seeds-c", type=int, default=48)
    r.add_argument("--min-n", type=int, default=20)
    r.add_argument("--min-gain", type=float, default=20.0)
    r.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    Finale3(a).run()


if __name__ == "__main__":
    main()
