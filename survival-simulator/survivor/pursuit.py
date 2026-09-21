"""
survivor.pursuit - a kinematic model of the predator/agent chase (transcribed from the simulator source) and a vectorised
sampling MPC planner for predator encounters. Also the physics-based VIABILITY test used by the terminal policy.

Predator rules used (src/elements/predator.py + environment.py), all per 0.1 s tick:
  * it chases the closest agent (here: us). It CHARGES if we look away (|rel_dir| > 90 deg) or are closer than 90 (= 1.5 x its hearing 60):
    heading turn = clip(0.5 * bearing, +-0.3), it moves min(15, distance) along heading + turn.
    Otherwise it PIVOTS: moves 15 at 45 deg off the line to us and turns to face us again (burns 2.55 energy/tick without closing fast).
  * energy: walking 0.05/unit, sprinting 0.5/unit above its walking speed 11; below 40 energy it is capped at 11; at <= 0 it sleeps.
  * kill when centre distance < 15 (sizes 10 + 5); the score loses (our energy / 100).
Agent rules: walk cost 0.05/unit, sprint 0.5/unit above the walking speed, sprint only while energy >= max_energy / 5, living cost 0.1/tick.
Timing: agents move first, then predators; our observation is taken BEFORE the predator's move of that tick, so it is one predator move
stale - the planner advances every predator one step before rolling out.
"""
import math

import numpy as np

PI = math.pi
KILL = 15.0
P_WALK, P_SPRINT, P_TIRED = 11.0, 15.0, 40.0
CHARGE_R = 90.0
TURN_MAX = 0.3
SAFE_M = 40.0        # margin (units) above the kill distance that counts as 'safe' within the horizon


def _wrap(a):
    return (a + PI) % (2 * PI) - PI


def pred_step(px, py, ph, pE, ax, ay, ah):
    """one predator tick (arrays broadcast to (K,P)); returns new px, py, ph, pE"""
    dx, dy = ax - px, ay - py
    d = np.hypot(dx, dy) + 1e-9
    ang = _wrap(np.arctan2(dy, dx) - ph)                    # bearing of the agent in the predator frame
    rel = _wrap(np.arctan2(py - ay, px - ax) - ah)          # the agent's looking direction relative to the line to the predator
    charge = (np.abs(rel) > PI / 2) | (d < CHARGE_R)
    ts = np.clip(ang * 0.5, -TURN_MAX, TURN_MAX)
    big = np.abs(ang) > 0.05
    c_dir = np.where(big, ts, ang)
    c_turn = np.where(big, ts, 0.0)
    c_dist = np.minimum(P_SPRINT, d)
    sign = -np.sign(rel)
    sign = np.where(sign == 0, 1.0, sign)
    p_dir = ang + sign * PI / 4
    xa, ya = d * np.cos(ang), d * np.sin(ang)
    p_turn = np.arctan2(ya - P_SPRINT * np.sin(p_dir), xa - P_SPRINT * np.cos(p_dir))
    dist = np.where(charge, c_dist, P_SPRINT)
    mdir = np.where(charge, c_dir, p_dir)
    turn = np.where(charge, c_turn, p_turn)
    dist = np.where(pE < P_TIRED, np.minimum(dist, P_WALK), dist)
    cost = np.where(dist <= P_WALK, dist * 0.05, P_WALK * 0.05 + (dist - P_WALK) * 0.5) + np.minimum(PI, np.abs(turn)) / (2 * PI)
    awake = pE > 0
    dist, turn, cost = np.where(awake, dist, 0.0), np.where(awake, turn, 0.0), np.where(awake, cost, 0.0)
    wd = ph + mdir
    return px + dist * np.cos(wd), py + dist * np.sin(wd), ph + turn, pE - cost


def _seg_hit(ax, ay, bx, by, E):
    """(K,) bool: does the move A->B cross any edge in E (Ne,4)? A,B are (K,)"""
    if E is None or len(E) == 0:
        return np.zeros(len(ax), bool)
    cx, cy, ex, ey = E[None, :, 0], E[None, :, 1], (E[:, 2] - E[:, 0])[None, :], (E[:, 3] - E[:, 1])[None, :]
    d1x, d1y = (bx - ax)[:, None], (by - ay)[:, None]
    den = d1x * ey - d1y * ex
    ok = np.abs(den) > 1e-9
    den_s = np.where(ok, den, 1.0)
    wx, wy = cx - ax[:, None], cy - ay[:, None]
    t = (wx * ey - wy * ex) / den_s
    u = (wx * d1y - wy * d1x) / den_s
    return (ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)).any(axis=1)


class Plan:
    __slots__ = ("theta", "sprint", "face_pred", "surv_frac", "best_captured", "t_cap", "dmin", "toward", "cands")


def plan(sp, spr, E0, ME, preds, edges, H=16, n_dirs=16, prev_theta=None):
    """
    sp/spr: our walking / sprint speed; E0, ME: energy and max energy; preds: list of (px, py, ph, pE) in OUR local frame
    (we sit at the origin with heading 0), already advanced for the observation latency; edges: (Ne,4) local segments or None.
    Returns Plan (the best first action) or None if there is nothing to plan against.
    """
    if not preds:
        return None
    th0 = np.linspace(-PI, PI, n_dirs, endpoint=False)
    modes = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]              # (speed mode: walk / sprint / sprint-then-walk, facing: predator / forward)
    th = np.repeat(th0, len(modes))
    sm = np.tile([m[0] for m in modes], n_dirs)
    fc = np.tile([m[1] for m in modes], n_dirs)
    K, P = len(th), len(preds)
    ax, ay, ah = np.zeros(K), np.zeros(K), np.zeros(K)
    E = np.full(K, float(E0))
    px = np.tile(np.array([p[0] for p in preds], float), (K, 1))
    py = np.tile(np.array([p[1] for p in preds], float), (K, 1))
    ph = np.tile(np.array([p[2] for p in preds], float), (K, 1))
    pE = np.tile(np.array([p[3] for p in preds], float), (K, 1))
    floor = ME / 5.0
    cap = np.zeros(K, bool)
    tcap = np.full(K, H, float)
    Ecap = np.zeros(K)
    dmin = np.full(K, 1e9)
    blocked = np.zeros(K)
    pivot = np.zeros(K)
    dlast = np.zeros(K)
    first_sprint = np.zeros(K, bool)
    cth, sth = np.cos(th), np.sin(th)
    for t in range(H):
        want = (sm == 1) | ((sm == 2) & (t < 6))
        sprint = want & (E >= floor)
        dist = np.where(sprint, spr, sp)
        if t == 0:
            first_sprint = sprint
        cost = np.where(dist <= sp, dist * 0.05, sp * 0.05 + (dist - sp) * 0.5)
        nx, ny = ax + dist * cth, ay + dist * sth
        hit = _seg_hit(ax, ay, nx + 5.0 * cth, ny + 5.0 * sth, edges)                         # 5 = agent radius
        alive = ~cap
        move = alive & ~hit
        ax, ay = np.where(move, nx, ax), np.where(move, ny, ay)
        blocked += (alive & hit)
        E = np.where(alive, E - cost - 0.1 - 0.01, E)
        # facing after the move
        dd = np.hypot(px - ax[:, None], py - ay[:, None])
        j = dd.argmin(axis=1)
        pxj, pyj = px[np.arange(K), j], py[np.arange(K), j]
        bear = np.arctan2(pyj - ay, pxj - ax)
        ah = np.where(fc == 0, bear, th)
        # predators react to our new position / facing
        npx, npy, nph, npE = pred_step(px, py, ph, pE, ax[:, None], ay[:, None], ah[:, None])
        px, py, ph, pE = np.where(alive[:, None], npx, px), np.where(alive[:, None], npy, py), np.where(alive[:, None], nph, ph), np.where(alive[:, None], npE, pE)
        d = np.hypot(px - ax[:, None], py - ay[:, None]).min(axis=1)
        newcap = alive & (d < KILL)
        tcap = np.where(newcap, t, tcap)
        Ecap = np.where(newcap, E, Ecap)
        cap |= newcap
        dmin = np.where(alive, np.minimum(dmin, d), dmin)
        rel = np.abs(_wrap(np.arctan2(pyj - ay, pxj - ax) - ah))                             # rel_dir: vector agent->predator vs our heading
        pivot += (alive & (d >= CHARGE_R) & (rel < PI / 2))
        dlast = np.where(alive, d, dlast)
    # lexicographic intent: (1) never be caught, (2) among plans that keep a safety margin choose the CHEAPEST in energy (sprinting from a predator
    # that is not closing wastes the reserve that decides the next encounter), (3) only if no plan keeps the margin, maximise the distance.
    spent = E0 - E
    safe = (~cap) & (dmin >= SAFE_M)
    cost_safe = 1.0 * spent + 25.0 * blocked / H - 3.0 * pivot / H - 0.02 * np.minimum(dlast, 200.0) + 30.0 * (E < floor)
    cost_unsafe = 200.0 + 3.0 * np.maximum(0.0, SAFE_M - dmin) - 0.3 * np.minimum(dlast, 200.0) + 0.15 * spent + 25.0 * blocked / H + 30.0 * (E < floor)
    cost_ok = np.where(safe, cost_safe, cost_unsafe)
    if prev_theta is not None:
        cost_ok = cost_ok + 1.5 * np.abs(_wrap(th - prev_theta)) / PI
    cost_cap = 1000.0 + 40.0 * (H - tcap) + 0.05 * Ecap                                      # later capture is better, and so is dying with less energy
    total = np.where(cap, cost_cap, cost_ok)
    b = int(total.argmin())
    pl = Plan()
    pl.theta = float(th[b])
    pl.sprint = bool(first_sprint[b])
    pl.face_pred = bool(fc[b] == 0)
    pl.surv_frac = float(1.0 - cap.mean())
    pl.best_captured = bool(cap[b])
    pl.t_cap = float(tcap[b])
    pl.dmin = float(dmin[b])
    pl.toward = float(np.mean(cap))
    pl.cands = K
    return pl
