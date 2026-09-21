"""
survivor.server - drop-in replacement for agent_server.py (same contract: POST /predict -> {"actions": [...]}).

  SURVIVOR_PARAMS=configs/v4a_lowE_frugal.json python -m survivor.server      # port 9052

RUN THIS ON THE GOOGLE-CLOUD VM ITSELF if you can (CPU only, ~0.5 ms/request, needs just fastapi + uvicorn):
every extra hop (ssh tunnel to a workstation) adds latency that counts against the wait budget.

What you see in the console (one line every ~15 s, plus events):
  [17:31:07] ep0 t=312.4 score=318.9 agents=9 E=141 spd=11.2 | gap med/p95 74/210 ms | body 12 ms cpu 0.6 ms |
  reqs 3124 (66/s) | wait-cap 1200s -> reachable sim_t ~1620s | flee 7% camp 55% forage 27% explore 4%
'gap' = time between consecutive requests as seen here (their simulator step + network + our work).
'reachable sim_t' = WAIT_CAP / mean gap * 0.1 s, i.e. how far the run can get before the wait cap if the
gap were entirely wait time (a conservative estimate of the latency cost).
"""
import json
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402

from survivor.policy import Controller, Params  # noqa: E402

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "9052"))
WAIT_CAP = float(os.environ.get("WAIT_CAP", "1200"))          # organisers' accumulated-wait limit (s)
STATUS_EVERY = float(os.environ.get("STATUS_EVERY", "15"))    # console status period (s)
LOG_DIR = os.environ.get("SURVIVOR_LOGS", os.path.join(ROOT, "logs"))
PARAMS_PATH = os.environ.get("SURVIVOR_PARAMS") or next(
    (p for p in (os.path.join(ROOT, "configs", "best.json"), os.path.join(ROOT, "configs", "v1_default.json")) if os.path.exists(p)), None)

try:
    import orjson

    def _loads(b):
        return orjson.loads(b)

    def _dumps(o):
        return orjson.dumps(o)
except ImportError:
    def _loads(b):
        return json.loads(b)

    def _dumps(o):
        return json.dumps(o).encode()

CTRL = Controller(Params.load(PARAMS_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"server_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
_buf = []
S = {"requests": 0, "errors": 0, "last_req": None, "episode": -1, "last_t": -1.0, "last_ok": None,
     "client": None, "ep_reqs": 0, "last_status": time.time(), "prev_stats": {}}
W = {"gaps": [], "body": [], "cpu": []}          # rolling windows for the console


def _pct(a, q):
    if not a:
        return float("nan")
    b = sorted(a)
    return b[min(len(b) - 1, int(q * len(b)))]


def _flush(force=False):
    global _buf
    if _buf and (force or len(_buf) >= 300):
        with open(LOG_PATH, "a") as f:
            f.write("\n".join(json.dumps(r, separators=(",", ":")) for r in _buf) + "\n")
        _buf = []


def _say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_actions(agents):
    return [{"agent_id": a["agent_id"], "move_distance": 0.0, "move_direction": 0.0, "turn_angle": 0.3, "spawn_agent": False}
            for a in agents]


def _status_line(t, n, score, E, spd):
    g, b, c = W["gaps"], W["body"], W["cpu"]
    mean_gap = (sum(g) / len(g)) if g else float("nan")
    reach = WAIT_CAP / (mean_gap / 1000.0) * 0.1 if g and mean_gap > 0 else float("nan")
    st = CTRL.stats
    prev = S["prev_stats"]
    d = {k: st.get(k, 0) - prev.get(k, 0) for k in ("agent_ticks", "flee", "flee_sprint", "camp", "forage", "seek_tree", "explore")}
    at = max(1, d["agent_ticks"])
    S["prev_stats"] = dict(st)
    rate = len(g) / STATUS_EVERY
    _say(f"ep{S['episode']} t={t:.1f} score={score:.1f} agents={n} E={E:.0f} spd={spd:.1f} | "
         f"gap med/p95 {_pct(g, .5):.0f}/{_pct(g, .95):.0f} ms | body {_pct(b, .5):.1f} ms cpu {_pct(c, .5):.2f} ms | "
         f"reqs {S['requests']} ({rate:.0f}/s) | wait-cap {WAIT_CAP:.0f}s -> reachable sim_t ~{reach:.0f}s | "
         f"flee {100 * (d['flee'] + d['flee_sprint']) / at:.0f}% camp {100 * d['camp'] / at:.0f}% "
         f"forage {100 * (d['forage'] + d['seek_tree']) / at:.0f}% explore {100 * d['explore'] / at:.0f}%")
    W["gaps"], W["body"], W["cpu"] = [], [], []


@asynccontextmanager
async def lifespan(app):
    P_ = CTRL.p
    layers = {k: getattr(P_, k) for k in ("colony_mgr", "senescence", "n_lifeboat", "emap", "flee_ray", "use_escape_model", "rest_frac", "shared_alarm", "patrol", "mpc_flee", "mpc_terminal", "kite", "vm_gate", "birth_gap_s") if getattr(P_, k) not in (0, 0.0, 1.01)}
    _say(f"layers on: {layers or 'none'} | escape model: {'loaded' if CTRL.escape is not None else 'no'}")
    _say(f"survivor server ready | params: {PARAMS_PATH} | log: {LOG_PATH} | wait cap {WAIT_CAP:.0f}s | waiting for the evaluation server ...")
    yield
    if S["last_ok"]:
        _say(f"shutdown | last ok state: t={S['last_ok'][0]:.1f} agents={S['last_ok'][1]} score={S['last_ok'][2]:.1f} | "
             f"requests {S['requests']} errors {S['errors']}")
    _buf.append({"event": "shutdown", "requests": S["requests"], "errors": S["errors"], "last_ok": S["last_ok"], "policy": CTRL.stats,
                 "track": CTRL.summary()})
    _flush(True)


app = FastAPI(title="Survival Simulator Agent Endpoint (survivor)", lifespan=lifespan)


@app.get("/")
def index():
    return {"message": "Agent endpoint running!", "params": PARAMS_PATH}


@app.get("/stats")
def stats():
    return {**{k: v for k, v in S.items() if k != "prev_stats"}, "policy": CTRL.stats}


@app.post("/predict")
async def predict(request: Request):
    t0 = time.perf_counter()
    raw = await request.body()
    t1 = time.perf_counter()
    step = None
    agents, t = [], S["last_t"]
    status = "ok"
    try:
        step = _loads(raw)
        agents = step.get("agent_status") or []
        t = float(step.get("sim_time", 0.0))
        status = step.get("game_status", "ok")
        if S["client"] is None and request.client:
            S["client"] = request.client.host
            _say(f"first request from {S['client']}  (ping this IP from the VM to see the network latency to the evaluation server)")
        if t < S["last_t"] - 1e-6 or (t <= 0.05 and S["last_t"] > 1.0) or S["episode"] < 0:
            if S["episode"] >= 0 and S["last_ok"]:
                _say(f"episode {S['episode']} ended: last ok state t={S['last_ok'][0]:.1f} agents={S['last_ok'][1]} score={S['last_ok'][2]:.1f}")
            S["episode"] += 1
            S["ep_reqs"] = 0
            _say(f"episode {S['episode']} started")
            _buf.append({"event": "episode_start", "episode": S["episode"], "wall": time.time()})
        S["last_t"] = t
        actions = CTRL.act(step)
        have = {a["agent_id"] for a in actions}
        missing = [a for a in agents if a["agent_id"] not in have]
        if missing:
            actions += _safe_actions(missing)
    except Exception:
        S["errors"] += 1
        if S["errors"] <= 5:
            traceback.print_exc()
        status = "error"
        actions = _safe_actions(agents)
    out = Response(content=_dumps({"actions": actions}), media_type="application/json")
    t2 = time.perf_counter()

    gap = None if S["last_req"] is None else (t0 - S["last_req"]) * 1000.0
    S["last_req"] = t2
    S["requests"] += 1
    S["ep_reqs"] += 1
    body_ms, cpu_ms = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
    if gap is not None:
        W["gaps"].append(gap)
    W["body"].append(body_ms)
    W["cpu"].append(cpu_ms)

    n = len(agents)
    score = float(step.get("score") or 0.0) if isinstance(step, dict) else 0.0
    E = sum(a["energy"] for a in agents) / n if n else 0.0
    spd = sum(a["speed"] for a in agents) / n if n else 0.0
    if status in ("ok",) and n:
        S["last_ok"] = (t, n, score)
    # log: every 5th request, every request when the colony is nearly gone (exact end state)
    if S["requests"] % 5 == 0 or n <= 3:
        _buf.append({"ep": S["episode"], "t": round(t, 1), "n": n, "score": round(score, 1), "status": status,
                     "E": round(E, 1), "age": round(sum(a["age"] for a in agents) / n, 1) if n else 0, "spd": round(spd, 2),
                     "cpu_ms": round(cpu_ms, 3), "body_ms": round(body_ms, 2), "gap_ms": None if gap is None else round(gap, 2),
                     "bytes": len(raw), "wall": round(time.time(), 2)})
        if len(_buf) >= 300:
            _flush(True)
    if status not in ("ok", "error", "running"):
        _say(f"GAME STATUS '{status}' at t={t:.1f} score={score:.1f}; last ok state: {S['last_ok']}")
        _buf.append({"event": "game_over", "ep": S["episode"], "status": status, "t": round(t, 1), "score": score,
                     "last_ok": S["last_ok"], "policy": CTRL.stats,
                     "track": CTRL.summary()})
        _flush(True)
    if time.time() - S["last_status"] >= STATUS_EVERY and n:
        S["last_status"] = time.time()
        _status_line(t, n, score, E, spd)
    return out


if __name__ == "__main__":
    import uvicorn
    kw = {}
    try:
        import uvloop  # noqa: F401
        kw["loop"] = "uvloop"
    except ImportError:
        pass
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", access_log=False, **kw)
