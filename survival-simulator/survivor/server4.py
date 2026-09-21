"""Compatible /predict endpoint; no changes to api.py or agent_server.py.
SURVIVOR_PARAMS=runs/fin4/final_config.json python -m survivor.server4
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import json
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from survivor.frontier import load_controller

CONFIG=os.environ.get('SURVIVOR_PARAMS',str(Path(__file__).resolve().parent.parent/'configs/fin4_baseline.json'))
DOC=json.loads(Path(CONFIG).read_text())
CTRL=load_controller(DOC,seed=0)
app=FastAPI(title='Survival fin4')
COUNTERS={'requests':0,'errors':0,'episode':0,'policy_seconds':0.,'max_policy_ms':0.,'last_t':-1.,'last_score':0.}

@app.get('/')
def index():
    return {'message':'Agent endpoint running!','controller':DOC.get('controller','fin3'),'config':CONFIG}

@app.get('/stats')
def stats():
    return dict(COUNTERS)

@app.post('/predict')
async def predict(request:Request):
    global CTRL
    body=await request.json(); t=float(body.get('sim_time',0)); agents=body.get('agent_status') or []
    status=body.get('game_status','ok')
    # Some old traces contained 'running'. Normalize it without interpreting it as game-over.
    if status=='running': body['game_status']='ok'
    if t<COUNTERS['last_t']-1e-6:
        print(json.dumps({'event':'episode_end',**COUNTERS}),flush=True)
        CTRL=load_controller(DOC,seed=0)
        COUNTERS['episode']+=1; COUNTERS['policy_seconds']=0.; COUNTERS['max_policy_ms']=0.
    ts=time.perf_counter()
    try:
        actions=CTRL.act(body)
        if body.get('game_status','ok')=='ok' and {x['agent_id'] for x in actions}!={a['agent_id'] for a in agents}:
            raise ValueError('action IDs do not match observations')
    except Exception:
        COUNTERS['errors']+=1
        import traceback
        traceback.print_exc()
        # Preserve endpoint continuity; errors remain visible and disqualify validation readiness.
        actions=[{'agent_id':a['agent_id'],'move_distance':0.,'move_direction':0.,'turn_angle':.3,'spawn_agent':False} for a in agents]
    elapsed=time.perf_counter()-ts
    COUNTERS['requests']+=1; COUNTERS['policy_seconds']+=elapsed
    COUNTERS['max_policy_ms']=max(COUNTERS['max_policy_ms'],1000*elapsed)
    COUNTERS['last_t']=t; COUNTERS['last_score']=body.get('score',0)
    if COUNTERS['requests']%1000==0 or status not in ('ok','running'):
        print(json.dumps(COUNTERS),flush=True)
    return JSONResponse({'actions':actions})

if __name__=='__main__':
    import uvicorn
    print(f'fin4 server config={CONFIG}; controller={DOC.get("controller","fin3")}; single process',flush=True)
    uvicorn.run(app,host=os.environ.get('HOST','0.0.0.0'),port=int(os.environ.get('PORT','9052')),workers=1,access_log=False)
