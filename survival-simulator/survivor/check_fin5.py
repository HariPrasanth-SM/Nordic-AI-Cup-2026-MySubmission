"""Short integration checks, not a performance benchmark."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse, json, math
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/fin4_baseline.json'); args=ap.parse_args()
    from survivor.qr_escape import Network,load_controller,INPUT,ACTIONS,QUANTILES,frame,maneuver
    from survivor.fin5 import Replay,make_net
    from survivor.policy import Params
    from survivor.runner import _Act
    from survivor import fastsim
    from src.core import SimulationCore
    cfg=json.loads(Path(args.config).read_text())
    assert not (set(cfg['params'])-set(Params.__dataclass_fields__)), 'Unknown parameter'
    fastsim.apply(); sim=SimulationCore(seed=921337,starting_predators=2)
    ctrl=load_controller(cfg); acts=[]; first=None
    for tick in range(80):
        st=sim.step(acts)
        payload={'game_status':'ok','sim_time':st['sim_time'],'score':st['score'],
                 'n_agents':st['num_agents'],'agent_status':[a for a in st['observations'] if a]}
        if first is None: first=payload
        out=ctrl.act(payload)
        assert {a['agent_id'] for a in out}=={a['agent_id'] for a in payload['agent_status']}
        for a in out:
            assert all(math.isfinite(a[k]) for k in ('move_distance','move_direction','turn_angle'))
            assert isinstance(a['spawn_agent'],bool)
        acts=[(a['agent_id'],_Act(a)) for a in out]
    z=np.zeros(INPUT,np.float32); r=Replay(8)
    r.add([(z,0,1,z,False),(z,1,-20,z,True)])
    assert np.all(r.g[:2]==0) and np.isclose(r.r[0],1-.995*20)
    import torch
    torch.set_num_threads(1); torch.manual_seed(9); net=make_net(torch).eval()
    x=np.random.default_rng(9).normal(size=(5,INPUT)).astype('float32')
    w={k:v.detach().numpy() for k,v in net.state_dict().items()}
    expected=net(torch.from_numpy(x)).detach().numpy().reshape(5,ACTIONS,QUANTILES)
    assert np.allclose(expected,Network(w).quantiles(x),atol=2e-6)
    a=dict(first['agent_status'][0]); a['observations']=[{'type':'Predator','distance':80.,'angle':.3,'rel_dir':0.}]
    base={'agent_id':a['agent_id'],'move_distance':0.,'move_direction':0.,'turn_angle':0.,'spawn_agent':False}
    assert frame(a).shape==(INPUT//2,)
    for index in range(ACTIONS):
        act=maneuver(a,index,base)
        assert 0<=act['move_distance']<=a['sprint_speed']+1e-6
        assert abs(act['turn_angle'])<=math.pi
    os.environ['SURVIVOR_PARAMS']=args.config
    from fastapi.testclient import TestClient
    from survivor.server5 import app
    with TestClient(app) as client:
        for _ in range(3):
            for t in (.1,.2):
                p=dict(first,sim_time=t)
                response=client.post('/predict',json=p)
                assert response.status_code==200
                assert len(response.json()['actions'])==len(p['agent_status'])
        assert client.get('/stats').json()['errors']==0
        assert client.post('/predict',json=dict(first,game_status='over')).json()['actions']==[]
    print('PASS: 80 official-engine ticks, action bounds, terminal replay, NumPy/Torch equivalence, /predict and episode resets')
if __name__=='__main__': main()
