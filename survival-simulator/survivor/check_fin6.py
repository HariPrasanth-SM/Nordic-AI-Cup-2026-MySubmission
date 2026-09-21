"""Focused tests of reward, temporal credit, export and action contract."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import copy,json,argparse,tempfile
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/fin4_baseline.json'); args=p.parse_args()
    import torch
    from survivor.colony_ppo import INPUT,ACTIONS,OPTIONS,Network,Supervisor
    from survivor.fin6 import make_net,weights,World,advantages,payload
    from survivor.qr_escape import load_controller
    from survivor.policy import Params
    from survivor import fastsim
    torch.set_num_threads(1); torch.manual_seed(4); fastsim.apply()
    for _,changes in OPTIONS: assert not(set(changes)-set(Params.__dataclass_fields__))
    model=make_net(torch).eval(); w=weights(model); nn=Network(w)
    x=np.random.default_rng(4).normal(size=(9,INPUT)).astype('float32')
    with torch.no_grad():
        z=model(torch.from_numpy(x)); expected=z[:,:ACTIONS].softmax(-1).numpy()
    probs,v=nn.forward(x)
    assert np.allclose(probs,expected,atol=2e-6)
    assert np.allclose(v,z[:,ACTIONS].numpy(),atol=2e-6)
    zero=np.zeros(INPUT,np.float32)
    rows=[(zero,0,0,3.,1.,4.,False),(zero,0,0,4.,2.,0.,True)]
    adv,ret=advantages(rows,lam=1.)
    assert np.allclose(ret,[3,2]) and np.allclose(adv,[0,-2])
    base=json.loads(Path(args.config).read_text())
    cfg=dict(controller='fin6_ppo',base=base,supervisor={'period':5})
    world=World(cfg,2560123,20)
    import time
    rows,episodes=world.collect(w,4,time.monotonic()+60)
    assert len(rows)==4 and len(episodes)==1
    assert np.isclose(sum(r[4] for r in rows)*100,episodes[0]['score']-.1,atol=1e-6)
    # Option zero must preserve the exact incumbent; selecting a different option then zero resets all knobs.
    step=payload(world.st); normal=load_controller(base)
    sup=Supervisor(cfg,network=nn); sup.choose(step,1); sup.choose(step,0)
    assert vars(sup.base.p)==vars(normal.p)
    assert sup.base.act(copy.deepcopy(step))==normal.act(copy.deepcopy(step))
    with tempfile.TemporaryDirectory() as td:
        path=Path(td); np.savez(path/'model.npz',**w)
        cfg['supervisor']['checkpoint']=str(path/'model.npz')
        (path/'config.json').write_text(json.dumps(cfg)); os.environ['SURVIVOR_PARAMS']=str(path/'config.json')
        from survivor.server6 import app
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            for t in (.1,.2,.1,.2):
                body=dict(step,sim_time=t)
                reply=client.post('/predict',json=body)
                assert reply.status_code==200
                assert {a['agent_id'] for a in reply.json()['actions']}=={a['agent_id'] for a in body['agent_status']}
            stats=client.get('/stats').json()
            assert stats['errors']==0 and stats['episode']==1 and stats['options']
            assert client.post('/predict',json=dict(step,game_status='over')).json()['actions']==[]
    print('PASS: score reward telescopes, GAE terminal handling, NumPy/Torch parity, option reset, endpoint and episode reset')
if __name__=='__main__': main()
