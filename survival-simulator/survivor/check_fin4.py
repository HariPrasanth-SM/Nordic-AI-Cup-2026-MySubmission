"""Focused mechanics and integration checks, not a score benchmark."""
import copy
import json
import math
from pathlib import Path
from survivor.frontier import load_controller
from survivor.fin4 import ROOT, Race, summary, paired, key


def agent(i=1,E=180,age=30,observations=None):
    return {'agent_id':i,'energy':E,'age':age,'speed':10.,'sprint_speed':20.,'max_energy':500.,
            'hearing_radius':50.,'vision_range':200.,'vision_angle':math.pi/3,'biome':'forest',
            'observations':observations or []}


def main():
    cfg=json.loads((ROOT/'configs/fin4_candidate.json').read_text())
    c=load_controller(cfg)
    def step(agents,t=1): return {'game_status':'ok','sim_time':t,'agent_status':agents,'score':t}
    a=agent(E=25,observations=[{'type':'Fruit','distance':35.,'angle':math.pi/2}])
    out=c.act(step([a])); assert len(out)==1 and abs(out[0]['move_direction']-math.pi/2)<1e-7
    assert 0<out[0]['move_distance']<=10
    print('PASS hungry agent moves toward food in relative coordinates')
    # Observation-only birth with old, low-rank parent must still replace its lineage.
    c=load_controller(cfg)
    a=agent(E=170,age=80,observations=[{'type':'Tree','distance':10.,'angle':0.}])
    ac=c.act(step([a]))[0]; assert ac['spawn_agent']; assert c.mem[1]['last_spawn']
    a['energy']=69.9; a['age']=80.1
    c.act(step([a],1.1)); assert not c.mem[1].get('aging',False)
    print('PASS approved birth bookkeeping does not falsely detect aging')
    c=load_controller(cfg); a=agent(E=90,age=70,observations=[{'type':'Tree','distance':10.,'angle':0.}])
    c.act(step([a])); a['energy']=89.2; a['age']=70.1; c.act(step([a],1.1))
    a['energy']=88.4; a['age']=70.2; c.act(step([a],1.2))
    assert c.mem[1].get('aging')
    print('PASS repeated unexplained age drain detected')
    a=agent(E=25,observations=[{'type':'Fruit','distance':35.,'angle':0.}])
    c=load_controller(cfg); first=c.act(step([a],.1)); c.act(step([a],5))
    assert c.act(step([a],.1))==first
    print('PASS repeated episodes reset to the same serving random stream')
    c=load_controller(cfg); a=agent(E=200,observations=[{'type':'Tree','distance':50.,'angle':0.},{'type':'Tree','distance':80.,'angle':.5}])
    c.act(step([a])); m=c.mem[1]; m['landmarks']=[('T',50.,0.),('T',80*math.cos(.5),80*math.sin(.5))]
    m['x']=10.;m['y']=0.;m['h']=0.
    c._correct_pose(a,m); assert abs(m['x'])<1e-6
    print('PASS landmark correction recovers a blocked move')
    # Fruit should not be yielded to a neighbour which cannot see/smell it.
    c.by_id={1:agent(1,E=200),2:agent(2,E=30)}
    a=agent(1,E=200,observations=[{'type':'Agent','id':2,'distance':20.,'angle':0.,'rel_dir':math.pi}])
    assert c._owned(a,40,0,[(20,0,2)])
    print('PASS food allocation requires the neighbour to observe the fruit')
    # HTTP contract, running status, and episode resets through the actual new endpoint.
    import os
    os.environ['SURVIVOR_PARAMS']=str(ROOT/'configs/fin4_candidate.json')
    from fastapi.testclient import TestClient
    from survivor.server4 import app
    with TestClient(app) as client:
        for t in (.1,3.,.1):
            payload=step([agent()],t); payload['game_status']='running'
            r=client.post('/predict',json=payload); assert r.status_code==200
            ac=r.json()['actions']; assert len(ac)==1 and ac[0]['agent_id']==1
            assert all(math.isfinite(ac[0][k]) for k in ('move_distance','move_direction','turn_angle'))
        assert client.get('/stats').json()['errors']==0
    print('PASS HTTP actions, status normalization, and three-request reset')
    # Cache keys actually change when the algorithm or relevant config changes.
    b=copy.deepcopy(cfg); b['frontier']={'food_wait':3.}
    assert key(cfg)!=key(b)
    print('ALL FIN4 CHECKS PASSED')

if __name__=='__main__': main()
