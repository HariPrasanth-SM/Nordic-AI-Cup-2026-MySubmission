"""Observation-only QR escape policy. NumPy inference; PyTorch is training-only."""
import json, math
from pathlib import Path
import numpy as np
from survivor.policy import Controller, Params, BIOME_PEN, wrap
ACTIONS, QUANTILES, FRAME = 49, 32, 53
INPUT = FRAME * 2

def frame(a):
    f = [a['energy']/350, a['energy']/max(1,a['max_energy']), a['max_energy']/1000,
         a['speed']/20, a['sprint_speed']/40, a['age']/120,
         a.get('vision_range',200)/400, a.get('hearing_radius',50)/100,
         a.get('vision_angle',1)/math.pi]
    f += [float(a['biome']==b) for b in ('grassland','forest','swamp','desert','river')]
    obs=a['observations']; preds=sorted((p for p in obs if p['type']=='Predator'),key=lambda p:p['distance'])[:3]
    for i in range(3):
        if i<len(preds):
            p=preds[i]; d=p['distance']; ang=p['angle']; h=ang+math.pi-p.get('rel_dir',0)
            f += [1,d*math.cos(ang)/300,d*math.sin(ang)/300,math.cos(h),math.sin(h)]
        else: f += [0]*5
    # Nearest point on each observed edge, preserving occlusion/visibility limitations.
    pts=[]
    for e in obs:
        if e['type']!='Edge': continue
        v=np.asarray(e['coords'],dtype=float).reshape(2,2); d=v[1]-v[0]
        p=v[0]+np.clip(-np.dot(v[0],d)/(np.dot(d,d)+1e-8),0,1)*d
        pts.append((float(p@p),float(p[0]),float(p[1])))
    pts.sort()
    for i in range(8):
        f += [1,pts[i][1]/300,pts[i][2]/300] if i<len(pts) else [0,0,0]
    return np.clip(np.asarray(f,dtype=np.float32),-4,4)

def maneuver(a,index,base):
    if index==48: return dict(base)
    preds=sorted((p for p in a['observations'] if p['type']=='Predator'),key=lambda p:p['distance'])
    if not preds: return dict(base)
    # Eight bearings relative to nearest predator: includes shoulder passes and crossing.
    angle=wrap(preds[0]['angle']+math.pi+(index//6)*math.pi/4)
    speed_mode=(index//2)%3; face=index%2
    walk=min(a['speed'],a['sprint_speed'])
    speed=(walk,min(a['sprint_speed'],max(walk,16.0)),a['sprint_speed'])[speed_mode]
    if a['energy']<a['max_energy']/5: speed=min(speed,walk)
    turn=wrap(preds[0]['angle'] if face else angle)
    return dict(agent_id=a['agent_id'],move_distance=float(speed),move_direction=float(angle),
                turn_angle=float(turn),spawn_agent=False)

class Network:
    def __init__(self, weights): self.w=weights
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f: return cls({k:f[k] for k in f.files})
    def quantiles(self,x):
        x=np.asarray(x,dtype=np.float32)
        for i in (0,2): x=np.maximum(0,x@self.w[f'{i}.weight'].T+self.w[f'{i}.bias'])
        return (x@self.w['4.weight'].T+self.w['4.bias']).reshape(-1,ACTIONS,QUANTILES)
    def choose(self,x,mode='mean'):
        q=self.quantiles(x)
        if mode!='mean':
            q=np.sort(q,axis=-1)
            q=q[...,:8] if mode=='risk' else q[...,-8:]
        return q.mean(-1).argmax(-1)

class EscapeController(Controller):
    def __init__(self,cfg,seed=0,network=None,epsilon=0):
        p=Params()
        for k,v in cfg.get('params',{}).items():
            if hasattr(p,k): setattr(p,k,float(v))
        super().__init__(p,seed=seed)
        self.cfg=cfg; self.qcfg=cfg.get('escape',{}); self.qprev={}; self.decisions={}
        self.network=network or (Network.load(self.qcfg['checkpoint']) if self.qcfg.get('checkpoint') else None)
        self.epsilon=epsilon; self.qrng=np.random.default_rng(seed); self.qtime=-1
    def trait_score(self,a):
        if self.cfg.get('breeding')=='speed_bank':
            speed=min(a['speed'],a['sprint_speed'])
            return speed/20 + .7*float(speed>15.5) + (.5 if speed>15.5 else -.1)*a['max_energy']/1000 + .05*a['hearing_radius']/100
        return super().trait_score(a)
    def act(self,step):
        t=step.get('sim_time',0)
        if t<self.qtime: self.qprev.clear()
        self.qtime=t; self.decisions={}
        alive={a['agent_id'] for a in step.get('agent_status',[])}
        self.qprev={k:v for k,v in self.qprev.items() if k in alive}
        return super().act(step)
    def _act_one(self,a,t,n_eff,cap,sel_ok):
        cur=frame(a); aid=a['agent_id']; x=np.concatenate([cur,self.qprev.get(aid,cur)])
        self.qprev[aid]=cur
        base=super()._act_one(a,t,n_eff,cap,sel_ok)
        preds=[p for p in a['observations'] if p['type']=='Predator']
        index=48
        if self.network is not None and preds and min(p['distance'] for p in preds)<self.qcfg.get('range',180):
            if self.qrng.random()<self.epsilon:
                index=48 if self.qrng.random()<.2 else int(self.qrng.integers(ACTIONS))
            else: index=int(self.network.choose(x[None],self.qcfg.get('mode','mean'))[0])
        self.decisions[aid]=(x,index)
        out=maneuver(a,index,base)
        if self.cfg.get('maneuver_mode')=='cruise16' and preds and min(p['distance'] for p in preds)>70 and a['biome'] in ('grassland','forest') and out['move_distance']>16:
            out['move_distance']=max(min(a['speed'],a['sprint_speed']),16.)
        if out!=base:
            m=self.mem[aid]; h=m['h']-base['turn_angle']; pen=BIOME_PEN.get(a['biome'],1)
            for key,fn in [('x',math.cos),('y',math.sin)]:
                m[key]+=pen*(out['move_distance']*fn(h+out['move_direction'])-base['move_distance']*fn(h+base['move_direction']))
            m['h']=h+out['turn_angle']; m['last_spawn']=bool(out['spawn_agent']); m['still']=0
            d=out['move_distance']; sp=a['speed']
            m['last_cost']=.1+min(math.pi,abs(out['turn_angle']))/(2*math.pi)+.05*min(d,sp)+.5*max(0,d-sp)
            m['flee_dir']=wrap(out['move_direction']-out['turn_angle']); m['flee_until']=t+self.p.flee_hold*.1
            m['threat_now']=True; m['doomed']=False; self.last_state[aid]='qr_escape'
        return out

def load_controller(cfg,seed=0):
    if not cfg.get('escape') and not cfg.get('breeding') and not cfg.get('maneuver_mode'):
        from survivor.frontier import load_controller as original
        return original(cfg,seed)
    return EscapeController(cfg,seed)
