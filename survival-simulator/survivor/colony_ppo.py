"""Observation-only PPO strategy supervisor; NumPy serving, no Torch dependency."""
import copy
from collections import Counter
import numpy as np
from survivor.qr_escape import load_controller as load_base

# Every selection is an overlay on original params, never on the previous option.
OPTIONS=[
 ('incumbent',{}),
 ('grow',dict(cap_early=60,cap_late=12,cap_tau=600,spawn_thr=145,spawn_thr_late=220,
              reserve_margin=20,young_frac=.75,birth_gap_s=1,gap_after_s=0)),
 ('bank',dict(cap_early=8,cap_late=2,cap_tau=600,n_min=1,sen_extra=1,dbed_extra=0,
              spawn_thr=280,spawn_thr_late=450,reserve_margin=60,rest_frac=.85,late_rest_frac=.85)),
 ('harvest_now',dict(ripe_age=5,ripe_unknown=0,urgent_abs=80,rest_frac=.95,late_rest_frac=.95,
                     camp_patience=12,crowd_max=1,spread_w=1.5)),
 ('ripen',dict(ripe_age=30,ripe_unknown=3,urgent_abs=40,camp_patience=40,
               rest_frac=.65,late_rest_frac=.65)),
 ('relay',dict(dbed_age=30,dbed_min_energy=140,dbed_min_late=180,cap_early=30,cap_late=8,
               reserve_margin=20,birth_gap_s=2,gap_after_s=0,young_frac=.7,age_bonus=2)),
 ('preserve_parents',dict(spawn_thr=380,spawn_thr_late=700,reserve_margin=110,
                          dbed_min_energy=220,dbed_min_late=350,cap_early=15,cap_late=4)),
 ('select_speed',dict(select_q=.85,w_speed=5,w_energy=.15,w_sprint=.2,
                      cap_early=45,cap_late=8,reserve_margin=30)),
]
ACTIONS=len(OPTIONS); FRAME=36; INPUT=2*FRAME+ACTIONS

def frame(step):
    agents=step.get('agent_status') or []; n=len(agents)
    if not n: return np.zeros(FRAME,np.float32)
    e=np.array([a['energy'] for a in agents]); me=np.array([a['max_energy'] for a in agents])
    age=np.array([a['age'] for a in agents]); sp=np.array([min(a['speed'],a['sprint_speed']) for a in agents])
    nearest=[]; food=[]; tree=[]; crowd=[]
    for a in agents:
        obs=a['observations']
        nearest.append(min([p['distance'] for p in obs if p['type']=='Predator'] or [300]))
        food.append(sum(p['type']=='Fruit' and p['distance']<100 for p in obs))
        tree.append(any(p['type']=='Tree' and p['distance']<70 for p in obs))
        crowd.append(sum(p['type']=='Agent' and p['distance']<80 for p in obs))
    nearest=np.asarray(nearest)
    f=[step.get('sim_time',0)/3000,n/60,np.log1p(n)/5,e.sum()/5000,e.mean()/350,
       np.quantile(e,.1)/350,np.median(e)/350,e.max()/1000,me.mean()/1000,
       np.mean(e>=me*.2+40),np.mean(e<50),age.mean()/120,np.mean(age<30),np.mean(age>=75),
       sp.mean()/20,sp.max()/20,np.mean(sp>15.5),np.mean([a['sprint_speed'] for a in agents])/40,
       np.mean([a['hearing_radius'] for a in agents])/100,np.mean([a['vision_range'] for a in agents])/400,
       np.mean([a['vision_angle'] for a in agents])/1.5707963,
       np.mean(nearest<180),np.mean(nearest<60),nearest.min()/300,
       np.mean(food)/8,np.mean(np.array(food)>0),np.mean(tree),np.mean(crowd)/10]
    f += [np.mean([a['biome']==b for a in agents]) for b in ('grassland','forest','swamp','desert','river')]
    f += [step.get('score',0)/3000,np.mean(e/(me+1e-6)),np.mean(me<375)]
    f=np.clip(np.asarray(f,np.float32),-5,5)
    if len(f)!=FRAME: raise ValueError('Feature schema mismatch')
    return f

class Network:
    def __init__(self,w): self.w=w
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f: return cls({k:f[k] for k in f.files})
    def forward(self,x):
        x=np.asarray(x,np.float32)
        for i in (0,2): x=np.tanh(x@self.w[f'{i}.weight'].T+self.w[f'{i}.bias'])
        z=x@self.w['4.weight'].T+self.w['4.bias']
        logits=z[..., :ACTIONS]; v=z[...,ACTIONS]
        p=np.exp(logits-logits.max(axis=-1,keepdims=True)); p/=p.sum(axis=-1,keepdims=True)
        return p,v

class Supervisor:
    def __init__(self,cfg,seed=0,network=None):
        self.cfg=copy.deepcopy(cfg); self.seed=seed; self.period=float(cfg.get('supervisor',{}).get('period',20))
        if self.period<=0: raise ValueError('period must be positive')
        self.network=network or Network.load(cfg['supervisor']['checkpoint'])
        self.reset()
    def reset(self):
        self.base=load_base(self.cfg['base'],self.seed); self.original=copy.copy(self.base.p)
        self.previous=None; self.last_option=0; self.next_decision=-1; self.last_time=-1
        self.extra=Counter(); self.rng=np.random.default_rng(self.seed)
    def features(self,step):
        now=frame(step); prev=now if self.previous is None else self.previous
        return np.concatenate([now,prev,np.eye(ACTIONS,dtype=np.float32)[self.last_option]])
    def choose(self,step,index):
        self.previous=frame(step); self.last_option=int(index)
        self.base.p=copy.copy(self.original)
        for k,v in OPTIONS[index][1].items(): setattr(self.base.p,k,float(v))
        self.next_decision=float(step['sim_time'])+self.period
        self.extra[OPTIONS[index][0]]+=1
    def act(self,step):
        t=float(step.get('sim_time',0))
        if t<self.last_time-1e-6: self.reset()
        self.last_time=t
        if step.get('game_status','ok')!='ok': return []
        if step.get('agent_status') and t>=self.next_decision-1e-6:
            probs,_=self.network.forward(self.features(step)[None])
            index=int(self.rng.choice(ACTIONS,p=probs[0].astype(float)/probs[0].sum(dtype=float))) if self.cfg.get('supervisor',{}).get('sample',False) else int(probs[0].argmax())
            self.choose(step,index)
        return self.base.act(step)

def load_controller(cfg,seed=0):
    if cfg.get('controller')=='fin6_ppo': return Supervisor(cfg,seed)
    return load_base(cfg,seed)
