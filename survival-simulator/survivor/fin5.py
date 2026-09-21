"""Train QR-DQN on official-engine encounters, then compare full games. See FIN5_RUNBOOK.md."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse, copy, json, math, time, random
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from survivor.qr_escape import Network, EscapeController, INPUT, ACTIONS, QUANTILES

def log(**kw): print(json.dumps(kw,allow_nan=False),flush=True)
def atomic_json(path,doc):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(doc,indent=2)); os.replace(tmp,path)

def arena_sim(seed):
    # Unlike legacy micro._sim, do not train/test on the same three maps.
    from src.core import SimulationCore
    from survivor import fastsim
    fastsim.apply()
    # Eight cached maps per process per seed block; train/eval blocks are disjoint.
    key=(seed//10000,seed%8)
    cache=getattr(arena_sim,'cache',{})
    if key not in cache:
        if len(cache)>=8: cache.clear()
        s=SimulationCore(seed=key[0]*1009+key[1]+5000)
        s.env._spawn_pred_orig=s.env.spawn_predator
        s.env.spawn_predator=lambda *a,**k:None
        s.env.spawn_tree=lambda *a,**k:None
        cache[key]=s; arena_sim.cache=cache
    return cache[key]

class ArenaArm:
    def __init__(self,cfg,weights,epsilon,seed,collect):
        self.cfg=copy.deepcopy(cfg); self.weights=weights; self.epsilon=epsilon; self.seed=seed
        self.collect=collect; self.rows=[]; self.pending=None; self.used=[]
    def start(self,seed,spec,world):
        p=self.cfg['params']; p.update(colony_mgr=0,spawn_thr=1e9,spawn_thr_late=1e9,
            dbed_min_energy=1e9,dbed_min_late=1e9,cap_early=0,cap_late=0,n_min=0,dbed_extra=0,sen_extra=0,doom_spawn=0)
        rng=random.Random(seed+98173)
        for agent in world.env.agents:
            agent.hearing_radius=rng.uniform(20,100); agent.vision_radius=rng.uniform(100,400)
            agent.cone_angle=rng.uniform(math.pi/4,math.pi/2)
        self.ctrl=EscapeController(self.cfg,self.seed,Network(self.weights) if self.weights else None,self.epsilon)
    def act(self,step,world):
        acts=self.ctrl.act(step)
        if step['agent_status']:
            a=step['agent_status'][0]; x,i=self.ctrl.decisions[a['agent_id']]
            if self.collect and self.pending is not None:
                old,j,e=self.pending
                self.rows.append((old,j,.1-.01*max(0,e-a['energy']),x,False))
            self.pending=(x,i,a['energy']); self.used.append(i)
        return acts
    def state(self,aid): return self.ctrl.last_state.get(aid,'?')
    def finish(self,raw):
        alive=bool(raw['alive'])
        if self.collect and not alive and self.pending is not None:
            x,i,e=self.pending; self.rows.append((x,i,-20.,np.zeros(INPUT,np.float32),True))
        # At the time limit the last selected action was not executed: discard it.
        return {'alive':alive,'seconds':float(raw['dead'][0][0]) if raw['dead'] else raw['seconds'],
                'energy':sum(v[0] for v in raw['alive'].values()),
                'baseline_fraction':self.used.count(48)/max(1,len(self.used)),
                'predators':len(raw['predator_energy'])}

def arena_job(job):
    cfg,weights,epsilon,seed,seconds,collect=job
    from survivor import micro
    micro._sim=arena_sim
    r=random.Random(seed); me=r.choice([180,250,350,500,750]); sp=r.uniform(7,20)
    spr=r.uniform(max(10,sp*.7),min(40,max(20,sp+12)))
    ef=r.uniform(.08,.95); n=2 if r.random()<.25 else 1
    angle=r.uniform(-math.pi,math.pi)
    spec={'seconds':seconds,'t_off':900,'agents':[{'tag':'a','dx':0,'dy':0,'E':me*ef,
          'maxE':me,'speed':sp,'sprint':spr,'age':r.uniform(0,80)}],
          'predators':[{'dist':r.uniform(45,165),'bearing':angle+i*r.uniform(.8,3.5),
                        'E':r.choice([35,80,150,195]),'aimed':r.random()<.8} for i in range(n)]}
    arm=ArenaArm(cfg,weights,epsilon,seed,collect)
    raw=micro.run_arena(spec,arm,seed,trace=False)
    stats=arm.finish(raw)
    # Failed spawns are not useful encounter samples; do not inflate survival with empty arenas.
    if not raw['init'] or stats['predators']==0: return [],None
    return arm.rows,stats

class Replay:
    def __init__(self,capacity=300000):
        self.x=np.zeros((capacity,INPUT),np.float32); self.y=np.zeros_like(self.x)
        self.a=np.zeros(capacity,np.int64); self.r=np.zeros(capacity,np.float32)
        self.g=np.zeros(capacity,np.float32); self.n=0; self.pos=0; self.cap=capacity
    def add(self,rows):
        # Three-step returns, shortened correctly at death and at collection boundaries.
        for j,(x,a,_,_,_) in enumerate(rows):
            reward=0.; discount=1.; nxt=None
            for _,_,r,y,done in rows[j:j+3]:
                reward+=discount*r; discount*=.995; nxt=y
                if done: discount=0.; break
            i=self.pos; self.x[i]=x; self.y[i]=nxt; self.a[i]=a; self.r[i]=reward; self.g[i]=discount
            self.pos=(i+1)%self.cap; self.n=min(self.n+1,self.cap)
    def sample(self,batch,rng):
        i=rng.integers(self.n,size=batch)
        return self.x[i],self.a[i],self.r[i],self.y[i],self.g[i]

def make_net(torch):
    return torch.nn.Sequential(torch.nn.Linear(INPUT,128),torch.nn.ReLU(),torch.nn.Linear(128,128),
                               torch.nn.ReLU(),torch.nn.Linear(128,ACTIONS*QUANTILES))
def weights(net): return {k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}

def snapshot(net,opt,cfg,out,steps,updates,torch):
    stem=f'model_{steps:09d}_{updates:07d}'; npz=out/(stem+'.npz')
    with open(str(npz)+'.tmp','wb') as f: np.savez_compressed(f,**weights(net))
    os.replace(str(npz)+'.tmp',npz)
    pt=out/'resume.pt'; torch.save({'net':net.state_dict(),'optimizer':opt.state_dict(),
        'steps':steps,'updates':updates},str(pt)+'.tmp'); os.replace(str(pt)+'.tmp',pt)
    for mode in ('mean','risk','upper'):
        doc=copy.deepcopy(cfg); doc['controller']='qr_dqn'; doc['escape']={'checkpoint':str(npz.resolve()),'mode':mode,'range':180}
        doc['experimental']=True; doc['training_steps']=steps
        atomic_json(out/f'latest_{mode}.json',doc)
        atomic_json(out/f'{stem}_{mode}.json',doc)
    log(event='checkpoint',steps=steps,updates=updates,config=str(out/'latest_mean.json'),validation_ready=True,
        note='Experimental snapshot; no full-game improvement established')

def train(args):
    import torch
    torch.set_num_threads(1)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Fix PyTorch or use --device cpu.')
    torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    net=make_net(torch).to(device)
    with torch.no_grad(): net[4].bias.reshape(ACTIONS,QUANTILES)[48].add_(.25)
    opt=torch.optim.Adam(net.parameters(),lr=3e-4); steps=updates=0
    if args.resume:
        cp=torch.load(args.resume,map_location=device,weights_only=True); net.load_state_dict(cp['net'])
        opt.load_state_dict(cp['optimizer']); steps=cp['steps']; updates=cp['updates']
    target=copy.deepcopy(net).eval(); replay=Replay(args.capacity)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.config).read_text()); cfg['escape']={'mode':'mean','range':180}
    tau=(torch.arange(QUANTILES,device=device)+.5)/QUANTILES
    deadline=time.monotonic()+args.minutes*60; start=time.monotonic(); lastsave=start; lastlog=start
    pool=get_context('spawn').Pool(args.workers)
    n_episodes=0; nextseed=args.seed+steps; recent=[]; losses=[]; credit=0.; newsteps=0
    log(event='train_start',device=device,workers=args.workers,minutes=args.minutes,input=INPUT,actions=ACTIONS,
        quantiles=QUANTILES,resumed_steps=steps,replay='fresh',note='Initial checkpoint after first 2048 fresh transitions')
    try:
        while time.monotonic()<deadline:
            epsilon=max(.05,.65*(1-min(1,steps/100000)))
            w=weights(net)
            jobs=[(cfg,w,epsilon,nextseed+i,args.arena_seconds,True) for i in range(args.workers*2)]
            nextseed+=len(jobs)
            futures=[pool.apply_async(arena_job,(j,)) for j in jobs]
            for f in futures:
                remaining=deadline-time.monotonic()
                if remaining<=0: break
                try: rows,stat=f.get(timeout=remaining)
                except __import__('multiprocessing').TimeoutError: break
                if stat is None: continue
                replay.add(rows); count=len(rows); steps+=count; newsteps+=count; credit+=count/32
                n_episodes+=1; recent.append(stat); recent=recent[-200:]
                if replay.n>=2048:
                    for _ in range(min(64,int(credit))):
                        if time.monotonic()>=deadline: break
                        x,a,r,y,g=[torch.as_tensor(z,device=device) for z in replay.sample(128,rng)]
                        q=net(x).reshape(-1,ACTIONS,QUANTILES)[torch.arange(len(x),device=device),a]
                        with torch.no_grad():
                            choice=net(y).reshape(-1,ACTIONS,QUANTILES).mean(-1).argmax(-1)
                            # Deployment delegates to baseline when no visible predator is within range.
                            outside=(y[:,14]<.5)|(torch.linalg.vector_norm(y[:,15:17],dim=1)>.6)
                            choice[outside]=48
                            z=target(y).reshape(-1,ACTIONS,QUANTILES)[torch.arange(len(x),device=device),choice]
                            z=r[:,None]+g[:,None]*z
                        delta=z[:,None,:]-q[:,:,None]; ad=delta.abs()
                        huber=torch.where(ad<=1,.5*delta.square(),ad-.5)
                        loss=((tau[None,:,None]-(delta.detach()<0).float()).abs()*huber).mean()
                        if not torch.isfinite(loss): raise RuntimeError('Nonfinite QR loss')
                        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),10)
                        opt.step(); updates+=1; credit-=1; losses.append(float(loss.detach())); losses=losses[-100:]
                        if updates%250==0: target.load_state_dict(net.state_dict())
                now=time.monotonic()
                if now-lastlog>=15:
                    log(event='train',minutes=round((now-start)/60,2),steps=steps,new_steps_per_s=round(newsteps/max(1,now-start),1),
                        episodes=n_episodes,replay=replay.n,updates=updates,epsilon=round(epsilon,3),
                        loss=round(float(np.mean(losses)),5) if losses else None,
                        exploratory_survival=round(float(np.mean([s['alive'] for s in recent])),3),
                        baseline_fraction=round(float(np.mean([s['baseline_fraction'] for s in recent])),3))
                    lastlog=now
                if updates and (not (out/'latest_mean.json').exists() or now-lastsave>=args.checkpoint_seconds):
                    snapshot(net,opt,cfg,out,steps,updates,torch); lastsave=now
    finally:
        pool.terminate(); pool.join()
        if updates: snapshot(net,opt,cfg,out,steps,updates,torch)
    log(event='train_end',steps=steps,updates=updates,wall_minutes=round((time.monotonic()-start)/60,2))

def arena_eval(args):
    cfg=json.loads(Path(args.config).read_text()); w=None
    if cfg.get('escape',{}).get('checkpoint'): w=Network.load(cfg['escape']['checkpoint']).w
    jobs=[(cfg,w,0,args.seed+i,args.arena_seconds,False) for i in range(args.seeds)]
    with get_context('spawn').Pool(args.workers) as pool: results=pool.map(arena_job,jobs)
    vals=[s for _,s in results if s is not None]
    log(event='arena_validation',config=args.config,n=len(vals),requested=args.seeds,
        survival=float(np.mean([s['alive'] for s in vals])) if vals else None,
        mean_seconds=float(np.mean([s['seconds'] for s in vals])) if vals else None,
        energy_per_initial_agent=float(np.mean([s['energy'] for s in vals])) if vals else None,
        baseline_fraction=float(np.mean([s['baseline_fraction'] for s in vals])) if vals else None)
    if args.out:
        Path(args.out).mkdir(parents=True,exist_ok=True)
        atomic_json(Path(args.out)/(Path(args.config).stem+'_arena.json'),dict(config=cfg,seeds=list(range(args.seed,args.seed+args.seeds)),results=[s for _,s in results]))

def game_job(job):
    from survivor import fin4
    from survivor.qr_escape import load_controller
    fin4.load_controller=load_controller
    return fin4.episode(job)

def evaluate(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # Snapshot configs before running: training may atomically replace latest configs.
    configs={Path(p).stem:json.loads(Path(p).read_text()) for p in args.configs}
    if len(configs)!=len(args.configs): raise ValueError('Config basenames must be distinct')
    atomic_json(out/'evaluated_configs.json',configs)
    baseline=next(iter(configs)); deadline=time.monotonic()+args.minutes*60
    jobs=[(name,(cfg,args.seed+i,args.horizon,deadline,True)) for i in range(args.seeds) for name,cfg in configs.items()]
    rows={k:{} for k in configs}; pool=get_context('spawn').Pool(args.workers)
    from survivor.fin4 import summary,paired
    try:
        futures=[(name,pool.apply_async(game_job,(j,))) for name,j in jobs]
        for name,f in futures:
            remain=deadline-time.monotonic()
            if remain<=0 and not f.ready(): continue
            try: r=f.get(timeout=max(.001,remain))
            except __import__('multiprocessing').TimeoutError: continue
            with open(out/'games.jsonl','a') as h: h.write(json.dumps({'config':name,'result':r})+'\n')
            if r.get('censored'): continue
            rows[name][r['seed']]=r
            log(event='game',arm=name,seed=r['seed'],score=round(r['score'],2),seconds=round(r['sim_time'],1),
                policy_ms=round(r['policy_ms_mean'],2),population=r['peak_population'])
    finally: pool.terminate(); pool.join()
    stats={k:summary(v) for k,v in rows.items()}
    pairs={k:paired(v,rows[baseline]) for k,v in rows.items() if k!=baseline}
    for pair in pairs.values():
        if pair['n']<2: pair['lo']=pair['hi']=None
    report={'baseline':baseline,'summary':stats,'paired_vs_baseline':pairs,'horizon':args.horizon,
            'note':'Pilot comparisons, not a promotion gate; censored episodes excluded.'}
    atomic_json(out/'report.json',report)
    for k,s in stats.items(): log(event='summary',arm=k,**s)
    for k,s in pairs.items(): print(json.dumps({'event':'paired','arm':k,**s}),flush=True)

def combine(args):
    cfg=json.loads(Path(args.strategy).read_text()); learned=json.loads(Path(args.escape).read_text())
    if not learned.get('escape',{}).get('checkpoint'): raise ValueError('Escape config has no model')
    cfg['escape']=learned['escape']; cfg['experimental']=True; cfg['controller']='qr_dqn_combined'
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); atomic_json(args.out,cfg)
    log(event='combined_config',config=args.out)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--config',default='configs/fin4_baseline.json')
    tr.add_argument('--out',default='runs/fin5'); tr.add_argument('--minutes',type=float,default=30)
    tr.add_argument('--workers',type=int,default=12); tr.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    tr.add_argument('--seed',type=int,default=200000); tr.add_argument('--arena-seconds',type=float,default=15)
    tr.add_argument('--capacity',type=int,default=300000); tr.add_argument('--resume'); tr.add_argument('--checkpoint-seconds',type=float,default=300)
    ev=sub.add_parser('arena'); ev.add_argument('--config',required=True); ev.add_argument('--seeds',type=int,default=128)
    ev.add_argument('--seed',type=int,default=900000); ev.add_argument('--workers',type=int,default=12)
    ev.add_argument('--arena-seconds',type=float,default=20); ev.add_argument('--out',default='runs/fin5_arena')
    ge=sub.add_parser('evaluate'); ge.add_argument('--configs',nargs='+',required=True)
    ge.add_argument('--out',default='runs/fin5_eval'); ge.add_argument('--minutes',type=float,default=15)
    ge.add_argument('--seeds',type=int,default=8); ge.add_argument('--seed',type=int,default=1910000)
    ge.add_argument('--horizon',type=float,default=3000); ge.add_argument('--workers',type=int,default=12)
    co=sub.add_parser('combine'); co.add_argument('--strategy',required=True); co.add_argument('--escape',required=True); co.add_argument('--out',required=True)
    args=ap.parse_args()
    if getattr(args,'workers',1)<1: ap.error('--workers must be positive')
    {'train':train,'arena':arena_eval,'evaluate':evaluate,'combine':combine}[args.cmd](args)
if __name__=='__main__': main()
