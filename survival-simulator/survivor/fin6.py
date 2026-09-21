"""PPO on actual full-game score, selecting a colony strategy every 20 seconds."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ.setdefault('SDL_VIDEODRIVER','dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse, copy, json, time, multiprocessing as mp
from pathlib import Path
import numpy as np
from survivor.colony_ppo import Network,Supervisor,INPUT,ACTIONS,OPTIONS
from survivor.fin5 import atomic_json,log


def payload(st):
    return dict(game_status='ok',sim_time=st['sim_time'],score=st['score'],n_agents=st['num_agents'],
                agent_status=[a for a in st['observations'] if a])

class World:
    def __init__(self,cfg,seed,horizon):
        self.cfg=cfg; self.seed=seed; self.episode=0; self.horizon=horizon
        self.rng=np.random.default_rng(seed); self.ctrl=None; self.sim=None; self.st=None
    def reset(self,network):
        from src.core import SimulationCore
        self.sim=SimulationCore(seed=self.seed+self.episode); self.episode+=1
        self.st=self.sim.step([])
        self.ctrl=Supervisor(self.cfg,network=network)
    def collect(self,w,length,deadline):
        from survivor.runner import _Act
        network=Network(w)
        if self.sim is None: self.reset(network)
        self.ctrl.network=network
        rows=[]; completed=[]
        for _ in range(length):
            if time.monotonic()>=deadline: break
            step=payload(self.st); x=self.ctrl.features(step); prob,val=network.forward(x[None])
            action=int(self.rng.choice(ACTIONS,p=prob[0].astype(float)/prob[0].sum(dtype=float)))
            oldscore=float(self.st['score']); self.ctrl.choose(step,action)
            end=step['sim_time']+self.ctrl.period; count=0
            while self.st['num_agents']>0 and self.st['sim_time']<min(end,self.horizon)-1e-7:
                acts=self.ctrl.base.act(payload(self.st))
                self.st=self.sim.step([(a['agent_id'],_Act(a)) for a in acts]); count+=1
                if count%50==0 and time.monotonic()>=deadline: break
            done=self.st['num_agents']==0 or self.st['sim_time']>=self.horizon-1e-7
            nxt=self.ctrl.features(payload(self.st)); _,nv=network.forward(nxt[None])
            reward=(float(self.st['score'])-oldscore)/100.
            rows.append((x,action,float(np.log(max(prob[0,action],1e-12))),float(val[0]),
                         reward,0. if done else float(nv[0]),done))
            if done:
                completed.append(dict(seed=self.seed+self.episode-1,score=float(self.st['score']),
                                      seconds=float(self.st['sim_time']),options=dict(self.ctrl.extra)))
                if time.monotonic()<deadline: self.reset(network)
                else: self.sim=None
        return rows,completed

_WORLD=None
def init_worker(cfg,seed,horizon):
    global _WORLD
    from survivor import fastsim
    fastsim.apply()
    # Each process owns persistent full games and a disjoint seed block.
    ident=mp.current_process()._identity[-1]
    _WORLD=World(cfg,seed+ident*100000,horizon)
def rollout(job): return _WORLD.collect(*job)

def advantages(rows,lam=.95):
    adv=np.zeros(len(rows),np.float32); acc=0.
    # gamma=1: target undiscounted official score; terminal prevents cross-episode credit.
    for i in range(len(rows)-1,-1,-1):
        _,_,_,value,reward,nextvalue,done=rows[i]
        delta=reward+nextvalue-value
        acc=delta+(0 if done else lam*acc); adv[i]=acc
    returns=adv+np.asarray([r[3] for r in rows],np.float32)
    return adv,returns

def make_net(torch):
    return torch.nn.Sequential(torch.nn.Linear(INPUT,64),torch.nn.Tanh(),torch.nn.Linear(64,64),
                               torch.nn.Tanh(),torch.nn.Linear(64,ACTIONS+1))
def weights(net): return {k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
def export(net,opt,cfg,out,steps,updates,torch):
    stem=f'colony_{steps:08d}_{updates:05d}'; path=out/(stem+'.npz')
    with open(str(path)+'.tmp','wb') as f: np.savez_compressed(f,**weights(net))
    os.replace(str(path)+'.tmp',path)
    doc=copy.deepcopy(cfg); doc['supervisor']['checkpoint']=str(path.resolve())
    atomic_json(out/'latest_colony.json',doc); atomic_json(out/(stem+'.json'),doc)
    sampled=copy.deepcopy(doc); sampled['supervisor']['sample']=True
    atomic_json(out/'latest_colony_sampled.json',sampled); atomic_json(out/(stem+'_sampled.json'),sampled)
    torch.save(dict(net=net.state_dict(),optimizer=opt.state_dict(),steps=steps,updates=updates,cfg=cfg),str(out/'resume.pt.tmp'))
    os.replace(out/'resume.pt.tmp',out/'resume.pt')
    log(event='checkpoint',macro_steps=steps,updates=updates,config=str(out/'latest_colony.json'),
        note='Loadable experiment; full-game holdout required')

def train(args):
    import torch
    torch.set_num_threads(1); torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    net=make_net(torch).to(device)
    with torch.no_grad():
        net[4].weight.mul_(.01); net[4].bias.zero_(); net[4].bias[0]=2.
    opt=torch.optim.Adam(net.parameters(),lr=2e-4,eps=1e-5); steps=updates=0
    cfg=dict(controller='fin6_ppo',base=json.loads(Path(args.base).read_text()),
             supervisor=dict(period=args.period),experimental=True)
    if cfg['base'].get('controller') not in ('fin3','qr_dqn'):
        raise ValueError('Use the exact fin3 baseline or an uncombined fin5 QR config; frontier and custom breeding are unsupported.')
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=True)
        net.load_state_dict(saved['net']); opt.load_state_dict(saved['optimizer'])
        steps=saved['steps']; updates=saved['updates']; cfg=saved['cfg']
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'training_config.json',cfg)
    start=time.monotonic(); deadline=start+args.minutes*60; lastsave=start; finished=[]; since=0
    pool=mp.get_context('spawn').Pool(args.workers,initializer=init_worker,
                                   initargs=(cfg,args.seed+updates*1000,args.horizon))
    log(event='train_start',device=device,workers=args.workers,period=cfg['supervisor']['period'],
        actions=[k for k,_ in OPTIONS],input=INPUT,minutes=args.minutes,
        horizon=args.horizon,objective='official score delta / 100; gamma=1; no energy shaping')
    try:
        while time.monotonic()<deadline:
            w=weights(net)
            futures=[pool.apply_async(rollout,((w,args.rollout_steps,deadline),)) for _ in range(args.workers)]
            rows=[]; aa=[]; rr=[]
            lastheartbeat=time.monotonic()
            for f in futures:
                while not f.ready() and time.monotonic()<deadline:
                    try: f.wait(timeout=min(5,max(0,deadline-time.monotonic())))
                    except KeyboardInterrupt: raise
                    if time.monotonic()-lastheartbeat>20:
                        log(event='collecting',minutes=round((time.monotonic()-start)/60,2),
                            completed_workers=sum(g.ready() for g in futures),workers=args.workers,updates=updates)
                        lastheartbeat=time.monotonic()
                if not f.ready(): continue
                trajectory,episodes=f.get()
                if trajectory:
                    a,r=advantages(trajectory); rows.extend(trajectory); aa.extend(a); rr.extend(r)
                for episode in episodes:
                    finished.append(episode); since+=1
                    with open(out/'training_games.jsonl','a') as h: h.write(json.dumps(episode)+'\n')
                    log(event='training_game',**episode)
            if not rows or time.monotonic()>=deadline: break
            steps+=len(rows)
            x=torch.as_tensor(np.stack([r[0] for r in rows]),device=device)
            action=torch.as_tensor([r[1] for r in rows],dtype=torch.long,device=device)
            oldlog=torch.as_tensor([r[2] for r in rows],device=device)
            oldval=torch.as_tensor([r[3] for r in rows],device=device)
            adv=torch.as_tensor(np.asarray(aa,np.float32),device=device); ret=torch.as_tensor(np.asarray(rr,np.float32),device=device)
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            losses=[]; kls=[]; entropies=[]; clipped=[]
            for epoch in range(4):
                if time.monotonic()>=deadline: break
                order=rng.permutation(len(rows)); epochkl=[]
                for i in range(0,len(rows),256):
                    ids=torch.as_tensor(order[i:i+256],device=device)
                    z=net(x[ids]); dist=torch.distributions.Categorical(logits=z[:,:ACTIONS])
                    lp=dist.log_prob(action[ids]); logratio=lp-oldlog[ids]; ratio=logratio.exp()
                    policy=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                    # Clipped critic updates; reward is scaled, never clipped.
                    value=z[:,ACTIONS]; vc=oldval[ids]+(value-oldval[ids]).clamp(-.2,.2)
                    vl=.5*torch.maximum((value-ret[ids]).square(),(vc-ret[ids]).square()).mean()
                    entropy=dist.entropy().mean(); loss=policy+.5*vl-.01*entropy
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite PPO loss')
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),.5); opt.step()
                    kl=float(((ratio-1)-logratio).mean().detach()); epochkl.append(kl); kls.append(kl)
                    losses.append(float(loss.detach())); entropies.append(float(entropy.detach()))
                    clipped.append(float(((ratio-1).abs()>.2).float().mean().detach()))
                if epochkl and np.mean(epochkl)>.03: break
            if not losses: break
            updates+=1; now=time.monotonic()
            log(event='ppo',minutes=round((now-start)/60,2),macro_steps=steps,updates=updates,batch=len(rows),
                completed_games=since,recent_training_score=float(np.mean([g['score'] for g in finished[-20:]])) if finished else None,
                loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),entropy=float(np.mean(entropies)),
                clip_fraction=float(np.mean(clipped)),option_counts=np.bincount([r[1] for r in rows],minlength=ACTIONS).tolist())
            if not (out/'latest_colony.json').exists() or now-lastsave>=args.checkpoint_seconds:
                export(net,opt,cfg,out,steps,updates,torch); lastsave=now
    finally:
        pool.terminate(); pool.join()
        if updates: export(net,opt,cfg,out,steps,updates,torch)
    log(event='train_end',macro_steps=steps,updates=updates,completed_games=since,
        wall_minutes=round((time.monotonic()-start)/60,2))

def game_job(job):
    from survivor import fin4
    from survivor.colony_ppo import load_controller
    fin4.load_controller=load_controller
    return fin4.episode(job)
def evaluate(args):
    from survivor import fin5
    fin5.game_job=game_job
    fin5.evaluate(args)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('train'); p.add_argument('--base',default='configs/fin4_baseline.json')
    p.add_argument('--out',default='runs/fin6'); p.add_argument('--minutes',type=float,default=35)
    p.add_argument('--workers',type=int,default=12); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seed',type=int,default=6000000); p.add_argument('--period',type=float,default=20)
    p.add_argument('--rollout-steps',type=int,default=12); p.add_argument('--horizon',type=float,default=3000)
    p.add_argument('--checkpoint-seconds',type=float,default=300); p.add_argument('--resume')
    p=sub.add_parser('evaluate'); p.add_argument('--configs',nargs='+',required=True)
    p.add_argument('--out',default='runs/fin6_eval'); p.add_argument('--minutes',type=float,default=15)
    p.add_argument('--seeds',type=int,default=8); p.add_argument('--seed',type=int,default=2400000)
    p.add_argument('--horizon',type=float,default=3000); p.add_argument('--workers',type=int,default=12)
    args=ap.parse_args()
    if args.workers<1 or args.minutes<=0: ap.error('workers and minutes must be positive')
    if args.cmd=='train' and (args.period<=0 or args.rollout_steps<1): ap.error('period and rollout-steps must be positive')
    {'train':train,'evaluate':evaluate}[args.cmd](args)
if __name__=='__main__': main()
