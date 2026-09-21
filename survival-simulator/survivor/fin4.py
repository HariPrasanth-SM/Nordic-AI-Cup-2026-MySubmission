"""Deadline-bounded full-horizon policy search. CPU only, no additional ML dependencies.
python -m survivor.fin4 train --minutes 140 --workers 14 --out runs/fin4
python -m survivor.fin4 eval --config runs/fin4/final_config.json --seeds 91 92 93
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import random
import statistics
import time
from pathlib import Path
from dataclasses import asdict
from survivor.frontier import load_controller, FrontierParams
from survivor.policy import Params

VERSION='fin4.3'
ROOT=Path(__file__).resolve().parent.parent
_CODE_FILES=[ROOT/'survivor'/n for n in ('frontier.py','fin4.py','policy.py','colony.py','pursuit.py','fastsim.py')]
_CODE_FILES+=sorted((ROOT/'src').rglob('*.py'))
CODE_HASH=hashlib.sha256(b''.join(p.read_bytes() for p in _CODE_FILES if p.exists())).hexdigest()[:16]


def atomic(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)


def key(cfg):
    # Config-only cache plus code version. Seeds and full horizon also belong in keys.
    d={k:v for k,v in cfg.items() if k in ('params','frontier','controller')}
    return hashlib.sha256((VERSION+CODE_HASH+json.dumps(d,sort_keys=True)).encode()).hexdigest()[:16]


def episode(job):
    cfg,seed,horizon,deadline,fast=job
    import numpy as np
    from src.core import SimulationCore
    from survivor.runner import _Act
    if fast:
        from survivor import fastsim
        fastsim.apply()
    t0=time.monotonic(); sim=SimulationCore(seed=seed); ctrl=load_controller(cfg,seed=0)
    actions=[]; pol=0.; n_ticks=0; times=[]; checkpoints={}; maxpop=0
    parts={'fruit':0.,'predation':0.}; births=0
    remove=sim.env.remove_fruit; kill=sim.env.kill_agent; spawn=sim.env.spawn_agent
    def remove_fruit(fruit):
        if fruit.age<=100: parts['fruit']+=fruit.energy/1000
        remove(fruit)
    def kill_agent(agent):
        if agent.energy>0: parts['predation']-=agent.energy/100
        kill(agent)
    def spawn_agent(*args,**kwargs):
        nonlocal births
        if kwargs.get('parent') is not None: births+=1
        return spawn(*args,**kwargs)
    sim.env.remove_fruit=remove_fruit; sim.env.kill_agent=kill_agent; sim.env.spawn_agent=spawn_agent
    while True:
        st=sim.step(actions); n_ticks+=1
        t=st['sim_time']; n=st['num_agents']; maxpop=max(maxpop,n)
        if n==0 or t>=horizon-1e-7: break
        if n_ticks%100==0 and time.monotonic()>deadline:
            return {'seed':seed,'censored':True,'sim_time':t,'wall_s':time.monotonic()-t0}
        agents=[a for a in st['observations'] if a is not None]
        payload={'game_status':'ok','score':st['score'],'sim_time':t,'n_agents':n,'agent_status':agents}
        ts=time.perf_counter(); out=ctrl.act(payload); elapsed=time.perf_counter()-ts
        pol+=elapsed; times.append(elapsed)
        actions=[(a['agent_id'],_Act(a)) for a in out]
        for mark in (300,600,900,1200,1500,1800,2100,2400,3000):
            if t>=mark and str(mark) not in checkpoints:
                checkpoints[str(mark)]={'n':n,'energy':sum(a['energy'] for a in agents)/n}
    return {'seed':seed,'score':float(st['score']),'sim_time':float(t),'full':t>=horizon-1e-7,
            'censored':False,'births':births,'peak_population':maxpop,'population':checkpoints,
            'fruit_bonus':parts['fruit'],'predation_penalty':parts['predation'],
            'policy_ms_mean':1000*pol/max(1,len(times)),'policy_ms_p99':1000*float(np.quantile(times,.99)) if times else 0.,
            'policy_seconds':pol,'wall_s':time.monotonic()-t0,'diagnostics':dict(getattr(ctrl,'extra',{}))}


def paired(a,b):
    seeds=sorted(set(a)&set(b)); dif=[a[s]['score']-b[s]['score'] for s in seeds]
    if not dif: return {'n':0,'delta':0,'lo':-1e9,'hi':1e9}
    avg=statistics.mean(dif)
    # Student t interval, fixed final comparison. Screening intervals are descriptive only.
    from scipy.stats import t
    half=float(t.ppf(.975,len(dif)-1))*statistics.stdev(dif)/math.sqrt(len(dif)) if len(dif)>1 else float('inf')
    return {'n':len(dif),'delta':avg,'lo':avg-half,'hi':avg+half}


def summary(rows):
    scores=[r['score'] for r in rows.values()]
    if not scores: return {'n':0,'mean':0,'cvar20':0}
    tail=sorted(scores)[:max(1,math.ceil(.2*len(scores)))]
    return {'n':len(scores),'mean':statistics.mean(scores),'cvar20':statistics.mean(tail),
            'median':statistics.median(scores),'full':sum(r['full'] for r in rows.values()),
            'max':max(scores),'policy_ms':statistics.mean(r['policy_ms_mean'] for r in rows.values())}


def ranking(rows):
    q=summary(rows)
    # Small downside regularizer in search; final selection uses mean official score.
    return .9*q['mean']+.1*q['cvar20']


class Race:
    def __init__(self,out,workers,horizon=3000,fast=True):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.workers=workers; self.horizon=horizon; self.fast=fast; self.cache={}
        self.path=self.out/'episodes.jsonl'; self.pool=None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r=json.loads(line)
                    if r.get('version')==VERSION and r.get('code_hash')==CODE_HASH and r.get('horizon')==horizon and r.get('fast')==fast:
                        self.cache[(r['key'],r['result']['seed'])]=r['result']
                except (ValueError,KeyError): pass
    def run(self,configs,seeds,deadline):
        jobs=[]; rows={name:{} for name in configs}
        for seed in seeds:
            for name,cfg in configs.items():
                ck=(key(cfg),seed)
                if ck in self.cache: rows[name][seed]=self.cache[ck]
                else: jobs.append((name,cfg,seed))
        if not jobs or time.monotonic()>=deadline: return rows
        ctx=mp.get_context('spawn'); self.pool=ctx.Pool(self.workers)
        pending={}; queue=iter(jobs)
        try:
            # Bounded queue avoids waiting behind already expired jobs at a stage boundary.
            exhausted=False
            while time.monotonic()<deadline:
                while len(pending)<self.workers and not exhausted:
                    try: name,cfg,seed=next(queue)
                    except StopIteration: exhausted=True; break
                    result=self.pool.apply_async(episode,((cfg,seed,self.horizon,deadline,self.fast),))
                    pending[(name,seed)]=(result,cfg)
                done=[]
                for (name,seed),(future,cfg) in pending.items():
                    if not future.ready(): continue
                    r=future.get(); done.append((name,seed)) # raise worker exceptions, never silently select broken code
                    if not r.get('censored'):
                        rows[name][seed]=r; self.cache[(key(cfg),seed)]=r
                        with self.path.open('a') as f:
                            f.write(json.dumps({'version':VERSION,'code_hash':CODE_HASH,'key':key(cfg),'horizon':self.horizon,'fast':self.fast,'result':r})+'\n')
                        print(f"  {name:22s} seed={seed} score={r['score']:7.1f} T={r['sim_time']:6.1f} cpu={r['policy_ms_mean']:.2f}ms",flush=True)
                for k in done: del pending[k]
                if exhausted and not pending: break
                time.sleep(.1)
        finally:
            self.pool.terminate(); self.pool.join(); self.pool=None
        return rows


def balanced(rows):
    common=set.intersection(*(set(r) for r in rows.values())) if rows else set()
    return {k:{s:r[s] for s in sorted(common)} for k,r in rows.items()}


def arms(base):
    result={'fin3':copy.deepcopy(base)}
    def add(name,frontier,params=None):
        c=copy.deepcopy(base); c['controller']='frontier'; c['frontier']=asdict(FrontierParams())
        c['frontier'].update(frontier); c['params'].update(params or {}); result[name]=c
    add('renewal',{'economy':False,'landmarks':False})
    add('food',{'renewal':False,'landmarks':False})
    add('memory',{'economy':False,'renewal':False})
    add('frontier',{})
    add('distributed',{'local_capacity':1.,'local_gap':18.,'birth_energy':200.,'rest_energy':150.},
        {'cap_early':24.,'cap_late':4.,'cap_tau':650.})
    add('fast_generations',{'replacement_age':48.,'birth_energy':185.,'local_gap':8.,'food_wait':6.,'rest_energy':150.})
    add('patient_orchard',{'food_wait':18.,'urgent_energy':65.,'local_capacity':1.,'rest_energy':220.,'explore_fraction':1.})
    add('heading',{'economy':False,'renewal':False,'landmarks':True,'exact_heading':True})
    add('active_sensing',{'scan_period':.5}, {'shared_alarm':1.,'pred_react_dist':290.})
    return result


# Search only 8 high-impact parameters, conditional on the selected architecture.
SPACE={'food_wait':(2.,20.),'urgent_energy':(45.,110.),'rest_energy':(110.,270.),
       'replacement_age':(42.,85.),'birth_energy':(160.,310.),'local_gap':(3.,24.),
       'local_capacity':(1.,3.),'explore_fraction':(.45,1.)}


def train(a):
    start=time.monotonic(); deadline=start+a.minutes*60
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path(a.base).read_text()); base={'controller':'fin3','params':asdict(Params.load(base))}
    atomic(out/'baseline.json',base)
    # A usable fallback exists immediately, even if training is interrupted.
    atomic(out/'final_config.json',{**base,'selection':{'status':'baseline_until_holdout_passes'}})
    race=Race(out,a.workers)
    bank=arms(base); atomic(out/'candidate_configs.json',bank)
    print('Stage 1: structural ablations; all scores use complete 3000s horizons.',flush=True)
    d1=min(deadline,start+a.minutes*60*.30)
    rows=balanced(race.run(bank,range(a.seed_start,a.seed_start+a.screen_seeds),d1))
    screen={k:summary(v) for k,v in rows.items()}; atomic(out/'screen.json',screen)
    rank=sorted(bank,key=lambda k:ranking(rows[k]),reverse=True)
    # Keep fin3 and top two architectures; all comparisons use complete seed blocks.
    finalists={k:bank[k] for k in rank[:2]}; finalists['fin3']=base
    eligible=[k for k in rank if k!='fin3' and summary(rows[k])['n']>=4]
    if eligible:
        best=copy.deepcopy(bank[eligible[0]])
        space={k:v for k,v in SPACE.items() if (best['frontier']['economy'] and k in ('food_wait','urgent_energy','rest_energy','explore_fraction','local_capacity')) or (best['frontier']['renewal'] and k in ('replacement_age','birth_energy','local_gap','local_capacity'))}
        means=[best['frontier'][k] for k in space]; std=[(hi-lo)*.22 for lo,hi in space.values()]
        rng=random.Random(a.seed_start+77)
        for generation in range(a.generations):
            if not space or time.monotonic()>=start+a.minutes*60*.62: break
            candidates={'incumbent':best,'fin3':base}
            for j in range(8):
                c=copy.deepcopy(best)
                for i,(name,(lo,hi)) in enumerate(space.items()):
                    c['frontier'][name]=min(hi,max(lo,rng.gauss(means[i],std[i])))
                candidates[f'g{generation}_{j}']=c
            d2=min(deadline,start+a.minutes*60*(.30+.32*(generation+1)/a.generations))
            seeds=range(a.seed_start+1000+generation*100,a.seed_start+1000+generation*100+a.tune_seeds)
            rr=balanced(race.run(candidates,seeds,d2))
            if min((len(v) for v in rr.values()),default=0)<3: break
            names=sorted(candidates,key=lambda k:ranking(rr[k]),reverse=True)
            elite=[k for k in names if k!='fin3'][:3]
            if elite:
                best=copy.deepcopy(candidates[elite[0]])
                for i,(name,(lo,hi)) in enumerate(space.items()):
                    vals=[candidates[k]['frontier'][name] for k in elite]
                    means[i]=.3*means[i]+.7*statistics.mean(vals)
                    std[i]=max((hi-lo)*.07,.3*std[i]+.7*(statistics.pstdev(vals)))
                finalists[f'cem{generation}']=copy.deepcopy(best)
            atomic(out/f'generation_{generation}.json',{k:summary(v) for k,v in rr.items()})
            atomic(out/'candidate_configs.json',{**bank,**finalists})
    print('Stage 3: choose ONE challenger on new selection seeds.',flush=True)
    ds=min(deadline,start+a.minutes*60*.78)
    rr=balanced(race.run(finalists,range(a.seed_start+10000,a.seed_start+10000+a.select_seeds),ds))
    atomic(out/'selection.json',{k:summary(v) for k,v in rr.items()})
    eligible=[k for k in finalists if k!='fin3' and len(rr[k])>=4]
    if not eligible:
        decision={'winner':'fin3','reason':'Insufficient complete selection seed blocks; baseline retained.'}
        atomic(out/'decision.json',decision)
        (out/'REPORT.md').write_text(report(decision,screen))
        print('Baseline retained: insufficient complete selection blocks.',flush=True); return
    chosen=max(eligible,key=lambda k:ranking(rr[k]))
    challenger=finalists[chosen]; atomic(out/'challenger.json',challenger)
    print(f'Stage 4: LOCKED challenger {chosen} vs fin3 on untouched holdout seeds.',flush=True)
    rh=balanced(race.run({'fin3':base,'challenger':challenger},range(a.seed_start+20000,a.seed_start+20000+a.holdout_seeds),deadline))
    effect=paired(rh['challenger'],rh['fin3']); sm={k:summary(v) for k,v in rh.items()}
    # No optional stopping: one decision after the prespecified stage ends.
    accepted=effect['n']>=max(a.min_holdout,a.holdout_seeds) and effect['delta']>=a.min_gain and effect['lo']>0
    winner=challenger if accepted else base
    decision={'winner':chosen if accepted else 'fin3','challenger':chosen,'paired_effect':effect,'holdout':sm,
              'accepted':accepted,'elapsed_minutes':(time.monotonic()-start)/60,
              'reason':'positive paired CI and minimum gain' if accepted else 'holdout gate not passed; fin3 retained',
              'protocol':'controller seed 0 every episode; ALL requested holdout pairs must complete; no holdout data enters search'}
    atomic(out/'decision.json',decision)
    atomic(out/'final_config.json',{**winner,'selection':decision})
    (out/'REPORT.md').write_text(report(decision,screen))
    print(json.dumps(decision,indent=2),flush=True)


def report(decision,screen):
    lines=['# Fin4 search result','',f"Selected: **{decision['winner']}**. {decision['reason']}.",'',
           'All runs use the original simulator dynamics, 3000s horizon, and controller seed 0.',
           'Screening is exploratory. Only the locked final challenger gets a holdout significance claim.','',
           '| Screening arm | n | Mean score | Worst-20% mean |','|---|---:|---:|---:|']
    for k,v in screen.items(): lines.append(f"| {k} | {v['n']} | {v['mean']:.1f} | {v['cvar20']:.1f} |")
    lines+=['','## Final holdout','', '```json',json.dumps(decision,indent=2),'```', '',
            'Check remote validation and response waiting budget before the single official evaluation.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--base',default=str(ROOT/'configs/fin4_baseline.json'))
    tr.add_argument('--out',default='runs/fin4'); tr.add_argument('--minutes',type=float,default=140)
    tr.add_argument('--workers',type=int,default=14); tr.add_argument('--seed-start',type=int,default=910001)
    tr.add_argument('--screen-seeds',type=int,default=12); tr.add_argument('--tune-seeds',type=int,default=5)
    tr.add_argument('--select-seeds',type=int,default=12); tr.add_argument('--holdout-seeds',type=int,default=24)
    tr.add_argument('--min-holdout',type=int,default=12); tr.add_argument('--min-gain',type=float,default=30)
    tr.add_argument('--generations',type=int,default=3)
    ev=sub.add_parser('eval'); ev.add_argument('--config',required=True); ev.add_argument('--seeds',type=int,nargs='+',required=True)
    ev.add_argument('--workers',type=int,default=4); ev.add_argument('--out',default='runs/fin4_eval')
    ev.add_argument('--minutes',type=float,default=30); ev.add_argument('--horizon',type=float,default=3000)
    ev.add_argument('--official',action='store_true',help='disable existing fastsim patch')
    a=p.parse_args()
    if a.workers<1 or a.minutes<=0: p.error('workers and minutes must be positive')
    if a.cmd=='train' and (a.generations<1 or min(a.screen_seeds,a.tune_seeds,a.select_seeds,a.holdout_seeds)<1): p.error('generation and seed counts must be positive')
    if a.cmd=='train': train(a)
    else:
        c=json.loads(Path(a.config).read_text()); race=Race(a.out,a.workers,a.horizon,not a.official)
        rows=race.run({'config':c},a.seeds,time.monotonic()+a.minutes*60)['config']
        result={'summary':summary(rows),'episodes':rows,'requested':len(a.seeds),'completed':len(rows)}
        atomic(Path(a.out)/'eval.json',result); print(json.dumps(result['summary'],indent=2))

if __name__=='__main__': main()
