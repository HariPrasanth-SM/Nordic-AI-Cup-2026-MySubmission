"""Fin4: observable-state resource allocation and local lineage renewal.
No simulator state is read here. The original fin3 Controller stays intact.
"""
import copy
import json
import math
from collections import Counter
from dataclasses import dataclass, fields
from survivor.policy import Controller, Params, BIOME_PEN, wrap


@dataclass
class FrontierParams:
    economy: bool = True
    renewal: bool = True
    landmarks: bool = True
    allocation: bool = True
    scan_period: float = 1.2
    food_wait: float = 12.0
    unknown_wait: float = 1.0
    urgent_energy: float = 85.0
    rest_energy: float = 180.0
    replacement_age: float = 62.0
    birth_energy: float = 220.0
    terminal_energy: float = 112.0
    local_gap: float = 12.0
    nursery_radius: float = 100.0
    local_capacity: float = 2.0
    explore_fraction: float = 0.7
    terminal_horizon: float = 2950.0
    exact_heading: bool = False

    @classmethod
    def load(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown frontier parameters: {sorted(unknown)}")
        return cls(**d)


def load_controller(config, seed=0):
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)
    if config.get('controller', 'fin3') == 'fin3':
        return Controller(Params.load(config), seed=seed)
    if config.get('controller') != 'frontier':
        raise ValueError('controller must be fin3 or frontier')
    return FrontierController(config, seed=seed)


class FrontierController(Controller):
    def __init__(self, config, seed=0):
        self.f = FrontierParams.load(config.get('frontier', {}))
        p = Params.load(copy.deepcopy(config))
        # Our birth arbiter makes the decision after movement cost is known.
        if self.f.renewal:
            p.colony_mgr = 0
            p.senescence = 0
        super().__init__(p, seed=seed)

    def reset(self):
        super().reset()
        # Serving must not change the policy's random stream for the second/third run.
        self.episode = 0
        self.by_id = {}
        self.extra = Counter()

    def act(self, step):
        t = float(step.get('sim_time', 0))
        if t < self.t_last - 1e-6:
            self.reset()
        self.by_id = {a['agent_id']: a for a in step.get('agent_status', [])}
        out = super().act(step)
        if self.f.renewal and out:
            self._renew(step, out)
        # The old colony manager approves births AFTER _act_one records last_spawn.
        # Keep the real issued action in memory, including for renewal=False variants.
        for ac in out:
            m = self.mem.get(ac['agent_id'])
            if m:
                m['last_spawn'] = ac['spawn_agent']
        return out

    def _want_spawn(self, *args, **kwargs):
        return False if self.f.renewal else super()._want_spawn(*args, **kwargs)

    def _act_one(self, a, t, *args):
        m = self.mem.get(a['agent_id'])
        if m:
            if self.f.landmarks:
                self._correct_pose(a, m)
            prev = m.get('E_prev')
            if prev is not None:
                # Compare with the action ACTUALLY approved last tick. Clipped food gains
                # can mask aging, so detection uses repeated positive residuals.
                loss = prev - a['energy'] - m.get('last_cost', .1) - 100 * m.get('last_spawn', False)
                old = a['age'] >= 60 and loss > max(.35, .006 * a['age'])
                m['age_hits'] = m.get('age_hits', 0) + 1 if old else 0
                if m['age_hits'] >= 2:
                    m['aging'] = True
        return super()._act_one(a, t, *args)

    def _correct_pose(self, a, m):
        # Static trees/edge endpoints correct translation after collision deflection.
        # Heading is exactly known from commanded turns. Require two agreeing landmarks.
        c, s = math.cos(m['h']), math.sin(m['h'])
        points = []
        for o in a['observations']:
            if o['type'] == 'Tree':
                x, y = o['distance'] * math.cos(o['angle']), o['distance'] * math.sin(o['angle'])
                points.append(('T', x * c - y * s + m['x'], x * s + y * c + m['y']))
            elif o['type'] == 'Edge':
                for x, y in o['coords']:
                    points.append(('E', x * c - y * s + m['x'], x * s + y * c + m['y']))
        points = list(dict.fromkeys(points))[:48]
        old = m.get('landmarks', [])
        offsets = []
        for tag, x, y in points:
            same = [(ox-x, oy-y) for ot, ox, oy in old if ot == tag and (ox-x)**2+(oy-y)**2 < 45**2]
            if same:
                offsets.append(min(same, key=lambda p: p[0]**2+p[1]**2))
        if len(offsets) >= 2:
            best = max(offsets, key=lambda p: sum((p[0]-q[0])**2+(p[1]-q[1])**2 < 4 for q in offsets))
            support = [q for q in offsets if (best[0]-q[0])**2+(best[1]-q[1])**2 < 4]
            if len(support) >= 2 and len(support) >= .6 * len(offsets):
                dx = sum(p[0] for p in support)/len(support)
                dy = sum(p[1] for p in support)/len(support)
                m['x'] += dx; m['y'] += dy
                points = [(tag, x+dx, y+dy) for tag, x, y in points]
                if math.hypot(dx, dy) > 2:
                    self.extra['pose_corrections'] += 1
        m['landmarks'] = points

    def _ledger(self, a, m, t, fruits):
        if not self.f.economy:
            return super()._ledger(a, m, t, fruits)
        # Nearest unique association with a tighter gate than fin3's first hit in 14px.
        # Newly heard does not mean newborn unless the previous hearing disc covered it.
        out, used = [], set()
        old = m.get('food_tracks', [])
        prev_pose = m.get('food_pose')
        current = []
        for d, ang in fruits:
            x, y = m['x']+d*math.cos(m['h']+ang), m['y']+d*math.sin(m['h']+ang)
            candidates = [(math.hypot(x-v[0], y-v[1]), i) for i, v in enumerate(old) if i not in used]
            hit = min(candidates, default=(1e9, -1))
            if hit[0] < 5:
                used.add(hit[1]); v = old[hit[1]]
                entry = [x, y, v[2], t, v[4]]
            else:
                certain = prev_pose is not None and math.hypot(x-prev_pose[0], y-prev_pose[1]) < a['hearing_radius']-3
                entry = [x, y, t, t, certain]
            current.append(entry)
            out.append((d, ang, t-entry[2], entry[4]))
        current.extend(v for i,v in enumerate(old) if i not in used and t-v[3] < 12 and math.hypot(v[0]-m['x'],v[1]-m['y']) > a['hearing_radius']+3)
        m['food_tracks'] = current[:120]
        m['food_pose'] = (m['x'], m['y'])
        return out

    def _priority(self, a, distance):
        sp = max(1., min(a['speed'],a['sprint_speed']) * BIOME_PEN.get(a['biome'],1.))
        E = a['energy']
        # Lower bid wins; travel cost and energy need, not agent ID, govern access.
        return distance/sp + .055*E + (4 if a['age'] > 100 else 0)

    def _owned(self, a, d, ang, mates):
        if not self.f.allocation or a['energy'] < 30:
            return True
        x,y = d*math.cos(ang), d*math.sin(ang)
        own = (self._priority(a,d), a['agent_id'])
        for md,ma,mid in mates:
            other = self.by_id.get(mid)
            if other is None or other['energy'] >= other['max_energy']-30:
                continue
            # Do not allocate food to an agent that is currently occupied evading.
            if any(o['type']=='Predator' and o['distance']<160 for o in other['observations']):
                continue
            dd = math.hypot(x-md*math.cos(ma), y-md*math.sin(ma))
            nb=next((o for o in a['observations'] if o['type']=='Agent' and o.get('id')==mid),None)
            if nb is None: continue
            heading=ma+math.pi-nb['rel_dir']
            bx,by=x-md*math.cos(ma),y-md*math.sin(ma)
            # Only yield to a neighbour which actually observes this same fruit.
            if not any(o['type']=='Fruit' and math.hypot(bx-o['distance']*math.cos(heading+o['angle']),by-o['distance']*math.sin(heading+o['angle']))<8 for o in other['observations']):
                continue
            if (self._priority(other,dd)+2,mid) < own:
                return False
        return True

    def _scan(self, a, m, t):
        if t < m.get('scan_at', -1):
            return 0.
        m['scan_at'] = t + self.f.scan_period
        return min(math.pi/2, max(.3, .9*a['vision_angle']))

    def _forage(self,a,m,t,fruits,trees,mates,edges,rng):
        if not self.f.economy:
            return super()._forage(a,m,t,fruits,trees,mates,edges,rng)
        f=self.f; E=a['energy']; sp=min(a['speed'],a['sprint_speed'])
        pen=BIOME_PEN.get(a['biome'],1.)
        urgent=E<f.urgent_energy or a['age']<12
        # An aging agent below birth cost should yield to nearby younger viable agents.
        retire=m.get('aging') and E<90 and any(self.by_id.get(mid,{}).get('age',999)<55 for md,ma,mid in mates if md<100)
        cand=[]; waiting=False
        for d,ang,age,known in fruits:
            if d>self.p.fruit_max_dist or E>a['max_energy']-25 or retire:
                continue
            if not self._owned(a,d,ang,mates):
                self.extra['food_yields']+=1; continue
            wait=f.food_wait if known else f.unknown_wait
            if not urgent and age<wait:
                waiting=True; continue
            # Direct collision radius is >=6; avoid paying for motion to fruit centre.
            travel=max(0,d-5)/max(pen,.2)
            if travel*.05>E-3 and not urgent:
                continue
            cand.append((travel,ang,d))
        if cand:
            _,ang,d=min(cand)
            self._bump('forage'); self._progress(m,t,d)
            phi=self._steer(ang,edges,None,m,18.) if d>25 else ang
            return min(sp,max(0.,(d-5)/pen)),phi,wrap(ang)
        if retire:
            self._bump('camp'); return 0.,0.,self._scan(a,m,t)
        # A tree's service rate is only ~0.1 fruit/s: share among local hungry agents.
        tree=None
        for d,ang in sorted(trees):
            tx,ty=d*math.cos(ang),d*math.sin(ang)
            occupied=0
            for md,ma,mid in mates:
                other=self.by_id.get(mid)
                if other and math.hypot(tx-md*math.cos(ma),ty-md*math.sin(ma))<65:
                    if (self._priority(other,0),mid)<(self._priority(a,0),a['agent_id']):
                        occupied+=1
            if occupied < f.local_capacity:
                tree=(d,ang); break
        if tree and a['biome']!='river':
            d,ang=tree
            if d>14:
                self._bump('seek_tree'); self._progress(m,t,d)
                phi=self._steer(ang,edges,None,m,20.)
                return min(sp,max(0,(d-10)/pen)),phi,phi
            self._bump('camp')
            return 0.,0.,self._scan(a,m,t)
        if waiting or E>=f.rest_energy:
            self._bump('camp'); return 0.,0.,self._scan(a,m,t)
        # No viable food anchor: explore persistently, but lower the metabolic travel bill.
        self._bump('explore')
        m['trv']=wrap(m['trv']+rng.gauss(0.,self.p.explore_wander))
        phi=self._steer(m['trv'],edges,mates,m,self.p.wall_margin)
        turn=self._scan(a,m,t)
        m['trv']=wrap(phi-turn)
        speed=sp*(1. if a['biome'] in ('river','desert') else f.explore_fraction)
        return speed,phi,turn

    def trait_score(self,a):
        value=super().trait_score(a)
        if self.f.renewal:
            # Low max_energy lowers the sprint floor, but <=100 can never reproduce.
            value-=2.*max(0.,(180.-a['max_energy'])/180.)
        return value

    def _renew(self,step,out):
        t=step.get('sim_time',0.); f=self.f
        remaining = 3000. - t
        if t>f.terminal_horizon and any(a['age']+remaining<60 and a['energy']>remaining+60 for a in self.by_id.values()):
            return  # only skip births when an existing young cohort can cover the remaining time
        n=len(out); cap=self.pop_cap(t); amap={x['agent_id']:x for x in out}
        candidates=[]
        for aid,a in self.by_id.items():
            m=self.mem[aid]; ac=amap[aid]
            E=a['energy']-(m['last_cost']-.1) # movement/turn occur before the spawn check
            if E<=101 or m.get('threat_now'):
                continue
            nearby=[]; food=0; trees=0
            for o in a['observations']:
                if o['type']=='Agent' and o['distance']<f.nursery_radius and o.get('id') in self.by_id:
                    nearby.append(self.by_id[o['id']])
                elif o['type']=='Fruit' and o['distance']<f.nursery_radius: food+=1
                elif o['type']=='Tree' and o['distance']<f.nursery_radius: trees+=1
            young=sum(b['age']<35 for b in nearby)
            aging=m.get('aging',False)
            replacement=a['age']>=f.replacement_age and young<1
            emergency=n<=2
            gap=t-m.get('birth_at',-1e9)
            local_recent=any(t-self.mem.get(b['agent_id'],{}).get('birth_at',-1e9)<f.local_gap for b in nearby)
            if gap<f.local_gap or (local_recent and not aging and not emergency): continue
            reserve=max(35.,.2*a['max_energy']+20)
            terminal=aging and E>=min(f.terminal_energy,max(103.,a['max_energy']-8.)) and young<2
            normal=E>=max(f.birth_energy,100+reserve) and food>=1 and len(nearby)<f.local_capacity+1
            renew=replacement and E>=min(max(145.,f.terminal_energy),max(105.,a['max_energy']-8.)) and (food>=1 or trees>=1)
            if not (terminal or normal or renew or (emergency and E>135)): continue
            if n>=cap+(3 if terminal or renew else 0) and not emergency: continue
            # Trait selection is a soft ranking, never a veto on a dying isolated lineage.
            rank=150*terminal+80*renew+30*emergency+E*.1+40*self.trait_score(a)-15*len(nearby)
            candidates.append((rank,aid, 'terminal' if terminal else 'replacement' if renew else 'growth'))
        granted=[]
        for _,aid,why in sorted(candidates,reverse=True):
            a=self.by_id[aid]
            if any(o['type']=='Agent' and o.get('id') in granted and o['distance']<f.nursery_radius for o in a['observations']): continue
            if len(granted)>=max(1,min(3,int(cap-n+1))): break
            amap[aid]['spawn_agent']=True
            self.mem[aid]['birth_at']=t
            granted.append(aid)
            self.extra['birth_'+why]+=1

    def _mpc_plan(self,m,preds,edges,sp,spr,E,ME):
        if not self.f.exact_heading:
            return super()._mpc_plan(m,preds,edges,sp,spr,E,ME)
        # rel_dir supplies predator heading directly; velocity-based heading is unreliable
        # during pivoting and while colliding. Do not identify low-biome velocity as sleep.
        nearest=min(preds,key=lambda v:v[0])
        olddir,oldv=m.get('pdir'),m.get('pv')
        m['pdir']=wrap(nearest[1]+math.pi-nearest[2]); m['pv']=None
        try:
            return super()._mpc_plan(m,preds,edges,sp,spr,E,ME)
        finally:
            m['pdir'],m['pv']=olddir,oldv

    def summary(self):
        out=super().summary(); out['frontier']=dict(self.extra); return out
