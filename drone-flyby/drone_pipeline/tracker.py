"""Source-pixel EKF-like box filter, two-stage assignment and visibility lifecycle.

State = [cx,cy,vx,vy,log(w),log(h)]. Residual velocity excludes camera motion.
Existence/class/localization scores are ranking heuristics, not calibrated probabilities.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from dtos import OBJECT_CLASSES
from .geometry import box_measurement, state_box, iou, corners, transform_points, envelope, clip_box, nms

OBS=np.eye(6)[[0,1,4,5]]

@dataclass
class Observation:
    box: np.ndarray
    cls: int
    score: float
    partial: bool
    scale: np.ndarray
    view_box: np.ndarray
    tile: int = 0
    appearance: object = None
    extra: object = None   # detector-attached evidence (DINO similarities)

@dataclass
class Track:
    id: int
    x: np.ndarray
    P: np.ndarray
    votes: np.ndarray
    existence: float
    score: float
    first: float
    last: float
    hits: int = 1
    misses: int = 0
    status: str = 'tentative'
    updated: bool = True
    partial: bool = False
    export_reason: str = ''
    export_score: float = 0.
    last_full: float | None = None
    appearance: object = None
    trail: list = field(default_factory=list)
    last_obs: object = None   # the detection that updated this track in the current step
    # Retain oriented corners between measurements to prevent AABB inflation.
    polygon: np.ndarray = field(default_factory=lambda:np.empty((0,2)))
    @property
    def box(self): return state_box(self.x)
    @property
    def cls(self): return int(self.votes.argmax())

class Tracker:
    def __init__(self,cfg):
        self.cfg=cfg
        self.tracks=[]
        self.next_id=1
        self.events=[]

    def measurement_noise(self,d):
        sigma=self.cfg.measurement_view_px*d.scale*(1+2*(1-d.score))
        return np.diag(np.r_[sigma*sigma,[.10**2,.10**2]])

    def predict(self,t,motion,dt):
        H=motion.H; A=H[:2,:2]
        old=t.x.copy()
        t.polygon=transform_points(t.polygon,H) + (A@old[2:4]*dt)[None,:]
        b=envelope(t.polygon)
        z=box_measurement(b)
        t.x=np.r_[z[:2],A@old[2:4],z[2:]]
        F=np.eye(6); F[:2,:2]=A; F[:2,2:4]=A*dt; F[2:4,2:4]=A
        sigma=motion.sigma
        if 'support_center' in motion.details:
            distance=np.linalg.norm(t.x[:2]-motion.details['support_center'])
            radius=max(100,motion.details['support_radius'])
            sigma*=1+max(0,distance/radius-1)
        q=self.cfg.process_px_per_second
        Q=np.diag([q*q*dt*dt+sigma*sigma]*2+[q*q*dt]*2+[.025**2*dt]*2)
        t.P=F@t.P@F.T+Q
        t.updated=False; t.last_obs=None
        t.existence*=np.exp(-.012*dt)

    def associate(self,tracks,dets,weak=False):
        if not tracks or not dets: return []
        costs=np.full((len(tracks),len(dets)+len(tracks)),1.05)
        costs[:,:len(dets)]=1e6
        for i,t in enumerate(tracks):
            for j,d in enumerate(dets):
                residual=box_measurement(d.box)-OBS@t.x
                S=OBS@t.P@OBS.T+self.measurement_noise(d)
                maha=float(residual[:2]@np.linalg.solve(S[:2,:2],residual[:2]))
                overlap=iou(t.box,d.box)
                size=max(8.,float(np.linalg.norm(t.box[2:]-t.box[:2])))
                distance=np.linalg.norm(residual[:2])/size
                size_ratio=np.exp(np.abs(residual[2:])).max()
                if maha>self.cfg.gate_mahalanobis or distance>self.cfg.max_center_distance_sizes or (size_ratio>3 and not d.partial):
                    continue
                if weak and (overlap<.15 or t.cls!=d.cls or t.hits<self.cfg.confirm_hits):
                    continue
                appearance_cost=0.
                if t.appearance is not None and d.appearance is not None:
                    similarity=float(np.minimum(t.appearance,d.appearance).sum())
                    # Dormant re-identification needs class + plausible geometry
                    # + appearance; histogram alone never creates a match.
                    if t.status=='dormant' and (t.cls!=d.cls or similarity<.35): continue
                    appearance_cost=.15*(1-similarity)
                certainty=float(t.votes.max()/max(t.votes.sum(),1e-8))
                class_cost=0 if t.cls==d.cls else .25*certainty
                # Soft class cost lets good geometry correct a previous class error.
                costs[i,j]=.40*min(1,maha/self.cfg.gate_mahalanobis)+.45*(1-overlap)+class_cost+appearance_cost
        rows,cols=linear_sum_assignment(costs)
        return [(tracks[i],dets[j]) for i,j in zip(rows,cols) if j<len(dets) and costs[i,j]<1.05]

    def update_track(self,t,d,now):
        before=t.x.copy()
        z=box_measurement(d.box); R=self.measurement_noise(d)
        H=OBS
        if d.partial:
            # A visible fragment has neither a reliable full-box centre nor size.
            # Preserve both; only class/existence evidence is updated.
            pass
        else:
            S=H@t.P@H.T+R
            K=np.linalg.solve(S,H@t.P).T
            t.x=t.x+K@(z-H@t.x)
            I=np.eye(6)-K@H
            t.P=I@t.P@I.T+K@R@K.T
            t.polygon=corners(t.box)
            t.last_full=now
            if d.appearance is not None:
                t.appearance=d.appearance.copy() if t.appearance is None else .8*t.appearance+.2*d.appearance
            # A fresh measurement may establish residual motion through cross covariance.
            t.x[2:4]=np.clip(t.x[2:4],-500,500)
        t.votes*=.82
        t.votes[d.cls]+=d.score*(.5 if d.partial else 1.)
        t.existence=min(.995,t.existence+.45*(1-t.existence))
        t.score=d.score; t.last=now; t.hits+=1; t.misses=0
        t.updated=True; t.partial=d.partial; t.last_obs=d
        t.status='confirmed' if t.hits>=self.cfg.confirm_hits else 'tentative'
        self.events.append({'event':'match','track_id':t.id,'class':OBJECT_CLASSES[d.cls],
            'weak':d.score<self.cfg.high_conf,'partial':d.partial,
            'innovation_xy':(z[:2]-before[:2]).tolist()})

    def observable(self,t,view,motion,detector_ok):
        if not detector_ok or not motion.trusted: return False
        b=t.box; r=np.array(view.source_region_xyxy)
        sigma=2*np.sqrt(np.maximum(np.diag(t.P)[:2],0))
        inside=(b[:2]-sigma>=r[:2]+2).all() and (b[2:]+sigma<=r[2:]-2).all()
        scale=(r[2:]-r[:2])/np.array([view.width,view.height])
        size=(b[2:]-b[:2])/scale
        return bool(inside and size.min()>=self.cfg.min_observable_view_px)

    def step(self,detections,motion,view,now,dt,detector_ok=True):
        self.events=[]
        for t in self.tracks: self.predict(t,motion,dt)
        high=[d for d in detections if d.score>=self.cfg.high_conf]
        low=[d for d in detections if d.score<self.cfg.high_conf]
        pairs=self.associate(self.tracks,high)
        matched={t.id for t,d in pairs}; used={id(d) for t,d in pairs}
        pairs+=self.associate([t for t in self.tracks if t.id not in matched],low,weak=True)
        matched={t.id for t,d in pairs}; used={id(d) for t,d in pairs}
        for t,d in pairs: self.update_track(t,d,now)
        for t in self.tracks:
            if t.id not in matched:
                informative=self.observable(t,view,motion,detector_ok)
                if informative:
                    p=self.cfg.miss_detection_probability
                    t.existence=t.existence*(1-p)/max(1e-8,1-t.existence*p)
                    t.misses+=1
                t.status='coasting' if t.hits>=self.cfg.confirm_hits else 'tentative'
                if now-t.last>self.cfg.max_coast_seconds: t.status='dormant'
                self.events.append({'event':'unmatched','track_id':t.id,'informative_miss':informative})
        for d in high:
            if id(d) in used or d.score<self.cfg.birth_conf: continue
            # Avoid a new competing class identity at essentially the same box.
            if any(iou(t.box,d.box)>.85 for t in self.tracks): continue
            if len(self.tracks)>=self.cfg.max_tracks: break
            z=box_measurement(d.box); x=np.r_[z[:2],[0.,0.],z[2:]]
            P=np.diag([*np.diag(self.measurement_noise(d))[:2],100.,100.,.04,.04])
            votes=np.zeros(len(OBJECT_CLASSES)); votes[d.cls]=d.score
            t=Track(self.next_id,x,P,votes,.75,d.score,now,now,partial=d.partial,polygon=corners(d.box))
            t.last_full=None if d.partial else now
            t.appearance=d.appearance; t.last_obs=d
            self.next_id+=1; self.tracks.append(t)
            self.events.append({'event':'birth','track_id':t.id})
        live=[]
        for t in self.tracks:
            age=now-t.last
            if age>self.cfg.memory_seconds or t.existence<.12 or (t.hits<self.cfg.confirm_hits and age>2):
                self.events.append({'event':'delete','track_id':t.id}); continue
            t.trail=(t.trail+[t.x[:2].tolist()])[-20:]
            live.append(t)
        self.tracks=live

    def export(self,now,width,height):
        candidates=[]
        for t in self.tracks:
            t.export_score=0.
            t.export_reason='unconfirmed'
            fresh=t.updated
            if t.hits<self.cfg.confirm_hits and not (fresh and t.score>=self.cfg.immediate_conf): continue
            age=now-(t.last_full if t.last_full is not None else t.first)
            size=t.box[2:]-t.box[:2]
            rel=float(np.max(np.sqrt(np.maximum(np.diag(t.P)[:2],0))/np.maximum(size,1)))
            if (not fresh or t.partial) and (age>self.cfg.max_coast_seconds or rel>self.cfg.max_relative_sigma):
                t.export_reason='localization_uncertain'; t.status='dormant'; continue
            b=clip_box(t.box,width,height)
            if b is None: t.export_reason='outside_frame'; continue
            class_support=float(t.votes.max()/max(t.votes.sum(),1e-8))
            localization=1. if fresh and not t.partial else np.exp(-2*rel)*np.exp(-age/3)
            score=float(np.clip(t.score*t.existence*(.5+.5*class_support)*localization,0,1))
            if score<self.cfg.min_export_score: t.export_reason='low_score'; continue
            t.export_score=score; t.export_reason='fresh' if fresh else 'propagated'
            candidates.append({'track_id':t.id,'box':b,'cls':t.cls,'score':score,'fresh':fresh})
        result=nms(candidates,self.cfg.duplicate_iou,box=lambda d:d['box'],label=lambda d:d['cls'],score=lambda d:d['score'])[:500]
        keep={d['track_id'] for d in result}
        for t in self.tracks:
            if t.export_score and t.id not in keep: t.export_reason='duplicate'; t.export_score=0.
        return result

    def snapshot(self,now):
        return [{'id':t.id,'box':t.box.tolist(),'class':OBJECT_CLASSES[t.cls],
            'status':t.status,'updated':t.updated,'hits':t.hits,'misses':t.misses,
            'existence':t.existence,'class_votes':t.votes.tolist(),'state':t.x.tolist(),
            'covariance':t.P.tolist(),'age_since_seen':now-t.last,'age_since_full_box':None if t.last_full is None else now-t.last_full,'partial':t.partial,
            'export_reason':t.export_reason,'export_score':t.export_score,'trail':t.trail}
            for t in self.tracks]
