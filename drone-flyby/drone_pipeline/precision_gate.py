"""Causal, strict YOLO+DINO+trajectory publication; delayed region revisit."""
import math,os
import numpy as np
from dtos import RequestedViewDto
from .geometry import iou,warp_box,nms


def env(name,default):return float(os.getenv(name,str(default)))

class PrecisionGate:
    def __init__(self,cfg):
        self.enabled=os.getenv('PRECISION_ENABLED','1')=='1';self.bank={};self.last_frame=None;self.cfg=cfg
        self.yolo=env('PRECISION_YOLO',.55);self.dino=env('PRECISION_DINO',.60)
        self.neg=env('PRECISION_NEG_MARGIN',.05);self.cls=env('PRECISION_CLASS_MARGIN',.03)
        self.hits=int(env('PRECISION_HITS',2));self.revisit=int(env('PRECISION_REVISIT_FRAMES',5))
        self.maxerror=env('PRECISION_MAX_TRAJECTORY_ERROR',.35);self.minscore=env('PRECISION_MIN_SCORE',.50)
        if not all(math.isfinite(x) for x in [self.yolo,self.dino,self.neg,self.cls,self.maxerror,self.minscore]):raise ValueError('Nonfinite precision thresholds')
        if self.hits<2 or self.revisit<1 or self.maxerror<=0:raise ValueError('Invalid precision gate settings')
        if not all(0<=x<=1 for x in [self.yolo,self.dino,self.minscore]) or min(self.neg,self.cls)<0:raise ValueError('Invalid precision thresholds')
    def process(self,r,detections,tracks,motion,now,dt,baseline,detector_info,**_):
        if not self.enabled:return baseline,{'enabled':False},None
        info={'enabled':True,'settings':{'yolo':self.yolo,'dino':self.dino,'negative_margin':self.neg,'class_margin':self.cls,'hits':self.hits,'min_score':self.minscore,'revisit_frames':self.revisit},'decisions':[]}
        verification=detector_info.get('verifier',{})
        candidates=verification.get('candidates',[])
        validids={t.id for t in tracks};self.bank={k:v for k,v in self.bank.items() if k in validids}
        previous={}
        # Advance stored full measurements with registered camera motion + residual velocity.
        for key,state in self.bank.items():
            previous[key]=(state['box'].copy(),state['velocity'].copy())
            if motion.trusted:
                shift=motion.H[:2,:2]@state['velocity']*dt
                state['box']=warp_box(state['box'],motion.H)+np.r_[shift,shift]
                state['velocity']=motion.H[:2,:2]@state['velocity']
            else:
                state['valid']=False;state['streak']=0
        approved=[]
        # One observation can support at most one track, regardless of matching order.
        used=set()
        for t in sorted(tracks,key=lambda t:(not t.updated,-t.score)):
            t.export_score=0.;t.export_reason='precision_rejected'
            reasons=[];row={'track_id':t.id,'class_id':t.cls,'published':False};info['decisions'].append(row)
            if not t.updated or t.partial:
                row['reasons']=['no_fresh_complete_observation'];continue
            options=[(iou(t.box,d.box),i,d) for i,d in enumerate(detections) if i not in used and d.cls==t.cls and not d.partial]
            if not options or max(options,key=lambda x:x[0])[0]<.5:
                row['reasons']=['no_matching_detection'];continue
            _,index,d=max(options,key=lambda x:x[0]);used.add(index)
            matches=[v for v in candidates if v['cls']==d.cls and iou(np.array(v['box']),d.view_box)>.99 and abs(v['score']-d.score)<.001]
            v=matches[0] if matches else None
            row.update(yolo=float(d.score),dino_status=v.get('status') if v else 'missing')
            if v:
                for key in ['target_similarity','background_similarity','best_other_similarity']:
                    value=v.get(key)
                    row[key]=float(value) if value is not None and math.isfinite(float(value)) else None
            if d.score<self.yolo:reasons.append('weak_yolo')
            if not v or v.get('status')!='verified' or verification.get('error'):
                reasons.append('no_valid_dino_evidence')
            elif not all(math.isfinite(float(v.get(k,float('nan')))) for k in ['target_similarity','background_similarity','best_other_similarity']):
                reasons.append('invalid_dino_scores')
            else:
                if v['target_similarity']<self.dino:reasons.append('weak_dino')
                if v['target_similarity']-v['background_similarity']<self.neg:reasons.append('background_ambiguity')
                if v['target_similarity']-v['best_other_similarity']<self.cls:reasons.append('class_ambiguity')
            support=float(t.votes.max()/max(t.votes.sum(),1e-9))
            rel=float(np.max(np.sqrt(np.maximum(np.diag(t.P)[:2],0))/np.maximum(t.box[2:]-t.box[:2],1)))
            if support<.8:reasons.append('unstable_track_class')
            if rel>.20:reasons.append('uncertain_localization')
            state=self.bank.get(t.id);error=None
            if state and state['class_id']!=t.cls:state=None;self.bank.pop(t.id,None)
            if state and state['valid'] and motion.trusted:
                predicted=state['box'];normal=max(8.,float(np.linalg.norm(d.box[2:]-d.box[:2])))
                fwd=np.linalg.norm((predicted[:2]+predicted[2:]-d.box[:2]-d.box[2:])/2)/normal
                old,velocity=previous[t.id];shift=motion.H[:2,:2]@velocity*dt
                try:
                    back=warp_box(d.box-np.r_[shift,shift],np.linalg.inv(motion.H))
                    backerr=np.linalg.norm((back[:2]+back[2:]-old[:2]-old[2:])/2)/max(8.,float(np.linalg.norm(old[2:]-old[:2])))
                    ratio=np.max(np.maximum((d.box[2:]-d.box[:2])/np.maximum(predicted[2:]-predicted[:2],1), (predicted[2:]-predicted[:2])/np.maximum(d.box[2:]-d.box[:2],1)))
                    error=float(max(fwd,backerr));row.update(forward_error=float(fwd),backprojection_error=float(backerr),size_ratio=float(ratio))
                    if error>self.maxerror or ratio>1.6:reasons.append('trajectory_mismatch')
                except np.linalg.LinAlgError:reasons.append('singular_motion')
            # Reacquisition can use a transported memory box, but only while registration stayed trusted.
            if not motion.trusted:reasons.append('untrusted_registration')
            if reasons:
                if state:state['streak']=0
                row['reasons']=reasons;continue
            streak=(state['streak']+1) if state and state['valid'] and error is not None else 1
            quality=float(math.exp(-2*(error or 0.)))
            score=float(min(d.score,max(0,v['target_similarity']),quality,support,t.existence))
            previous_due=state.get('due') if state else None
            state={'box':d.box.copy(),'velocity':t.x[2:4].copy(),'class_id':t.cls,'streak':streak,'valid':True,
                   'attempts':0,'next_focus':0,'last_good_frame':r.frame,'due':previous_due,'source_level':r.view.resolution_level,'score':score}
            self.bank[t.id]=state
            if streak<self.hits:reasons.append('awaiting_temporal_confirmation')
            if score<self.minscore:reasons.append('weak_combined_score')
            row.update(reasons=reasons,strong_hits=streak,yolo=float(d.score),dino=float(v['target_similarity']),trajectory_quality=quality,combined_score=score)
            if not reasons:
                row['published']=True;t.export_score=score;t.export_reason='precision_confirmed'
                approved.append({'track_id':t.id,'box':d.box.copy(),'cls':d.cls,'score':score,'fresh':True})
                if previous_due is None or r.frame>=previous_due:state['due']=r.frame+self.revisit
        # Seek a confirmed object's CURRENT predicted position after five source-frame steps.
        due=[(s['due'],t,s) for t in tracks if (s:=self.bank.get(t.id)) and s['valid'] and s['due'] is not None and r.frame>=s['due'] and r.frame>=s.get('next_focus',0) and t.existence>=.5]
        focus=None
        if due:
            _,t,state=min(due,key=lambda x:x[0]);level=state['source_level']
            if abs(level-r.view.resolution_level)>1:level=r.view.resolution_level+(1 if level>r.view.resolution_level else -1)
            focus=RequestedViewDto(resolution_level=int(level),center_x=int(round(t.x[0])),center_y=int(round(t.x[1])))
            # Keep focus due until observed strongly again, but expire old targets.
            if r.frame-state['last_good_frame']>max(15,3*self.revisit):state['due']=None;focus=None
            if focus is not None:
                state['attempts']=state.get('attempts',0)+1;state['next_focus']=r.frame+2
                if state['attempts']>=3:state['due']=None
                info['revisit_track_id']=t.id
        # Keep the current view for one more observation when a strong birth awaits confirmation.
        if focus is None and any(d.get('reasons')==['awaiting_temporal_confirmation'] for d in info['decisions']):
            focus=RequestedViewDto(resolution_level=r.view.resolution_level,center_x=r.view.center_x,center_y=r.view.center_y)
            info['hold_for_confirmation']=True
        self.last_frame=r.frame
        approved=nms(approved,.55,box=lambda x:x['box'],label=lambda x:x['cls'],score=lambda x:x['score'])
        kept={e['track_id'] for e in approved}
        for row in info['decisions']:
            if row['published'] and row['track_id'] not in kept:
                row['published']=False;row['reasons'].append('duplicate_suppression')
        for t in tracks:
            if t.id not in kept:t.export_score=0.;t.export_reason='precision_rejected'
        return approved,info,focus
