"""One serialized GPU/state owner. Every sequence has an independent state bank."""
from collections import OrderedDict
import hashlib
import logging
import threading
import time
import numpy as np
import cv2
from dtos import OBJECT_CLASSES, DroneFlybyPredictResponseDto, DroneFlybyPredictionDto
from utils import decode_view
from .geometry import view_transform, warp_box, clip_box, nms
from .detector import create_detector
from .motion import MotionEstimator, Motion
from .tracker import Tracker, Observation
from .camera import CameraPolicy, legal
from .camera_shadow import CameraShadow
from .precision_camera import guard_command
from .ego import EgoPrior
from .fusion import make_publisher
# realtime-fusion-v1
from .trace import TraceWriter

log=logging.getLogger(__name__)

class Session:
    def __init__(self,cfg,key):
        self.key=key; self.motion=MotionEstimator(cfg.motion); self.tracker=Tracker(cfg.tracker)
        self.camera=CameraPolicy(cfg.camera); self.cache=OrderedDict()
        self.publisher=make_publisher(cfg); self.shadow=CameraShadow(cfg.camera); self.ego=EgoPrior(cfg.ego)
        self.last_publish=None   # immutable snapshot of the last response, used by the watchdog fallback
        self.last_frame=None; self.last_index=None; self.now=0.

class Pipeline:
    def __init__(self,cfg,detector=None):
        self.cfg=cfg; cv2.setNumThreads(cfg.opencv_threads)
        self.lock=threading.RLock(); self.sessions=OrderedDict(); self.generation=0
        self.detector=detector if detector is not None else create_detector(cfg.detector)
        self.trace=TraceWriter(cfg.trace,cfg)

    def close(self): self.trace.close()

    def reset(self):
        """Explicit local/offline reset; never exposed as an unauthenticated HTTP route."""
        with self.lock: self.sessions.clear()

    def _session(self,r):
        if r.sequence_id not in self.sessions:
            self.generation+=1
            key=hashlib.sha256(r.sequence_id.encode()).hexdigest()[:10]+f'-{self.generation}'
            self.sessions[r.sequence_id]=Session(self.cfg,key)
            while len(self.sessions)>self.cfg.max_sessions: self.sessions.popitem(last=False)
        self.sessions.move_to_end(r.sequence_id)
        return self.sessions[r.sequence_id]

    def predict(self,r):
        arrival=time.perf_counter()
        with self.lock:
            start=time.perf_counter(); s=self._session(r)
            if r.request_id in s.cache:
                frame,index,response=s.cache[r.request_id]
                if frame!=r.frame or index!=r.frame_index: raise ValueError('request_id reused for a different frame')
                return response.model_copy(deep=True)
            if s.last_frame is not None and (r.frame<=s.last_frame or r.frame_index<=s.last_index):
                # Never rewind or advance on out-of-order input. Restart local
                # server between stock evaluator runs, which reuse sequence/request IDs.
                log.warning('Out-of-order request: %s frame %s',r.sequence_id,r.frame)
                return DroneFlybyPredictResponseDto(request_id=r.request_id,frame=r.frame,annotations=[])
            shadow_info=s.shadow.observe(r) if self.cfg.camera.shadow else None
            image=decode_view(r.view)
            if image.shape[:2]!=(r.view.height,r.view.width): raise ValueError('Decoded image dimensions disagree with view')
            region=r.view.source_region_xyxy
            if not (0<=region[0]<region[2]<=r.original_width and 0<=region[1]<region[3]<=r.original_height):
                raise ValueError('Invalid source region')
            decoded=time.perf_counter(); errors=[]; detector_ok=True
            try:
                raw=self.detector.detect(image,r.view.resolution_level) if getattr(self.detector,'supports_levels',False) else self.detector.detect(image)
                if getattr(self.detector,'last_info',{}).get('skipped'):
                    detector_ok=False  # skipped L0 is not evidence that the view is empty
                valid=[]
                for d in raw:
                    b=clip_box(d.box,r.view.width,r.view.height)
                    if b is not None and isinstance(d.cls,(int,np.integer)) and 0<=d.cls<len(OBJECT_CLASSES) and np.isfinite(d.score) and 0<=d.score<=1:
                        d.box=b; valid.append(d)
                raw=nms(valid,self.cfg.detector.iou)
            except Exception as exc:
                log.exception('Detector failure'); raw=[]; detector_ok=False; errors.append(f'detector: {exc}')
            detected=time.perf_counter()
            T=view_transform(r.view)
            detections=[Observation(warp_box(d.box,T),int(d.cls),float(d.score),bool(d.partial),
                np.array([T[0,0],T[1,1]]),d.box.copy(),d.tile,extra=getattr(d,'extra',None)) for d in raw]
            for d in detections:
                if not d.partial:
                    x,y,x2,y2=np.rint(d.view_box).astype(int)
                    patch=image[max(0,y):min(image.shape[0],y2),max(0,x):min(image.shape[1],x2)]
                    if patch.size:
                        hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
                        hist=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).ravel()
                        d.appearance=hist/max(float(hist.sum()),1.)
            try:
                motion=s.motion.estimate(image,r.view,r.frame,raw)
            except Exception as exc:
                log.exception('Registration failure')
                errors.append(f'motion: {exc}')
                gap=1 if s.last_frame is None else max(1,r.frame-s.last_frame)
                sigma=self.cfg.motion.failure_sigma_px*np.sqrt(gap)
                motion=Motion(np.eye(3),False,float(sigma),{'reason':'registration_exception',
                    'trusted':False,'sigma_px':float(sigma),'H':np.eye(3).tolist()})
                # Invalid keyframe links cannot be reused after an exceptional update.
                s.motion=MotionEstimator(self.cfg.motion)
            registered=time.perf_counter()
            gap=1 if s.last_frame is None else max(1,r.frame-s.last_frame)
            motion,ego_info=s.ego.refine(motion,gap)   # constant-velocity prior replaces failed/outlying registration
            # frame is source capture order; index independently diagnoses skipped delivery.
            dt=0. if s.last_frame is None else (r.frame-s.last_frame)*r.frame_interval_ms/1000
            now=s.now+dt
            if self.cfg.tracker.enabled:
                s.tracker.step(detections,motion,r.view,now,dt,detector_ok)
                exported=s.tracker.export(now,r.original_width,r.original_height)
            else:
                exported=[{'track_id':None,'box':d.box,'cls':d.cls,'score':d.score,'fresh':True} for d in detections]
            if self.cfg.ego.enabled:
                innovations=[e['innovation_xy'] for e in s.tracker.events if e['event']=='match' and not e['partial'] and not e['weak']
                             and np.linalg.norm(e['innovation_xy'])<self.cfg.ego.max_innovation_px] if self.cfg.tracker.enabled else []
                s.ego.update(motion,gap,innovations)
            if self.cfg.tracker.enabled and self.cfg.export_fresh and self.cfg.publish.mode!='fusion':
                # Keep current detector recall; tracking adds only supported memory.
                # Prefer full measured boxes over filtered boxes for the same identity.
                from .geometry import iou
                from .level_detector import containment
                matched_classes={e['track_id']:e['class'] for e in s.tracker.events if e['event']=='match'}
                exported=[e for e in exported if e['track_id'] not in matched_classes or matched_classes[e['track_id']]==OBJECT_CLASSES[e['cls']]]
                for d in sorted(detections,key=lambda d:d.score,reverse=True):
                    if d.score<self.cfg.fresh_conf: continue
                    duplicates=[e for e in exported if e['cls']==d.cls and (iou(e['box'],d.box)>.5 or (d.partial and containment(e['box'],d.box)>.8))]
                    if d.partial and duplicates: continue
                    exported=[e for e in exported if not any(e is old for old in duplicates)]
                    exported.append({'track_id':None,'box':d.box,'cls':d.cls,'score':d.score,'fresh':True})
                exported=nms(exported,self.cfg.detector.iou,box=lambda e:e['box'],label=lambda e:e['cls'],score=lambda e:e['score'])
            exported,precision_info,focus=s.publisher.process(r,detections,s.tracker.tracks,motion,now,dt,exported,getattr(self.detector,'last_info',{}),ego=s.ego,detector_ok=detector_ok)
            cams=s.shadow.candidates(r) if self.cfg.camera.shadow else None
            command,camera_info=s.camera.choose(r,s.tracker.tracks,motion,now,cams=cams)
            if focus is not None:
                command=focus
                camera_info['precision_revisit']=precision_info.get('revisit_track_id')
            command,audit=s.shadow.guard(r,command) if self.cfg.camera.shadow else guard_command(r,command)
            camera_info['audit']=audit; camera_info['shadow']=shadow_info
            # annotations=[]; clean=[]; sent=[]
            # for e in sorted(exported,key=lambda e:e['score'],reverse=True)[:500]:
            #     b=clip_box(e['box'],r.original_width,r.original_height)
            #     if b is None: continue
            #     normalized=b/np.array([r.original_width,r.original_height]*2)
            #     annotations.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[e['cls']],bbox=normalized.tolist(),confidence=float(e['score'])))
            #     clean.append({**e,'box':b.tolist(),'class':OBJECT_CLASSES[e['cls']]})
            #     sent.append({'box':b.tolist(),'bbox_norm':normalized.tolist(),'class':OBJECT_CLASSES[e['cls']],'score':float(e['score']),
            #                  'track_id':e.get('track_id'),'source':e.get('source','exported'),'fresh':bool(e.get('fresh'))})
            
            annotations=[]; clean=[]; sent=[]

            for e in sorted(exported,key=lambda e:e['score'],reverse=True)[:500]:
                b=clip_box(e['box'],r.original_width,r.original_height)
                if b is None:
                    continue

                # Final bbox pixel-area filter
                box_width = b[2] - b[0]
                box_height = b[3] - b[1]
                box_area = box_width * box_height

                if box_area < 2500 or box_area > 350000:
                    continue

                normalized=b/np.array([r.original_width,r.original_height]*2)

                annotations.append(
                    DroneFlybyPredictionDto(
                        object_id=OBJECT_CLASSES[e['cls']],
                        bbox=normalized.tolist(),
                        confidence=float(e['score'])
                    )
                )

                clean.append({
                    **e,
                    'box':b.tolist(),
                    'class':OBJECT_CLASSES[e['cls']]
                })

                sent.append({
                    'box':b.tolist(),
                    'bbox_norm':normalized.tolist(),
                    'class':OBJECT_CLASSES[e['cls']],
                    'score':float(e['score']),
                    'track_id':e.get('track_id'),
                    'source':e.get('source','exported'),
                    'fresh':bool(e.get('fresh'))
                })

                #### Added till here
            response=DroneFlybyPredictResponseDto(request_id=r.request_id,frame=r.frame,annotations=annotations,requested_view=command)
            ready=time.perf_counter()
            record={'session_key':s.key,'sequence_id':r.sequence_id,'request_id':r.request_id,
                'frame':r.frame,'frame_index':r.frame_index,'source_time':now,
                'frame_gap':0 if s.last_index is None else r.frame_index-s.last_index-1,
                'request':r.model_dump(exclude={'view':{'image'}}),'response':response.model_dump(),
                'detections':[{'box':d.box.tolist(),'view_box':d.view_box.tolist(),'class':OBJECT_CLASSES[d.cls],
                    'cls':d.cls,'score':d.score,'partial':d.partial,'tile':d.tile,'dino':d.extra} for d in detections],
                'tracks':s.tracker.snapshot(now),'events':s.tracker.events,'exports':clean,'sent':sent,'ego':ego_info,
                'motion':motion.details,'camera':camera_info,'errors':errors,'precision':precision_info,
                'detector_info':getattr(self.detector,'last_info',{}),
                'timing_ms':{'queue':(start-arrival)*1000,'decode':(decoded-start)*1000,
                    'detector':(detected-decoded)*1000,'motion':(registered-detected)*1000,
                    'tracking_camera':(ready-registered)*1000,'response_ready':(ready-arrival)*1000}}
            s.last_frame=r.frame; s.last_index=r.frame_index; s.now=now
            s.last_publish={'frame':r.frame,'items':[{'box':x['box'],'cls':x['cls'],'score':x['score']} for x in clean]}
            s.cache[r.request_id]=(r.frame,r.frame_index,response.model_copy(deep=True))
            while len(s.cache)>self.cfg.response_cache: s.cache.popitem(last=False)
            try: self.trace.submit(record,image)
            except Exception: log.exception('Trace snapshot failed; prediction remains valid')
            return response

    def abandon(self,r):
        """The response for `r` was not delivered in time; keep the camera shadow honest."""
        s=self.sessions.get(r.sequence_id)
        if s is not None: s.shadow.abandon(r.request_id)

    def fallback(self,r):
        """Valid, lock-free response from the last published snapshot moved by the ego prior. Never moves the camera."""
        annotations=[]
        try:
            s=self.sessions.get(r.sequence_id); snap=None if s is None else s.last_publish
            if snap is not None and r.frame>=snap['frame']:
                gap=r.frame-snap['frame']; v=s.ego.velocity if s.ego.locked else None
                shift=np.zeros(2) if v is None else v*gap; decay=self.cfg.publish.fallback_decay**gap
                # for it in snap['items']:
                #     b=clip_box(np.array(it['box'],float)+np.r_[shift,shift],r.original_width,r.original_height)
                #     if b is None: continue
                #     n=b/np.array([r.original_width,r.original_height]*2)
                for it in snap['items']:
                    b=clip_box(
                        np.array(it['box'],float)+np.r_[shift,shift],
                        r.original_width,
                        r.original_height
                    )
                    if b is None:
                        continue

                    # Final bbox pixel-area filter
                    box_width = b[2] - b[0]
                    box_height = b[3] - b[1]
                    box_area = box_width * box_height

                    if box_area < 2500 or box_area > 350000:
                        continue

                    n=b/np.array([r.original_width,r.original_height]*2)
                    annotations.append(DroneFlybyPredictionDto(object_id=OBJECT_CLASSES[it['cls']],bbox=n.tolist(),
                                                               confidence=float(min(1.,max(0.,it['score']*decay)))))
                    if len(annotations)>=500: break
        except Exception:
            log.exception('Fallback failed; answering with no detections'); annotations=[]
        return DroneFlybyPredictResponseDto(request_id=r.request_id,frame=r.frame,annotations=annotations,requested_view=None)
