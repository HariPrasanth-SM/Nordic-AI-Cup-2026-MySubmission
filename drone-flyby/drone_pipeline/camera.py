"""Reachable-view utility search; coverage memory follows registered source motion."""
import math
import numpy as np
from dtos import RequestedViewDto, MAXIMUM_CENTER_DELTA_PIXELS, ALLOWED_RESOLUTION_LEVELS
from .geometry import transform_points
from .camera_shadow import view_cam, is_legal, project


def movement_limit(request):
    # Defense in depth against stale/permissive constraint metadata. The current
    # level governs movement, including when the requested level is lower.
    return max(0., min(float(request.camera_constraints.maximum_center_delta),
                       MAXIMUM_CENTER_DELTA_PIXELS[request.view.resolution_level]) - .01)  # precision-navigation-v1


def legal(request,command,cams=None):
    """Legal from EVERY candidate camera (default: the received view only)."""
    if command is None: return True
    return is_legal(cams or [view_cam(request)],request.camera_constraints,command.resolution_level,command.center_x,command.center_y)


def reachable(request,level,target,cams=None):
    """Nearest integer command at `level` legal from every candidate camera, else None."""
    xy=project(cams or [view_cam(request)],request.camera_constraints,level,np.asarray(target,float))
    return None if xy is None else RequestedViewDto(resolution_level=int(level),center_x=xy[0],center_y=xy[1])

class CameraPolicy:
    def __init__(self,cfg):
        self.cfg=cfg; self.coverage=[]; self.last_overview=None; self.count=0; self.history=[]

    def choose(self,request,tracks,motion,now,cams=None):
        c=self.cfg; view=request.view; self.count+=1
        cams=cams or [view_cam(request)]; base=cams[0]; cur=base.level  # where the camera really is / will be
        width,height=request.original_width,request.original_height
        if motion.trusted:
            for mark in self.coverage:
                mark['center']=transform_points([mark['center']],motion.H)[0]
        else: self.coverage=[]
        r=np.array(view.source_region_xyxy); level=view.resolution_level
        self.coverage.append({'center':(r[:2]+r[2:])/2,'size':r[2:]-r[:2], 'time':now,'level':level})
        self.coverage=self.coverage[-40:]
        if level==0: self.last_overview=request.frame_index
        if c.mode=='hold': return None,{'reason':'hold'}
        if level==0 and c.mode!='full':
            cmd=reachable(request,1,[width/2,height/2],cams)
            return cmd,{'reason':'enter_trained_L1'}
        if c.mode=='sweep_l1':
            # Repeat a coarse L1 survey. reachable() clips every move legally.
            positions=[(width*.25,height*.25),(width*.75,height*.25),
                       (width*.75,height*.75),(width*.25,height*.75),(width*.5,height*.5)]
            target=positions[(self.count-1)%len(positions)]
            return reachable(request,1,target,cams),{'reason':'sweep_L1'}
        due=self.last_overview is None or request.frame_index-self.last_overview>=c.overview_every
        if c.mode=='full' or due or (not motion.trusted and cur==2):
            target=max(0,cur-1)
            cmd=reachable(request,target,[width/2,height/2],cams)
            return cmd,{'reason':'full_mode' if c.mode=='full' else 'registration_recovery' if not motion.trusted else 'overview_due'}
        # Explore periodically even when known-object clusters are attractive.
        exploration=self.count%c.explore_every==0 or not tracks or not motion.trusted
        candidates=[]
        targets=[np.array([(gx+.5)*width/c.grid_columns,(gy+.5)*height/c.grid_rows])
                 for gy in range(c.grid_rows) for gx in range(c.grid_columns)]
        targets += [t.x[:2]+t.x[2:4]*request.frame_interval_ms/1000 for t in tracks if t.existence>.3]
        for target in targets:
                for candidate_level in request.camera_constraints.allowed_resolution_levels:
                    if candidate_level==0 or (exploration and candidate_level!=1): continue
                    cmd=reachable(request,candidate_level,target,cams)
                    if cmd is None: continue
                    center=np.array([cmd.center_x,cmd.center_y],float)
                    footprint=np.array([width,height],float)/(2**candidate_level)
                    age=10.
                    for mark in self.coverage:
                        if mark['level']>=candidate_level and (np.abs(center-mark['center'])<mark['size']*.35).all():
                            age=min(age,now-mark['time'])
                    utility=min(age,10)/10*(2. if exploration else .5)
                    for t in tracks:
                        future=t.x[:2]+t.x[2:4]*request.frame_interval_ms/1000
                        if (np.abs(future-center)<footprint*.42).all():
                            stale=min(2.,max(0,now-t.last))
                            uncertainty=float(np.sqrt(max(t.P[0,0],t.P[1,1]))/max(1,min(t.box[2:]-t.box[:2])))
                            ambiguity=1-float(t.votes.max()/max(t.votes.sum(),1e-8))
                            # Prefer L2 where objects are small at L1; avoid spending
                            # every request revisiting a large, already-certain target.
                            view_size=np.min((t.box[2:]-t.box[:2])*np.array([960,540])/footprint)
                            detail_gain=.7 if candidate_level==2 and view_size<40 else 0.
                            utility+=(.1+stale+min(1,uncertainty)+ambiguity+detail_gain)*t.existence*(0.1 if exploration else 1.)
                    travel=np.linalg.norm(center-[base.x,base.y])/max(width,height)
                    utility-=.35*travel+.10*abs(candidate_level-cur)
                    # A one-step zoom-out from L2 reaches L0 on the following request.
                    utility+=.10*candidate_level
                    candidates.append((utility,cmd))
        cmd=max(candidates,key=lambda x:x[0])[1] if candidates else None
        self.history=(self.history+[[view.center_x,view.center_y]])[-30:]
        return cmd,{'reason':'explore' if exploration else 'refresh','candidates':len(candidates)}


def guard_command(request, command, cams=None):
    """Legacy stateless last gate; the pipeline uses CameraShadow.guard instead."""
    cams=cams or [view_cam(request)]
    original=None if command is None else command.model_dump()
    repaired=False
    if not legal(request,command,cams):
        repaired=True
        command=reachable(request,command.resolution_level,[command.center_x,command.center_y],cams)
    if not legal(request,command,cams): command=None
    audit={'current_level':request.view.resolution_level,
           'current_center':[request.view.center_x,request.view.center_y],
           'limit_px':movement_limit(request),'proposed':original,'repaired':repaired,
           'final':None if command is None else command.model_dump()}
    return command,audit
