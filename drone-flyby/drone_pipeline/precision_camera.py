"""One authoritative camera gate for pipeline, API and diagnostic middleware."""
import math
from dtos import MAXIMUM_CENTER_DELTA_PIXELS,ALLOWED_RESOLUTION_LEVELS,RequestedViewDto
REVISION='precision-navigation-v1'


def constrain(req,res):
    original=res.get('requested_view');audit={'revision':REVISION,'original':original,'action':'unchanged'}
    if res.get('request_id')!=req['request_id'] or res.get('frame')!=req['frame']:
        return {'request_id':req['request_id'],'frame':req['frame'],'annotations':[],'requested_view':None},dict(audit,action='identity_mismatch')
    if original is None:return res,audit
    v=req['view'];c=req['camera_constraints'];level=v['resolution_level'];center=[v['center_x'],v['center_y']]
    r=v['source_region_xyxy'];regioncenter=[(r[0]+r[2])/2,(r[1]+r[3])/2]
    audit.update(current_level=level,current_center=center,region_center=regioncenter)
    def hold(reason):res['requested_view']=None;return res,dict(audit,action=reason)
    if center!=regioncenter:return hold('inconsistent_request_center')
    desired=original['resolution_level'];target=desired
    # Adjacent-level transition: a requested L2->L0 becomes L2->L1.
    if abs(target-level)>1:target=level+(1 if target>level else -1)
    if target not in c['allowed_resolution_levels'] or target not in ALLOWED_RESOLUTION_LEVELS[level]:return hold('unavailable_level')
    b=next((b for b in c['center_bounds'] if b['resolution_level']==target),None)
    if b is None:return hold('missing_bounds')
    width,height=req['original_width'],req['original_height'];foot=[width/(2**target),height/(2**target)]
    low=[max(b['minimum_center_x'],math.ceil(foot[0]/2)),max(b['minimum_center_y'],math.ceil(foot[1]/2))]
    high=[min(b['maximum_center_x'],math.floor(width-foot[0]/2)),min(b['maximum_center_y'],math.floor(height-foot[1]/2))]
    if any(l>h for l,h in zip(low,high)):return hold('inconsistent_bounds')
    limit=min(float(c['maximum_center_delta']),MAXIMUM_CENTER_DELTA_PIXELS[level])
    if not math.isfinite(limit) or limit<0:return hold('invalid_delta')
    # A 2px margin can make L2-corner -> L1 impossible (distance ~550.73px).
    radius=max(0,limit-.01);reset=target==0 and c['full_view_reset_exempt_from_delta']
    audit.update(limit_px=limit,safe_limit_px=radius,target_level=target)
    x,y=original['center_x'],original['center_y']
    if desired!=target:x,y=center # zoom one step toward overview without adding a pan
    end=[max(low[i],min(high[i],p)) for i,p in enumerate([x,y])]
    near=[max(low[i],min(high[i],center[i])) for i in range(2)]
    if not reset and math.dist(near,center)>radius:return hold('no_reachable_integer_target')
    if not reset and math.dist(end,center)>radius:
        lo,hi=0.,1.
        for _ in range(45):
            mid=(lo+hi)/2;p=[near[i]+mid*(end[i]-near[i]) for i in range(2)]
            if math.dist(p,center)<=radius:lo=mid
            else:hi=mid
        end=[near[i]+lo*(end[i]-near[i]) for i in range(2)]
    def valid(x,y):return low[0]<=x<=high[0] and low[1]<=y<=high[1] and (reset or math.hypot(x-center[0],y-center[1])<=limit)
    points=[(x,y) for x in {math.floor(end[0]),math.ceil(end[0])} for y in {math.floor(end[1]),math.ceil(end[1])} if valid(x,y)]
    if not points:return hold('rounding_unreachable')
    x,y=min(points,key=lambda p:math.dist(p,end));cmd={'resolution_level':int(target),'center_x':int(x),'center_y':int(y)}
    res['requested_view']=cmd;audit.update(final=cmd,final_distance_px=math.dist([x,y],center),original_distance_px=math.dist([original['center_x'],original['center_y']],center),action='corrected' if cmd!=original else 'unchanged')
    return res,audit


def guard_command(request,command):
    response={'request_id':request.request_id,'frame':request.frame,'annotations':[],'requested_view':command.model_dump() if command else None}
    result,audit=constrain(request.model_dump(exclude={'view':{'image'}}),response)
    return RequestedViewDto.model_validate(result['requested_view']) if result['requested_view'] else None,audit


def enforce_response(request,response):
    from dtos import DroneFlybyPredictResponseDto
    result,audit=constrain(request.model_dump(exclude={'view':{'image'}}),response.model_dump())
    if audit['action']!='unchanged':
        import logging
        logging.warning('FINAL_CAMERA %s frame=%s %s',request.request_id,request.frame,audit)
    return DroneFlybyPredictResponseDto.model_validate(result)


def identity_only(req,res):
    """Egress check for the API layer. Camera legality is decided ONCE, statefully, by CameraShadow inside the
    pipeline; re-clamping here against the (possibly one-command-stale) received view would undo it."""
    audit={'revision':'realtime-fusion-v1','action':'pipeline_guarded','original':res.get('requested_view')}
    if res.get('request_id')!=req['request_id'] or res.get('frame')!=req['frame']:
        return {'request_id':req['request_id'],'frame':req['frame'],'annotations':[],'requested_view':None},dict(audit,action='identity_mismatch')
    return res,audit
