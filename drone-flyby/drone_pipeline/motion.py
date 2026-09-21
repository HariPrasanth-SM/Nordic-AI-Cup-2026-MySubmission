"""Robust similarity registration in source pixels, with bounded keyframe memory.

No-overlap registration failure is explicitly uncertain, never a confident identity.
Similarity is deliberately used instead of unconstrained homography extrapolation.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import view_transform, transform_points

@dataclass
class Motion:
    H: np.ndarray
    trusted: bool
    sigma: float
    details: dict

@dataclass
class Keyframe:
    frame: int
    points: np.ndarray
    descriptors: np.ndarray
    to_previous: np.ndarray
    chain_variance: float = 0.

class MotionEstimator:
    def __init__(self,cfg):
        self.cfg=cfg
        self.features=cv2.SIFT_create(nfeatures=cfg.features)
        self.matcher=cv2.BFMatcher(cv2.NORM_L2)
        self.keys=[]
        self.previous_frame=None

    def estimate(self,image,view,frame,detections):
        c=self.cfg
        mask=np.full(image.shape[:2],255,np.uint8)
        for d in detections:
            b=d.box.astype(int)
            cv2.rectangle(mask,(max(0,b[0]-5),max(0,b[1]-5)),(b[2]+5,b[3]+5),0,-1)
        kp,des=self.features.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
        local=np.array([p.pt for p in kp],float).reshape(-1,2)
        points=transform_points(local,view_transform(view))
        first=self.previous_frame is None
        gap=1 if first else max(1,frame-self.previous_frame)
        best=None
        if c.enabled and des is not None and len(des)>=c.min_inliers:
            for key in reversed(self.keys):
                if frame-key.frame>c.max_keyframe_age or key.descriptors is None or len(key.descriptors)<2:
                    continue
                pairs=self.matcher.knnMatch(key.descriptors,des,k=2)
                matches=[p[0] for p in pairs if len(p)==2 and p[0].distance<0.72*p[1].distance]
                # One current descriptor cannot support several independent inliers.
                unique={}
                for m in sorted(matches,key=lambda m:m.distance):
                    unique.setdefault(m.trainIdx,m)
                matches=list(unique.values())
                if len(matches)<c.min_inliers:
                    continue
                a=np.array([key.points[m.queryIdx] for m in matches])
                b=np.array([points[m.trainIdx] for m in matches])
                A,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,
                    ransacReprojThreshold=c.ransac_source_px,maxIters=1200,confidence=.995)
                if A is None or inliers is None:
                    continue
                keep=inliers.ravel().astype(bool)
                count=int(keep.sum()); ratio=count/len(matches)
                coverage=float(cv2.contourArea(cv2.convexHull(b[keep].astype(np.float32))))/max(1,
                    (view.source_region_xyxy[2]-view.source_region_xyxy[0])*(view.source_region_xyxy[3]-view.source_region_xyxy[1]))
                if count<c.min_inliers or ratio<c.min_ratio or coverage<c.min_coverage:
                    continue
                Hkt=np.vstack([A,[0,0,1]])
                H=Hkt@np.linalg.inv(key.to_previous)
                scale=float(np.linalg.norm(H[:2,0]))
                angle=abs(float(np.degrees(np.arctan2(H[1,0],H[0,0]))))
                if not 1/c.max_scale_change**gap<scale<c.max_scale_change**gap:
                    continue
                if angle>c.max_rotation_deg*gap or np.linalg.norm(H[:2,2])>c.max_translation_px_per_frame*gap:
                    continue
                residual=np.linalg.norm(transform_points(a[keep],Hkt)-b[keep],axis=1)
                sigma=max(1.,float(np.percentile(residual,80)))
                # A direct old-keyframe fit cannot erase uncertainty accumulated in
                # the keyframe->previous link used to derive this incremental motion.
                sigma=float(np.sqrt(sigma*sigma+key.chain_variance))
                details={'reason':'registered','reference_frame':key.frame,'inliers':count,
                    'inlier_ratio':ratio,'coverage':coverage,'residual_p80_px':float(np.percentile(residual,80)),
                    'support_center':b[keep].mean(axis=0).tolist(),'support_radius':float(np.linalg.norm(np.ptp(b[keep],axis=0))/2)}
                candidate=Motion(H,True,sigma,details)
                rank=count*ratio/(sigma*(1+0.05*(frame-key.frame)))
                if best is None or rank>best[0]:
                    best=(rank,candidate)
        if first:
            result=Motion(np.eye(3),True,0.,{'reason':'initial'})
        elif best is not None:
            result=best[1]
        else:
            result=Motion(np.eye(3),False,c.failure_sigma_px*np.sqrt(gap),
                {'reason':'disabled' if not c.enabled else 'registration_failed'})
        for key in self.keys:
            key.to_previous=result.H@key.to_previous
            key.chain_variance+=result.sigma**2
        self.keys=[k for k in self.keys if frame-k.frame<=c.max_keyframe_age]
        self.keys.append(Keyframe(frame,points,des,np.eye(3)))
        self.keys=self.keys[-c.keyframes:]
        self.previous_frame=frame
        result.details.update(trusted=result.trusted,sigma_px=result.sigma,H=result.H.tolist())
        return result
