import numpy as np


def view_transform(view):
    a,b,c,d = view.source_region_xyxy
    return np.array([[(c-a)/view.width,0,a],[0,(d-b)/view.height,b],[0,0,1.]], float)


def corners(box):
    a,b,c,d = box
    return np.array([[a,b],[c,b],[c,d],[a,d]], float)


def transform_points(points, H):
    p = np.c_[np.asarray(points), np.ones(len(points))] @ H.T
    return p[:,:2] / p[:,2:3]


def envelope(points):
    return np.r_[points.min(axis=0), points.max(axis=0)]


def warp_box(box, H):
    return envelope(transform_points(corners(box), H))


def clip_box(box, width, height):
    b = np.asarray(box, float).copy()
    if not np.isfinite(b).all():
        return None
    b[[0,2]] = np.clip(b[[0,2]], 0, width)
    b[[1,3]] = np.clip(b[[1,3]], 0, height)
    return b if (b[2:] > b[:2]).all() else None


def iou(a,b):
    lo,hi = np.maximum(a[:2],b[:2]), np.minimum(a[2:],b[2:])
    inter = float(np.prod(np.maximum(hi-lo,0)))
    union = float(np.prod(np.maximum(a[2:]-a[:2],0)) + np.prod(np.maximum(b[2:]-b[:2],0)) - inter)
    return inter / max(union,1e-9)


def box_measurement(b):
    return np.r_[(b[:2]+b[2:])/2, np.log(np.maximum(b[2:]-b[:2],1))]


def state_box(x):
    size = np.exp(np.clip(x[4:6],0,10))
    return np.r_[x[:2]-size/2, x[:2]+size/2]


def nms(items, threshold, box=lambda d:d.box, label=lambda d:d.cls, score=lambda d:d.score):
    result=[]
    for item in sorted(items,key=score,reverse=True):
        if not any(label(item)==label(old) and iou(box(item),box(old))>threshold for old in result):
            result.append(item)
    return result
