"""Frozen DINOv2 (S/B/L, /14) candidate verifier with optional multi-prototype scoring. No SAM, box changes, or relabeling."""
import hashlib,json,logging,os,time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

PREPROCESS='rgb-letterbox224-context10-v1'
log=logging.getLogger(__name__)


def resolve_path(explicit,defaults):
    """First existing path among an explicit setting and the defaults; explicit-but-missing is returned so the caller can report it."""
    if explicit:return Path(explicit)
    for d in defaults:
        if Path(d).is_file():return Path(d)
    return None


def crop(image,box):
    h,w=image.shape[:2]; x,y,r,b=map(float,box); dx=(r-x)*.1;dy=(b-y)*.1
    x,y=max(0,int(np.floor(x-dx))),max(0,int(np.floor(y-dy)))
    r,b=min(w,int(np.ceil(r+dx))),min(h,int(np.ceil(b+dy)))
    return image[y:b,x:r].copy()


def prepare(image):
    if image is None or image.size==0: raise ValueError('Empty verifier crop')
    h,w=image.shape[:2]; scale=224/max(h,w)
    im=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_LINEAR)
    # BGR input, RGB encoder. Neutral letterbox; identical offline and online.
    canvas=np.full((224,224,3),127,np.uint8); y=(224-im.shape[0])//2;x=(224-im.shape[1])//2
    canvas[y:y+im.shape[0],x:x+im.shape[1]]=im
    a=canvas[:,:,::-1].copy().transpose(2,0,1).astype(np.float32)/255
    return (a-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]


MODELS={'dinov2_vits14':384,'dinov2_vitb14':768,'dinov2_vitl14':1024}


class Encoder:
    def __init__(self,device='0',batch=32,model='dinov2_vits14'):
        if model not in MODELS:raise ValueError(f'Unsupported DINOv2 model {model}; use one of {sorted(MODELS)}')
        self.name=model;self.dim=MODELS[model]
        self.device=torch.device('cpu' if str(device)=='cpu' else str(device) if str(device).startswith('cuda') else 'cuda:'+str(device).split(',')[0])
        local=os.getenv('DINO_REPO')
        self.model=torch.hub.load(local or 'facebookresearch/dinov2',model,source='local' if local else 'github',pretrained=True,trust_repo=True).eval()
        h=hashlib.sha256()
        for name,t in sorted(self.model.state_dict().items()):h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        self.sha=h.hexdigest(); self.model.to(self.device);self.batch=batch
        for p in self.model.parameters():p.requires_grad_(False)
    @torch.inference_mode()
    def encode(self,images):
        rows=[]
        for start in range(0,len(images),self.batch):
            x=torch.from_numpy(np.stack([prepare(im) for im in images[start:start+self.batch]])).to(self.device)
            with torch.autocast(device_type=self.device.type,enabled=self.device.type=='cuda',dtype=torch.float16):
                z=self.model.forward_features(x)['x_norm_clstoken']
            z=F.normalize(z.float(),dim=1);rows.append(z.cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0,self.dim),np.float32)


def scores(embeddings,refs,labels,nclasses):
    similarities=embeddings@refs.T
    out=np.full((len(embeddings),nclasses+1),-1.,np.float32)
    for cls in range(-1,nclasses):
        which=labels==cls
        if which.any():out[:,cls+1]=similarities[:,which].max(axis=1)
    return out


def decision(row,cls,minimum,negative_margin,class_margin):
    own=float(row[cls+1]); bg=float(row[0]);other=float(np.max(np.delete(row[1:],cls)))
    reasons=[]
    if own<minimum:reasons.append('low_target_similarity')
    if own-bg<negative_margin:reasons.append('background_match')
    if own-other<class_margin:reasons.append('ambiguous_class')
    return not reasons,{'target_similarity':own,'background_similarity':bg,'best_other_similarity':other,'reasons':reasons}


class VerifiedDetector:
    supports_levels=True
    def __init__(self,cfg):
        from .level_detector import LevelDetector
        from dtos import OBJECT_CLASSES
        self.base=LevelDetector(cfg);self.cfg=cfg;self.names=list(OBJECT_CLASSES)
        self.mode=os.getenv('DINO_MODE',getattr(cfg,'dino_mode','shadow'))
        if self.mode not in ('off','shadow','filter'):raise ValueError('DINO_MODE must be off, shadow or filter')
        self.maximum=int(os.getenv('DINO_MAX_CANDIDATES',str(getattr(cfg,'dino_max_candidates',48))))
        if self.maximum<1:raise ValueError('DINO_MAX_CANDIDATES must be positive')
        self.minimum=float(os.getenv('DINO_MIN_SIM','0.55'));self.negmargin=float(os.getenv('DINO_NEG_MARGIN','0.03'));self.classmargin=float(os.getenv('DINO_CLASS_MARGIN','0.00'))
        self.last_info={};self.protos=None
        if not (-1<=self.minimum<=1 and 0<=self.negmargin<=2 and 0<=self.classmargin<=2):raise ValueError('Invalid similarity thresholds')
        if self.mode=='off':return
        self.model_name=os.getenv('DINO_MODEL',getattr(cfg,'dino_model','dinov2_vits14'))
        self.encoder=Encoder(cfg.device,model=self.model_name)
        bank=resolve_path(os.getenv('DINO_BANK',getattr(cfg,'dino_bank','')),[f'weights/dino_bank_{self.model_name}.npz']+(['weights/dino_bank.npz'] if self.model_name=='dinov2_vits14' else []))
        if bank is None or not bank.is_file():raise FileNotFoundError(f'DINO bank not found for {self.model_name}: build it with  python scripts/build_dino_bank.py --model {self.model_name} --output weights/dino_bank_{self.model_name}.npz ...')
        with np.load(bank,allow_pickle=False) as z:
            self.refs=z['features'].astype(np.float32);self.labels=z['labels'].astype(int);self.meta=json.loads(str(z['metadata'].item()))
        if self.meta.get('model','dinov2_vits14')!=self.model_name:raise ValueError(f"DINO bank was built with {self.meta.get('model','dinov2_vits14')} but {self.model_name} is configured; rebuild the bank")
        if self.meta['classes']!=self.names or self.meta['encoder_sha256']!=self.encoder.sha or self.meta['preprocess']!=PREPROCESS:raise ValueError('DINO bank/encoder/preprocessing mismatch; rebuild bank')
        if self.refs.ndim!=2 or self.refs.shape[1]!=self.encoder.dim or len(self.refs)!=len(self.labels) or not np.isfinite(self.refs).all():raise ValueError('Invalid DINO bank')
        if not np.allclose(np.linalg.norm(self.refs,axis=1),1,atol=.001):raise ValueError('Bank embeddings must be normalized')
        if any(not (self.labels==i).any() for i in range(-1,len(self.names))):raise ValueError('Need every class and background references')
        self.banksha=hashlib.sha256(bank.read_bytes()).hexdigest()
        protos=resolve_path(os.getenv('DINO_PROTOS',getattr(cfg,'dino_protos','')),[f'weights/dino_protos_{self.model_name}.npz'])
        if protos is not None and protos.is_file():
            from .prototypes import load,ProtoScorer
            data=load(protos)
            if data['meta'].get('dim')!=self.encoder.dim or data['meta'].get('encoder_sha256',self.encoder.sha)!=self.encoder.sha:raise ValueError('Prototype file does not match the encoder; rebuild with scripts/build_prototypes.py')
            self.protos=ProtoScorer(data,self.names)
        else:log.warning('No prototype file for %s: multi-prototype scoring disabled (scripts/build_prototypes.py)',self.model_name)
        self.encoder.encode([np.zeros((60,120,3),np.uint8)])
    def detect(self,image,level):
        raw=self.base.detect(image,level);self.last_info=dict(self.base.last_info)
        if self.mode=='off':return raw
        start=time.perf_counter();logs=[];chosen=[];images=[]
        for i,d in enumerate(raw):
            log={'index':i,'box':d.box.tolist(),'cls':int(d.cls),'class':self.names[d.cls],'score':float(d.score),'partial':bool(d.partial),'would_keep':True,'status':'bypass'};logs.append(log)
            d.extra={'status':'bypass'}
            if d.partial:log['reason']='partial_box';continue
            if min(d.box[2:]-d.box[:2])<8:log['reason']='too_small';continue
            if len(chosen)>=self.maximum:log['reason']='candidate_budget';continue
            im=crop(image,d.box)
            if not im.size:log['reason']='empty_crop';continue
            chosen.append(i);images.append(im)
        error=None
        try:
            embeddings=self.encoder.encode(images)
            if not np.isfinite(embeddings).all():raise ValueError('Nonfinite encoder features')
            values=scores(embeddings,self.refs,self.labels,len(self.names))
            for i,row in zip(chosen,values):
                keep,info=decision(row,raw[i].cls,self.minimum,self.negmargin,self.classmargin)
                logs[i].update(info,would_keep=keep,status='verified')
                raw[i].extra={'status':'verified','target_similarity':info['target_similarity'],'background_similarity':info['background_similarity'],
                    'best_other_similarity':info['best_other_similarity'],'class_similarities':[round(float(v),4) for v in row[1:]]}
            protos=getattr(self,'protos',None)
            if protos is not None:
                pm=protos.class_max(embeddings)
                for i,row in zip(chosen,pm):
                    raw[i].extra.update(protos.summarize(row,raw[i].cls));logs[i].update({k:v for k,v in raw[i].extra.items() if k.startswith('proto_') and k!='proto_class'})
        except Exception as exc:
            # Deliberate fail-open: preserve baseline recall, expose error in trace.
            error=repr(exc)
            for i in chosen:logs[i].update(would_keep=True,status='error',reason=error);raw[i].extra={'status':'error'}
        self.last_info['verifier']={'mode':self.mode,'elapsed_ms':(time.perf_counter()-start)*1000,'bank_sha256':self.banksha,
            'encoder_sha256':self.encoder.sha,'model':getattr(self,'model_name',None),'prototypes':getattr(self,'protos',None) is not None,'thresholds':[self.minimum,self.negmargin,self.classmargin],
            'candidates':logs,'error':error,'rejected':sum(not x['would_keep'] for x in logs)}
        return [d for d,log in zip(raw,logs) if self.mode!='filter' or log['would_keep']]


def create(cfg):return VerifiedDetector(cfg)
