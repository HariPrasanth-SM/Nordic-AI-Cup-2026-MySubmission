"""Timestamp-native span geometry. No labels or training data in inference."""
import bisect,math
import numpy as np

def transcript(words):
    parts=[];bounds=[];pos=0
    for w in words:
        text=w.text.strip();parts.append(text);bounds.append((pos,pos+len(text)));pos+=len(text)+1
    return ' '.join(parts),bounds

def windows(tokenizer,question,words,cfg):
    text,bounds=transcript(words)
    if not text:return []
    enc=tokenizer(question,text,truncation='only_second',max_length=cfg['max_length'],
        stride=cfg['stride'],return_overflowing_tokens=True,return_offsets_mapping=True,padding='max_length')
    features=[]
    for n,offsets in enumerate(enc['offset_mapping']):
        mapping={};seq=enc.sequence_ids(n)
        ends=[b[1] for b in bounds]
        for ti,((lo,hi),sid) in enumerate(zip(offsets,seq)):
            if sid!=1 or hi<=lo:continue
            wi=bisect.bisect_right(ends,lo)
            if wi<len(bounds) and bounds[wi][0]<hi:
                mapping.setdefault(wi,[]).append((ti,lo,hi))
        # Discard a word split at a window edge; predictions use full words only.
        indices=[wi for wi,ts in mapping.items() if ts[0][1]<=bounds[wi][0] and ts[-1][2]>=bounds[wi][1]]
        if not indices:continue
        features.append(dict(inputs={k:enc[k][n] for k in tokenizer.model_input_names if k in enc},
            word_ids=indices,start_tokens=[mapping[i][0][0] for i in indices],end_tokens=[mapping[i][-1][0] for i in indices],
            cls=enc['input_ids'][n].index(tokenizer.cls_token_id)))
    return features

def geometry(words,feature,cfg):
    ids=np.asarray(feature['word_ids']);starts=np.array([words[i].start for i in ids])[:,None]
    ends=np.array([words[i].end for i in ids])[None,:];duration=ends-starts
    s=starts+cfg['start_fraction']*duration;e=np.broadcast_to(ends+cfg['end_offset_s'],s.shape)
    valid=(ids[None,:]>=ids[:,None]) & (ids[None,:]-ids[:,None]<cfg['max_words']) & (duration>0) & (duration<=cfg['max_duration_s'])
    return np.round(s,2),np.round(e,2),valid

def overlaps(s,e,gold):
    intersection=np.maximum(0,np.minimum(e,gold[1])-np.maximum(s,gold[0]))
    return intersection/np.maximum(1e-9,np.maximum(e,gold[1])-np.minimum(s,gold[0]))

def supervision(words,feature,cfg,gold):
    s,e,valid=geometry(words,feature,cfg);ious=np.where(valid,overlaps(s,e,gold),0)
    best=float(ious.max(initial=0))
    # Boundary-truncated and unrelated windows are trained as null, at lower weight.
    if best<.5:return dict(null=True,weight=.25,oracle=best)
    logits=np.where(valid,(ious-best)/.08,-1e9)
    # Only near-best pairs receive soft mass; avoid thousands of distant negatives.
    logits=np.where(ious>=best-.15,logits,-1e9)
    p=np.exp(logits-logits.max());p/=p.sum()
    return dict(null=False,weight=1.,oracle=best,start=p.sum(axis=1),end=p.sum(axis=0))

def decode(words,features,logits,cfg,baseline=None):
    pool={};base_score=-math.inf;base_error=math.inf
    for f,(sl,el) in zip(features,logits):
        s,e,valid=geometry(words,f,cfg)
        a=np.asarray(sl)[f['start_tokens']];b=np.asarray(el)[f['end_tokens']]
        null=float(sl[f['cls']]+el[f['cls']]);score=a[:,None]+b[None,:]-null
        score=np.where(valid,score,-np.inf)
        if baseline is not None:
            distance=np.where(valid,np.abs(s-baseline[0])+np.abs(e-baseline[1]),np.inf)
            low=float(distance.min())
            if math.isfinite(low) and low<=cfg['max_baseline_projection_error_s']:
                nearest=np.isclose(distance,low,atol=.00001)
                value=float(score[nearest].max())
                if low<base_error-.00001:base_error=low;base_score=value
                elif abs(low-base_error)<.00001:base_score=max(base_score,value)
        count=min(5,int(valid.sum()))
        if not count:continue
        flat=score.ravel();top=np.argpartition(flat,-count)[-count:]
        for ix in top:
            a,b=np.unravel_index(ix,score.shape);lo=f['word_ids'][a];hi=f['word_ids'][b]
            key=(round(float(s[a,b]),2),round(float(e[a,b]),2));value=float(score[a,b])
            if key not in pool or value>pool[key]['score']:
                pool[key]=dict(start=key[0],end=key[1],score=value,text=' '.join(w.text for w in words[lo:hi+1]),start_word=lo,end_word=hi)
    candidates=sorted(pool.values(),key=lambda c:c['score'],reverse=True)[:5]
    best=candidates[0] if candidates else None
    margin=None if best is None or not math.isfinite(base_score) else best['score']-base_score
    accept=bool(best and best['score']>=cfg['min_null_margin'] and
        ((baseline is None) or (margin is not None and margin>=cfg['margin'])))
    return dict(candidate=best,accepted=accept,margin=margin,baseline_score=base_score if math.isfinite(base_score) else None,
        baseline_projection_error_s=base_error if math.isfinite(base_error) else None,candidates=candidates)
