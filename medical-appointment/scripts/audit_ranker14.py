"""Offline five-fold grouped model test. This model failed the local quality gate; never loaded by serving code."""
import argparse,sys,json,csv,time,statistics as st,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.span_quality14 import build_pool,matrix
from solution.types import Word
from solution.local_diagnostics import iou
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
threadpool_limits(2)
p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path);p.add_argument('--csv',type=Path,default=ROOT/'data/question_train.csv');p.add_argument('--out',type=Path,default=ROOT/'reports/exp14-ranker-audit');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
tr=list(map(json.loads,args.trace.read_text().splitlines()));gold=list(csv.DictReader(args.csv.open()))
labels={(r['transcript_id'],r['question']):int(r['label']) for r in gold}
correct=sum(int(bool(y)==labels[(t['audio_filename'][13:-4],q)]) for t in tr for q,y in zip(t['questions'],t['response']['answers']))
accuracy=correct/len(gold)
for expand in [False,True]:
    X=[];y=[];groups=[];weights=[];records=[]
    for t in tr:
        gs=[r for r in gold if 'conversation_'+r['transcript_id']+'.mp3'==t['audio_filename']]
        words=[Word(**w) for s in t['segments'] for w in s['words']]
        for i,r in enumerate(gs):
            if r['label']!='1':continue
            span=None if t['response']['evidence_start'][i] is None else(t['response']['evidence_start'][i],t['response']['evidence_end'][i]);g=(float(r['evidence_start']),float(r['evidence_end']))
            seeds=next((e['candidates'] for e in t['events'] if e['stage']=='exp13_candidates' and e['index']==i),[])
            base=next(e['results'][i]['span'] for e in t['events'] if e['stage']=='exp13_baseline')
            pool=build_pool(words,seeds,expand) if seeds else []
            lo=len(y);features=matrix(r['question'],words,pool,base)
            X.extend(features);y.extend(iou((c['start'],c['end']),g) for c in pool);groups.extend([r['transcript_id']]*len(pool));weights.extend([1/max(1,len(pool))]*len(pool))
            records.append(dict(id=r['question_id'],conv=r['transcript_id'],lo=lo,hi=len(y),baseline=iou(span,g)))
    X=np.array(X);y=np.array(y);weights=np.array(weights);pred=np.zeros(len(y))
    for train,test in GroupKFold(5).split(X,y,groups):
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=5,learning_rate=.06,random_state=42)
        # Mean weight one, equal weight per question.
        w=weights[train];model.fit(X[train],y[train],sample_weight=w/w.mean());pred[test]=model.predict(X[test])
    vals=[];oracle=[]
    for r in records:
        lo,hi=r['lo'],r['hi'];r['selected']=float(y[lo+np.argmax(pred[lo:hi])]) if hi>lo else 0
        r['oracle']=float(max(y[lo:hi])) if hi>lo else 0;vals.append(r['selected']);oracle.append(r['oracle'])
    output=dict(expanded=expand,candidates=len(y),tiou=st.mean(vals),score=.4*accuracy+.6*st.mean(vals),oracle=st.mean(oracle),records=records)
    (args.out/(('expanded' if expand else 'original')+'.json')).write_text(json.dumps(output,indent=2))
    print({k:v for k,v in output.items() if k!='records'},flush=True)
