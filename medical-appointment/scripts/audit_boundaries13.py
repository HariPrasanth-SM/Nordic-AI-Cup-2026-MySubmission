"""Offline gold-assisted oracle audit and coarse grouped calibration comparison. Never used in inference."""
import argparse,csv,json,statistics as st
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path);p.add_argument('--csv',type=Path,default=Path('data/question_train.csv'));p.add_argument('--out',type=Path,default=Path('reports/exp12-boundary-audit'));args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
rows=list(csv.DictReader(args.csv.open()))
tr={r['audio_filename']:r for r in map(json.loads,args.trace.open())}
def iou(a,b):
 if a is None:return 0
 return max(0,min(a[1],b[1])-max(a[0],b[0]))/max(1e-9,max(a[1],b[1])-min(a[0],b[0]))
res=[]
for r in rows:
 if r['label']!='1':continue
 t=tr['conversation_'+r['transcript_id']+'.mp3'];idx=t['questions'].index(r['question']);w=[w for s in t['segments'] for w in s['words']]
 s=t['response']['evidence_start'][idx];e=t['response']['evidence_end'][idx];p=None if s is None else(s,e);g=(float(r['evidence_start']),float(r['evidence_end']))
 d=dict(id=r['question_id'],conv=r['transcript_id'],question=r['question'],gold=g,pred=p,tiou=iou(p,g),crop_oracle=0,local_oracle=0)
 if p:
  ix=[i for i,x in enumerate(w) if s<=(x['start']+x['end'])/2<=e]
  if ix:
   a,b=ix[0],ix[-1]
   d['crop_oracle']=max(iou((w[i]['start'],w[j]['end']),g) for i in range(a,b+1) for j in range(i,b+1))
   d['local_oracle']=max(iou((w[i]['start'],w[j]['end']),g) for i in range(max(0,a-8),min(len(w),b+9)) for j in range(i,min(len(w),b+9)))
  d.update(start_err=s-g[0],end_err=e-g[1],ratio=(e-s)/(g[1]-g[0]),gold_text=' '.join(x['text'] for x in w if g[0]<=(x['start']+x['end'])/2<=g[1]),pred_text=' '.join(x['text'] for x in w if s<=(x['start']+x['end'])/2<=e))
 res.append(d)
summary={k:st.mean(d[k] for d in res) for k in ['tiou','crop_oracle','local_oracle']}
over=[d for d in res if d['tiou']>0]
summary.update(overlap_count=len(over),median_start_error_overlapping=st.median(d['start_err'] for d in over),median_end_error_overlapping=st.median(d['end_err'] for d in over),median_duration_ratio_overlapping=st.median(d['ratio'] for d in over),crop_gain_at_least_point1=sum(d['crop_oracle']-d['tiou']>=.1 for d in res))
# One predeclared coarse family; compare leave-one-conversation-out selection.
policies=[(a,b) for a in [0,.1,.2] for b in [0,.1,.2]]
def score(d,ab):
 if not d['pred']:return 0
 s,e=d['pred'];L=e-s;return iou((s+ab[0]*L,e-ab[1]*L),d['gold'])
cv=[];choices=[]
for c in sorted({d['conv'] for d in res}):
 train=[d for d in res if d['conv']!=c];best=max(policies,key=lambda ab:st.mean(score(d,ab) for d in train));choices.append(best)
 cv.extend(score(d,best) for d in res if d['conv']==c)
summary['simple_fractional_trim_LOCO_tiou']=st.mean(cv);summary['trim_choices']=dict((str(p),choices.count(p)) for p in set(choices))
(args.out/'audit.json').write_text(json.dumps(summary,indent=2));(args.out/'positive_details.json').write_text(json.dumps(res,indent=2))
print(json.dumps(summary,indent=2))
print('\nExample excess clauses:')
for d in sorted(over,key=lambda x:x['crop_oracle']-x['tiou'],reverse=True)[:8]:print(d['id'],round(d['tiou'],3),round(d['crop_oracle'],3),'\nG:',d['gold_text'],'\nP:',d['pred_text'])
