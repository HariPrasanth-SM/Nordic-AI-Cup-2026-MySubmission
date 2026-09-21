"""Offline nine-policy timing comparison with leave-one-conversation-out selection."""
import argparse,csv,json,statistics as st
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--csv',type=Path,required=True);args=p.parse_args()
rows=list(csv.DictReader(args.csv.open()))
rs=[r for r in rows if r['label']=='1']
def calc(r,ds,de):
 if not r['pred_start']:return 0
 s=round(max(0,float(r['pred_start'])+ds),2);e=round(max(s+.01,float(r['pred_end'])+de),2);a,b=float(r['gold_start']),float(r['gold_end'])
 return max(0,min(e,b)-max(s,a))/max(1e-9,max(e,b)-min(s,a))
policies=[(s,e) for s in [-.1,0,.1] for e in [0,.1,.2]]
print('baseline',st.mean(calc(r,0,0) for r in rs))
for p in policies:print(p,st.mean(calc(r,*p) for r in rs))
cv=[];choices=[]
for c in sorted({r['conversation'] for r in rs}):
 train=[r for r in rs if r['conversation']!=c];p=max(policies,key=lambda p:st.mean(calc(r,*p) for r in train));choices.append(p);cv.extend(calc(r,*p) for r in rs if r['conversation']==c)
print('LOCO',st.mean(cv),{str(p):choices.count(p) for p in set(choices)})
print('errors',st.median(float(r['start_error_s']) for r in rs if float(r['tiou'])>0),st.median(float(r['end_error_s']) for r in rs if float(r['tiou'])>0))
