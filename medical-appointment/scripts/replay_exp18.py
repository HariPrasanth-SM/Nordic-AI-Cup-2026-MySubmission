"""Replay rescue on ACTUAL Exp15 responses; gold labels are used only after inference."""
import argparse,copy,csv,hashlib,json,os,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.rescue18 import RescueVerifier,seed_pools,eligibility
from solution.types import Segment,Word,Span,VerifierResult
from solution import experiment_trace
from solution.local_diagnostics import analyze
from solution.diagnostics18 import rescue_analysis

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path);p.add_argument('--run-name',default='exp18-cached')
    p.add_argument('--csv',default='data/question_train.csv',type=Path);p.add_argument('--screen-only',action='store_true')
    a=p.parse_args();a.trace=a.trace.resolve();a.csv=a.csv.resolve();os.chdir(ROOT)
    if Path(a.run_name).name!=a.run_name:raise SystemExit('Use a simple run name')
    ts=[json.loads(l) for l in a.trace.read_text().splitlines() if l.strip()]
    forbidden={'exp16_config','exp17_config','exp18_config'}
    if not ts or any(not any(e['stage']=='exp15_calibration' for e in t['events']) or any(e['stage'] in forbidden for e in t['events']) for t in ts):
        raise SystemExit('Input must be an Exp15 trace, not a later experiment')
    ids={};expected={}
    with a.csv.open() as f:
        for row in csv.DictReader(f):
            name='conversation_'+row['transcript_id']+'.mp3';ids.setdefault(name,[]).append(row['question_id']);expected.setdefault(name,[]).append(row['question'])
    if len({t['audio_filename'] for t in ts})!=len(ts):raise SystemExit('Duplicate conversations')
    for t in ts:
        if expected.get(t['audio_filename'])!=t['questions']:raise SystemExit('Trace/CSV question order mismatch')
    v=RescueVerifier()
    if not a.screen_only:v.client.warmup()
    out=ROOT/'reports'/a.run_name;out.mkdir(parents=True,exist_ok=False)
    os.environ['MEDICAL_LOCAL_TRACE']=str(out/'trace.jsonl')
    manifest=dict(mode='screen_only' if a.screen_only else 'cached_exp15_new_rescue_calls',input_trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
                  config=v.config,calibration=v.calibration,model_config=v.client.config,
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['solution/rescue18.py','scripts/replay_exp18.py']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (out/'predictions.jsonl').open('w') as pf,(out/'trace.jsonl').open('w') as tf:
        for t in ts:
            experiment_trace.reset();v.audio_hash=t['audio_sha256']
            seg=[Segment(s['start'],s['end'],s['text'],[Word(**w) for w in s['words']]) for s in t['segments']]
            response=t['response'];base=[]
            for i,r in enumerate(t['results']):
                s,e=response['evidence_start'][i],response['evidence_end'][i]
                base.append(VerifierResult(float(response['answers'][i]),Span(s,e) if s is not None and e is not None else None,r['expected_tiou'],r.get('quote')))
            if len(base)!=len(t['questions']):raise ValueError('Incomplete baseline')
            start=time.monotonic();pools=seed_pools(t['events'])
            if a.screen_only:
                v.last_events=[];records=eligibility(t['questions'],seg,base,pools,v.config['max_questions'])
                v.emit('exp18_baseline',results=[asdict(r) for r in base]);v.emit('exp18_config',config=v.config,calibration=v.calibration)
                v.emit('exp18_screen',cases=records);v.emit('exp18_screen_only')
                v.emit('exp18_coverage',eligible=len(records),selected=sum(r['review_selected'] for r in records),changed=0,classification_changes=0,elapsed_s=0)
                result=base
            else:result=v.rescue(t['questions'],seg,base,start+v.config['max_rescue_s']+1.5,pools)
            elapsed=time.monotonic()-start
            response=copy.deepcopy(t['response'])
            for i,(old,new) in enumerate(zip(base,result)):
                if old.span==new.span:continue
                response['evidence_start'][i]=round(max(0,new.span.start),2)
                response['evidence_end'][i]=round(max(new.span.end,response['evidence_start'][i]+.01),2)
            record=copy.deepcopy(t);record.update(results=[asdict(r) for r in result],response=response,elapsed_s=elapsed,events=list(experiment_trace._EVENTS))
            tf.write(json.dumps(record)+'\n');tf.flush()
            pf.write(json.dumps(dict(audio_filename=t['audio_filename'],questions=t['questions'],question_ids=ids[t['audio_filename']],
                predictions=[int(x) for x in response['answers']],spans=[None if s is None else [s,e] for s,e in zip(response['evidence_start'],response['evidence_end'])],
                latency_ms=elapsed*1000,error=None,timed_out=False))+'\n');pf.flush()
            print(t['audio_filename'],round(elapsed,2),'rescue seconds',flush=True)
    print(json.dumps(analyze(out,a.csv),indent=2));audit=rescue_analysis(out);print(json.dumps(audit,indent=2))
    print('Screen-only measures eligibility, not model quality.' if a.screen_only else 'Cached timing excludes the Exp15 pipeline. Full live evaluation is required before validation.')
    return 0 if audit['execution_gate_passed'] else 2
if __name__=='__main__':raise SystemExit(main())
