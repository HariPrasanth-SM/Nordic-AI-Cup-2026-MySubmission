"""CPU-only calibration of a saved Exp14 trace. No model calls and no labels in inference."""
import argparse,copy,csv,json,sys,hashlib
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.calibration15 import extend_end as calibrate,load_settings
from solution.types import Span,VerifierResult
from solution.local_diagnostics import analyze
from solution.diagnostics15 import calibration_analysis

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--csv',type=Path,default=ROOT/'data/question_train.csv');p.add_argument('--run-name',default='exp15-cached');args=p.parse_args()
    if Path(args.run_name).name!=args.run_name:raise SystemExit('Simple run name required')
    out=ROOT/'reports'/args.run_name;out.mkdir(parents=True,exist_ok=False);cfg=load_settings()
    traces=list(map(json.loads,args.trace.read_text().splitlines()))
    if not traces:raise SystemExit('Empty trace')
    for t in traces:
        stages={e['stage'] for e in t.get('events',[])}
        if 'exp15_calibration' in stages:raise SystemExit('Refusing to apply calibration twice')
        if 'exp14_calibration' not in stages:raise SystemExit('Input must be an Exp14 trace')
    with args.csv.open() as f:gold=list(csv.DictReader(f))
    groups={}
    for r in gold:groups.setdefault('conversation_'+r['transcript_id']+'.mp3',[]).append(r)
    manifest=dict(mode='saved_prediction_calibration_only',config=cfg,input_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (out/'trace.jsonl').open('w') as tf,(out/'predictions.jsonl').open('w') as pf:
        for t in traces:
            rows=groups[t['audio_filename']]
            if [r['question'] for r in rows]!=t['questions']:raise SystemExit('Question order mismatch')
            old=[VerifierResult(float(y),None if s is None else Span(s,e),r.get('expected_tiou',0),r.get('quote'))
                for y,s,e,r in zip(t['response']['answers'],t['response']['evidence_start'],t['response']['evidence_end'],t['results'])]
            new=calibrate(old,cfg);answers=[r.p_yes>.5 for r in new]
            starts=[round(max(0,r.span.start),2) if y and r.span else None for y,r in zip(answers,new)]
            ends=[round(max(r.span.end,starts[i]+.01),2) if y and r.span else None for i,(y,r) in enumerate(zip(answers,new))]
            rec=copy.deepcopy(t);rec.update(results=[asdict(r) for r in new],response=dict(answers=answers,evidence_start=starts,evidence_end=ends))
            rec['elapsed_s']=None
            rec['events'] += [dict(stage='exp15_baseline',results=[asdict(r) for r in old]),dict(stage='exp15_calibration',config=cfg,changed=sum(a.span!=b.span for a,b in zip(old,new)),classification_changes=0)]
            tf.write(json.dumps(rec)+'\n')
            pf.write(json.dumps(dict(audio_filename=t['audio_filename'],questions=t['questions'],question_ids=[r['question_id'] for r in rows],predictions=[int(x) for x in answers],spans=[None if s is None else [s,e] for s,e in zip(starts,ends)],latency_ms=None,error=None,timed_out=False))+'\n')
    print(json.dumps(analyze(out,args.csv),indent=2));gate=calibration_analysis(out);print(json.dumps(gate,indent=2));return 0 if gate['execution_gate_passed'] else 2
if __name__=='__main__':raise SystemExit(main())
