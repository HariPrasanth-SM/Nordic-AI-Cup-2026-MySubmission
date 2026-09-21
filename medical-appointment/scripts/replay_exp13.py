"""Replay ONLY Exp13 refinement on saved Exp12 predictions. No audio/ASR/base-LLM pass.
Not a competition latency measurement. Gold is used only after predictions.
"""
import argparse,copy,csv,hashlib,json,os,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solution.refinement13 import ContrastiveVerifier
from solution.types import Segment,Word,Span,VerifierResult
from solution import experiment_trace
from solution.local_diagnostics import analyze
from solution.diagnostics13 import refinement_analysis

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--run-name',default='exp13-cached');p.add_argument('--csv',default='data/question_train.csv',type=Path)
    a=p.parse_args();a.trace=a.trace.resolve();a.csv=a.csv.resolve();os.chdir(ROOT)
    if Path(a.run_name).name!=a.run_name:raise SystemExit('Use a simple run name')
    out=ROOT/'reports'/a.run_name;out.mkdir(parents=True,exist_ok=False)
    ts=[json.loads(l) for l in a.trace.read_text().splitlines()]
    if not ts or any(not any(e.get('stage')=='exp12_config' for e in t.get('events',[])) or
                     any(e.get('stage')=='exp13_baseline' for e in t.get('events',[])) for t in ts):
        raise SystemExit('Input must be a saved Exp12 trace, not an older experiment or an Exp13 output.')
    verifier=ContrastiveVerifier();verifier.client.warmup()
    os.environ['MEDICAL_LOCAL_TRACE']=str(out/'trace.jsonl')
    manifest=dict(mode='cached_exp12_predictions_refinement_only',input_trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
                  config=verifier.config,model_config=verifier.client.config,python=sys.executable,
                  bank_sha256=hashlib.sha256((ROOT/verifier.config['bank_path']).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Row IDs are transport bookkeeping only; not passed to the verifier.
    with a.csv.open() as f:rows=list(csv.DictReader(f))
    ids={};expected_questions={}
    for row in rows:
        name='conversation_'+row['transcript_id']+'.mp3'
        ids.setdefault(name,[]).append(row['question_id'])
        expected_questions.setdefault(name,[]).append(row['question'])
    for t in ts:
        if expected_questions.get(t['audio_filename']) != t['questions']:
            raise SystemExit('Trace/CSV question order mismatch: '+t['audio_filename'])
    with (out/'predictions.jsonl').open('w') as predictions,(out/'trace.jsonl').open('w') as traces:
        for t in ts:
            experiment_trace.reset();verifier.audio_hash=t['audio_sha256']
            segments=[Segment(s['start'],s['end'],s['text'],[Word(**w) for w in s['words']]) for s in t['segments']]
            base=[VerifierResult(float(yes),None if s is None else Span(s,e),r.get('expected_tiou',0),r.get('quote'))
                  for yes,s,e,r in zip(t['response']['answers'],t['response']['evidence_start'],
                                       t['response']['evidence_end'],t['results'])]
            if len(base)!=len(t['questions']):raise ValueError('Incomplete base trace')
            start=time.monotonic();results=verifier.refine(t['questions'],segments,base,start+38)
            elapsed=time.monotonic()-start
            answers=[r.p_yes>.5 for r in results]
            starts=[round(max(0,r.span.start),2) if yes and r.span else None for yes,r in zip(answers,results)]
            ends=[round(max(r.span.end,starts[i]+.01),2) if yes and r.span else None for i,(yes,r) in enumerate(zip(answers,results))]
            response=dict(answers=answers,evidence_start=starts,evidence_end=ends)
            record=copy.deepcopy(t);record.update(results=[asdict(r) for r in results],response=response,
                elapsed_s=elapsed,events=list(experiment_trace._EVENTS))
            record['config']['threshold']={'mode':'fixed','fixed_value':.5}
            record['config']['exp13_refinement']=verifier.config
            traces.write(json.dumps(record)+'\n');traces.flush()
            client_record=dict(audio_filename=t['audio_filename'],questions=t['questions'],
                question_ids=ids[t['audio_filename']],predictions=[int(x) for x in answers],
                spans=[None if x is None else [x,y] for x,y in zip(starts,ends)],latency_ms=elapsed*1000,
                error=None,timed_out=False)
            predictions.write(json.dumps(client_record)+'\n');predictions.flush()
            print(t['audio_filename'],round(elapsed,2),'seconds',flush=True)
            if any(e['stage']=='exp13_failed' for e in verifier.last_events):
                print('Stopping cached replay after a grounding-call failure; remaining rows are scored unsent.')
                break
    summary=analyze(out,a.csv);gate=refinement_analysis(out)
    print(json.dumps(summary,indent=2));print(json.dumps(gate,indent=2))
    print('Cached replay excludes ASR, Exp12 base calls, and HTTP overhead. Do full local replay before deployment.')
    return 0 if gate['execution_gate_passed'] else 2
if __name__=='__main__':raise SystemExit(main())
