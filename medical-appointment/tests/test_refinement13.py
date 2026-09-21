import json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from solution.types import Word,Segment,Span,VerifierResult
from solution.refinement13 import ContrastiveVerifier,proposal_schema,selection_schema
from solution.grounding12 import make_sources
from solution.diagnostics13 import refinement_analysis
from solution.evidence_experiment import transcript_hash

CFG=dict(bank_path='unused',proposal_max_tokens=1000,selection_max_tokens=200,call_timeout_s=24,
    min_remaining_s=22,selection_min_remaining_s=7,examples_per_question=2,
    source_forward_segments=4,max_prompt_chars=43000,max_review_chars=48000)
WORDS=[Word(t,i,i+1) for i,t in enumerate(['Hello.','Take','10','mg','daily.','Goodbye.'])]
SEGS=[Segment(0,6,' '.join(w.text for w in WORDS),WORDS)]
BASE=[VerifierResult(1,Span(0,6),.6,'Hello. Take 10 mg daily. Goodbye.'),VerifierResult(0,None,0)]
DEMO=dict(question_id='other',question='Take ten?',context='Take ten daily',gold_quote='ten',contrast={},audio_sha256='other',transcript_hash='other')
class Client:
    config={}
    def __init__(self,mode='ok'):self.mode=mode;self.calls=[]
    def call(self,messages,schema,deadline,max_tokens=None):
        self.calls.append((messages,schema))
        if self.mode=='fail':raise TimeoutError('fixture')
        if len(self.calls)==1:
            q=dict(source='S000',quote='Take 10 mg daily.' if self.mode!='bad_quote' else 'Invented words')
            return {'q0':dict(core=q,context=q,alternative=None)}
        if self.mode=='bad_id':return {'q0':dict(candidate='C999',covers_all_details=True)}
        shown=json.loads(messages[-1]['content'].split('CANDIDATES TO COMPARE:\n')[1])
        cs=shown['q0'];pick=next((c for c in cs if c['text']=='Take 10 mg daily.'),cs[0])
        return {'q0':dict(candidate=pick['id'],covers_all_details=self.mode!='insufficient')}

def verifier(client=None,bank=None):
    c=client or Client();return ContrastiveVerifier(base=SimpleNamespace(client=c),client=c,bank=bank or [DEMO],config=CFG)

class RefinementTests(unittest.TestCase):
    def test_refines_and_preserves_every_classification(self):
        v=verifier();r=v.refine(['Take 10 mg daily?','Take 20 mg?'],SEGS,BASE,time.monotonic()+60)
        self.assertEqual([x.p_yes for x in r],[1,0]);self.assertEqual(r[0].span,Span(1,5));self.assertIs(r[1],BASE[1])
        self.assertEqual(v.client.calls[0][1]['required'],['q0'])
        self.assertEqual(v.last_events[-1]['changed'],1)
    def test_quote_failure_preserves_baseline_and_is_visible(self):
        v=verifier(Client('bad_quote'));r=v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60)
        self.assertEqual(r[0].span,BASE[0].span);self.assertEqual(v.last_events[-1]['proposed'],0)
        self.assertTrue(any(e['stage']=='exp13_rejection' for e in v.last_events))
    def test_model_failure_preserves_baseline(self):
        v=verifier(Client('fail'));r=v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60)
        self.assertEqual(r,BASE);self.assertEqual(v.last_events[-1]['stage'],'exp13_failed')
    def test_invalid_selection_preserves_baseline(self):
        v=verifier(Client('bad_id'));r=v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60)
        self.assertEqual(r,BASE);self.assertEqual(v.last_events[-1]['reviewed'],0)
    def test_sufficiency_veto(self):
        v=verifier(Client('insufficient'));r=v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60)
        self.assertEqual(r,BASE)
    def test_deadline_skips_without_model_call(self):
        v=verifier();r=v.refine(['Q','N'],SEGS,BASE,time.monotonic()+2)
        self.assertEqual(r,BASE);self.assertFalse(v.client.calls);self.assertEqual(v.last_events[-1]['stage'],'exp13_skipped')
    def test_current_consultation_excluded(self):
        own=dict(DEMO,question_id='OWN',audio_sha256='current',transcript_hash=transcript_hash(WORDS))
        v=verifier(bank=[own,DEMO]);v.audio_hash='current';v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60)
        e=next(e for e in v.last_events if e['stage']=='exp13_examples');self.assertNotIn('OWN',e['question_ids'])
    def test_schema_every_key_and_candidate_enum(self):
        _,src=make_sources(SEGS);s=proposal_schema(['q0','q3'],src)
        self.assertEqual(s['required'],['q0','q3']);self.assertIn('anyOf',s['properties']['q0']['properties']['alternative'])
        s=selection_schema({'q3':[{'id':'C1'}]});self.assertEqual(s['properties']['q3']['properties']['candidate']['enum'],['KEEP','C1'])
    def test_pipeline_wraps_exp12_and_keeps_threshold(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,{'MEDICAL_APPT_SKIP_MODEL_LOAD':'1','MEDICAL_APPT_EXPERIMENT':'exp13'}):
            from solution.config import load_config
            import solution.pipeline as pipeline
            cfg=load_config('workstation_32gb')
            base=SimpleNamespace(client=SimpleNamespace(warmup=lambda:None))
            wrapped=object()
            with patch('solution.grounding12.AnchoredVerifier',return_value=base), patch('solution.refinement13.ContrastiveVerifier',return_value=wrapped) as ctor:
                self.assertIs(pipeline._build_verifier(cfg,SimpleNamespace()),wrapped)
                ctor.assert_called_once_with(base=base)
            self.assertEqual(cfg.threshold.fixed_value,.5)

    def test_diagnostics_full_denominator_and_missing_stage(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=[dict(question_id='a',conversation='x',question='Q',label='1',prediction='1',gold_start='1',gold_end='5',tiou='1',error_bucket='strong'),dict(question_id='b',conversation='x',question='N',label='1',prediction='0',gold_start='1',gold_end='5',tiou='0',error_bucket='false_negative')]
            with (p/'diagnosis.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            v=verifier();v.refine(['Q','N'],SEGS,BASE,time.monotonic()+60)
            (p/'trace.jsonl').write_text(json.dumps(dict(audio_filename='conversation_x.mp3',questions=['Q','N'],events=v.last_events))+'\n')
            r=refinement_analysis(p);self.assertEqual(r['candidate_oracle_all_gold_positives'],.5);self.assertTrue(r['execution_gate_passed'])
            (p/'trace.jsonl').write_text('');self.assertFalse(refinement_analysis(p)['execution_gate_passed'])
if __name__=='__main__':unittest.main()
