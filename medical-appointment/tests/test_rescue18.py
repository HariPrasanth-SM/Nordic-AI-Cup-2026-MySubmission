import os,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
from solution.rescue18 import RescueVerifier,eligibility,distinct
from solution.types import Word,Segment,Span,VerifierResult
CFG=dict(max_questions=2,max_alternatives=2,max_rescue_s=20,min_remaining_s=4,context_segments=1,max_prompt_chars=44000)
CAL=dict(base_experiment='exp13',start_fraction=.1,end_fraction=0,end_offset_s=.1)
SEG=[Segment(0,3,'Pain yesterday',[Word('Pain',0,1),Word('yesterday',1,3)]),Segment(10,13,'Pain today',[Word('Pain',10,11),Word('today',11,13)])]
BASE=[VerifierResult(1,Span(.3,3.1),.6,'Pain yesterday')]
POOLS={0:[dict(start=10,end=13,text='Pain today')]}
class Fake:
    def __init__(self,mode='switch'):self.mode=mode;self.calls=0;self.last_response={}
    def call(self,messages,schema,deadline,max_tokens=None):
        import json
        self.calls+=1;p=json.loads(messages[1]['content'])
        if self.mode=='fail':raise TimeoutError('mock')
        if self.calls==1:return {'0':[dict(source='S001',quote='Pain today')]}
        if self.mode=='late_fail' and self.calls==3:raise TimeoutError('mock')
        out={}
        for k,pair in p.items():
            winner='A' if pair['A']['marked_evidence']=='Pain today' else 'B'
            d=dict(verdict=winner+'_better',reason='wrong_time',a_supports=winner=='A',b_supports=winner=='B',
                   a_evidence=pair['A']['marked_evidence'],b_evidence=pair['B']['marked_evidence'])
            if self.mode=='both':d.update(verdict='both_valid',a_supports=True,b_supports=True)
            if self.mode=='unanchored':d['a_evidence']='invented quotation'
            if self.mode=='reverse_disagrees' and self.calls==3:d['verdict']='uncertain'
            if self.mode=='different_reason' and self.calls==3:d['reason']='wrong_subject'
            out[k]=d
        return out
class Tests(unittest.TestCase):
    def run_case(self,mode):
        f=Fake(mode);v=RescueVerifier(client=f,config=CFG,calibration=CAL)
        result=v.rescue(['Does the patient have pain today?'],SEG,BASE,time.monotonic()+30,POOLS)
        return result,v,f
    def test_replacement_requires_two_consistent_orders(self):
        result,v,f=self.run_case('switch');self.assertEqual(f.calls,3)
        self.assertEqual(result[0].span,Span(10.3,13.1));self.assertEqual(result[0].p_yes,BASE[0].p_yes)
    def test_keep_modes_preserve_actual_exp15_object(self):
        for mode in ['fail','late_fail','both','unanchored','reverse_disagrees','different_reason']:
            with self.subTest(mode=mode):self.assertIs(self.run_case(mode)[0][0],BASE[0])
    def test_same_occurrence_is_not_alternative(self):
        self.assertFalse(distinct(dict(start=1,end=5),dict(start=2,end=4)))
    def test_no_trigger_no_calls(self):
        f=Fake();v=RescueVerifier(client=f,config=CFG,calibration=CAL)
        self.assertIs(v.rescue(['Fever?'],SEG,BASE,time.monotonic()+30,{})[0],BASE[0]);self.assertEqual(f.calls,0)
    def test_negative_missing_span_not_reviewed(self):
        b=[VerifierResult(0,None,0),VerifierResult(1,None,.5)]
        self.assertEqual(eligibility(['Pain?']*2,SEG,b,POOLS),[])
    def test_expired_deadline_keeps(self):
        f=Fake();v=RescueVerifier(client=f,config=CFG,calibration=CAL)
        self.assertIs(v.rescue(['Pain today?'],SEG,BASE,time.monotonic(),POOLS)[0],BASE[0]);self.assertEqual(f.calls,0)
    def test_runtime_not_dependent_on_trace(self):
        with patch.dict(os.environ,{},clear=True):self.assertNotEqual(self.run_case('switch')[0][0].span,BASE[0].span)
    def test_pipeline_builds_exp15_then_rescue_without_reader(self):
        with patch.dict(os.environ,{'MEDICAL_APPT_SKIP_MODEL_LOAD':'1','MEDICAL_APPT_EXPERIMENT':'exp18'}):
            from solution.config import load_config
            import solution.pipeline as p
            b=SimpleNamespace(client=SimpleNamespace(warmup=lambda:None));refined=object()
            with patch('solution.grounding12.AnchoredVerifier',return_value=b),patch('solution.refinement13.ContrastiveVerifier',return_value=refined),patch('solution.calibration15.load_settings',return_value=CAL):
                v=p._build_verifier(load_config('workstation_32gb'),SimpleNamespace())
            self.assertIsInstance(v,RescueVerifier);self.assertIs(v.base.base,refined)
    def test_live_receives_exp13_seed_pool(self):
        base=SimpleNamespace(base=SimpleNamespace(last_events=[dict(stage='exp13_candidates',index=0,candidates=POOLS[0])]),verify_batch=lambda *args:BASE)
        v=RescueVerifier(base=base,client=Fake(),config=CFG,calibration=CAL)
        self.assertEqual(v.verify_batch(['Pain today?'],SEG,[],None,time.monotonic()+30)[0].span,Span(10.3,13.1))
if __name__=='__main__':unittest.main()
