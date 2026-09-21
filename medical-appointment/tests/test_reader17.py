import time,unittest
from types import SimpleNamespace
import numpy as np
from solution.types import Word,Span,VerifierResult,Segment
from solution.span17 import geometry,decode,supervision,windows
from solution.reader17 import ReaderVerifier
CFG=dict(max_words=10,max_duration_s=20,start_fraction=.1,end_offset_s=.1,margin=2.,min_null_margin=2.,max_baseline_projection_error_s=.5,min_remaining_s=4,max_length=30,stride=3)
WORDS=[Word(x,i,i+1) for i,x in enumerate(['Hello.','Take','10','mg','daily.','Goodbye.'])]
FEATURE=dict(word_ids=list(range(6)),start_tokens=list(range(1,7)),end_tokens=list(range(1,7)),cls=0)
class Tests(unittest.TestCase):
    def test_calibration_once_and_valid_order(self):
        s,e,v=geometry(WORDS,FEATURE,CFG);self.assertAlmostEqual(s[1,4],1.4);self.assertAlmostEqual(e[1,4],5.1);self.assertFalse(v[4,1])
    def test_supervision_mass_on_target(self):
        t=supervision(WORDS,FEATURE,CFG,(1.4,5.1));self.assertFalse(t['null']);self.assertAlmostEqual(t['oracle'],1.)
        self.assertEqual(np.argmax(t['start']),1);self.assertEqual(np.argmax(t['end']),4)
        self.assertAlmostEqual(sum(t['start']),1.)
    def test_off_window_null(self):self.assertTrue(supervision(WORDS,FEATURE,CFG,(20,22))['null'])
    def test_high_margin_switch(self):
        s=np.zeros(8);e=np.zeros(8);s[2]=5;e[5]=5
        d=decode(WORDS,[FEATURE],[(s,e)],CFG,(.6,6.1));self.assertTrue(d['accepted']);self.assertEqual(d['candidate']['start'],1.4)
    def test_low_margin_keep(self):
        s=np.zeros(8);e=np.zeros(8);s[2]=.4;e[5]=.4
        self.assertFalse(decode(WORDS,[FEATURE],[(s,e)],CFG,(.6,6.1))['accepted'])
    def test_projection_missing_does_not_force_switch(self):
        s=np.ones(8)*10;e=np.ones(8)*10;s[0]=e[0]=0
        self.assertFalse(decode(WORDS,[FEATURE],[(s,e)],CFG,(99,100))['accepted'])
    def test_wrapper_failure_preserves_actual_exp15(self):
        class Fake:
            cfg=CFG
            def infer(self,*a):raise RuntimeError('fixture')
        base=[VerifierResult(1,Span(2.1,4.1),.7,'10 mg'),VerifierResult(0,None,0)]
        v=ReaderVerifier(None,Fake());out=v.refine(['Q','N'],[Segment(0,6,'',WORDS)],base,time.monotonic()+20)
        self.assertIs(out,base)
    def test_wrapper_frozen_classification(self):
        class Fake:
            cfg=CFG
            def infer(self,*a):return [dict(accepted=True,candidate=dict(start=1.4,end=5.1,text='Take 10 mg daily.'),candidates=[])]
        base=[VerifierResult(1,Span(2.1,4.1),.7,'10 mg'),VerifierResult(0,None,0)]
        out=ReaderVerifier(None,Fake()).refine(['Q','N'],[Segment(0,6,'',WORDS)],base,time.monotonic()+20)
        self.assertEqual([x.p_yes for x in out],[1,0]);self.assertIs(out[1],base[1]);self.assertEqual(out[0].span,Span(1.4,5.1))
    def test_token_mapping_without_optional_ml_dependencies(self):
        class Encoding(dict):
            def sequence_ids(self,n):return [None,0,None,1,1,1,None]
        class Tokenizer:
            model_input_names=['input_ids','attention_mask'];cls_token_id=0
            def __call__(self,*a,**kw):return Encoding(input_ids=[[0,2,3,4,5,6,3]],attention_mask=[[1]*7],offset_mapping=[[(0,0),(0,1),(0,0),(0,3),(4,6),(7,9),(0,0)]])
        f=windows(Tokenizer(),'Q',[Word('one',0,1),Word('22',1,2),Word('mg',2,3)],CFG)[0]
        self.assertEqual(f['word_ids'],[0,1,2]);self.assertEqual(f['start_tokens'],[3,4,5])
if __name__=='__main__':unittest.main()
