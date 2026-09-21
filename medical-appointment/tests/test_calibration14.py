import os,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
from solution.calibration14 import calibrate,CalibratedVerifier
from solution.types import Span,VerifierResult
CFG=dict(base_experiment='exp13',start_fraction=.1,end_fraction=0)
class CalibrationTests(unittest.TestCase):
    def test_calibrates_and_preserves_decisions(self):
        rs=[VerifierResult(1,Span(10,20),.6,'quote'),VerifierResult(0,None,0),VerifierResult(1,None,0)]
        out=calibrate(rs,CFG);self.assertEqual(out[0].span,Span(11,20));self.assertEqual([r.p_yes for r in out],[1,0,1]);self.assertIs(out[1],rs[1]);self.assertIs(out[2],rs[2]);self.assertEqual(out[0].quote,'quote')
    def test_tiny_spans_unchanged(self):
        r=VerifierResult(1,Span(1,1.01),.6);self.assertIs(calibrate([r],CFG)[0],r)
    def test_invalid_settings_rejected(self):
        for a in [-.1,.3,float('nan')]:
            with self.assertRaises(ValueError):calibrate([],dict(CFG,start_fraction=a))
    def test_audio_hash_propagates(self):
        b=SimpleNamespace(verify_batch=lambda *args:[VerifierResult(1,Span(0,10),.6)])
        v=CalibratedVerifier(b,CFG);v.audio_hash='abc';r=v.verify_batch([],[],[],None,time.monotonic()+3)
        self.assertEqual(b.audio_hash,'abc');self.assertEqual(r[0].span,Span(1,10))
    def test_pipeline_wraps_exp13_once(self):
        with patch.dict(os.environ,{'MEDICAL_APPT_SKIP_MODEL_LOAD':'1','MEDICAL_APPT_EXPERIMENT':'exp14'}):
            from solution.config import load_config
            import solution.pipeline as p
            b=SimpleNamespace(client=SimpleNamespace(warmup=lambda:None));refined=object()
            with patch('solution.grounding12.AnchoredVerifier',return_value=b),patch('solution.refinement13.ContrastiveVerifier',return_value=refined),patch('solution.calibration14.load_settings',return_value=CFG):
                v=p._build_verifier(load_config('workstation_32gb'),SimpleNamespace())
            self.assertIsInstance(v,CalibratedVerifier);self.assertIs(v.base,refined)
if __name__=='__main__':unittest.main()
