import os,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
from solution.calibration15 import extend_end,CalibratedVerifier
from solution.types import Span,VerifierResult
CFG=dict(base_experiment='exp13',start_fraction=.1,end_fraction=0,end_offset_s=.1)
class EndCalibrationTests(unittest.TestCase):
    def test_only_end_changes(self):
        x=VerifierResult(1,Span(11,20),.6,'text');y=extend_end([x],CFG)[0]
        self.assertEqual(y.span,Span(11,20.1));self.assertEqual(y.p_yes,x.p_yes)
    def test_no_and_missing_preserved(self):
        rs=[VerifierResult(0,None,0),VerifierResult(1,None,0)]
        out=extend_end(rs,CFG);self.assertIs(out[0],rs[0]);self.assertIs(out[1],rs[1])
    def test_live_and_cached_apply_start_once(self):
        b=SimpleNamespace(verify_batch=lambda *a:[VerifierResult(1,Span(10,20),.6)])
        v=CalibratedVerifier(b,CFG);v.audio_hash='abc';live=v.verify_batch([],[],[],None,time.monotonic()+2)
        cached=extend_end([VerifierResult(1,Span(11,20),.6)],CFG)
        self.assertEqual(live[0].span,cached[0].span);self.assertEqual(b.audio_hash,'abc')
    def test_invalid_offset(self):
        for d in [-.1,.3,float('nan')]:
            with self.assertRaises(ValueError):extend_end([],dict(CFG,end_offset_s=d))
    def test_pipeline_uses_new_wrapper(self):
        with patch.dict(os.environ,{'MEDICAL_APPT_SKIP_MODEL_LOAD':'1','MEDICAL_APPT_EXPERIMENT':'exp15'}):
            from solution.config import load_config
            import solution.pipeline as p
            b=SimpleNamespace(client=SimpleNamespace(warmup=lambda:None));refined=object()
            with patch('solution.grounding12.AnchoredVerifier',return_value=b),patch('solution.refinement13.ContrastiveVerifier',return_value=refined),patch('solution.calibration15.load_settings',return_value=CFG):
                v=p._build_verifier(load_config('workstation_32gb'),SimpleNamespace())
            self.assertIsInstance(v,CalibratedVerifier);self.assertIs(v.base,refined)
if __name__=='__main__':unittest.main()
