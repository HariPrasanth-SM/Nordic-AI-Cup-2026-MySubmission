"""Exp15: preserve Exp14 onset calibration; correct a small early end bias."""
import json,os
from pathlib import Path
from dataclasses import asdict
from solution.calibration14 import calibrate,validate
from solution.types import Span,VerifierResult
from solution.experiment_trace import event
ROOT=Path(__file__).resolve().parents[1]
def load_settings():
    p=Path(os.environ.get('MEDICAL_EXP15_CONFIG',str(ROOT/'configs/exp15_calibration.json')))
    cfg=json.loads(p.read_text());validate(cfg)
    if not 0<=cfg['end_offset_s']<=.2:raise ValueError('End offset must be between 0 and 0.2 seconds')
    return cfg

def extend_end(results,cfg):
    offset=cfg['end_offset_s']
    if not isinstance(offset,(int,float)) or not 0<=offset<=.2:raise ValueError('Invalid end offset')
    out=[]
    for r in results:
        if r.p_yes<=.5 or r.span is None:out.append(r);continue
        out.append(VerifierResult(r.p_yes,Span(r.span.start,r.span.end+offset),r.expected_tiou,r.quote))
    return out

class CalibratedVerifier:
    def __init__(self,base,settings=None):
        self.base=base;self.settings=settings or load_settings();self.audio_hash=''
    def verify_batch(self,questions,segments,windows,vectors,deadline):
        self.base.audio_hash=self.audio_hash
        base=self.base.verify_batch(questions,segments,windows,vectors,deadline)
        old=calibrate(base,self.settings)
        event('exp15_baseline',results=[asdict(r) for r in old])
        out=extend_end(old,self.settings)
        event('exp15_calibration',config=self.settings,changed=sum(a.span!=b.span for a,b in zip(old,out)),
              classification_changes=sum(a.p_yes!=b.p_yes for a,b in zip(old,out)))
        return out
