"""Exp14: continuous endpoint calibration; classification and API unchanged."""
import json,os
from dataclasses import asdict
from pathlib import Path
from solution.types import Span,VerifierResult
from solution.experiment_trace import event
ROOT=Path(__file__).resolve().parents[1]

def load_settings():
    p=Path(os.environ.get('MEDICAL_EXP14_CONFIG',str(ROOT/'configs/exp14_calibration.json')))
    cfg=json.loads(p.read_text());validate(cfg);return cfg

def validate(cfg):
    if cfg['base_experiment'] not in {'exp12','exp13'}:raise ValueError('Base must be exp12 or exp13')
    a,b=cfg['start_fraction'],cfg['end_fraction']
    if not isinstance(a,(int,float)) or not isinstance(b,(int,float)) or not 0<=a<=.2 or not 0<=b<=.2:
        raise ValueError('Calibration fractions must lie in [0, 0.2]')

def calibrate(results,cfg):
    validate(cfg);out=[]
    for r in results:
        if r.p_yes<=.5 or r.span is None:out.append(r);continue
        s,e=r.span.start,r.span.end;length=e-s
        # Tiny spans remain unchanged; avoid collapse after API rounding.
        if length<=.03:out.append(r);continue
        out.append(VerifierResult(r.p_yes,Span(s+cfg['start_fraction']*length,
                    e-cfg['end_fraction']*length),r.expected_tiou,r.quote))
    return out

class CalibratedVerifier:
    def __init__(self,base,settings=None):
        self.base=base;self.settings=settings or load_settings();validate(self.settings);self.audio_hash=''
    def verify_batch(self,questions,segments,windows,vectors,deadline):
        self.base.audio_hash=self.audio_hash
        old=self.base.verify_batch(questions,segments,windows,vectors,deadline)
        event('exp14_baseline',results=[asdict(r) for r in old])
        out=calibrate(old,self.settings)
        event('exp14_calibration',config=self.settings,changed=sum(a.span!=b.span for a,b in zip(old,out)),
              classification_changes=sum(a.p_yes!=b.p_yes for a,b in zip(old,out)))
        return out
