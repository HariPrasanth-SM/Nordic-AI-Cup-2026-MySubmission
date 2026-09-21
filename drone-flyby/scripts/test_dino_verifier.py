#!/usr/bin/env python3
"""Fast logic tests; fake encoder, no downloaded model or GPU required."""
import importlib.util,sys
from pathlib import Path
import numpy as np
# Load standalone module so tests can run even before overlay installation.
p=Path(__file__).resolve().parents[1]/'drone_pipeline/dino_verifier.py'
s=importlib.util.spec_from_file_location('verifier_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
red=np.zeros((12,20,3),np.uint8);red[:,:,2]=255
z=m.prepare(red);assert z.shape==(3,224,224) and z[0,112,112]>z[2,112,112]
assert m.crop(red,[-5,-5,10,10]).size>0
refs=np.eye(3,dtype=np.float32);labels=np.array([-1,0,1]);v=m.scores(refs,refs,labels,2)
assert m.decision(v[1],0,.55,.03,0)[0]
assert not m.decision(v[0],0,.55,.03,0)[0]
assert not m.decision(v[2],0,.55,.03,0)[0]
from types import SimpleNamespace
class Base:
 last_info={'skipped':False}
 def detect(self,im,level):return self.dets
class Enc:
 def encode(self,images):return np.array([[1.,0,0],[0,1.,0]],np.float32)[:len(images)]
a=m.VerifiedDetector.__new__(m.VerifiedDetector);a.base=Base();a.encoder=Enc();a.refs=refs;a.labels=labels;a.names=['target','other'];a.maximum=48;a.minimum=.55;a.negmargin=.03;a.classmargin=0.;a.banksha='test';a.encoder.sha='test'
a.base.dets=[SimpleNamespace(box=np.array([10,10,30,30]),cls=0,score=.9,partial=False),SimpleNamespace(box=np.array([40,40,60,60]),cls=0,score=.8,partial=False),SimpleNamespace(box=np.array([0,0,4,4]),cls=0,score=.7,partial=True)]
a.mode='shadow';im=np.zeros((100,100,3),np.uint8);assert len(a.detect(im,1))==3
assert not a.last_info['verifier']['candidates'][0]['would_keep']
a.mode='filter';kept=a.detect(im,1);assert len(kept)==2 and kept[0] is a.base.dets[1]
a.maximum=1;assert len(a.detect(im,1))==2 # unverified overflow passes, never silently drops
class Bad:
 sha='test'
 def encode(self,images):raise RuntimeError('fixture error')
a.encoder=Bad();assert len(a.detect(im,1))==3 and a.last_info['verifier']['error']
print('PASS: color, crop, normalization shape, target/background/competitor rejection, shadow/filter, partial/budget bypass, fail-open error logging.')
