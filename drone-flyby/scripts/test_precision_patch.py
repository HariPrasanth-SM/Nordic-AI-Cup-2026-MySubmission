#!/usr/bin/env python3
"""CPU-only official-camera regressions and synthetic gate behavior tests."""
import sys,os,unittest,random,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['PRECISION_ENABLED']='1'
import numpy as np
from types import SimpleNamespace as NS
from local_evaluator import Camera,build_request
from dtos import DroneFlybyPredictRequestDto
from drone_pipeline.precision_camera import constrain
from drone_pipeline.precision_gate import PrecisionGate
from drone_pipeline.tracker import Track,Observation
from drone_pipeline.geometry import box_measurement
CASES=[(1,1680,540,2,2640,1350),(1,960,540,2,2482,1292),(1,960,1350,1,2160,1620),(2,1669,1295,1,1680,540),(2,2018,1718,1,1680,810),(2,1680,1890,1,1570,1239),(2,1680,1890,1,2423,1444),(2,720,1890,2,1200,1350),(2,1621,1890,1,1200,1350),(2,1200,1350,2,1849,1129),(2,2640,1350,1,1680,1350),(2,1880,1890,1,1916,1031),(1,2160,810,2,2640,1890),(2,2640,810,0,1920,1080),(1,2880,1620,1,1680,1350)]
def req(frame=0,camera=None):return build_request(frame,frame,camera or Camera(1,1920,1080),'',None)
def response(r,level,x,y):return dict(request_id=r['request_id'],frame=r['frame'],annotations=[],requested_view=dict(resolution_level=level,center_x=x,center_y=y))
class CameraTests(unittest.TestCase):
    def check(self,case):
        l,x,y,n,u,v=case;c=Camera(l,x,y);r=req(camera=c);out,a=constrain(r,response(r,n,u,v))
        cmd=out['requested_view']
        self.assertIsNotNone(cmd,case)
        c.apply(**cmd) # Official local evaluator, independently validates command.
        again,_=constrain(r,copy.deepcopy(out));self.assertEqual(again,out)
    def test_reported_violations(self):
        for case in CASES:
            with self.subTest(case=case):self.check(case)
    def test_corner_zoom_out(self):
        self.check((2,480,270,1,960,540));self.check((2,3360,1890,0,1920,1080))
    def test_policy_can_zoom_out_from_corner(self):
        from drone_pipeline.camera import reachable
        r=DroneFlybyPredictRequestDto.model_validate(req(camera=Camera(2,480,270)))
        c=reachable(r,1,[1920,1080]);self.assertIsNotNone(c)
        Camera(2,480,270).apply(**c.model_dump())
    def test_random_moves(self):
        g=random.Random(42)
        for _ in range(2000):
            l=g.randrange(3);hw=1920//2**l;hh=1080//2**l
            self.check((l,g.randint(hw,3840-hw),g.randint(hh,2160-hh),g.randrange(3),g.randint(0,3840),g.randint(0,2160)))
    def test_tighter_request_limit(self):
        r=req();r['camera_constraints']['maximum_center_delta']=10
        out,_=constrain(r,response(r,1,2800,1600));c=out['requested_view']
        self.assertLessEqual(np.hypot(c['center_x']-1920,c['center_y']-1080),10)
    def test_wrong_identity(self):
        r=req();s=response(r,1,1920,1080);s['frame']=100
        out,a=constrain(r,s);self.assertEqual(a['action'],'identity_mismatch');self.assertEqual(out['annotations'],[])
    def test_inconsistent_metadata(self):
        r=req();r['view']['center_x']+=1
        out,_=constrain(r,response(r,1,2000,1100));self.assertIsNone(out['requested_view'])
class GateTests(unittest.TestCase):
    def setUp(self):
        self.g=PrecisionGate(None);self.g.yolo=.55;self.g.dino=.60;self.g.neg=.05;self.g.cls=.03;self.g.hits=2;self.g.revisit=5
        b=np.array([1100,700,1180,780.]);z=box_measurement(b)
        self.t=Track(1,np.r_[z[:2],0.,0.,z[2:]],np.eye(6),np.array([1.,0.]),.95,.9,0.,0.)
        self.d=Observation(b,0,.9,False,np.ones(2),np.array([100,100,180,180.]))
        self.v=dict(cls=0,box=self.d.view_box.tolist(),score=.9,status='verified',target_similarity=.8,background_similarity=.2,best_other_similarity=.3)
    def runframe(self,frame,updated=True,trusted=True):
        self.t.updated=updated
        return self.g.process(DroneFlybyPredictRequestDto.model_validate(req(frame)),[self.d] if updated else [],[self.t],NS(H=np.eye(3),trusted=trusted),frame*.333,.333,[],{'verifier':{'candidates':[self.v]}})
    def test_confirm_then_publish_then_no_ghost(self):
        out,info,focus=self.runframe(0);self.assertEqual(out,[]);self.assertIsNotNone(focus)
        out,_,_=self.runframe(1);self.assertEqual(len(out),1)
        out,info,_=self.runframe(2,updated=False);self.assertEqual(out,[])
        self.assertEqual(info['decisions'][0]['reasons'],['no_fresh_complete_observation'])
    def test_dino_bypass_rejected(self):
        self.v['status']='bypass'
        for i in range(3):self.assertEqual(self.runframe(i)[0],[])
    def test_nan_rejected(self):
        self.v['target_similarity']=float('nan');self.assertEqual(self.runframe(0)[0],[])
    def test_weak_yolo_rejected(self):
        self.d.score=.3;self.v['score']=.3;self.assertEqual(self.runframe(0)[0],[])
    def test_ambiguous_class_rejected(self):
        self.v['best_other_similarity']=.79;self.assertEqual(self.runframe(0)[0],[])
    def test_trajectory_jump_rejected(self):
        self.runframe(0);self.t.x[0]+=100;self.d.box=self.d.box+np.array([100,0,100,0])
        out,info,_=self.runframe(1);self.assertEqual(out,[]);self.assertIn('trajectory_mismatch',info['decisions'][0]['reasons'])
    def test_lost_registration_restarts_confirmation(self):
        self.runframe(0);self.assertEqual(self.runframe(1,trusted=False)[0],[])
        self.assertEqual(self.runframe(2)[0],[]);self.assertEqual(len(self.runframe(3)[0]),1)
    def test_five_frame_revisit_and_bounded_attempts(self):
        self.runframe(0);self.runframe(1)
        self.assertIsNone(self.runframe(5,updated=False)[2]);self.assertIsNotNone(self.runframe(6,updated=False)[2])
        self.assertIsNone(self.runframe(7,updated=False)[2]);self.runframe(8,updated=False);self.runframe(10,updated=False)
        self.assertIsNone(self.runframe(12,updated=False)[2])
class PipelineIntegrationTests(unittest.TestCase):
    def test_actual_pipeline_cannot_bypass_gate_with_fresh_exports(self):
        from drone_pipeline.pipeline import Pipeline
        from drone_pipeline.config import PipelineConfig
        from drone_pipeline.detector import Detection
        from drone_pipeline.motion import Motion
        from utils import encode_image
        class Detector:
            def detect(self,image):
                self.last_info={'verifier':{'candidates':[dict(cls=0,box=[100,100,180,180],score=.9,status='verified',target_similarity=.8,background_similarity=.2,best_other_similarity=.3)]}}
                return [Detection(np.array([100,100,180,180.]),0,.9)]
        class Registration:
            def estimate(self,*a,**kw):return Motion(np.eye(3),True,0.,{'reason':'synthetic'})
        cfg=PipelineConfig();cfg.trace.enabled=False;cfg.export_fresh=True
        pipeline=Pipeline(cfg,Detector())
        encoded=encode_image(np.zeros((540,960,3),np.uint8))
        def request(n):
            r=req(n);r['view']['image']=encoded
            return DroneFlybyPredictRequestDto.model_validate(r)
        first=pipeline.predict(request(0));self.assertEqual(first.annotations,[])
        session=next(iter(pipeline.sessions.values()))
        # Instance method name deliberately checked against runtime below.
        session.motion=Registration()
        second=pipeline.predict(request(1));self.assertEqual(len(second.annotations),1)
        duplicate=pipeline.predict(request(1));self.assertEqual(second,duplicate)
        pipeline.close()
if __name__=='__main__':unittest.main(verbosity=2)
