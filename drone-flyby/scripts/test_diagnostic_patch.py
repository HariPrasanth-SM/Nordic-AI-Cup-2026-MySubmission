"""Run from repo root: python scripts/test_diagnostic_patch.py"""
import asyncio,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from api_diagnostic import constrain,AuditMiddleware
from local_evaluator import Camera,build_request
CASES=[(2,1637,1634,1,2880,1236),(2,1680,1890,2,2438,1634),(2,1200,1890,1,1873,1110),(2,2046,1890,1,1470,1217),(2,1200,1890,1,2592,1005),(2,1200,1890,1,1680,1350),(1,1680,810,1,2761,1421),(1,2348,1620,1,960,810),(2,1645,1670,1,1680,540),(2,1670,1890,1,1316,1231),(1,1200,810,2,2109,1461),(1,1200,1620,1,2312,1620),(1,1200,810,2,1516,1890),(1,960,810,1,2655,959),(2,2640,1890,1,2586,1170)]
import math
for lev,x,y,dest,tx,ty in CASES:
 req=build_request(0,0,Camera(lev,x,y),'',None)
 original={'request_id':req['request_id'],'frame':0,'annotations':[], 'requested_view':{'resolution_level':dest,'center_x':tx,'center_y':ty}}
 res,audit=constrain(req,original)
 cmd=res['requested_view']; assert cmd is not None
 assert math.hypot(cmd['center_x']-x,cmd['center_y']-y)<=req['camera_constraints']['maximum_center_delta']-2
 assert audit['action']=='projected_to_reachable'
req=build_request(1,1,Camera(2,1200,1890),'',None)
bad={'request_id':'wrong','frame':1,'annotations':[{}]}
fixed,a=constrain(req,bad); assert not fixed['annotations'] and a['action']=='response_identity_mismatch'
class Journal:
 def __init__(self):self.rows=[]
 def add(self,row):self.rows.append(row)
async def mock_app(scope,receive,send):
 await receive()
 res={'request_id':req['request_id'],'frame':1,'annotations':[{'object_id':'tank','bbox':[.1,.1,.2,.2],'confidence':.9}], 'requested_view':{'resolution_level':1,'center_x':2880,'center_y':540}}
 await send({'type':'http.response.start','status':200,'headers':[(b'content-length',b'999')]})
 await send({'type':'http.response.body','body':json.dumps(res).encode()})
async def check():
 j=Journal(); app=AuditMiddleware(mock_app,j); messages=[{'type':'http.request','body':json.dumps(req).encode()}]; sent=[]
 async def receive():return messages.pop(0)
 async def send(msg):sent.append(msg)
 await app({'type':'http','path':'/predict'},receive,send)
 res=json.loads(sent[1]['body']); assert len(res['annotations'])==1
 assert int(dict(sent[0]['headers'])[b'content-length'])==len(sent[1]['body'])
 assert [r['event'] for r in j.rows]==['received','response_ready','send_completed']
 assert not app.active
asyncio.run(check())
print('PASS: all 15 reported moves corrected; wire payload, identity, annotations, timing stages and content length checked.')
