"""Diagnostic entrypoint. Wraps existing API; does not change detection/tracking."""
import os
os.environ['DRONE_TRACE_IMAGES']='1'
import asyncio,base64,hashlib,json,logging,math,queue,threading,time,uuid
from pathlib import Path
import uvicorn
from dtos import MAXIMUM_CENTER_DELTA_PIXELS,ALLOWED_RESOLUTION_LEVELS

ROOT=Path(os.getenv('DRONE_AUDIT_DIR','results/api_audit'))/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
ROOT.mkdir(parents=True)
REVISION='realtime-fusion-v1'


# precision-navigation-v1
from drone_pipeline.precision_camera import identity_only as constrain  # camera legality is decided by CameraShadow in the pipeline

class Journal:
    def __init__(self):
        self.q=queue.Queue(128); self.dropped=0; self.errors=0
        self.t=threading.Thread(target=self.work,daemon=True); self.t.start()
    def add(self,row):
        try:self.q.put_nowait(row)
        except queue.Full:self.dropped+=1; logging.error('AUDIT QUEUE FULL: %s',row.get('event'))
    def work(self):
        with (ROOT/'http.jsonl').open('a',buffering=1) as f:
            while True:
                row=self.q.get()
                try:
                    if row is None:return
                    image=row.pop('_image',None)
                    if image:
                        raw=base64.b64decode(image,validate=True)
                        name=row['call_id']+'.png'; (ROOT/name).write_bytes(raw)
                        row['image']=name; row['image_sha256']=hashlib.sha256(raw).hexdigest()
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                except Exception:self.errors+=1; logging.exception('Audit write failed')
                finally:self.q.task_done()
    def close(self):
        self.q.join(); self.q.put(None); self.t.join()
        (ROOT/'writer_status.json').write_text(json.dumps({'dropped':self.dropped,'errors':self.errors}))


class AuditMiddleware:
    def __init__(self,app,journal):self.app=app; self.journal=journal; self.active=set()
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['path']!='/predict':return await self.app(scope,receive,send)
        call=uuid.uuid4().hex; start=time.perf_counter(); wall=time.time(); messages=[]; body=b''
        concurrent=list(self.active); self.active.add(call)
        try:
            while True:
                msg=await receive(); messages.append(msg)
                if msg['type']=='http.disconnect':return
                body+=msg.get('body',b'')
                if not msg.get('more_body'):break
            body_ms=(time.perf_counter()-start)*1000
            req=json.loads(body); snapshot=json.loads(body); encoded=snapshot.get('view',{}).pop('image',None)
            self.journal.add({'event':'received','call_id':call,'unix_received':wall,'body_received_ms':body_ms,'request':snapshot,
                              'concurrent_calls':concurrent,'request_sha256':hashlib.sha256(body).hexdigest(),'_image':encoded})
            captured=[]
            async def replay():
                if messages:return messages.pop(0)
                return await receive()
            async def capture(msg):captured.append(msg)
            await self.app(scope,replay,capture)
            header=next(m for m in captured if m['type']=='http.response.start')
            original=b''.join(m.get('body',b'') for m in captured if m['type']=='http.response.body')
            audit={}; res=None; final=original
            if header['status']==200:
                res=json.loads(original); res,audit=constrain(req,res)
                final=json.dumps(res,separators=(',',':'),allow_nan=False).encode()
            ready=(time.perf_counter()-start)*1000
            header=dict(header); header['headers']=[(k,v) for k,v in header['headers'] if k.lower()!='content-length'.encode()]+[(b'content-length',str(len(final)).encode()),(b'x-drone-audit',call.encode())]
            self.journal.add({'event':'response_ready','call_id':call,'elapsed_ms':ready,'unix_ready':time.time(),
                              'status':header['status'],'camera_audit':audit,'response':res,
                              'original_response_body':original.decode(errors='replace'),'response_body':final.decode(errors='replace'),'response_sha256':hashlib.sha256(final).hexdigest(),
                              'over_frame_interval':ready>req.get('frame_interval_ms',333),'over_timeout':ready>req.get('response_timeout_ms',3333)})
            await send(header); await send({'type':'http.response.body','body':final,'more_body':False})
            self.journal.add({'event':'send_completed','call_id':call,'elapsed_ms':(time.perf_counter()-start)*1000,'unix_sent':time.time()})
        except Exception as exc:
            self.journal.add({'event':'exception','call_id':call,'elapsed_ms':(time.perf_counter()-start)*1000,'error':repr(exc)})
            raise
        finally:self.active.discard(call)


def main():
    from example import get_pipeline
    from api import app
    p=get_pipeline()
    if not p.cfg.trace.enabled or not p.cfg.trace.images:raise RuntimeError('Enable trace.enabled and trace.images in the selected config')
    files=[Path(__file__),*Path('drone_pipeline').glob('*.py'),Path('api.py')]
    manifest={'revision':REVISION,'pid':os.getpid(),'pipeline_trace':str(p.trace.root.resolve()),
              'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              'note':'ASGI send completion is not organizer receipt or acceptance. Record remote result separately.'}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    @app.get('/diagnostics')
    def diagnostics():return {'revision':REVISION,'pid':os.getpid(),'audit_directory':str(ROOT),'files':manifest['files']}
    journal=Journal(); print('HTTP AUDIT:',ROOT.resolve(),flush=True)
    try:uvicorn.run(AuditMiddleware(app,journal),host='0.0.0.0',port=int(os.getenv('DRONE_PORT','9053')),workers=1)
    finally:journal.close(); p.close()
if __name__=='__main__':main()
