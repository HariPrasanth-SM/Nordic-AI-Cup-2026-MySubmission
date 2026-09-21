"""Bounded background writer. Drop optional images first; never block inference."""
import atexit
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import queue
import threading
import time
import cv2

log=logging.getLogger(__name__)

class TraceWriter:
    def __init__(self,cfg,full_config):
        self.cfg=cfg; self.dropped_images=0; self.dropped_records=0; self.errors=0
        self.root=Path(cfg.directory)/time.strftime('%Y%m%d-%H%M%S')
        self.root=self.root.with_name(self.root.name+'-'+str(time.time_ns()%1000000))
        self.q=queue.Queue(cfg.queue_size)
        self.thread=None
        if not cfg.enabled: return
        self.root.mkdir(parents=True,exist_ok=True)
        packages={}
        for package in ('numpy','scipy','opencv-python','opencv-python-headless','ultralytics','torch','pydantic'):
            try: packages[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        hashes={}
        for field in ('weights','weights_l1','weights_l2'):
            weights=Path(getattr(full_config.detector,field))
            if weights.is_file():
                h=hashlib.sha256()
                with weights.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
                hashes[field]={'path':str(weights),'sha256':h.hexdigest()}
        (self.root/'manifest.json').write_text(json.dumps({'config':full_config.model_dump(),
            'packages':packages,'weights':hashes,'created_unix':time.time()},indent=2))
        self.thread=threading.Thread(target=self._worker,daemon=True,name='trace-writer'); self.thread.start()
        atexit.register(self.close)
        log.info('Trace directory: %s',self.root)

    def submit(self,record,image):
        if not self.cfg.enabled: return
        # Snapshot bytes/JSON; worker never owns mutable tracker state.
        record=json.loads(json.dumps(record,allow_nan=False))
        wants=self.cfg.images and record['frame_index']%self.cfg.every==0
        if wants and self.q.qsize()>self.cfg.queue_size*.65:
            wants=False; self.dropped_images+=1
        record['trace_dropped_images']=self.dropped_images
        record['trace_dropped_records']=self.dropped_records
        try: self.q.put_nowait((record,image.copy() if wants else None))
        except queue.Full: self.dropped_records+=1

    def _worker(self):
        with (self.root/'trace.jsonl').open('a',buffering=1) as f:
            while True:
                item=self.q.get()
                try:
                    if item is None: return
                    record,image=item
                    if image is not None:
                        filename=f"{record['session_key']}_{record['frame_index']:06d}_{record['frame']:06d}.png"
                        path=self.root/'frames'/filename; path.parent.mkdir(exist_ok=True)
                        if not cv2.imwrite(str(path),image): raise IOError(f'Could not save {path}')
                        record['image']='frames/'+filename
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                except Exception:
                    self.errors+=1; log.exception('Trace writer failed')
                finally: self.q.task_done()

    def close(self):
        if self.thread is None: return
        self.q.join(); self.q.put(None); self.thread.join(timeout=5); self.thread=None
        (self.root/'writer_status.json').write_text(json.dumps({'dropped_images':self.dropped_images,
            'dropped_records':self.dropped_records,'writer_errors':self.errors},indent=2))
