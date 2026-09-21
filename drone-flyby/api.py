"""Official routes and DTOs, with model warmup before accepting requests and a response deadline."""
import concurrent.futures
import datetime
import logging
import os
import time
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from example import predict,get_pipeline
from utils import validate_response
from drone_pipeline.precision_camera import identity_only
# realtime-fusion-v1
try:
    import ultralytics
    ULTRALYTICS=ultralytics.__version__
except ImportError:
    ULTRALYTICS=None

logging.basicConfig(level=logging.INFO)
start_time=time.time()
REVISION='realtime-fusion-v1'
CODE_SHA256=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.joinpath('drone_pipeline').glob('*.py')))).hexdigest()
# One worker: the pipeline owns GPU state and is serialized by its own lock anyway. The endpoint thread only waits,
# so it can answer with a valid fallback at the deadline even if the worker is stuck.
_WORKER=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='predict')

@asynccontextmanager
async def lifespan(app):
    pipeline=get_pipeline()  # fail fast on missing weights or incompatible class names
    yield
    pipeline.close()

app=FastAPI(lifespan=lifespan)

def deadline_seconds(request):
    """Leave room for network latency: never hold a response past the request's own budget minus 700 ms."""
    configured=int(os.getenv('DRONE_DEADLINE_MS',get_pipeline().cfg.deadline_ms))
    return max(.1,min(configured,request.response_timeout_ms-700))/1000

@app.post('/predict',response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    pipeline=get_pipeline()
    try:
        future=_WORKER.submit(predict,request)
        try:
            response=future.result(timeout=deadline_seconds(request))
        except concurrent.futures.TimeoutError:
            future.cancel(); pipeline.abandon(request)
            logging.error('DEADLINE: %s frame=%s not ready in time; sending fallback',request.request_id,request.frame)
            response=pipeline.fallback(request)
        checked,_=identity_only(request.model_dump(exclude={'view':{'image'}}),response.model_dump())
        response=DroneFlybyPredictResponseDto.model_validate(checked)
        validate_response(response)
        return response
    except Exception:
        logging.exception('Prediction failed for %s; returning a valid empty response',request.request_id)
        if pipeline.lock.acquire(timeout=.5):   # never block the reply behind a stuck worker
            try: pipeline.sessions.pop(request.sequence_id,None)
            finally: pipeline.lock.release()
        return DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,annotations=[])

@app.get('/api')
def hello():
    return {'service':'drone-flyby-usecase','revision':REVISION,'code_sha256':CODE_SHA256,
            'ultralytics':ULTRALYTICS,'uptime':str(datetime.timedelta(seconds=time.time()-start_time))}

@app.get('/')
def index(): return 'Your endpoint is running!'

if __name__=='__main__':
    # Exactly one worker: state and the GPU model belong to this process.
    uvicorn.run(app,host='0.0.0.0',port=int(os.getenv('DRONE_PORT','9053')),workers=1)
