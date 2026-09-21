"""The file api.py imports predict() from -- README calls this "the file to
replace". A thin shim into solution/worker_client.py, which spawns and
talks to a dedicated GPU worker process rather than doing GPU work in this
(FastAPI/uvicorn) process directly -- see solution/gpu_worker.py's
docstring for why. solution/pipeline.py has the actual model/prediction
logic and runs inside that separate worker process, never here.
"""

from solution.worker_client import predict

__all__ = ["predict"]
