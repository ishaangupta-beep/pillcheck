"""AWS Lambda entry point.

One function serves three kinds of events:
  * HTTP requests from the Lambda Function URL -> the FastAPI app (via Mangum);
  * {"pillcheck": "tick"} from EventBridge Scheduler every 15 minutes ->
    reminders, missed-dose alerts and refill checks for every real patient;
  * {"pillcheck": "warm"} -> loads the OpenCV 5 models (optional pre-warming).
"""
from __future__ import annotations

import json
import logging
import time

from mangum import Mangum

from . import service
from .api.main import app
from .storage import get_store
from .vision.models import get_engine

log = logging.getLogger("pillcheck.lambda")
_http = Mangum(app, lifespan="off")


def handler(event, context):
    if isinstance(event, dict) and "pillcheck" in event:
        task = event["pillcheck"]
        t0 = time.perf_counter()
        if task == "tick":
            runs = service.tick_all(get_store(), event.get("at"))
            out = {"task": "tick", "runs": len(runs)}
        elif task == "warm":
            out = {"task": "warm", "engine": get_engine().name}
        else:
            out = {"error": f"unknown task {task!r}"}
        out["elapsed_ms"] = round(1000 * (time.perf_counter() - t0))
        log.info(json.dumps({"event": "scheduled_task", **out}))
        return out
    return _http(event, context)
