"""PillCheck HTTP API (FastAPI). Runs under uvicorn locally and inside AWS Lambda.

Run locally:  uvicorn pillcheck.api.main:app --reload
Docs:         http://localhost:8000/docs

Access model
  * PILLCHECK_API_TOKEN unset (local development): everything is open.
  * Token set (deployed): the token holder (admin) can do everything. Without
    the token, visitors can only start demo sessions and act on demo patients,
    within per-IP and global rate limits. Real patients are never reachable
    without the token.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import __version__, demo, service
from ..config import BACKEND_ROOT, settings
from ..storage import get_store
from ..vision.models import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("pillcheck.api")

app = FastAPI(title="PillCheck API", version=__version__,
              description="Camera-verified medication adherence: OpenCV 5 perception + an agent that acts on it.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Elapsed-Ms"])


# ---------------------------------------------------------------- access & limits


@dataclass
class Caller:
    admin: bool
    ip: str


def caller(request: Request, authorization: str | None = Header(default=None),
           token: str | None = Query(default=None)) -> Caller:
    """Bearer token, or `?token=` (for <img> tags)."""
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    expected = settings.api_token
    if not expected:
        return Caller(True, ip)
    given = (authorization or "").removeprefix("Bearer ").strip() or (token or "")
    return Caller(bool(given) and secrets.compare_digest(given, expected), ip)


def require_admin(c: Caller = Depends(caller)) -> Caller:
    if not c.admin:
        raise HTTPException(401, "this endpoint needs the admin token")
    return c


def guard(c: Caller, patient_id: str | None) -> dict:
    """Admins reach every patient; visitors only demo patients."""
    if not patient_id:
        raise HTTPException(404, "not found")
    patient = service._get(get_store(), "patients", patient_id)
    if not c.admin and not patient.get("demo"):
        raise HTTPException(401, "this patient needs the admin token")
    return patient


def guard_doc(c: Caller, collection: str, doc_id: str) -> dict:
    doc = service._get(get_store(), collection, doc_id)
    guard(c, doc.get("patient_id"))
    return doc


def limit(c: Caller, name: str, per_hour: int) -> None:
    if c.admin:
        return
    hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    if get_store().incr(f"rl_{name}_{c.ip}_{hour}", 2 * 3600) > per_hour:
        raise HTTPException(429, f"Too many requests. The demo allows {per_hour} of these per hour; please try again later.")


def global_check_budget() -> None:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    if get_store().incr(f"global_checks_{day}", 2 * 86400) > settings.limits.checks_per_day_global:
        raise HTTPException(429, "The public demo has reached today's photo limit. Please try again tomorrow.")


# ---------------------------------------------------------------- plumbing


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round(1000 * (time.perf_counter() - t0))
    response.headers["X-Elapsed-Ms"] = str(ms)
    log.info('{"event": "http", "method": "%s", "path": "%s", "status": %d, "ms": %d}',
             request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(service.NotFound)
async def _nf(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(service.Conflict)
async def _cf(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(ValueError)
async def _ve(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


async def read_image(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    max_bytes = int(settings.limits.max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(413, f"image too large (max {settings.limits.max_upload_mb:g} MB); resize it before uploading")
    return data


def blob_response(key: str | None, media_type: str) -> Response:
    data = get_store().get_blob(key) if key else None
    if data is None:
        raise HTTPException(404, "image not found")
    return Response(data, media_type=media_type, headers={"Cache-Control": "private, max-age=86400"})


# ---------------------------------------------------------------- schemas


class Caregiver(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""


class PatientIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    language: str = Field(default="en", pattern="^(en|hi)$")
    timezone: str | None = None
    caregiver: Caregiver = Caregiver()


class ScheduleEntry(BaseModel):
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["09:00"])
    pills: int = Field(default=1, ge=1, le=4)


class MedicationIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    strength: str = ""
    instructions: str = ""
    schedule: list[ScheduleEntry] = Field(min_length=1)


class PocketIn(BaseModel):
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float = 0.0
    initially_empty: bool = False


class ConfirmIn(BaseModel):
    pockets: list[PocketIn] | None = Field(default=None, description="Omit to accept the detected pockets as they are.")
    expiry: str | None = Field(default=None, description="YYYY-MM printed on the strip. Omit to keep the date read from the photo; empty string clears it.")


class ExpiryIn(BaseModel):
    expiry: str | None = Field(default=None, description="YYYY-MM, or empty/null when unknown")


class DecisionIn(BaseModel):
    approve: bool
    note: str | None = None


class TickIn(BaseModel):
    at: str | None = Field(default=None, description="Simulated time (ISO 8601), for demos and tests.")


class DemoIn(BaseModel):
    language: str = Field(default="en", pattern="^(en|hi)$")
    name: str | None = Field(default=None, max_length=40)


# ---------------------------------------------------------------- health & demo


@app.get("/health")
def health():
    return {"ok": True, "version": __version__, "vision_engine": get_engine().name,
            "storage": settings.storage.backend, "agent": "bedrock" if settings.agent.bedrock_model_id else "rules",
            "auth": "token" if settings.api_token else "open"}


@app.post("/demo/session", status_code=201)
async def demo_session(body: DemoIn | None = None, c: Caller = Depends(caller)):
    """Start a sandbox: a demo patient with two registered strips and sample photos."""
    limit(c, "session", settings.limits.demo_sessions_per_ip_hour)
    body = body or DemoIn()
    return await run_in_threadpool(demo.create_session, get_store(), body.language, body.name)


@app.get("/demo/samples")
def demo_samples():
    return demo.samples()


@app.get("/demo/assets/{file_name}")
def demo_asset(file_name: str):
    return FileResponse(demo.asset_path(file_name), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------- patients & medications


@app.post("/patients", status_code=201)
def create_patient(body: PatientIn, c: Caller = Depends(require_admin)):
    return service.create_patient(get_store(), body.model_dump())


@app.get("/patients")
def list_patients(c: Caller = Depends(require_admin), include_demo: bool = False):
    return [p for p in get_store().list("patients") if include_demo or not p.get("demo")]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str, c: Caller = Depends(caller)):
    return guard(c, patient_id)


@app.post("/patients/{patient_id}/medications", status_code=201)
def create_medication(patient_id: str, body: MedicationIn, c: Caller = Depends(caller)):
    guard(c, patient_id)
    return service.create_medication(get_store(), patient_id, body.model_dump())


@app.get("/patients/{patient_id}/medications")
def list_medications(patient_id: str, c: Caller = Depends(caller)):
    guard(c, patient_id)
    return get_store().find("medications", patient_id, status="active")


# ---------------------------------------------------------------- reference strips


@app.post("/medications/{medication_id}/reference", status_code=201)
async def upload_reference(medication_id: str, file: UploadFile = File(...), c: Caller = Depends(caller)):
    """Step 1 of strip setup: photo of a FULL strip. Returns detected pockets to confirm."""
    guard_doc(c, "medications", medication_id)
    limit(c, "reference", settings.limits.references_per_ip_hour)
    data = await read_image(file)
    return await run_in_threadpool(service.upload_reference, get_store(), medication_id, data)


@app.put("/references/{reference_id}/confirm")
def confirm_reference(reference_id: str, body: ConfirmIn, c: Caller = Depends(caller)):
    """Step 2: accept (or correct) the pockets. The strip becomes active for checks."""
    guard_doc(c, "references", reference_id)
    pockets = [p.model_dump() for p in body.pockets] if body.pockets is not None else None
    return service.confirm_reference(get_store(), reference_id, pockets, body.expiry)


@app.put("/medications/{medication_id}/expiry")
def set_expiry(medication_id: str, body: ExpiryIn, c: Caller = Depends(caller)):
    """Caregiver sets or corrects the expiry date printed on the strip."""
    guard_doc(c, "medications", medication_id)
    try:
        return service.set_expiry(get_store(), medication_id, body.expiry)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/references/{reference_id}")
def get_reference(reference_id: str, c: Caller = Depends(caller)):
    return guard_doc(c, "references", reference_id)


@app.get("/references/{reference_id}/image")
def reference_image(reference_id: str, c: Caller = Depends(caller)):
    return blob_response(guard_doc(c, "references", reference_id)["image_key"], "image/png")


@app.get("/references/{reference_id}/preview.jpg")
def reference_preview(reference_id: str, c: Caller = Depends(caller)):
    return blob_response(guard_doc(c, "references", reference_id)["preview_key"], "image/jpeg")


# ---------------------------------------------------------------- checks


@app.post("/patients/{patient_id}/checks", status_code=201)
async def submit_check(patient_id: str, file: UploadFile | None = File(default=None),
                       sample_id: str | None = Form(default=None), medication_id: str | None = Form(default=None),
                       at: str | None = Form(default=None), c: Caller = Depends(caller)):
    """The daily photo: an upload, or (demo) a sample photo id. Runs perception, then the agent."""
    patient = guard(c, patient_id)
    limit(c, "check", settings.limits.checks_per_ip_hour)
    if sample_id:
        if not (patient.get("demo") or c.admin):
            raise HTTPException(400, "sample photos are only available in demo sessions")
        data = demo.sample_bytes(sample_id)
    elif file is not None:
        data = await read_image(file)
    else:
        raise HTTPException(400, "send a photo (file) or a sample_id")
    if at and not (c.admin or patient.get("demo")):
        raise HTTPException(400, "simulated time is only available in demo sessions")
    global_check_budget()
    return await run_in_threadpool(service.submit_check, get_store(), patient_id, data, medication_id or None, at or None)


@app.get("/checks/{check_id}")
def get_check(check_id: str, c: Caller = Depends(caller)):
    return guard_doc(c, "checks", check_id)


@app.get("/checks/{check_id}/evidence.jpg")
def check_evidence(check_id: str, c: Caller = Depends(caller)):
    return blob_response(guard_doc(c, "checks", check_id).get("evidence_key"), "image/jpeg")


@app.get("/checks/{check_id}/photo.jpg")
def check_photo(check_id: str, c: Caller = Depends(caller)):
    return blob_response(guard_doc(c, "checks", check_id)["photo_key"], "image/jpeg")


# ---------------------------------------------------------------- views


@app.get("/patients/{patient_id}/today")
def today(patient_id: str, at: str | None = None, c: Caller = Depends(caller)):
    guard(c, patient_id)
    return service.today(get_store(), patient_id, at)


@app.get("/patients/{patient_id}/timeline")
def timeline(patient_id: str, limit_: int = Query(default=100, le=500, alias="limit"), c: Caller = Depends(caller)):
    guard(c, patient_id)
    return service.timeline(get_store(), patient_id, limit_)


@app.get("/patients/{patient_id}/notifications")
def notifications(patient_id: str, to: str | None = Query(default=None, pattern="^(patient|caregiver)$"),
                  c: Caller = Depends(caller)):
    guard(c, patient_id)
    items = get_store().list("notifications", patient_id)
    return [n for n in reversed(items) if to is None or n["to"] == to]


@app.get("/patients/{patient_id}/actions")
def actions(patient_id: str, status: str | None = None, c: Caller = Depends(caller)):
    guard(c, patient_id)
    items = get_store().list("actions", patient_id)
    return [a for a in reversed(items) if status is None or a["status"] == status]


@app.get("/patients/{patient_id}/runs")
def runs(patient_id: str, c: Caller = Depends(caller)):
    guard(c, patient_id)
    return list(reversed(get_store().list("runs", patient_id)))


@app.post("/actions/{action_id}/decision")
def decide(action_id: str, body: DecisionIn, at: str | None = None, c: Caller = Depends(caller)):
    """Human-in-the-loop: the caregiver approves or rejects a pending agent action."""
    guard_doc(c, "actions", action_id)
    return service.decide_action(get_store(), action_id, body.approve, body.note, at)


# ---------------------------------------------------------------- scheduler


@app.post("/agent/tick")
def tick(body: TickIn | None = None, c: Caller = Depends(require_admin)):
    """All real patients: reminders, missed-dose alerts, refills. Run every 15 min by EventBridge Scheduler."""
    runs_ = service.tick_all(get_store(), body.at if body else None)
    return {"runs": runs_, "count": len(runs_)}


@app.post("/patients/{patient_id}/tick")
def tick_patient(patient_id: str, body: TickIn | None = None, c: Caller = Depends(caller)):
    """One patient (demo: fast-forward the clock to see reminders and missed-dose alerts)."""
    patient = guard(c, patient_id)
    if body and body.at and not (c.admin or patient.get("demo")):
        raise HTTPException(400, "simulated time is only available in demo sessions")
    return {"run": service.tick_patient(get_store(), patient_id, body.at if body else None)}


# ---------------------------------------------------------------- web app
# The built frontend (frontend/dist) is served from the same origin, last, so
# API routes always win. In the Lambda image it lives in ./static.
_static = next((p for p in (BACKEND_ROOT / "static", BACKEND_ROOT.parent / "frontend" / "dist") if (p / "index.html").exists()), None)
if _static is not None:
    app.mount("/app", StaticFiles(directory=_static, html=True), name="app")
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(_static / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon():
        return FileResponse(_static / "favicon.svg")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")
