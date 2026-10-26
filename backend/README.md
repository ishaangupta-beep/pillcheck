# PillCheck backend

Camera-verified medication adherence for elderly patients. A parent photographs
their medicine strip after taking a dose; **OpenCV 5** works out which pockets
are now empty, and an **agent** decides what happens next: record the dose,
remind, ask for a better photo, alert the caregiver, or ask a human to decide.

Built for the OpenCV AI Competition 2026 (powered by AWS).

```
photo ─► quality gates ─► duplicate check (pHash) ─► ALIKED features
      ─► LightGlue matching ─► MAGSAC++ homography ─► identity check (tiled print correlation)
      ─► warp into reference frame ─► per-pocket change (Lab + SSIM, lighting-corrected)
      ─► reconcile with dose schedule ─► agent (Bedrock or rules) ─► policy guard ─► actions
```

## Quick start (macOS / Linux)

Needs **Python 3.12 or 3.13** (`brew install python@3.12` on a Mac).

```bash
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_models.py          # ALIKED + LightGlue ONNX, checksum-verified (52 MB)
```

**Try it on your own photos** (no server):

```bash
python scripts/try_photos.py full_strip.jpg after_dose_1.jpg after_dose_2.jpg
open try_output/        # reference_pockets.jpg, check_N_evidence.jpg
```

**Run the API:**

```bash
uvicorn pillcheck.api.main:app --reload
open http://localhost:8000/docs           # interactive API docs
```

**Tests and evaluation:**

```bash
pytest -q                                              # 21 tests: vision, agent, API, demo/limits, Lambda, AWS store
python eval/run_eval.py                                # synthetic benchmark, ALIKED+LightGlue
PILLCHECK_FEATURES=orb python eval/run_eval.py --tag orb   # classic ORB baseline
python eval/run_eval.py --real eval/real               # your labelled real photos
```

## How it works

### Perception (`pillcheck/vision/`)

| Step | OpenCV | File |
|---|---|---|
| Capture quality: blur, exposure, glare | `Laplacian`, HSV thresholds, morphology | `quality.py` |
| Reused-photo detection | `img_hash.PHash` + SHA-256 | `dedupe.py` |
| Pocket detection on the reference | CLAHE, `Canny`, `findContours`, `fitEllipse`, `HoughCircles(HOUGH_GRADIENT_ALT)` | `pockets.py` |
| Strip outline | `grabCut` seeded by the pockets | `pockets.py` |
| Keypoints + descriptors | **`cv.ALIKED`** (OpenCV 5 DNN engine) | `models.py` |
| Matching | **`cv.LightGlueMatcher`** | `models.py` |
| Geometry | `findHomography(..., cv.USAC_MAGSAC)`, sanity + framing checks | `align.py` |
| Identity ("is this the right medicine?") | tiled gradient correlation after alignment | `classify.py` |
| Pocket state | trimmed-LSQ photometric fit, CIE Lab distance, SSIM, ±3 px search, noise floor | `classify.py` |
| Evidence image | side-by-side reference vs aligned photo | `annotate.py` |

Registration does the heavy lifting: once today's photo is warped onto the day-1
photo of the full strip, each pocket is compared with *its own* earlier
appearance, so the method works for any pill colour, foil or layout without
training data. Identity uses the same alignment: a photo of a different strip
cannot make every printed region line up, even when two products share one
layout and differ only in the name.

### Agent (`pillcheck/agent/`)

* `context.py` builds the facts. Pill arithmetic (which dose a removed pill
  belongs to, what is extra, what is still due) is computed in `domain.py`,
  never by the model.
* `bedrock.py` lets a foundation model on Amazon Bedrock choose tools through
  the Converse API (`PILLCHECK_BEDROCK_MODEL_ID`). `rules.py` is the
  deterministic planner used offline and as the **safety floor**.
* `policy.py` validates every tool call: no dose recorded from an untrusted
  photo, no dosing advice to the patient, rate-limited caregiver alerts, refills
  only when supply is low. Mandatory actions a model skips (for example the
  alert for a possible double dose) are executed by the floor and marked as
  such in the trace.
* Tools: `record_dose`, `send_patient_message`, `request_retake`,
  `alert_caregiver`, `request_human_review` (pending until the caregiver
  decides), `draft_refill_request` (pending approval).
* Every run stores a trace: facts → each tool call (source, input, policy
  verdict, result) → summary. This is the "visual evidence changes the next
  action" record for the Agentic Vision award.

Escalation ladder for a dose: due → photo shows not taken → reminder (+60 min,
scheduler) → caregiver alert and slot marked missed (+180 min). A pill more
than 6 h late is never credited to the old slot, so it shows up as extra.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | engine, storage, agent mode |
| POST | `/patients` | create patient (+ caregiver, language `en`/`hi`, timezone) |
| POST | `/patients/{id}/medications` | name, strength, schedule `[{"time":"09:00","pills":1}]` |
| POST | `/medications/{id}/reference` | photo of the **full** strip; returns detected pockets |
| PUT | `/references/{id}/confirm` | accept or correct pockets (`initially_empty` for a used strip) |
| GET | `/references/{id}/preview.jpg` | pockets overlay for the confirmation screen |
| POST | `/patients/{id}/checks` | daily photo (+ `medication_id` expected); perception + agent run |
| GET | `/checks/{id}` · `/checks/{id}/evidence.jpg` | result, trace, evidence image |
| GET | `/patients/{id}/today` | today's dose slots, supply, pending decisions |
| GET | `/patients/{id}/timeline` | checks, doses, messages, actions (newest first) |
| GET | `/patients/{id}/notifications?to=caregiver` | messages sent |
| GET | `/patients/{id}/actions?status=pending_approval` | human-in-the-loop queue |
| POST | `/actions/{id}/decision` | `{"approve": true, "note": "..."}` |
| POST | `/agent/tick` | scheduler: reminders, missed doses, refills (admin; EventBridge every 15 min) |
| POST | `/patients/{id}/tick` | run the scheduler for one patient (demo: fast-forward the clock) |
| POST | `/demo/session` | public sandbox patient with registered strips |
| GET | `/demo/samples` · `/demo/assets/{file}` | sample photos for visitors without a strip |

`at` (ISO time) on checks, `today` and `tick` simulates the clock for demos.
With `PILLCHECK_API_TOKEN` set, real patients need `Authorization: Bearer <token>`; visitors use demo sessions.

## AWS (serverless, pay-per-use)

```
Browser ──HTTPS──► Lambda Function URL ──► Lambda (arm64 container: FastAPI + OpenCV 5 + agent)
                                              ├─► DynamoDB  (records, rate-limit counters with TTL)
                                              ├─► S3        (photos, evidence, reference features; private, encrypted)
                                              ├─► Bedrock   (agent planner, optional, daily cap)
                                              ├─► SNS       (caregiver alert emails)
                                              └─► CloudWatch (JSON logs, dashboard, error alarm)
EventBridge Scheduler ──every 15 min──► same Lambda {"pillcheck": "tick"}
```

Everything is defined in `infra/template.yaml` (CloudFormation, cfn-lint clean).
Nothing runs while idle: no servers, no load balancer, no fixed hourly cost.
Estimated cost for the hackathon period: about $1-4.

```bash
./infra/deploy.sh      # shows the plan and costs, asks y/n, builds arm64 image, deploys, smoke-tests
./infra/teardown.sh    # deletes everything (type 'delete' to confirm)
```

Spending protection for a public URL:

* Admin token (`PILLCHECK_API_TOKEN`, generated by deploy.sh into `.deploy.env`):
  real patients are reachable only with it.
* Visitors get **demo sessions** (`POST /demo/session`): a sandbox patient with two
  registered strips and sample photos (`GET /demo/samples`, submit with `sample_id`),
  simulated time (`at`) and a per-patient scheduler (`POST /patients/{id}/tick`).
  Demo alerts stay in the app; they never send email.
* Per-visitor hourly limits (sessions, checks, reference uploads), a global daily
  photo-check cap, and a Bedrock daily cap after which the free rule planner is used.
* Reserved Lambda concurrency where the account allows it; 4 MB upload limit.

Local Lambda simulation (no Docker): `tests/test_public.py` drives the real
handler with Function URL events; DynamoDB and S3 are emulated with moto.

## Responsible use

PillCheck supports adherence; it does not give medical advice. The policy layer
blocks dosing instructions in patient messages, uncertain pockets are never
guessed (the system abstains, asks for a retake, then asks the caregiver), and
caregivers approve refills and resolve unclear cases. Photos stay in the
operator's own AWS account, encrypted at rest.
