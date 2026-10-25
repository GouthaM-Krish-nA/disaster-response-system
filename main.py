"""
main.py

The whole pipeline in one process, for local development:

  POST /report  --->  asyncio.Queue  --->  background worker  --->  SQLite
  (ingestion)         (stands in for       (stands in for          (stands in
                        SQS)                 Lambda + decision       for DynamoDB)
                                             engine)

  GET /reports  <---  dashboard polls this to render the live map

Run it with:  uvicorn main:app --reload

When you're ready to port this to real AWS:
  - POST /report's body validation -> API Gateway request validation
  - asyncio.Queue -> SQS queue
  - the background worker loop -> a Lambda triggered by SQS
  - db.py's SQLite calls -> boto3 calls to DynamoDB
  - GET /reports -> a second Lambda behind API Gateway, or read directly from
    DynamoDB with a GSI on priority_score
The application logic (decision_engine.py) doesn't need to change at all.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import db
from decision_engine import score_report
from routing import find_route, RESCUE_HQ


app = FastAPI(title="Disaster Response Decision System")

# Allow the dashboard (served from a different origin/file) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

report_queue: asyncio.Queue = asyncio.Queue()


class Location(BaseModel):
    lat: float
    lng: float


class IncomingReport(BaseModel):
    type: Literal["road_block", "hospital_status", "victim_triage"]
    location: Location
    severity: int = Field(ge=1, le=5)
    description: str = ""


@app.on_event("startup")
async def startup():
    db.init_db()
    # Launch the background worker that consumes the queue.
    # This is the "Lambda triggered by SQS" equivalent.
    asyncio.create_task(worker_loop())


async def worker_loop():
    while True:
        report = await report_queue.get()
        try:
            duplicate_count = db.count_recent_similar(
                report["type"], report["location"]["lat"], report["location"]["lng"]
            )
            decision = score_report(
                report["type"], report["severity"], duplicate_count,
                report.get("description", ""),
            )
            db.insert_report(report, decision)
        except Exception as e:
            # In production this would go to a dead-letter queue. For the demo,
            # just log it so a bad report doesn't kill the worker.
            print(f"[worker] failed to process report {report.get('report_id')}: {e}")
        finally:
            report_queue.task_done()


@app.post("/report")
async def submit_report(report: IncomingReport):
    """Ingestion endpoint. Validates and enqueues — does NOT score inline,
    so a burst of thousands of requests doesn't block the API."""
    payload = {
        "report_id": str(uuid.uuid4()),
        "type": report.type,
        "location": {"lat": report.location.lat, "lng": report.location.lng},
        "severity": report.severity,
        "description": report.description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await report_queue.put(payload)
    return {"status": "queued", "report_id": payload["report_id"], "queue_depth": report_queue.qsize()}


@app.get("/reports")
async def list_reports(status: str | None = None):
    """Used by the dashboard to poll for the latest scored reports."""
    return db.get_all_reports(status)


@app.post("/reports/{report_id}/resolve")
async def resolve(report_id: str):
    ok = db.resolve_report(report_id)
    if not ok:
        raise HTTPException(status_code=404, detail="report not found")
    return {"status": "resolved", "report_id": report_id}


@app.get("/route/{report_id}")
async def get_route(report_id: str):
    """
    Compute the fastest safe route from Rescue HQ to a report's location,
    using A* search that avoids areas near active CRITICAL/HIGH road blocks.
    """
    report = db.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="report not found")

    active_blocks = db.get_active_road_blocks()
    result = find_route(
        start_latlng=RESCUE_HQ,
        goal_latlng=(report["lat"], report["lng"]),
        active_road_blocks=active_blocks,
    )
    return {
        "report_id": report_id,
        "waypoints": result.waypoints,
        "total_cost": result.total_cost,
        "blocked_nodes_avoided": result.blocked_nodes_avoided,
        "nodes_expanded": result.nodes_expanded,
        "road_blocks_considered": len(active_blocks),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "queue_depth": report_queue.qsize()}


# Three pages, all served by this same app/URL so there's one deployment
# and nothing to hardcode across origins:
#   /           -> home page (choose: report an emergency, or view dashboard)
#   /report     -> public report-submission form (GET; separate from the
#                  POST /report API route above — same path, different verb)
#   /dashboard  -> the responder-facing live map (unchanged from before)
@app.get("/")
async def home_page():
    return FileResponse("frontend/home.html")


@app.get("/report")
async def report_form_page():
    return FileResponse("frontend/report.html")


@app.get("/dashboard")
async def dashboard_page():
    return FileResponse("dashboard/index.html")