# Disaster Response Decision System

A local, runnable prototype of an intelligent cloud-based response system for
a Cloud Architecture Design project. Simulates disaster reports (road blocks,
hospital status, victim triage), scores them for priority, and shows them
live on a map dashboard.

## Architecture

```
generator/load test -> POST /report -> asyncio.Queue -> background worker
                                          (= API Gateway)   (= SQS)     (= Lambda)
                                                                          |
                                                                          v
                                                                  decision_engine.py
                                                                   (priority scoring)
                                                                          |
                                                                          v
                                                                     SQLite (db.py)
                                                                      (= DynamoDB)
                                                                          |
                                                                          v
                                                              dashboard/index.html
                                                              (polls GET /reports)
```

This runs entirely on your laptop with no AWS account needed, so you can
build and demo the *logic* fast. Section "Porting to AWS" below explains how
each local piece maps to the real service, for when you're ready to deploy
for the actual submission.

## Setup

```bash
pip install -r requirements.txt
```

## Running it

**Terminal 1 — start the API + worker:**
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2 — open the dashboard:**
Just open `dashboard/index.html` directly in a browser (no server needed for
the dashboard itself — it calls the API via fetch).

**Terminal 3 — send test reports:**

Single report:
```bash
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d '{"type":"victim_triage","location":{"lat":13.08,"lng":80.27},"severity":5,"description":"trapped on rooftop"}'
```

Burst load test (this is the demo moment — shows the queue absorbing spikes):
```bash
python load_test.py 200
```

Watch the dashboard update live as the burst gets processed.

## The AI components

This system uses two distinct AI techniques, which is worth highlighting in
a viva rather than just having one:

**1. A* route-finding (`routing.py`)** — search algorithm, deterministic:

`routing.py` finds the fastest **safe** route from a fixed Rescue HQ to any
report's location, using A* search:
- Grid of intersections over the disaster zone.
- Areas near active CRITICAL/HIGH road-block reports are heavily penalized
  (not deleted — a real route usually still exists, just risky/slow).
- Heuristic = straight-line distance to the goal (admissible, guarantees the
  optimal route).

Try it: click any marker on the dashboard and hit "Show route from HQ" — you'll
see the blue path bend around any active road blocks instead of going straight
through them.

API: `GET /route/{report_id}` returns the waypoints, total path cost, how many
blocked nodes were avoided, and how many nodes A* expanded during search
(useful if you want to discuss search efficiency).

**2. A trained text classifier (`text_urgency_model.py`)** — supervised ML,
learned from data:

Every report has a free-text `description` field (e.g. "trapped under rubble,
child on rooftop"). `text_urgency_model.py` trains a small TF-IDF +
logistic-regression classifier on a labeled seed set of phrases, then predicts
whether a new description's wording should raise or lower the priority score —
distinct signal the structured `type`/`severity` fields alone can't see. It
also surfaces which specific words drove the prediction (explainability),
shown right in the dashboard popup as "AI text analysis: ... — key terms: ...".

The model trains itself automatically on first run (saved to
`urgency_model.joblib`); delete that file and restart to retrain after editing
`TRAINING_DATA`. `decision_engine.py` combines both techniques: the rule-based
score (type/severity/duplicates) plus the ML text-classifier's boost.


## Files

- `main.py` — FastAPI app: ingestion endpoint, the in-memory queue, the
  background worker, and the reports API for the dashboard.
- `decision_engine.py` — combines the rule-based structured scorer with the
  trained text classifier's boost into a final priority score.
- `text_urgency_model.py` — the trained TF-IDF + logistic-regression text
  classifier over report descriptions (the ML component).
- `db.py` — SQLite storage layer (stands in for DynamoDB).
- `data_generator.py` — generates synthetic disaster reports clustered around
  a configurable location (defaults to Chennai).
- `load_test.py` — fires a concurrent burst of reports at the API to
  demonstrate the queue handling load spikes.
- `dashboard/index.html` — live map dashboard (Leaflet.js + CartoDB tiles),
  color-coded by priority, polling every 3 seconds.

## Porting to AWS

When you're ready to deploy the real version for submission:

| Local piece | AWS equivalent |
|---|---|
| FastAPI `POST /report` | API Gateway (REST) + Lambda for validation |
| `asyncio.Queue` | SQS standard queue |
| background `worker_loop()` | Lambda triggered by SQS (event source mapping) |
| `db.py` SQLite calls | boto3 calls to DynamoDB (same table shape) |
| `GET /reports` | Lambda + API Gateway, or DynamoDB Query with a GSI on `priority_score` |
| dashboard polling | Same frontend, just point `API_BASE` at your API Gateway URL |

`decision_engine.py` doesn't need to change at all — it's pure Python with no
framework dependency, so it becomes your Lambda's handler logic unchanged.

## Ideas for extending (if you want extra credit / more depth)

- Grow `TRAINING_DATA` in `text_urgency_model.py` with more/real examples,
  or swap logistic regression for a different classifier.
- Add a WebSocket endpoint for true real-time push instead of polling.
- Add authentication so only verified responders can POST reports.
- Add a "resolved reports" view using the existing `/reports/{id}/resolve`
  endpoint and `?status=resolved` filter.
