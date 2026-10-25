"""
db.py

SQLite storage layer. This stands in for DynamoDB in the local prototype —
same role in the architecture (durable store of scored reports), zero setup cost.
"""

import sqlite3
from contextlib import contextmanager

DB_PATH = "reports.db"


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                severity INTEGER NOT NULL,
                description TEXT,
                timestamp TEXT NOT NULL,
                priority_score REAL,
                priority_label TEXT,
                recommended_action TEXT,
                status TEXT DEFAULT 'pending',
                text_urgency_class INTEGER,
                text_urgency_confidence REAL,
                text_urgency_terms TEXT
            )
        """)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_report(report: dict, decision) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO reports
            (report_id, type, lat, lng, severity, description, timestamp,
             priority_score, priority_label, recommended_action, status,
             text_urgency_class, text_urgency_confidence, text_urgency_terms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                report["report_id"],
                report["type"],
                report["location"]["lat"],
                report["location"]["lng"],
                report["severity"],
                report.get("description", ""),
                report["timestamp"],
                decision.priority_score,
                decision.priority_label,
                decision.recommended_action,
                decision.text_urgency_class,
                decision.text_urgency_confidence,
                ", ".join(decision.text_urgency_terms),
            ),
        )
        conn.commit()


def count_recent_similar(report_type: str, lat: float, lng: float, radius_deg: float = 0.05) -> int:
    """Rough duplicate-detection: count reports of the same type within a small
    lat/lng box. Good enough for a demo; a real system would use geospatial queries."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT COUNT(*) as c FROM reports
            WHERE type = ?
              AND ABS(lat - ?) < ?
              AND ABS(lng - ?) < ?
            """,
            (report_type, lat, radius_deg, lng, radius_deg),
        )
        return cur.fetchone()["c"]


def get_all_reports(status: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if status:
            cur = conn.execute(
                "SELECT * FROM reports WHERE status = ? ORDER BY priority_score DESC", (status,)
            )
        else:
            cur = conn.execute("SELECT * FROM reports ORDER BY priority_score DESC")
        return [dict(row) for row in cur.fetchall()]


def get_report(report_id: str) -> dict | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_active_road_blocks() -> list[dict]:
    """Road blocks that are still pending and serious enough to route around."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT * FROM reports
            WHERE type = 'road_block'
              AND status = 'pending'
              AND priority_label IN ('CRITICAL', 'HIGH')
            """
        )
        return [dict(row) for row in cur.fetchall()]


def resolve_report(report_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE reports SET status = 'resolved' WHERE report_id = ?", (report_id,)
        )
        conn.commit()
        return cur.rowcount > 0
