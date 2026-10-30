"""Database manager for TruthLens AI scan history.

Manages SQLite storage for past analyses, enabling users to audit prior checks,
track prediction distribution, and clear their history on demand.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Establish and configure an SQLite database connection."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Initialize the scan_history table if it doesn't already exist."""
    try:
        with get_db_connection(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    headline TEXT NOT NULL,
                    prediction_label TEXT,
                    confidence REAL,
                    fake_prob REAL,
                    real_prob REAL,
                    live_status TEXT,
                    articles_count INTEGER DEFAULT 0,
                    top_source TEXT
                )
            """)
            conn.commit()
            logger.info("Database initialized successfully at: %s", db_path)
    except Exception as exc:
        logger.exception("Failed to initialize database: %s", exc)


def add_scan_record(
    db_path: str,
    headline: str,
    prediction_label: Optional[str],
    confidence: Optional[float],
    fake_prob: Optional[float] = None,
    real_prob: Optional[float] = None,
    live_status: Optional[str] = None,
    articles_count: int = 0,
    top_source: Optional[str] = None
) -> Optional[int]:
    """Insert a new analysis scan record into SQLite history."""
    try:
        init_db(db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scan_history (
                    headline, prediction_label, confidence,
                    fake_prob, real_prob, live_status,
                    articles_count, top_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                headline,
                prediction_label,
                confidence,
                fake_prob,
                real_prob,
                live_status,
                articles_count,
                top_source
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as exc:
        logger.exception("Failed to save scan record: %s", exc)
        return None


def get_scan_history(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve the most recent scan records ordered by timestamp descending."""
    try:
        init_db(db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    id,
                    created_at,
                    headline,
                    prediction_label,
                    confidence,
                    fake_prob,
                    real_prob,
                    live_status,
                    articles_count,
                    top_source
                FROM scan_history
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.exception("Failed to read scan history: %s", exc)
        return []


def clear_scan_history(db_path: str) -> bool:
    """Clear all records from scan history."""
    try:
        init_db(db_path)
        with get_db_connection(db_path) as conn:
            conn.execute("DELETE FROM scan_history")
            conn.commit()
            logger.info("Scan history cleared successfully.")
            return True
    except Exception as exc:
        logger.exception("Failed to clear scan history: %s", exc)
        return False


def get_history_stats(db_path: str) -> Dict[str, Any]:
    """Compute aggregate analytics on past scans."""
    try:
        init_db(db_path)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM scan_history")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS fake_cnt FROM scan_history WHERE prediction_label LIKE '%FAKE%'")
            fake_cnt = cursor.fetchone()["fake_cnt"]

            cursor.execute("SELECT COUNT(*) AS real_cnt FROM scan_history WHERE prediction_label LIKE '%REAL%'")
            real_cnt = cursor.fetchone()["real_cnt"]

            cursor.execute("SELECT AVG(confidence) AS avg_conf FROM scan_history WHERE confidence IS NOT NULL")
            avg_conf_row = cursor.fetchone()
            avg_conf = round(avg_conf_row["avg_conf"], 1) if avg_conf_row and avg_conf_row["avg_conf"] else 0.0

            return {
                "total_scans": total,
                "fake_scans": fake_cnt,
                "real_scans": real_cnt,
                "average_confidence": avg_conf
            }
    except Exception as exc:
        logger.exception("Failed to query history stats: %s", exc)
        return {"total_scans": 0, "fake_scans": 0, "real_scans": 0, "average_confidence": 0.0}
