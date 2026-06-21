"""SQLite store for prediction history.

Each prediction made through the ``/predict`` endpoint is persisted so the
classification history can be reviewed later (exposed via ``GET /history``).
The module uses the standard-library ``sqlite3`` and is thread-safe by
opening a fresh connection per call, which fits Flask's default threaded
server.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional


class PredictionDatabase:
    def __init__(self, db_path: str = "spam_predictions.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    body TEXT,
                    is_spam INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at "
                "ON predictions(predicted_at DESC)"
            )

    def save_prediction(
        self,
        subject: str,
        body: str,
        is_spam: bool,
        confidence: float,
        predicted_at: Optional[str] = None,
    ) -> int:
        """Insert a prediction record and return its row id."""
        predicted_at = predicted_at or datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO predictions (subject, body, is_spam, confidence, "
                "predicted_at) VALUES (?, ?, ?, ?, ?)",
                (
                    subject,
                    body,
                    1 if is_spam else 0,
                    float(confidence),
                    predicted_at,
                ),
            )
            return int(cursor.lastrowid)

    def get_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return recent predictions, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, subject, body, is_spam, confidence, predicted_at "
                "FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "subject": row["subject"],
                "body": row["body"],
                "is_spam": bool(row["is_spam"]),
                "confidence": row["confidence"],
                "predicted_at": row["predicted_at"],
            }
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()
            return int(row["n"]) if row else 0
