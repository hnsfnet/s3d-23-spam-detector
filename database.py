"""SQLite store for prediction history, user feedback, and training logs.

Three tables are maintained:

* ``predictions`` -- every classification made via ``/predict`` (and each
  item of ``/predict_batch``), so history can be reviewed later.
* ``feedback`` -- user corrections submitted via ``/feedback``. Each record
  stores what the model predicted and what the user says is correct, plus an
  ``applied`` flag indicating whether it has already been folded into a
  retraining run.
* ``trainings`` -- one row per (re)training event with a timestamp and the
  source that triggered it (startup / train API / feedback auto-retrain).

The module uses the standard-library ``sqlite3`` and is thread-safe by
opening a fresh connection per call, which fits Flask's default threaded
server.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

VALID_LABELS = ("ham", "spam")


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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    subject TEXT,
                    body TEXT,
                    predicted_label TEXT NOT NULL,
                    correct_label TEXT NOT NULL,
                    applied INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_applied "
                "ON feedback(applied)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_created_at "
                "ON feedback(created_at DESC)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trainings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    feedback_incorporated INTEGER NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

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

    def get_prediction(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, subject, body, is_spam, confidence, predicted_at "
                "FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "subject": row["subject"],
            "body": row["body"],
            "is_spam": bool(row["is_spam"]),
            "confidence": row["confidence"],
            "predicted_at": row["predicted_at"],
        }

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

    def count_since(self, since_iso: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE predicted_at >= ?",
                (since_iso,),
            ).fetchone()
            return int(row["n"]) if row else 0

    def spam_count_since(self, since_iso: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM predictions "
                "WHERE is_spam = 1 AND predicted_at >= ?",
                (since_iso,),
            ).fetchone()
            return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ #
    # Feedback
    # ------------------------------------------------------------------ #

    def save_feedback(
        self,
        subject: str,
        body: str,
        predicted_label: str,
        correct_label: str,
        prediction_id: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """Insert a feedback record and return its row id.

        ``predicted_label`` is what the model said; ``correct_label`` is the
        user-supplied ground truth. Both must be ``'ham'`` or ``'spam'``.
        """
        if predicted_label not in VALID_LABELS:
            raise ValueError(f"predicted_label must be one of {VALID_LABELS}")
        if correct_label not in VALID_LABELS:
            raise ValueError(f"correct_label must be one of {VALID_LABELS}")
        created_at = created_at or datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO feedback (prediction_id, subject, body, "
                "predicted_label, correct_label, applied, created_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (prediction_id, subject, body, predicted_label,
                 correct_label, created_at),
            )
            return int(cursor.lastrowid)

    def count_feedback(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()
            return int(row["n"]) if row else 0

    def count_unapplied_feedback(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM feedback WHERE applied = 0"
            ).fetchone()
            return int(row["n"]) if row else 0

    def get_unapplied_feedback(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, subject, body, correct_label FROM feedback "
                "WHERE applied = 0 ORDER BY id"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "subject": row["subject"],
                "body": row["body"],
                "correct_label": row["correct_label"],
            }
            for row in rows
        ]

    def mark_feedback_applied(self, feedback_ids: List[int]) -> int:
        if not feedback_ids:
            return 0
        placeholders = ",".join("?" for _ in feedback_ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE feedback SET applied = 1 WHERE id IN ({placeholders})",
                feedback_ids,
            )
            return cursor.rowcount

    def feedback_accuracy(self) -> Optional[Dict[str, Any]]:
        """Accuracy of the model computed from all labelled feedback.

        Returns ``None`` when there is no feedback yet.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT "
                "COUNT(*) AS total, "
                "SUM(CASE WHEN predicted_label = correct_label "
                "THEN 1 ELSE 0 END) AS correct "
                "FROM feedback"
            ).fetchone()
        total = int(row["total"]) if row and row["total"] else 0
        correct = int(row["correct"]) if row and row["correct"] is not None else 0
        if total == 0:
            return None
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 6),
        }

    # ------------------------------------------------------------------ #
    # Training log
    # ------------------------------------------------------------------ #

    def record_training(
        self,
        source: str,
        total_samples: int,
        feedback_incorporated: int,
        trained_at: Optional[str] = None,
    ) -> int:
        trained_at = trained_at or datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO trainings (trained_at, source, total_samples, "
                "feedback_incorporated) VALUES (?, ?, ?, ?)",
                (trained_at, source, total_samples, feedback_incorporated),
            )
            return int(cursor.lastrowid)

    def get_last_trained_at(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT trained_at FROM trainings ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["trained_at"] if row else None

    def get_last_training(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT trained_at, source, total_samples, feedback_incorporated "
                "FROM trainings ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {
            "trained_at": row["trained_at"],
            "source": row["source"],
            "total_samples": row["total_samples"],
            "feedback_incorporated": row["feedback_incorporated"],
        }

    # ------------------------------------------------------------------ #
    # Aggregate stats for the dashboard
    # ------------------------------------------------------------------ #

    def get_stats(self, since_iso: Optional[str] = None) -> Dict[str, Any]:
        """Return dashboard statistics.

        Prediction totals/ratios honour ``since_iso`` (an ISO timestamp) so
        callers can scope them to a recent window. Feedback accuracy and the
        last training time are always global so they remain stable when the
        recent window is small.
        """
        if since_iso:
            total = self.count_since(since_iso)
            spam_total = self.spam_count_since(since_iso)
        else:
            total = self.count()
            spam_total = self._spam_total()

        ham_total = total - spam_total
        spam_ratio = round(spam_total / total, 6) if total else 0.0

        accuracy = self.feedback_accuracy()
        last_training = self.get_last_training()

        return {
            "since": since_iso,
            "total_predictions": total,
            "spam_predictions": spam_total,
            "ham_predictions": ham_total,
            "spam_ratio": spam_ratio,
            "feedback_total": self.count_feedback(),
            "feedback_unapplied": self.count_unapplied_feedback(),
            "feedback_accuracy": accuracy,
            "last_training": last_training,
        }

    def _spam_total(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE is_spam = 1"
            ).fetchone()
            return int(row["n"]) if row else 0
