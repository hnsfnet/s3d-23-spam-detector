"""Tests for the storage module (SQLite database operations)."""

from __future__ import annotations

import pytest


class TestPredictionDatabase:
    """Tests for PredictionDatabase."""

    def test_init_creates_tables(self, db):
        """Verify __init__ creates all three tables."""
        assert db.count() == 0
        assert db.count_feedback() == 0

    def test_save_prediction_and_count(self, db):
        db.save_prediction("Hello", "World", False, 0.9)
        assert db.count() == 1

        db.save_prediction("Free money", "Click now", True, 0.95)
        assert db.count() == 2

    def test_save_prediction_returns_id(self, db):
        id1 = db.save_prediction("A", "B", False, 0.9)
        id2 = db.save_prediction("C", "D", True, 0.95)
        assert id1 == 1
        assert id2 == 2

    def test_get_prediction(self, db):
        pid = db.save_prediction("Test Subject", "Test Body", True, 0.85)
        record = db.get_prediction(pid)
        assert record is not None
        assert record["id"] == pid
        assert record["subject"] == "Test Subject"
        assert record["body"] == "Test Body"
        assert record["is_spam"] is True
        assert abs(record["confidence"] - 0.85) < 1e-6
        assert "predicted_at" in record

    def test_get_prediction_not_found(self, db):
        assert db.get_prediction(999) is None

    def test_get_history(self, db):
        for i in range(5):
            db.save_prediction(f"Subj{i}", f"Body{i}", i % 2 == 0, 0.8)

        history = db.get_history(limit=3)
        assert len(history) == 3
        assert history[0]["subject"] == "Subj4"
        assert history[1]["subject"] == "Subj3"
        assert history[2]["subject"] == "Subj2"

        history2 = db.get_history(limit=3, offset=3)
        assert len(history2) == 2
        assert history2[0]["subject"] == "Subj1"
        assert history2[1]["subject"] == "Subj0"

    def test_count_since(self, db):
        db.save_prediction("Old", "Email", False, 0.9, predicted_at="2020-01-01T00:00:00")
        db.save_prediction("New", "Email", True, 0.95, predicted_at="2025-01-01T00:00:00")

        assert db.count_since("2024-01-01T00:00:00") == 1
        assert db.count_since("2019-01-01T00:00:00") == 2
        assert db.count_since("2030-01-01T00:00:00") == 0

    def test_spam_count_since(self, db):
        db.save_prediction("Ham1", "Body", False, 0.9, predicted_at="2025-01-01T00:00:00")
        db.save_prediction("Spam1", "Body", True, 0.95, predicted_at="2025-01-02T00:00:00")
        db.save_prediction("Spam2", "Body", True, 0.9, predicted_at="2025-01-03T00:00:00")
        db.save_prediction("Ham2", "Body", False, 0.85, predicted_at="2025-01-04T00:00:00")

        assert db.spam_count_since("2025-01-01T00:00:00") == 2

    def test_save_feedback(self, db):
        fid = db.save_feedback(
            subject="Test",
            body="Body",
            predicted_label="ham",
            correct_label="spam",
        )
        assert fid == 1
        assert db.count_feedback() == 1
        assert db.count_unapplied_feedback() == 1

    def test_save_feedback_with_prediction_id(self, db):
        pid = db.save_prediction("Test", "Body", False, 0.9)
        fid = db.save_feedback(
            subject="Test",
            body="Body",
            predicted_label="ham",
            correct_label="spam",
            prediction_id=pid,
        )
        assert fid == 1

    def test_save_feedback_invalid_predicted_label(self, db):
        with pytest.raises(ValueError, match="predicted_label must be"):
            db.save_feedback("S", "B", "invalid", "spam")

    def test_save_feedback_invalid_correct_label(self, db):
        with pytest.raises(ValueError, match="correct_label must be"):
            db.save_feedback("S", "B", "ham", "invalid")

    def test_get_unapplied_feedback(self, db):
        db.save_feedback("S1", "B1", "ham", "spam")
        db.save_feedback("S2", "B2", "spam", "ham")

        unapplied = db.get_unapplied_feedback()
        assert len(unapplied) == 2
        assert unapplied[0]["subject"] == "S1"
        assert unapplied[1]["subject"] == "S2"

    def test_mark_feedback_applied(self, db):
        fid1 = db.save_feedback("S1", "B1", "ham", "spam")
        fid2 = db.save_feedback("S2", "B2", "spam", "ham")
        fid3 = db.save_feedback("S3", "B3", "ham", "spam")

        assert db.count_unapplied_feedback() == 3

        updated = db.mark_feedback_applied([fid1, fid3])
        assert updated == 2
        assert db.count_unapplied_feedback() == 1

        remaining = db.get_unapplied_feedback()
        assert remaining[0]["id"] == fid2

    def test_mark_feedback_applied_empty(self, db):
        assert db.mark_feedback_applied([]) == 0

    def test_feedback_accuracy_no_feedback(self, db):
        assert db.feedback_accuracy() is None

    def test_feedback_accuracy(self, db):
        db.save_feedback("S1", "B1", "ham", "ham")
        db.save_feedback("S2", "B2", "ham", "spam")
        db.save_feedback("S3", "B3", "spam", "spam")
        db.save_feedback("S4", "B4", "spam", "ham")

        result = db.feedback_accuracy()
        assert result["total"] == 4
        assert result["correct"] == 2
        assert abs(result["accuracy"] - 0.5) < 1e-6

    def test_record_training(self, db):
        tid = db.record_training("startup", 100, 5)
        assert tid == 1

        last = db.get_last_training()
        assert last is not None
        assert last["source"] == "startup"
        assert last["total_samples"] == 100
        assert last["feedback_incorporated"] == 5
        assert "trained_at" in last

    def test_get_last_training_empty(self, db):
        assert db.get_last_training() is None

    def test_get_last_trained_at(self, db):
        assert db.get_last_trained_at() is None
        db.record_training("test", 10, 0)
        assert db.get_last_trained_at() is not None

    def test_get_stats_empty(self, db):
        stats = db.get_stats()
        assert stats["total_predictions"] == 0
        assert stats["spam_predictions"] == 0
        assert stats["spam_ratio"] == 0.0
        assert stats["feedback_total"] == 0
        assert stats["feedback_unapplied"] == 0
        assert stats["feedback_accuracy"] is None
        assert stats["last_training"] is None

    def test_get_stats_with_data(self, db):
        db.save_prediction("S1", "B1", False, 0.9)
        db.save_prediction("S2", "B2", True, 0.95)
        db.save_prediction("S3", "B3", True, 0.9)

        db.save_feedback("S1", "B1", "ham", "ham")
        db.record_training("test", 10, 0)

        stats = db.get_stats()
        assert stats["total_predictions"] == 3
        assert stats["spam_predictions"] == 2
        assert stats["ham_predictions"] == 1
        assert abs(stats["spam_ratio"] - 2/3) < 1e-3
        assert stats["feedback_total"] == 1
        assert stats["feedback_unapplied"] == 1
        assert stats["feedback_accuracy"]["accuracy"] == 1.0
        assert stats["last_training"]["source"] == "test"

    def test_get_stats_with_since(self, db):
        db.save_prediction("Old", "Email", False, 0.9, predicted_at="2020-01-01T00:00:00")
        db.save_prediction("New", "Email", True, 0.95, predicted_at="2025-01-01T00:00:00")

        stats = db.get_stats(since_iso="2024-01-01T00:00:00")
        assert stats["total_predictions"] == 1
        assert stats["spam_predictions"] == 1
