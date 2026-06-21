"""Tests for the HTTP API routes (using Flask test client and mock classifier)."""

from __future__ import annotations

import json


class TestHealthEndpoint:
    """Tests for GET /."""

    def test_health_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_returns_expected_fields(self, client):
        resp = client.get("/")
        data = resp.get_json()
        assert data["service"] == "spam-detector"
        assert data["status"] == "ok"
        assert "classifier_type" in data
        assert "model_trained" in data
        assert "training_stats" in data
        assert "predictions_logged" in data
        assert "feedback_total" in data
        assert "feedback_unapplied" in data


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_ham_returns_200(self, client):
        resp = client.post(
            "/predict",
            json={"subject": "Meeting tomorrow", "body": "Let's meet at 10am."},
        )
        assert resp.status_code == 200

    def test_predict_spam_returns_200(self, client):
        resp = client.post(
            "/predict",
            json={"subject": "FREE MONEY", "body": "Click now to win free money guaranteed!!!"},
        )
        assert resp.status_code == 200

    def test_predict_returns_correct_format(self, client):
        resp = client.post(
            "/predict",
            json={"subject": "Meeting", "body": "Hello world"},
        )
        data = resp.get_json()
        assert "is_spam" in data
        assert isinstance(data["is_spam"], bool)
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert "label" in data
        assert data["label"] in ("spam", "ham")
        assert "subject" in data

    def test_predict_ham_label_correct(self, client):
        resp = client.post(
            "/predict",
            json={"subject": "Meeting", "body": "Let's discuss the project"},
        )
        data = resp.get_json()
        assert data["is_spam"] is False
        assert data["label"] == "ham"

    def test_predict_spam_label_correct(self, client):
        resp = client.post(
            "/predict",
            json={"subject": "FREE MONEY NOW", "body": "Click here for free money!!!"},
        )
        data = resp.get_json()
        assert data["is_spam"] is True
        assert data["label"] == "spam"

    def test_predict_empty_subject_and_body_returns_400(self, client):
        resp = client.post("/predict", json={"subject": "", "body": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "邮件内容不能为空" in data["error"]

    def test_predict_whitespace_only_returns_400(self, client):
        resp = client.post("/predict", json={"subject": "   ", "body": "  \t\n  "})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_predict_empty_body_ok(self, client):
        resp = client.post("/predict", json={"subject": "Meeting tomorrow", "body": ""})
        assert resp.status_code == 200

    def test_predict_empty_subject_ok(self, client):
        resp = client.post("/predict", json={"subject": "", "body": "Meeting tomorrow at 10am"})
        assert resp.status_code == 200

    def test_predict_saves_to_database(self, client, mock_state):
        db = mock_state["database"]
        assert db.count() == 0

        client.post(
            "/predict",
            json={"subject": "Test", "body": "Hello world"},
        )
        assert db.count() == 1

        history = db.get_history()
        assert history[0]["subject"] == "Test"
        assert history[0]["body"] == "Hello world"

    def test_predict_form_data(self, client):
        resp = client.post(
            "/predict",
            data={"subject": "Meeting", "body": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["is_spam"] is False

    def test_predict_no_body_returns_400(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 400


class TestPredictBatchEndpoint:
    """Tests for POST /predict_batch."""

    def test_predict_batch_returns_200(self, client):
        resp = client.post(
            "/predict_batch",
            json={
                "emails": [
                    {"subject": "Meeting", "body": "Hello"},
                    {"subject": "Free money", "body": "Click now"},
                ]
            },
        )
        assert resp.status_code == 200

    def test_predict_batch_returns_correct_format(self, client):
        resp = client.post(
            "/predict_batch",
            json={
                "emails": [
                    {"subject": "Meeting", "body": "Hello"},
                    {"subject": "Free money", "body": "Click now"},
                    {"subject": "Lunch", "body": "Want to grab lunch?"},
                ]
            },
        )
        data = resp.get_json()
        assert data["count"] == 3
        assert data["spam_count"] == 1
        assert data["ham_count"] == 2
        assert len(data["results"]) == 3
        for r in data["results"]:
            assert "index" in r
            assert "subject" in r
            assert "is_spam" in r
            assert "confidence" in r
            assert "label" in r

    def test_predict_batch_empty_emails_returns_400(self, client):
        resp = client.post("/predict_batch", json={"emails": []})
        assert resp.status_code == 400

    def test_predict_batch_missing_emails_returns_400(self, client):
        resp = client.post("/predict_batch", json={})
        assert resp.status_code == 400

    def test_predict_batch_empty_email_returns_400(self, client):
        resp = client.post(
            "/predict_batch",
            json={
                "emails": [
                    {"subject": "Meeting", "body": "Hello"},
                    {"subject": "", "body": ""},
                ]
            },
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "邮件内容不能为空" in data["error"]

    def test_predict_batch_non_dict_item_returns_400(self, client):
        resp = client.post(
            "/predict_batch",
            json={"emails": ["not a dict", {"subject": "Test", "body": "Test"}]},
        )
        assert resp.status_code == 400

    def test_predict_batch_raw_list(self, client):
        resp = client.post(
            "/predict_batch",
            json=[
                {"subject": "Meeting", "body": "Hello"},
                {"subject": "Free money", "body": "Click now"},
            ],
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2

    def test_predict_batch_saves_all_to_database(self, client, mock_state):
        db = mock_state["database"]
        assert db.count() == 0

        client.post(
            "/predict_batch",
            json={
                "emails": [
                    {"subject": "First", "body": "Body 1"},
                    {"subject": "Second", "body": "Body 2"},
                    {"subject": "Third", "body": "Body 3"},
                ]
            },
        )
        assert db.count() == 3


class TestTrainEndpoint:
    """Tests for POST /train."""

    def test_train_no_data_returns_400(self, client):
        resp = client.post("/train", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "No training data supplied" in data["error"]

    def test_train_with_json_data(self, client, temp_data_dir, mock_state):
        mock_state["data_dir"] = temp_data_dir
        resp = client.post(
            "/train",
            json={
                "ham": ["Subject: Good email\n\nHello world", "Subject: Meeting\n\nLet's meet"],
                "spam": ["Subject: Free money\n\nClick now free guaranteed", "Subject: Win\n\nClick here to win"],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "trained"
        assert data["added"]["ham"] == 2
        assert data["added"]["spam"] == 2

    def test_train_empty_string_data_ignored(self, client):
        resp = client.post(
            "/train",
            json={
                "ham": ["   ", ""],
                "spam": ["  \n  ", ""],
            },
        )
        assert resp.status_code == 400

    def test_train_non_string_ignored(self, client):
        resp = client.post(
            "/train",
            json={
                "ham": [None, 123, True, {}],
                "spam": ["Subject: Free\n\nClick"],
            },
        )
        assert resp.status_code == 400


class TestHistoryEndpoint:
    """Tests for GET /history."""

    def test_history_empty_returns_200(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 0
        assert data["records"] == []

    def test_history_with_data(self, client, mock_state):
        db = mock_state["database"]
        db.save_prediction("Subj1", "Body1", False, 0.9)
        db.save_prediction("Subj2", "Body2", True, 0.95)

        resp = client.get("/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert len(data["records"]) == 2
        assert data["records"][0]["subject"] == "Subj2"

    def test_history_limit_and_offset(self, client, mock_state):
        db = mock_state["database"]
        for i in range(10):
            db.save_prediction(f"Subj{i}", f"Body{i}", False, 0.8)

        resp = client.get("/history?limit=3&offset=2")
        data = resp.get_json()
        assert len(data["records"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 2

    def test_history_invalid_limit_returns_400(self, client):
        resp = client.get("/history?limit=abc")
        assert resp.status_code == 400


class TestFeedbackEndpoint:
    """Tests for POST /feedback."""

    def test_feedback_by_prediction_id(self, client, mock_state):
        db = mock_state["database"]
        pid = db.save_prediction("Test", "Body", False, 0.9)

        resp = client.post(
            "/feedback",
            json={"prediction_id": pid, "correct_label": "spam"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "recorded"
        assert data["predicted_label"] == "ham"
        assert data["correct_label"] == "spam"
        assert data["corrected"] is True

    def test_feedback_by_prediction_id_updates_database(self, client, mock_state):
        db = mock_state["database"]
        pid = db.save_prediction("Test", "Body", False, 0.9)

        assert db.count_feedback() == 0
        client.post(
            "/feedback",
            json={"prediction_id": pid, "correct_label": "spam"},
        )
        assert db.count_feedback() == 1

    def test_feedback_by_subject_body(self, client, mock_state):
        resp = client.post(
            "/feedback",
            json={
                "subject": "Free money",
                "body": "Click now for free money",
                "correct_label": "spam",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["predicted_label"] == "spam"
        assert data["correct_label"] == "spam"
        assert data["corrected"] is False

    def test_feedback_missing_correct_label_returns_400(self, client):
        resp = client.post(
            "/feedback",
            json={"subject": "Test", "body": "Test"},
        )
        assert resp.status_code == 400

    def test_feedback_invalid_correct_label_returns_400(self, client):
        resp = client.post(
            "/feedback",
            json={"subject": "Test", "body": "Test", "correct_label": "invalid"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "'correct_label' must be" in data["error"]

    def test_feedback_missing_everything_returns_400(self, client):
        resp = client.post(
            "/feedback",
            json={"correct_label": "spam"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "prediction_id" in data["error"]

    def test_feedback_invalid_prediction_id_returns_400(self, client):
        resp = client.post(
            "/feedback",
            json={"prediction_id": "not-an-int", "correct_label": "spam"},
        )
        assert resp.status_code == 400

    def test_feedback_nonexistent_prediction_id_returns_404(self, client):
        resp = client.post(
            "/feedback",
            json={"prediction_id": 9999, "correct_label": "spam"},
        )
        assert resp.status_code == 404

    def test_feedback_no_json_returns_400(self, client):
        resp = client.post("/feedback", data="not json")
        assert resp.status_code == 400

    def test_feedback_auto_retrain_when_threshold_hit(self, client, mock_state, temp_data_dir):
        db = mock_state["database"]
        mock_state["feedback_threshold"] = 2
        mock_state["data_dir"] = temp_data_dir

        import os
        ham_dir = os.path.join(temp_data_dir, "ham")
        spam_dir = os.path.join(temp_data_dir, "spam")
        os.makedirs(ham_dir, exist_ok=True)
        os.makedirs(spam_dir, exist_ok=True)

        for i in range(5):
            with open(os.path.join(ham_dir, f"ham_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Subject: Good email {i}\n\nThis is a normal ham email about meeting and work.")
        for i in range(5):
            with open(os.path.join(spam_dir, f"spam_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Subject: FREE MONEY {i}\n\nClick now for free money guaranteed!!! Urgent!!! Win free prize!!!")

        p1 = db.save_prediction("S1", "B1", False, 0.9)
        p2 = db.save_prediction("S2", "B2", False, 0.9)

        client.post("/feedback", json={"prediction_id": p1, "correct_label": "spam"})
        resp = client.post("/feedback", json={"prediction_id": p2, "correct_label": "spam"})

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["auto_retrained"] is True
        assert "retrain_stats" in data

    def test_feedback_no_auto_retrain_below_threshold(self, client, mock_state):
        db = mock_state["database"]
        mock_state["feedback_threshold"] = 10

        p1 = db.save_prediction("S1", "B1", False, 0.9)
        resp = client.post("/feedback", json={"prediction_id": p1, "correct_label": "spam"})

        data = resp.get_json()
        assert data["auto_retrained"] is False
        assert "retrain_threshold" in data
        assert data["retrain_threshold"] == 10


class TestStatsEndpoint:
    """Tests for GET /stats."""

    def test_stats_empty_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_predictions" in data
        assert "spam_predictions" in data
        assert "ham_predictions" in data
        assert "spam_ratio" in data
        assert "feedback_total" in data
        assert "feedback_unapplied" in data
        assert "feedback_accuracy" in data
        assert "last_training" in data
        assert "window_days" in data
        assert "model_trained" in data
        assert "training_stats" in data

    def test_stats_with_data(self, client, mock_state):
        db = mock_state["database"]
        db.save_prediction("S1", "B1", False, 0.9)
        db.save_prediction("S2", "B2", True, 0.95)
        db.save_prediction("S3", "B3", True, 0.9)
        db.save_feedback("S1", "B1", "ham", "ham")
        db.record_training("test", 10, 0)

        resp = client.get("/stats")
        data = resp.get_json()
        assert data["total_predictions"] == 3
        assert data["spam_predictions"] == 2
        assert data["ham_predictions"] == 1
        assert data["spam_ratio"] > 0.6
        assert data["feedback_total"] == 1
        assert data["feedback_accuracy"]["accuracy"] == 1.0
        assert data["last_training"]["source"] == "test"

    def test_stats_custom_days(self, client, mock_state):
        db = mock_state["database"]
        db.save_prediction("Old", "Body", False, 0.9, predicted_at="2020-01-01T00:00:00")
        db.save_prediction("New", "Body", True, 0.95)

        resp = client.get("/stats?days=30")
        data = resp.get_json()
        assert data["window_days"] == 30

    def test_stats_invalid_days_returns_400(self, client):
        resp = client.get("/stats?days=abc")
        assert resp.status_code == 400
