"""HTTP API route definitions for the spam detector service.

All routes are registered via ``register_routes(app, state)`` where ``state`` is
a dict holding shared mutable state (classifier instance, database, lock, etc.).
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict

from flask import Flask, jsonify, request

import config
import data_loader
from classifier import BaseClassifier, create_classifier, load_classifier
from storage import PredictionDatabase


def _extract_subject_body():
    """Read subject/body from a JSON or form-encoded request body."""
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        subject = (data.get("subject") or "").strip()
        body = (data.get("body") or "").strip()
        return subject, body

    subject = (request.form.get("subject") or "").strip()
    body = (request.form.get("body") or "").strip()
    return subject, body


def _ensure_data(data_dir: str, app: Flask) -> dict:
    """Make sure usable training data exists, downloading/seeding if needed."""
    _, stats = data_loader.load_dataset(data_dir)
    if stats["ham"] > 0 and stats["spam"] > 0:
        return stats

    try:
        data_loader.download_enron_dataset(data_dir)
    except Exception:
        pass
    _, stats = data_loader.load_dataset(data_dir)
    if stats["ham"] > 0 and stats["spam"] > 0:
        return stats

    data_loader.seed_sample_data(data_dir)
    _, stats = data_loader.load_dataset(data_dir)
    return stats


def _retrain(state: Dict[str, Any], source: str, app: Flask) -> dict:
    """Reload all disk data plus any unapplied feedback, then retrain.

    Replaces ``state["classifier"]`` with the freshly trained instance, saves
    it atomically, and updates the database. Not thread-safe; callers must hold
    the state lock.
    """
    data_dir = state["data_dir"]
    model_path = state["model_path"]
    database: PredictionDatabase = state["database"]

    emails, stats = data_loader.load_dataset(data_dir)
    raw_texts = [text for text, _ in emails]
    labels = [label for _, label in emails]

    unapplied = database.get_unapplied_feedback()
    for fb in unapplied:
        raw_texts.append(f"Subject: {fb['subject']}\n\n{fb['body']}")
        labels.append(fb["correct_label"])

    if not raw_texts:
        raise RuntimeError("Not enough training data: need at least one email.")

    label_set = {str(l).lower() for l in labels}
    if not {"ham", "spam"}.issubset(label_set):
        raise RuntimeError(
            "Not enough training data: need both ham and spam emails."
        )

    classifier: BaseClassifier = create_classifier()
    train_stats = classifier.train(raw_texts, labels)
    train_stats["data"] = stats
    train_stats["feedback_incorporated"] = len(unapplied)
    train_stats["source"] = source

    try:
        classifier.save(model_path)
        train_stats["model_saved"] = True
        train_stats["model_path"] = os.path.abspath(model_path)
    except Exception as exc:
        train_stats["model_saved"] = False
        train_stats["model_save_error"] = str(exc)
        app.logger.warning("Failed to save model to %s: %s", model_path, exc)

    state["classifier"] = classifier
    database.mark_feedback_applied([fb["id"] for fb in unapplied])
    database.record_training(source, len(raw_texts), len(unapplied))
    return train_stats


def init_app_state(app: Flask, state: Dict[str, Any]) -> None:
    """Prepare data and train or load the model before serving requests.

    Tries to load a previously saved model first. If the model file is missing
    or corrupted, falls back to training from scratch.
    """
    data_dir = state["data_dir"]
    model_path = state["model_path"]
    lock: threading.Lock = state["lock"]

    data_stats = _ensure_data(data_dir, app)
    app.logger.info(
        "Training data ready: ham=%s spam=%s total=%s",
        data_stats["ham"], data_stats["spam"], data_stats["total"],
    )

    loaded = False
    if os.path.exists(model_path):
        try:
            state["classifier"] = load_classifier(model_path)
            app.logger.info(
                "Model loaded from %s: vocabulary=%s features=%s",
                model_path,
                state["classifier"].training_stats.get("vocabulary_size"),
                state["classifier"].training_stats.get("feature_count"),
            )
            loaded = True
        except (FileNotFoundError, ValueError, RuntimeError, Exception) as exc:
            app.logger.warning(
                "Failed to load model from %s (%s), will retrain from scratch.",
                model_path, exc,
            )
            try:
                os.unlink(model_path)
                app.logger.info("Removed corrupted model file %s", model_path)
            except OSError:
                pass

    if not loaded:
        with lock:
            stats = _retrain(state, "startup", app)
        app.logger.info(
            "Model trained: vocabulary=%s features=%s feedback=%s saved=%s",
            stats["vocabulary_size"], stats["feature_count"],
            stats["feedback_incorporated"], stats.get("model_saved"),
        )


def register_routes(app: Flask, state: Dict[str, Any]) -> None:
    """Register all API routes on ``app`` using the shared ``state`` dict."""

    database: PredictionDatabase = state["database"]
    lock: threading.Lock = state["lock"]

    def get_cls() -> BaseClassifier:
        return state["classifier"]

    @app.route("/", methods=["GET"])
    def health():
        return jsonify({
            "service": "spam-detector",
            "status": "ok",
            "classifier_type": config.CLASSIFIER_TYPE,
            "model_trained": get_cls().is_trained,
            "training_stats": get_cls().training_stats,
            "last_training": database.get_last_training(),
            "predictions_logged": database.count(),
            "feedback_total": database.count_feedback(),
            "feedback_unapplied": database.count_unapplied_feedback(),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        subject, body = _extract_subject_body()
        if not subject and not body:
            return jsonify({"error": "邮件内容不能为空"}), 400

        try:
            with lock:
                is_spam, confidence = get_cls().predict(subject, body)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

        database.save_prediction(subject, body, is_spam, confidence)
        return jsonify({
            "is_spam": bool(is_spam),
            "confidence": round(float(confidence), 6),
            "label": "spam" if is_spam else "ham",
            "subject": subject,
        })

    @app.route("/predict_batch", methods=["POST"])
    def predict_batch():
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            emails_payload = payload.get("emails")
        elif isinstance(payload, list):
            emails_payload = payload
        else:
            emails_payload = None

        if not isinstance(emails_payload, list) or not emails_payload:
            return jsonify({
                "error": "Provide a non-empty 'emails' list of {subject, body}."
            }), 400

        emails = []
        for index, item in enumerate(emails_payload):
            if not isinstance(item, dict):
                return jsonify({
                    "error": f"emails[{index}] must be an object with subject/body."
                }), 400
            subject = (item.get("subject") or "").strip()
            body = (item.get("body") or "").strip()
            if not subject and not body:
                return jsonify({
                    "error": f"emails[{index}] 邮件内容不能为空"
                }), 400
            emails.append((subject, body))

        try:
            with lock:
                results = get_cls().predict_batch(emails)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

        serialized = []
        for (subject, body), (is_spam, confidence) in zip(emails, results):
            database.save_prediction(subject, body, is_spam, confidence)
            serialized.append({
                "index": len(serialized),
                "subject": subject,
                "is_spam": bool(is_spam),
                "confidence": round(float(confidence), 6),
                "label": "spam" if is_spam else "ham",
            })

        spam_count = sum(1 for r in serialized if r["is_spam"])
        return jsonify({
            "count": len(serialized),
            "spam_count": spam_count,
            "ham_count": len(serialized) - spam_count,
            "results": serialized,
        })

    @app.route("/train", methods=["POST"])
    def train():
        data_dir = state["data_dir"]
        saved = {"ham": 0, "spam": 0}

        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            for label in ("ham", "spam"):
                items = payload.get(label) or []
                if isinstance(items, str):
                    items = [items]
                for index, text in enumerate(items):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    name = f"json_{label}_{stamp}_{index}.txt"
                    data_loader.save_uploaded_email(data_dir, label, name, text)
                    saved[label] += 1

        for label in ("ham", "spam"):
            uploads = request.files.getlist(label)
            for index, storage in enumerate(uploads):
                content = storage.read()
                if not content:
                    continue
                text = content.decode("utf-8", errors="replace")
                if not text.strip():
                    continue
                name = f"upload_{label}_{index}_{storage.filename or 'email.txt'}"
                data_loader.save_uploaded_email(data_dir, label, name, text)
                saved[label] += 1

        if saved["ham"] == 0 and saved["spam"] == 0:
            return jsonify({
                "error": "No training data supplied. Send files under 'ham'/'spam' "
                         "fields or JSON {\"ham\":[...], \"spam\":[...]}.",
            }), 400

        try:
            with lock:
                stats = _retrain(state, "train_api", app)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({
            "status": "trained",
            "added": saved,
            "training_stats": stats,
        })

    @app.route("/history", methods=["GET"])
    def history():
        try:
            limit = max(1, min(int(request.args.get("limit", 100)), 1000))
            offset = max(0, int(request.args.get("offset", 0)))
        except ValueError:
            return jsonify({"error": "'limit' and 'offset' must be integers."}), 400

        return jsonify({
            "count": database.count(),
            "limit": limit,
            "offset": offset,
            "records": database.get_history(limit=limit, offset=offset),
        })

    @app.route("/feedback", methods=["POST"])
    def feedback():
        feedback_threshold = state["feedback_threshold"]

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        correct_label = (payload.get("correct_label") or "").strip().lower()
        if correct_label not in config.VALID_LABELS:
            return jsonify({
                "error": "'correct_label' must be 'spam' or 'ham'."
            }), 400

        prediction_id = payload.get("prediction_id")
        subject = body = ""
        predicted_label = None

        if prediction_id is not None:
            try:
                prediction_id = int(prediction_id)
            except (TypeError, ValueError):
                return jsonify({"error": "'prediction_id' must be an integer."}), 400
            record = database.get_prediction(prediction_id)
            if record is None:
                return jsonify({"error": f"prediction_id {prediction_id} not found."}), 404
            subject = record["subject"] or ""
            body = record["body"] or ""
            predicted_label = "spam" if record["is_spam"] else "ham"
        else:
            subject = (payload.get("subject") or "").strip()
            body = (payload.get("body") or "").strip()
            if not subject and not body:
                return jsonify({
                    "error": "Provide 'prediction_id' or a non-empty 'subject'/'body'."
                }), 400
            try:
                with lock:
                    is_spam, _ = get_cls().predict(subject, body)
            except RuntimeError as exc:
                return jsonify({"error": str(exc)}), 503
            predicted_label = "spam" if is_spam else "ham"

        feedback_id = database.save_feedback(
            subject=subject,
            body=body,
            predicted_label=predicted_label,
            correct_label=correct_label,
            prediction_id=prediction_id,
        )

        retrain_result = None
        retrain_error = None
        with lock:
            if database.count_unapplied_feedback() >= feedback_threshold:
                try:
                    retrain_result = _retrain(state, "feedback_auto", app)
                except RuntimeError as exc:
                    retrain_error = str(exc)

        response = {
            "status": "recorded",
            "feedback_id": feedback_id,
            "predicted_label": predicted_label,
            "correct_label": correct_label,
            "corrected": predicted_label != correct_label,
            "unapplied_feedback": database.count_unapplied_feedback(),
        }
        if retrain_result is not None:
            response["auto_retrained"] = True
            response["retrain_stats"] = retrain_result
        elif retrain_error is not None:
            response["auto_retrained"] = False
            response["retrain_error"] = retrain_error
        else:
            response["auto_retrained"] = False
            response["retrain_threshold"] = feedback_threshold
        return jsonify(response)

    @app.route("/stats", methods=["GET"])
    def stats():
        try:
            days = int(request.args.get("days", 7))
        except ValueError:
            return jsonify({"error": "'days' must be an integer."}), 400
        days = max(1, min(days, 3650))
        since_iso = (datetime.utcnow() - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        data = database.get_stats(since_iso=since_iso)
        data["window_days"] = days
        data["model_trained"] = get_cls().is_trained
        data["training_stats"] = get_cls().training_stats
        return jsonify(data)
