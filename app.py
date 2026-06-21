"""Flask HTTP service for spam email detection.

Endpoints
---------
GET  /               health check + model status
POST /predict        classify a single email (subject + body)
POST /predict_batch  classify many emails in one request
POST /train          add new training data and retrain the model
POST /feedback       correct a prediction; auto-retrains after enough feedback
GET  /history        browse stored prediction history
GET  /stats          dashboard statistics (totals, spam ratio, accuracy)

On startup the service loads every email under ``data/`` (ham/spam folders),
retraining the Naive Bayes model. If no data is present it first tries to
download the Enron-Spam corpus and, failing that, falls back to the bundled
synthetic sample emails so the API is always usable.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta

from flask import Flask, jsonify, request

import data_loader
from classifier import SpamClassifier
from database import PredictionDatabase, VALID_LABELS

DATA_DIR = os.environ.get("SPAM_DATA_DIR", "data")
DB_PATH = os.environ.get("SPAM_DB_PATH", "spam_predictions.db")
# Number of new, unapplied feedback records that trigger an automatic
# retraining so the model "learns" from user corrections.
FEEDBACK_RETRAIN_THRESHOLD = 50

app = Flask(__name__)

classifier = SpamClassifier()
database = PredictionDatabase(DB_PATH)
_lock = threading.Lock()


def _ensure_data() -> dict:
    """Make sure usable training data exists, downloading/seeding if needed."""
    _, stats = data_loader.load_dataset(DATA_DIR)
    if stats["ham"] > 0 and stats["spam"] > 0:
        return stats

    # No local data: try the real Enron-Spam corpus first.
    try:
        data_loader.download_enron_dataset(DATA_DIR)
    except Exception:
        pass
    _, stats = data_loader.load_dataset(DATA_DIR)
    if stats["ham"] > 0 and stats["spam"] > 0:
        return stats

    # Last resort: bundled synthetic samples.
    data_loader.seed_sample_data(DATA_DIR)
    _, stats = data_loader.load_dataset(DATA_DIR)
    return stats


def _retrain(source: str) -> dict:
    """Reload all disk data plus any unapplied feedback, then retrain.

    User feedback is folded in by reconstructing a ``Subject: ...`` email from
    each corrected (subject, body) pair so it shares the same parsing path as
    corpus emails. After training, the consumed feedback is marked applied and
    a row is written to the training log. Not thread-safe; callers must hold
    ``_lock``.
    """
    emails, stats = data_loader.load_dataset(DATA_DIR)
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

    train_stats = classifier.train(raw_texts, labels)
    train_stats["data"] = stats
    train_stats["feedback_incorporated"] = len(unapplied)
    train_stats["source"] = source

    database.mark_feedback_applied([fb["id"] for fb in unapplied])
    database.record_training(source, len(raw_texts), len(unapplied))
    return train_stats


def init_app() -> None:
    """Prepare data and train the model before serving requests."""
    data_stats = _ensure_data()
    app.logger.info(
        "Training data ready: ham=%s spam=%s total=%s",
        data_stats["ham"], data_stats["spam"], data_stats["total"],
    )
    with _lock:
        stats = _retrain("startup")
    app.logger.info(
        "Model trained: vocabulary=%s features=%s feedback=%s",
        stats["vocabulary_size"], stats["feature_count"],
        stats["feedback_incorporated"],
    )


def _extract_subject_body():
    """Read subject/body from a JSON or form-encoded request body."""
    data = request.get_json(silent=True)
    if data is not None:
        subject = (data.get("subject") or "").strip()
        body = (data.get("body") or "").strip()
        return subject, body

    subject = (request.form.get("subject") or "").strip()
    body = (request.form.get("body") or "").strip()
    return subject, body


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "spam-detector",
        "status": "ok",
        "model_trained": classifier.is_trained,
        "training_stats": classifier.training_stats,
        "last_training": database.get_last_training(),
        "predictions_logged": database.count(),
        "feedback_total": database.count_feedback(),
        "feedback_unapplied": database.count_unapplied_feedback(),
    })


@app.route("/predict", methods=["POST"])
def predict():
    subject, body = _extract_subject_body()
    if not subject and not body:
        return jsonify({"error": "Both 'subject' and 'body' are empty."}), 400

    try:
        with _lock:
            is_spam, confidence = classifier.predict(subject, body)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    database.save_prediction(subject, body, is_spam, confidence)
    return jsonify({
        "is_spam": bool(is_spam),
        "confidence": round(float(confidence), 6),
        "label": "spam" if is_spam else "ham",
        "subject": subject,
    })


@app.route("/train", methods=["POST"])
def train():
    """Add uploaded training data, then retrain on the whole corpus.

    Accepted inputs (combinable):
      * multipart files under form fields ``ham`` and/or ``spam``;
      * JSON ``{"ham": ["raw email", ...], "spam": ["raw email", ...]}``.
    """
    saved = {"ham": 0, "spam": 0}

    # 1. JSON payloads of raw email texts.
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
                data_loader.save_uploaded_email(DATA_DIR, label, name, text)
                saved[label] += 1

    # 2. Multipart file uploads grouped by field name ham/spam.
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
            data_loader.save_uploaded_email(DATA_DIR, label, name, text)
            saved[label] += 1

    if saved["ham"] == 0 and saved["spam"] == 0:
        return jsonify({
            "error": "No training data supplied. Send files under 'ham'/'spam' "
                     "fields or JSON {\"ham\":[...], \"spam\":[...]}.",
        }), 400

    try:
        with _lock:
            stats = _retrain("train_api")
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


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Classify many emails in a single request using one model pass.

    Request body (JSON)::

        {"emails": [{"subject": "...", "body": "..."}, ...]}

    A bare list ``[{...}, ...]`` is also accepted. Every email is classified
    in one batch (one feature matrix, one ``predict_proba`` call) and each
    result is persisted to history just like ``/predict``.
    """
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
                "error": f"emails[{index}] has both subject and body empty."
            }), 400
        emails.append((subject, body))

    try:
        with _lock:
            results = classifier.predict_batch(emails)
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


@app.route("/feedback", methods=["POST"])
def feedback():
    """Submit a correction for a prediction.

    Accepted JSON shapes::

        {"prediction_id": 12, "correct_label": "ham"}
        {"subject": "...", "body": "...", "correct_label": "spam"}

    When ``prediction_id`` is given the stored prediction supplies the
    subject/body and the model's original label. Otherwise the subject/body
    are taken from the request and the model is queried for its current label
    so that accuracy can be measured. Once the number of unapplied feedback
    records reaches ``FEEDBACK_RETRAIN_THRESHOLD`` the model is retrained
    automatically with the feedback folded in.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    correct_label = (payload.get("correct_label") or "").strip().lower()
    if correct_label not in VALID_LABELS:
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
            with _lock:
                is_spam, _ = classifier.predict(subject, body)
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
    with _lock:
        if database.count_unapplied_feedback() >= FEEDBACK_RETRAIN_THRESHOLD:
            try:
                retrain_result = _retrain("feedback_auto")
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
        response["retrain_threshold"] = FEEDBACK_RETRAIN_THRESHOLD
    return jsonify(response)


@app.route("/stats", methods=["GET"])
def stats():
    """Dashboard statistics over a recent window (default 7 days)."""
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
    data["model_trained"] = classifier.is_trained
    data["training_stats"] = classifier.training_stats
    return jsonify(data)


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
