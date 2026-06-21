"""Flask HTTP service for spam email detection.

Endpoints
---------
GET  /           health check + model status
POST /predict    classify a single email (subject + body)
POST /train      add new training data and retrain the model
GET  /history    browse stored prediction history

On startup the service loads every email under ``data/`` (ham/spam folders),
retraining the Naive Bayes model. If no data is present it first tries to
download the Enron-Spam corpus and, failing that, falls back to the bundled
synthetic sample emails so the API is always usable.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime

from flask import Flask, jsonify, request

import data_loader
from classifier import SpamClassifier
from database import PredictionDatabase

DATA_DIR = os.environ.get("SPAM_DATA_DIR", "data")
DB_PATH = os.environ.get("SPAM_DB_PATH", "spam_predictions.db")

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


def _train_from_disk() -> dict:
    """Load all data from disk and (re)train the classifier. Not thread-safe."""
    emails, stats = data_loader.load_dataset(DATA_DIR)
    if not emails or stats["ham"] == 0 or stats["spam"] == 0:
        raise RuntimeError(
            "Not enough training data: need both ham and spam emails."
        )
    raw_texts = [text for text, _ in emails]
    labels = [label for _, label in emails]
    train_stats = classifier.train(raw_texts, labels)
    train_stats["data"] = stats
    return train_stats


def init_app() -> None:
    """Prepare data and train the model before serving requests."""
    data_stats = _ensure_data()
    app.logger.info(
        "Training data ready: ham=%s spam=%s total=%s",
        data_stats["ham"], data_stats["spam"], data_stats["total"],
    )
    with _lock:
        stats = _train_from_disk()
    app.logger.info(
        "Model trained: vocabulary=%s features=%s",
        stats["vocabulary_size"], stats["feature_count"],
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
        "predictions_logged": database.count(),
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
            stats = _train_from_disk()
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


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
