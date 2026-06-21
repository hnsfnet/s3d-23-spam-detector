"""Flask application entry point for the spam detector service.

This module wires everything together: it reads configuration, creates the
shared state (classifier instance, database connection, thread lock),
registers the API routes, and provides ``init_app()`` to bootstrap the model
before serving requests.

All route handlers, business logic, and data access live in separate modules
(``routes``, ``classifier``, ``storage``, ``preprocessor``, ``feature_extractor``).
"""

from __future__ import annotations

import threading

from flask import Flask

import config
from classifier import create_classifier
from routes import init_app_state, register_routes
from storage import PredictionDatabase

app = Flask(__name__)

_state = {
    "classifier": create_classifier(),
    "database": PredictionDatabase(config.DB_PATH),
    "lock": threading.Lock(),
    "data_dir": config.DATA_DIR,
    "model_path": config.MODEL_PATH,
    "feedback_threshold": config.FEEDBACK_RETRAIN_THRESHOLD,
}

register_routes(app, _state)


def init_app() -> None:
    """Load or train the model before serving requests.

    Tries to load a saved model from disk first; if that fails, trains from
    scratch using the configured data directory.
    """
    init_app_state(app, _state)


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
