"""Shared fixtures for the spam detector test suite."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Dict, List, Tuple

import pytest
from flask import Flask

import config
from classifier import BaseClassifier, create_classifier
from routes import register_routes
from storage import PredictionDatabase


# ---------------------------------------------------------------------------
# Small email dataset used for classifier training tests
# ---------------------------------------------------------------------------

TEST_EMAILS: List[Tuple[str, str]] = [
    ("Meeting tomorrow at 10am", "Hi team, let's meet tomorrow at 10am to discuss the project. Thanks, John"),
    ("Lunch invitation", "Hey, want to grab lunch today? There's a new place downtown that looks great."),
    ("Project update", "The latest deployment went well. Here's the summary of what was shipped."),
    ("Your invoice is ready", "Please find attached the invoice for this month's services."),
    ("Weekly report", "Attached is the weekly report for your review. Let me know if you have questions."),
    ("Birthday party", "You're invited to my birthday party this Saturday! Bring your friends."),
    ("Code review request", "Could you please review my PR when you get a chance? Thanks!"),
    ("Vacation request", "I'd like to request vacation from July 1st to July 10th. Thanks for your consideration."),
    ("Team building event", "We're having a team building event next Friday. Please RSVP by Wednesday."),
    ("Welcome to the team", "Welcome aboard! We're excited to have you join the team."),
    ("WINNER!!! Claim your free prize now!!!", "Congratulations!!! You have been selected as the WINNER of $1,000,000! Click here to claim your free money now!!! Guaranteed!!! No risk!!! Amazing offer!!! Urgent!!! Reply immediately for your free cash prize and credit card offer. This is a limited time deal you cannot miss. Click this link right now to claim your winnings: http://scam.example.com/free-money"),
    ("Free money!!! Click now!!!", "You won $5000!!! Click here to claim your free cash!!! Guaranteed!!! Urgent!!! Free credit card!!! No risk!!! Amazing offer!!! Limited time only!!! Reply now for your free money and guaranteed prize. This is your chance to get rich quick! Click now!!! http://evil.example.com/free"),
    ("Get rich quick!!! Free credit card!!!", "Make $10,000 per week working from home!!! No experience needed!!! Guaranteed income!!! Click now for your free credit card and amazing loan offer!!! Risk free!!! Limited spots available!!! Reply immediately to claim your free money and start earning today!!!"),
    ("Hot singles in your area!!! Click now!!!", "Meet hot singles tonight!!! 100% free!!! No credit card needed!!! Click now for instant access!!! Guaranteed results!!! Urgent offer expires soon!!! Reply now for your free membership and amazing discount!!! This is a limited time deal you don't want to miss!!! Click immediately!!! http://fake.example.com/dating"),
    ("Your account has been suspended!!! Verify now!!!", "URGENT!!! Your bank account has been suspended!!! Click here to verify your password and account information immediately!!! Failure to respond within 24 hours will result in account closure!!! Guaranteed security!!! Risk free!!! Reply now to restore your account access and claim your free bonus!!! This is an amazing opportunity!!! Click immediately or lose everything!!! http://phish.example.com/verify"),
    ("Free gift card!!! Claim now!!!", "You won a free $1000 gift card!!! Click here to claim your amazing prize now!!! Guaranteed delivery!!! No purchase necessary!!! Limited time offer expires soon!!! Reply now for your free gift card and bonus credit card offer!!! This deal won't last!!! Click immediately to claim your free money and gifts!!! http://scam.example.com/giftcard"),
    ("Lose 30 pounds in 1 week!!! Miracle pill!!!", "Amazing weight loss miracle!!! Lose 30 pounds in just 1 week with our revolutionary new pill!!! Guaranteed results!!! No diet or exercise needed!!! Click now for your free trial and amazing discount!!! Risk free!!! Limited stock available!!! Reply immediately to claim your free bottle and start losing weight today!!! This is incredible!!! http://fake.example.com/diet"),
    ("You are pre-approved for a $50,000 loan!!!", "Great news!!! You are pre-approved for a $50,000 loan with 0% interest!!! Guaranteed approval no matter your credit score!!! Click now to claim your free money and amazing credit card offer!!! No risk!!! Limited time only!!! Reply immediately to get your free loan and start spending today!!! This offer won't last!!! Click now!!! http://evil.example.com/loan"),
]

TEST_LABELS: List[str] = ["ham"] * 10 + ["spam"] * 8


@pytest.fixture
def temp_db(tmp_path) -> str:
    """Return a path to a temporary SQLite database file."""
    return str(tmp_path / "test_spam.db")


@pytest.fixture
def db(temp_db) -> PredictionDatabase:
    """Return an empty PredictionDatabase backed by a temp file."""
    return PredictionDatabase(temp_db)


@pytest.fixture
def temp_model_path(tmp_path) -> str:
    """Return a path for a temporary model file."""
    return str(tmp_path / "test_model.joblib")


@pytest.fixture
def temp_data_dir(tmp_path) -> str:
    """Return a path to a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def trained_classifier() -> BaseClassifier:
    """Return a NaiveBayesClassifier trained on the TEST_EMAILS dataset."""
    clf = create_classifier()
    raw_texts = [f"Subject: {s}\n\n{b}" for s, b in TEST_EMAILS]
    clf.train(raw_texts, TEST_LABELS)
    return clf


@pytest.fixture
def mock_state(temp_db, temp_model_path, temp_data_dir) -> Dict:
    """Create a state dict suitable for route testing, with a mock classifier."""

    class MockClassifier(BaseClassifier):
        classifier_type = "mock"

        def __init__(self):
            self.is_trained = True
            self.classes_ = [0, 1]
            self.training_stats = {"total": 18, "ham": 10, "spam": 8}

        def train(self, raw_texts, labels):
            self.training_stats = {
                "total": len(raw_texts),
                "ham": sum(1 for l in labels if l == "ham"),
                "spam": sum(1 for l in labels if l == "spam"),
            }
            return self.training_stats

        def predict(self, subject, body):
            text = f"{subject} {body}".lower()
            spammy = any(kw in text for kw in ["free", "win", "click", "urgent", "guaranteed", "free money"])
            if spammy:
                return True, 0.95
            return False, 0.9

        def predict_batch(self, emails):
            return [self.predict(s, b) for s, b in emails]

        def save(self, path):
            pass

        @classmethod
        def load(cls, path):
            return cls()

    return {
        "classifier": MockClassifier(),
        "database": PredictionDatabase(temp_db),
        "lock": threading.Lock(),
        "data_dir": temp_data_dir,
        "model_path": temp_model_path,
        "feedback_threshold": 2,
    }


@pytest.fixture
def app(mock_state) -> Flask:
    """Return a Flask test app wired up with our routes and mock state."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    register_routes(flask_app, mock_state)
    return flask_app


@pytest.fixture
def client(app) -> Flask.test_client_class:
    """Return a Flask test client."""
    return app.test_client()
