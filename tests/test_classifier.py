"""Tests for the classifier module (BaseClassifier ABC, NaiveBayes, factory, save/load)."""

from __future__ import annotations

import os
import tempfile
from abc import ABC
from typing import Dict, List, Tuple

import pytest

from classifier import (
    BaseClassifier,
    NaiveBayesClassifier,
    create_classifier,
    load_classifier,
    register_classifier,
    _CLASSIFIER_REGISTRY,
)
import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_raw_emails():
    """Return a small list of raw email strings (with Subject: headers) for training."""
    return [
        "Subject: Meeting tomorrow\n\nHi team, let's meet tomorrow at 10am.",
        "Subject: Lunch invitation\n\nHey, want to grab lunch today?",
        "Subject: Project update\n\nThe latest deployment went well.",
        "Subject: Your invoice\n\nPlease find attached the invoice.",
        "Subject: Code review\n\nCould you please review my PR?",
        "Subject: Welcome\n\nWelcome to the team, glad you're here.",
        "Subject: Report\n\nAttached is the weekly report.",
        "Subject: Vacation\n\nI'd like to request vacation next week.",
        "Subject: FREE MONEY NOW!!! Click here!!!\n\nCongratulations you won $1000000!!! Click now to claim your free money!!! Guaranteed!!! No risk!!! Urgent!!! Reply immediately for your free cash prize!!! This is a limited time offer you cannot miss!!! Click right now to claim your winnings!!!",
        "Subject: Click now for free credit card!!!\n\nYou won $5000!!! Click here to claim your free cash!!! Guaranteed!!! Urgent!!! Free credit card!!! No risk!!! Amazing offer!!! Limited time only!!! Reply now for your free money!!!",
        "Subject: Get rich quick!!! Free loan!!!\n\nMake $10000 per week working from home!!! No experience needed!!! Guaranteed income!!! Click now for your free credit card and amazing loan offer!!! Risk free!!! Reply immediately to claim your free money!!!",
    ]


@pytest.fixture
def sample_labels():
    return ["ham", "ham", "ham", "ham", "ham", "ham", "ham", "ham", "spam", "spam", "spam"]


# ---------------------------------------------------------------------------
# BaseClassifier ABC tests
# ---------------------------------------------------------------------------

class TestBaseClassifierAbc:
    """Verify the BaseClassifier enforces the interface contract."""

    def test_base_classifier_is_abstract(self):
        assert issubclass(BaseClassifier, ABC)
        with pytest.raises(TypeError, match="abstract methods"):
            BaseClassifier()

    def test_naive_bayes_implements_interface(self):
        assert issubclass(NaiveBayesClassifier, BaseClassifier)
        clf = NaiveBayesClassifier()
        assert hasattr(clf, "train")
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_batch")
        assert hasattr(clf, "save")
        assert hasattr(clf, "load")
        assert hasattr(clf, "is_trained")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "training_stats")

    def test_new_classifier_must_subclass_base_classifier(self):
        """Verify register_classifier enforces subclassing."""

        class NotASubclass:
            pass

        with pytest.raises(TypeError, match="must subclass BaseClassifier"):
            register_classifier("bad", NotASubclass)


# ---------------------------------------------------------------------------
# Factory pattern tests
# ---------------------------------------------------------------------------

class TestClassifierFactory:
    """Tests for create_classifier() and the registry."""

    def test_create_naive_bayes_default(self):
        clf = create_classifier()
        assert isinstance(clf, NaiveBayesClassifier)
        assert clf.is_trained is False

    def test_create_naive_bayes_explicit(self):
        clf = create_classifier("naive_bayes")
        assert isinstance(clf, NaiveBayesClassifier)

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier_type"):
            create_classifier("unknown_type")

    def test_register_and_create_new_classifier(self):
        """Verify the extension mechanism works: register a custom classifier, create it."""

        class DummyClassifier(BaseClassifier):
            classifier_type = "dummy"

            def __init__(self):
                self.is_trained = False
                self.classes_ = []
                self.training_stats = {}

            def train(self, raw_texts, labels):
                self.is_trained = True
                self.classes_ = [0, 1]
                self.training_stats = {"total": len(raw_texts)}
                return self.training_stats

            def predict(self, subject, body):
                return False, 0.9

            def predict_batch(self, emails):
                return [(False, 0.9) for _ in emails]

            def save(self, path):
                pass

            @classmethod
            def load(cls, path):
                return cls()

        try:
            register_classifier("dummy", DummyClassifier)
            assert "dummy" in _CLASSIFIER_REGISTRY

            clf = create_classifier("dummy")
            assert isinstance(clf, DummyClassifier)
        finally:
            if "dummy" in _CLASSIFIER_REGISTRY:
                del _CLASSIFIER_REGISTRY["dummy"]


# ---------------------------------------------------------------------------
# NaiveBayesClassifier tests
# ---------------------------------------------------------------------------

class TestNaiveBayesClassifier:
    """Tests for the NaiveBayesClassifier concrete implementation."""

    def test_initial_state(self):
        clf = NaiveBayesClassifier()
        assert clf.is_trained is False
        assert clf.classes_ == []
        assert clf.training_stats == {}

    def test_train_sets_is_trained(self, sample_raw_emails, sample_labels):
        clf = NaiveBayesClassifier()
        stats = clf.train(sample_raw_emails, sample_labels)
        assert clf.is_trained is True
        assert stats["total"] == len(sample_raw_emails)
        assert stats["ham"] == 8
        assert stats["spam"] == 3
        assert stats["classifier_type"] == "naive_bayes"
        assert stats["vocabulary_size"] > 0
        assert stats["feature_count"] > 0

    def test_train_empty_dataset_raises(self):
        clf = NaiveBayesClassifier()
        with pytest.raises(ValueError, match="empty dataset"):
            clf.train([], [])

    def test_train_single_class_raises(self):
        clf = NaiveBayesClassifier()
        with pytest.raises(ValueError, match="both ham and spam"):
            clf.train(
                ["Subject: One\n\nBody", "Subject: Two\n\nBody"],
                ["ham", "ham"],
            )

    def test_predict_requires_trained_model(self):
        clf = NaiveBayesClassifier()
        with pytest.raises(RuntimeError, match="not trained yet"):
            clf.predict("test", "test")

    def test_predict_batch_requires_trained_model(self):
        clf = NaiveBayesClassifier()
        with pytest.raises(RuntimeError, match="not trained yet"):
            clf.predict_batch([("test", "test")])

    def test_predict_returns_correct_types(self, trained_classifier):
        is_spam, confidence = trained_classifier.predict("Hello", "World")
        assert isinstance(is_spam, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_batch_returns_list(self, trained_classifier):
        emails = [
            ("Hello", "Normal email content"),
            ("FREE MONEY", "Click now for free money!!!"),
            ("Meeting", "Let's meet tomorrow"),
        ]
        results = trained_classifier.predict_batch(emails)
        assert len(results) == 3
        for is_spam, confidence in results:
            assert isinstance(is_spam, bool)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0

    def test_predict_batch_empty_returns_empty(self, trained_classifier):
        assert trained_classifier.predict_batch([]) == []

    def test_can_correctly_classify_training_data(self, trained_classifier):
        """Verify the classifier can at least memorise its training data."""
        from tests.conftest import TEST_EMAILS, TEST_LABELS

        correct = 0
        for (subject, body), label in zip(TEST_EMAILS, TEST_LABELS):
            is_spam, _ = trained_classifier.predict(subject, body)
            predicted = "spam" if is_spam else "ham"
            if predicted == label:
                correct += 1

        accuracy = correct / len(TEST_EMAILS)
        assert accuracy >= 0.8, f"Training accuracy only {accuracy:.2f}, expected >= 0.8"

    def test_save_requires_trained_model(self, temp_model_path):
        clf = NaiveBayesClassifier()
        with pytest.raises(RuntimeError, match="untrained classifier"):
            clf.save(temp_model_path)

    def test_save_and_load_roundtrip(self, trained_classifier, temp_model_path):
        """Verify save -> load produces identical predictions."""
        trained_classifier.save(temp_model_path)
        assert os.path.exists(temp_model_path)

        loaded = load_classifier(temp_model_path)
        assert isinstance(loaded, NaiveBayesClassifier)
        assert loaded.is_trained is True
        assert loaded.classes_ == trained_classifier.classes_
        assert loaded.training_stats == trained_classifier.training_stats

        test_cases = [
            ("Hello world", "This is a normal email about meeting."),
            ("FREE MONEY NOW", "Click here for free guaranteed money!!!"),
            ("Project update", "Let's discuss the project timeline tomorrow."),
            ("Urgent!!! Win free iPhone", "Click now to claim your free iPhone!!!"),
        ]

        for subject, body in test_cases:
            orig_is_spam, orig_conf = trained_classifier.predict(subject, body)
            load_is_spam, load_conf = loaded.predict(subject, body)
            assert orig_is_spam == load_is_spam
            assert abs(orig_conf - load_conf) < 1e-6

    def test_naive_bayes_load_wrong_type_raises(self, trained_classifier, temp_model_path):
        trained_classifier.save(temp_model_path)

        import joblib
        payload = joblib.load(temp_model_path)
        payload["classifier_type"] = "other"
        joblib.dump(payload, temp_model_path)

        with pytest.raises(ValueError, match="unexpected classifier_type"):
            NaiveBayesClassifier.load(temp_model_path)

    def test_load_classifier_dispatches_by_type(self, trained_classifier, temp_model_path):
        trained_classifier.save(temp_model_path)
        loaded = load_classifier(temp_model_path)
        assert isinstance(loaded, NaiveBayesClassifier)
        assert loaded.classifier_type == "naive_bayes"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_classifier("/nonexistent/path/model.joblib")

    def test_custom_alpha_parameter(self):
        clf = NaiveBayesClassifier(alpha=0.5)
        assert clf.model.alpha == 0.5

    def test_default_alpha_from_config(self):
        clf = NaiveBayesClassifier()
        assert clf.model.alpha == config.NB_ALPHA

    def test_atomic_save_does_not_leave_temp_file(self, trained_classifier, temp_model_path):
        trained_classifier.save(temp_model_path)
        directory = os.path.dirname(temp_model_path)
        temp_files = [f for f in os.listdir(directory) if f.startswith(".spam_model_")]
        assert temp_files == [], f"Leftover temp files: {temp_files}"


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------

class TestClassifierHelpers:
    """Tests for the static helper methods on BaseClassifier."""

    def test_labels_to_y(self):
        y = BaseClassifier._labels_to_y(["ham", "spam", "Ham", "SPAM", "ham"])
        assert list(y) == [0, 1, 0, 1, 0]

    def test_proba_to_result_spam(self):
        classes = [0, 1]
        proba = [0.2, 0.8]
        is_spam, confidence = BaseClassifier._proba_to_result(proba, classes)
        assert is_spam is True
        assert abs(confidence - 0.8) < 1e-6

    def test_proba_to_result_ham(self):
        classes = [0, 1]
        proba = [0.85, 0.15]
        is_spam, confidence = BaseClassifier._proba_to_result(proba, classes)
        assert is_spam is False
        assert abs(confidence - 0.85) < 1e-6

    def test_proba_to_result_threshold(self):
        classes = [0, 1]
        proba = [0.5, 0.5]
        is_spam, confidence = BaseClassifier._proba_to_result(proba, classes)
        assert is_spam is True
        assert abs(confidence - 0.5) < 1e-6

    def test_parse_emails(self):
        raw = ["Subject: Test\n\nHello world", "Subject: Hi\n\nHow are you?"]
        subjects, bodies, headers_list = BaseClassifier._parse_emails(raw)
        assert subjects == ["Test", "Hi"]
        assert bodies == ["Hello world", "How are you?"]
        assert len(headers_list) == 2
