"""Spam classifier abstraction and concrete implementations.

The ``BaseClassifier`` defines the interface every classifier must satisfy.
``NaiveBayesClassifier`` is the Multinomial Naive Bayes implementation.
``create_classifier()`` is the factory that returns a classifier by type name,
making it easy to add new algorithms without touching the rest of the codebase.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

import config
from data_loader import parse_email
from feature_extractor import FeatureExtractor

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class BaseClassifier(ABC):
    """Abstract base class for spam classifiers.

    Every concrete classifier must be trainable from raw email texts, able to
    predict spam/ham for single and batch inputs, and support atomic
    save/load for persistence.
    """

    is_trained: bool
    classes_: List[int]
    training_stats: Dict[str, int]

    @abstractmethod
    def train(self, raw_texts: List[str], labels: List[str]) -> Dict[str, int]:
        """Train on raw email texts with ``'ham'``/``'spam'`` labels."""
        ...

    @abstractmethod
    def predict(self, subject: str, body: str) -> Tuple[bool, float]:
        """Return ``(is_spam, confidence)`` for a single email."""
        ...

    @abstractmethod
    def predict_batch(
        self, emails: List[Tuple[str, str]]
    ) -> List[Tuple[bool, float]]:
        """Classify many ``(subject, body)`` pairs in one pass."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Atomically save the trained classifier to ``path``."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Load a previously saved classifier from ``path``."""
        ...

    @staticmethod
    def _parse_emails(raw_texts: List[str]) -> Tuple[List[str], List[str], List[dict]]:
        subjects: List[str] = []
        bodies: List[str] = []
        headers_list: List[dict] = []
        for raw in raw_texts:
            subject, body, headers = parse_email(raw)
            subjects.append(subject)
            bodies.append(body)
            headers_list.append(headers)
        return subjects, bodies, headers_list

    @staticmethod
    def _labels_to_y(labels: List[str]) -> np.ndarray:
        return np.array([1 if str(l).lower() == "spam" else 0 for l in labels])

    @staticmethod
    def _proba_to_result(prob_row: np.ndarray, classes_: List[int]) -> Tuple[bool, float]:
        spam_index = classes_.index(1)
        spam_proba = float(prob_row[spam_index])
        is_spam = spam_proba >= 0.5
        confidence = spam_proba if is_spam else (1.0 - spam_proba)
        return is_spam, confidence

    @staticmethod
    def _atomic_save(payload: dict, path: str) -> None:
        """Write ``payload`` to ``path`` atomically using temp + rename."""
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required to save/load models.")
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".joblib.tmp",
            prefix=".spam_model_",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                joblib.dump(payload, handle)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_load(path: str, required_keys: Tuple[str, ...]) -> dict:
        if not JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required to save/load models.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        for key in required_keys:
            if key not in payload:
                raise ValueError(f"Corrupted model file: missing '{key}'")
        return payload


class NaiveBayesClassifier(BaseClassifier):
    """Multinomial Naive Bayes spam classifier.

    Combines TF-IDF text features with engineered numeric features, both of
    which are non-negative and well-suited to MultinomialNB.
    """

    classifier_type = "naive_bayes"

    def __init__(self, alpha: float | None = None) -> None:
        from sklearn.naive_bayes import MultinomialNB
        self.features = FeatureExtractor()
        self.model = MultinomialNB(alpha=alpha if alpha is not None else config.NB_ALPHA)
        self.is_trained = False
        self.classes_: List[int] = []
        self.training_stats: Dict[str, int] = {}

    # -- training ---------------------------------------------------------

    def train(self, raw_texts: List[str], labels: List[str]) -> Dict[str, int]:
        if not raw_texts:
            raise ValueError("Cannot train on an empty dataset.")

        subjects, bodies, headers_list = self._parse_emails(raw_texts)
        y = self._labels_to_y(labels)
        if len(set(y)) < 2:
            raise ValueError("Training data must contain both ham and spam.")

        x = self.features.fit_transform(subjects, bodies, headers_list)
        self.model.fit(x, y)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True

        self.training_stats = {
            "total": len(raw_texts),
            "ham": int(np.sum(y == 0)),
            "spam": int(np.sum(y == 1)),
            "vocabulary_size": self.features.vocabulary_size,
            "feature_count": self.features.feature_count,
            "classifier_type": self.classifier_type,
        }
        return self.training_stats

    # -- prediction --------------------------------------------------

    def predict(self, subject: str, body: str) -> Tuple[bool, float]:
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")
        x = self.features.transform([subject], [body], [{}])
        proba = self.model.predict_proba(x)[0]
        return self._proba_to_result(proba, self.classes_)

    def predict_batch(
        self, emails: List[Tuple[str, str]]
    ) -> List[Tuple[bool, float]]:
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")
        if not emails:
            return []
        subjects = [s for s, _ in emails]
        bodies = [b for _, b in emails]
        headers_list = [{} for _ in emails]
        x = self.features.transform(subjects, bodies, headers_list)
        proba = self.model.predict_proba(x)
        return [self._proba_to_result(row, self.classes_) for row in proba]

    # -- persistence -------------------------------------------------

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained classifier.")
        payload = {
            "classifier_type": self.classifier_type,
            "features": self.features.to_dict(),
            "model": self.model,
            "classes_": self.classes_,
            "training_stats": self.training_stats,
            "is_trained": True,
        }
        self._atomic_save(payload, path)

    @classmethod
    def load(cls, path: str) -> "NaiveBayesClassifier":
        required = ("classifier_type", "features", "model", "classes_", "training_stats")
        payload = cls._atomic_load(path, required)
        ctype = payload.get("classifier_type")
        if ctype != "naive_bayes":
            raise ValueError(f"Model has unexpected classifier_type: {ctype}")
        instance = cls()
        instance.features = FeatureExtractor.from_dict(payload["features"])
        instance.model = payload["model"]
        instance.classes_ = payload["classes_"]
        instance.training_stats = payload["training_stats"]
        instance.is_trained = bool(payload.get("is_trained", True))
        return instance


_CLASSIFIER_REGISTRY = {
    "naive_bayes": NaiveBayesClassifier,
}


def create_classifier(classifier_type: str | None = None) -> BaseClassifier:
    """Factory: return a new untrained classifier of the requested type.

    ``classifier_type`` defaults to ``config.CLASSIFIER_TYPE``. Raises
    ``ValueError`` when the type is not registered.
    """
    ctype = classifier_type or config.CLASSIFIER_TYPE
    cls = _CLASSIFIER_REGISTRY.get(ctype)
    if cls is None:
        raise ValueError(
            f"Unknown classifier_type '{ctype}'. "
            f"Available: {', '.join(_CLASSIFIER_REGISTRY)}"
        )
    return cls()


def load_classifier(path: str) -> BaseClassifier:
    """Load a saved classifier from ``path``, dispatching by stored type."""
    if not JOBLIB_AVAILABLE:
        raise RuntimeError("joblib is required to save/load models.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    ctype = payload.get("classifier_type", "naive_bayes")
    cls = _CLASSIFIER_REGISTRY.get(ctype)
    if cls is None:
        raise ValueError(f"Unknown saved classifier_type '{ctype}'")
    return cls.load(path)


def register_classifier(name: str, cls: type) -> None:
    """Register a new classifier implementation (for plugins / future extensions)."""
    if not issubclass(cls, BaseClassifier):
        raise TypeError("Classifier must subclass BaseClassifier")
    _CLASSIFIER_REGISTRY[name] = cls
