"""Naive Bayes spam classifier with hand-crafted email feature extraction.

The classifier combines two kinds of features:

* **Text features** -- TF-IDF over the lower-cased subject + body, which
  captures word-frequency signals (the classic bag-of-words approach).
* **Numeric features** -- engineered counts and ratios that are strong spam
  indicators: exclamation/dollar/percent counts, uppercase and digit ratios,
  URL/HTML tag counts, spam keywords, and email-header signals.

Both feature groups are non-negative and scaled to comparable ranges, so they
can be fed directly to a ``MultinomialNB`` classifier.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from data_loader import parse_email

# Spam-suggestive keywords (matched case-insensitively, as whole tokens).
_SPAM_KEYWORDS = (
    "free", "win", "winner", "prize", "guarantee", "guaranteed", "click",
    "urgent", "now", "congratulations", "lottery", "cash", "credit", "loan",
    "miracle", "amazing", "incredible", "risk", "selected", "limited",
    "offer", "discount", "deal", "money", "income", "profit", "bonus",
)

# Caps to keep numeric features in a range comparable with TF-IDF values.
_CAP_COUNT = 20.0
_CAP_LENGTH = 5000.0
_CAP_HEADERS = 20.0

# Patterns reused across emails.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CURRENCY_RE = re.compile(r"[$€£¥]")
_DIGIT_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Ratio clamped to [0, 1]; returns 0.0 when the denominator is 0."""
    if denominator <= 0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


def extract_numeric_features(subject: str, body: str, headers: dict) -> List[float]:
    """Return a fixed-length list of engineered, scaled numeric features.

    Every value lies in [0, 1] so the numeric block stays comparable in
    magnitude to the TF-IDF block, preventing length-style features from
    drowning out the text signal in the Naive Bayes model.
    """
    text = f"{subject}\n{body}"

    exclamations = text.count("!")
    dollars = len(_CURRENCY_RE.findall(text))
    percents = text.count("%")
    digits = len(_DIGIT_RE.findall(text))
    uppercase_letters = sum(1 for ch in text if ch.isupper())
    total_letters = sum(1 for ch in text if ch.isalpha())

    words = _WORD_RE.findall(text)
    total_words = len(words)
    caps_words = sum(1 for w in words if len(w) >= 3 and w.isupper())

    urls = len(_URL_RE.findall(text))
    html_tags = len(_HTML_TAG_RE.findall(text))
    has_html = 1.0 if html_tags > 0 else 0.0

    lowered = text.lower()
    keyword_hits = sum(lowered.count(kw) for kw in _SPAM_KEYWORDS)

    subject_letters = sum(1 for ch in subject if ch.isalpha())
    subject_upper = sum(1 for ch in subject if ch.isupper())
    subject_caps_ratio = _safe_ratio(subject_upper, subject_letters)

    body_length = len(body)

    features = [
        min(exclamations, _CAP_COUNT) / _CAP_COUNT,           # exclamation density
        min(dollars, _CAP_COUNT) / _CAP_COUNT,                # currency symbols
        min(percents, _CAP_COUNT) / _CAP_COUNT,               # percent signs
        _safe_ratio(uppercase_letters, total_letters),        # uppercase ratio
        _safe_ratio(caps_words, total_words),                 # ALL-CAPS word ratio
        _safe_ratio(digits, max(len(text), 1)),                # digit ratio
        min(urls, _CAP_COUNT) / _CAP_COUNT,                   # url count
        min(html_tags, _CAP_COUNT) / _CAP_COUNT,              # html tag count
        has_html,                                              # has html flag
        min(keyword_hits, _CAP_COUNT) / _CAP_COUNT,           # spam keyword hits
        subject_caps_ratio,                                    # subject caps ratio
        min(body_length, _CAP_LENGTH) / _CAP_LENGTH,          # body length
        1.0 if headers.get("Reply-To") else 0.0,              # reply-to present
        min(len(headers), _CAP_HEADERS) / _CAP_HEADERS,       # header count
    ]
    return features


NUMERIC_FEATURE_NAMES = [
    "exclamation_density",
    "currency_symbols",
    "percent_signs",
    "uppercase_ratio",
    "caps_word_ratio",
    "digit_ratio",
    "url_count",
    "html_tag_count",
    "has_html",
    "spam_keyword_hits",
    "subject_caps_ratio",
    "body_length",
    "has_reply_to",
    "header_count",
]


class SpamClassifier:
    """Multinomial Naive Bayes spam classifier."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
            norm="l2",
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        self.model = MultinomialNB(alpha=alpha)
        self.is_trained = False
        self.classes_: List[int] = []
        self.training_stats: Dict[str, int] = {}

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _document(subject: str, body: str) -> str:
        return f"{subject}\n{body}".lower()

    def _build_matrix(
        self, subjects, bodies, headers_list, fit: bool
    ) -> csr_matrix:
        documents = [self._document(s, b) for s, b in zip(subjects, bodies)]
        numeric = np.asarray(
            [
                extract_numeric_features(s, b, h)
                for s, b, h in zip(subjects, bodies, headers_list)
            ],
            dtype=np.float64,
        )
        if fit:
            x_text = self.vectorizer.fit_transform(documents)
        else:
            x_text = self.vectorizer.transform(documents)
        x_num = csr_matrix(numeric)
        return hstack([x_text, x_num], format="csr")

    # -- public API -------------------------------------------------------

    def train(self, raw_texts: List[str], labels: List[str]) -> Dict[str, int]:
        """Train on a list of raw email texts and ``'ham'``/``'spam'`` labels."""
        if not raw_texts:
            raise ValueError("Cannot train on an empty dataset.")

        subjects: List[str] = []
        bodies: List[str] = []
        headers_list: List[dict] = []
        for raw in raw_texts:
            subject, body, headers = parse_email(raw)
            subjects.append(subject)
            bodies.append(body)
            headers_list.append(headers)

        y = np.array([1 if str(l).lower() == "spam" else 0 for l in labels])
        if len(set(y)) < 2:
            raise ValueError("Training data must contain both ham and spam.")

        x = self._build_matrix(subjects, bodies, headers_list, fit=True)
        self.model.fit(x, y)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True

        self.training_stats = {
            "total": len(raw_texts),
            "ham": int(np.sum(y == 0)),
            "spam": int(np.sum(y == 1)),
            "vocabulary_size": int(len(self.vectorizer.vocabulary_)),
            "feature_count": int(x.shape[1]),
        }
        return self.training_stats

    def predict(self, subject: str, body: str) -> Tuple[bool, float]:
        """Return ``(is_spam, confidence)`` for a single email.

        ``confidence`` is the probability the model assigns to the predicted
        class, always in the range [0, 1].
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")

        x = self._build_matrix([subject], [body], [{}], fit=False)
        proba = self.model.predict_proba(x)[0]
        spam_index = self.classes_.index(1)
        spam_proba = float(proba[spam_index])
        is_spam = spam_proba >= 0.5
        confidence = spam_proba if is_spam else (1.0 - spam_proba)
        return is_spam, confidence

    def predict_batch(
        self, emails: List[Tuple[str, str]]
    ) -> List[Tuple[bool, float]]:
        """Classify many emails in one pass, reusing the trained model.

        A single feature matrix is built and a single ``predict_proba`` call
        is made, which is far cheaper than looping over ``predict`` when the
        batch is large.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained yet.")
        if not emails:
            return []

        subjects = [s for s, _ in emails]
        bodies = [b for _, b in emails]
        x = self._build_matrix(subjects, bodies, [{} for _ in emails], fit=False)
        proba = self.model.predict_proba(x)
        spam_index = self.classes_.index(1)
        results: List[Tuple[bool, float]] = []
        for row in proba:
            spam_proba = float(row[spam_index])
            is_spam = spam_proba >= 0.5
            confidence = spam_proba if is_spam else (1.0 - spam_proba)
            results.append((is_spam, confidence))
        return results
