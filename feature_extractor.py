"""Feature extraction for spam email classification.

Two feature groups are combined:

* **Text features** -- TF-IDF over subject + body, capturing word-frequency
  signals (bag-of-words).
* **Numeric features** -- engineered counts and ratios that are strong spam
  indicators (exclamation marks, uppercase ratio, URL counts, keyword hits,
  email-header signals, etc.).

The feature extractor is model-agnostic: it produces sparse feature matrices
that can be fed to any scikit-learn compatible classifier.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from preprocessor import tokenize, make_document

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CURRENCY_RE = re.compile(r"[$€£¥]")
_DIGIT_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[A-Za-z]+")


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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(max(numerator / denominator, 0.0), 1.0)


def extract_numeric_features(subject: str, body: str, headers: dict) -> List[float]:
    """Return a fixed-length list of engineered, scaled numeric features.

    Every value lies in [0, 1] so the numeric block stays comparable in
    magnitude to the TF-IDF block.
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
    keyword_hits = sum(lowered.count(kw) for kw in config.SPAM_KEYWORDS)

    subject_letters = sum(1 for ch in subject if ch.isalpha())
    subject_upper = sum(1 for ch in subject if ch.isupper())
    subject_caps_ratio = _safe_ratio(subject_upper, subject_letters)

    body_length = len(body)

    features = [
        min(exclamations, config.CAP_COUNT) / config.CAP_COUNT,
        min(dollars, config.CAP_COUNT) / config.CAP_COUNT,
        min(percents, config.CAP_COUNT) / config.CAP_COUNT,
        _safe_ratio(uppercase_letters, total_letters),
        _safe_ratio(caps_words, total_words),
        _safe_ratio(digits, max(len(text), 1)),
        min(urls, config.CAP_COUNT) / config.CAP_COUNT,
        min(html_tags, config.CAP_COUNT) / config.CAP_COUNT,
        has_html,
        min(keyword_hits, config.CAP_COUNT) / config.CAP_COUNT,
        subject_caps_ratio,
        min(body_length, config.CAP_LENGTH) / config.CAP_LENGTH,
        1.0 if headers.get("Reply-To") else 0.0,
        min(len(headers), config.CAP_HEADERS) / config.CAP_HEADERS,
    ]
    return features


class FeatureExtractor:
    """Combined TF-IDF + numeric feature extractor.

    The extractor must be ``fit`` on training data before it can ``transform``
    new emails.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            tokenizer=tokenize,
            token_pattern=None,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            norm=config.TFIDF_NORM,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=(config.TFIDF_NGRAM_MIN, config.TFIDF_NGRAM_MAX),
        )
        self._is_fitted = False

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._is_fitted else 0

    @property
    def feature_count(self) -> int:
        return self.vocabulary_size + len(NUMERIC_FEATURE_NAMES)

    def fit_transform(
        self,
        subjects: List[str],
        bodies: List[str],
        headers_list: List[dict],
    ) -> csr_matrix:
        documents = [make_document(s, b) for s, b in zip(subjects, bodies)]
        x_text = self.vectorizer.fit_transform(documents)
        self._is_fitted = True
        return self._stack_numeric(x_text, subjects, bodies, headers_list)

    def transform(
        self,
        subjects: List[str],
        bodies: List[str],
        headers_list: List[dict],
    ) -> csr_matrix:
        if not self._is_fitted:
            raise RuntimeError("FeatureExtractor is not fitted yet.")
        documents = [make_document(s, b) for s, b in zip(subjects, bodies)]
        x_text = self.vectorizer.transform(documents)
        return self._stack_numeric(x_text, subjects, bodies, headers_list)

    @staticmethod
    def _stack_numeric(
        x_text: csr_matrix,
        subjects: List[str],
        bodies: List[str],
        headers_list: List[dict],
    ) -> csr_matrix:
        numeric = np.asarray(
            [
                extract_numeric_features(s, b, h)
                for s, b, h in zip(subjects, bodies, headers_list)
            ],
            dtype=np.float64,
        )
        x_num = csr_matrix(numeric)
        return hstack([x_text, x_num], format="csr")

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "vectorizer": self.vectorizer,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureExtractor":
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance._is_fitted = bool(data.get("is_fitted", True))
        return instance
