"""Tests for the feature_extractor module (TF-IDF + numeric feature extraction)."""

from __future__ import annotations

import numpy as np
import pytest

from feature_extractor import (
    FeatureExtractor,
    NUMERIC_FEATURE_NAMES,
    extract_numeric_features,
)


class TestExtractNumericFeatures:
    """Tests for the standalone extract_numeric_features function."""

    def test_html_tag_count(self):
        """Verify HTML tags are detected and counted correctly."""
        subject = "Hello"
        body = "<p>Click <a href='http://example.com'>here</a> now!</p>"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["html_tag_count"] > 0
        assert feats["has_html"] == 1.0

    def test_no_html_tags(self):
        subject = "Hello"
        body = "Plain text email with no markup."
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["html_tag_count"] == 0.0
        assert feats["has_html"] == 0.0

    def test_exclamation_density(self):
        subject = "URGENT!!!"
        body = "Click now!!! Free money!!!"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["exclamation_density"] > 0

    def test_no_exclamations(self):
        subject = "Hello"
        body = "A calm email with no shouting."
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["exclamation_density"] == 0.0

    def test_url_count(self):
        subject = "Check this out"
        body = "Visit http://example.com and also www.test.com"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["url_count"] > 0

    def test_uppercase_ratio(self):
        subject = "ALL CAPS SUBJECT"
        body = "This email has MIXED case text WITH SOME uppercase."
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["uppercase_ratio"] > 0
        assert feats["uppercase_ratio"] <= 1.0

    def test_caps_word_ratio(self):
        subject = "HELLO WORLD"
        body = "THIS EMAIL HAS MANY CAPITALIZED WORDS THAT ARE LOUD."
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["caps_word_ratio"] > 0

    def test_spam_keyword_hits(self):
        subject = "Free money now!!!"
        body = "Click here for your guaranteed free prize, winner!!!"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["spam_keyword_hits"] > 0

    def test_no_spam_keywords(self):
        subject = "Meeting notes"
        body = "Let's discuss the project timeline and budget."
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["spam_keyword_hits"] == 0.0

    def test_chinese_spam_keywords(self):
        subject = "免费中奖恭喜!!!"
        body = "恭喜您中奖了，点击这里领取免费大奖，限时优惠！！！"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["spam_keyword_hits"] > 0

    def test_subject_caps_ratio(self):
        subject = "ALL CAPS"
        body = "normal body text"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["subject_caps_ratio"] == 1.0

    def test_currency_symbols(self):
        subject = "Make money fast"
        body = "Earn $1000 per week with our amazing system!"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["currency_symbols"] > 0

    def test_digit_ratio(self):
        subject = "Win 1000000 dollars"
        body = "Call 1234567890 now for your 100% guaranteed prize!"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["digit_ratio"] > 0

    def test_percent_signs(self):
        subject = "50% OFF EVERYTHING"
        body = "Get 100% free! 90% discount! Limited time!"
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["percent_signs"] > 0

    def test_reply_to_header(self):
        subject = "Hello"
        body = "Reply to me please."
        headers = {"Reply-To": "spam@example.com"}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["has_reply_to"] == 1.0

    def test_no_reply_to_header(self):
        subject = "Hello"
        body = "Just a normal email."
        headers = {"From": "user@example.com", "To": "me@example.com"}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["has_reply_to"] == 0.0

    def test_header_count(self):
        subject = "Hello"
        body = "Testing headers."
        headers = {
            "From": "a@b.com",
            "To": "c@d.com",
            "Date": "2024-01-01",
            "Subject": "Hello",
            "X-Spam": "no",
        }
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["header_count"] > 0

    def test_body_length(self):
        subject = "Short"
        body = "Very short."
        headers = {}
        feats = dict(zip(NUMERIC_FEATURE_NAMES, extract_numeric_features(subject, body, headers)))
        assert feats["body_length"] >= 0
        assert feats["body_length"] <= 1.0

    def test_all_features_in_range_zero_to_one(self):
        """Every numeric feature must be in [0, 1]."""
        subject = "!!! FREE MONEY NOW !!! click click click $$$"
        body = "<p>Click <a href='http://evil.com'>here</a> NOW for FREE money!!! URGENT!!! WINNER!!! Guaranteed 100%!!! $$$€€€£££¥¥¥ Call 1-800-SCAM-NOW!!! Limited time offer expires SOON!!! Don't miss this AMAZING opportunity!!! FREE FREE FREE!!!</p>"
        headers = {"Reply-To": "scam@evil.com", "From": "spam@evil.com", "X-Priority": "1 (Highest)"}
        feats = extract_numeric_features(subject, body, headers)
        for name, value in zip(NUMERIC_FEATURE_NAMES, feats):
            assert 0.0 <= value <= 1.0, f"{name} = {value} is out of [0, 1]"

    def test_empty_email(self):
        feats = extract_numeric_features("", "", {})
        assert len(feats) == len(NUMERIC_FEATURE_NAMES)
        for value in feats:
            assert 0.0 <= value <= 1.0

    def test_feature_count_matches_names(self):
        feats = extract_numeric_features("test", "test body", {})
        assert len(feats) == len(NUMERIC_FEATURE_NAMES) == 14


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class (TF-IDF + numeric stacking)."""

    def test_initial_state_not_fitted(self):
        extractor = FeatureExtractor()
        assert extractor._is_fitted is False
        assert extractor.vocabulary_size == 0
        with pytest.raises(RuntimeError, match="not fitted yet"):
            extractor.transform(["subj"], ["body"], [{}])

    def test_fit_transform_returns_correct_shape(self):
        extractor = FeatureExtractor()
        subjects = ["Meeting tomorrow", "Free money now", "Project update", "Win big prize"]
        bodies = [
            "Let's meet tomorrow to discuss the project.",
            "Click here for free money, guaranteed!!!",
            "Here's the latest update on the project timeline.",
            "You won! Click now to claim your free prize!!!",
        ]
        headers = [{}, {}, {}, {}]

        x = extractor.fit_transform(subjects, bodies, headers)

        assert extractor._is_fitted is True
        assert extractor.vocabulary_size > 0
        assert x.shape[0] == 4
        assert x.shape[1] == extractor.feature_count
        assert x.shape[1] == extractor.vocabulary_size + len(NUMERIC_FEATURE_NAMES)

    def test_transform_after_fit_returns_same_shape(self):
        extractor = FeatureExtractor()
        subjects = ["Meeting tomorrow", "Free money now"]
        bodies = [
            "Let's meet tomorrow to discuss the project.",
            "Click here for free money, guaranteed!!!",
        ]
        headers = [{}, {}]

        x1 = extractor.fit_transform(subjects, bodies, headers)
        x2 = extractor.transform(["New subject"], ["New body text"], [{}])

        assert x1.shape[1] == x2.shape[1]

    def test_transform_returns_sparse_matrix(self):
        from scipy.sparse import csr_matrix

        extractor = FeatureExtractor()
        extractor.fit_transform(
            ["subj1", "subj2", "subj3"],
            ["body1 content", "body2 different", "body3 unique"],
            [{}, {}, {}],
        )
        x = extractor.transform(["subj_new"], ["body new content"], [{}])
        assert isinstance(x, csr_matrix)

    def test_persistence_roundtrip(self):
        extractor = FeatureExtractor()
        extractor.fit_transform(
            ["Hello world", "Free money"],
            ["This is a test email", "Click now for free stuff"],
            [{}, {}],
        )

        data = extractor.to_dict()
        restored = FeatureExtractor.from_dict(data)

        assert restored._is_fitted == extractor._is_fitted
        assert restored.vocabulary_size == extractor.vocabulary_size
        assert restored.feature_count == extractor.feature_count

        x1 = extractor.transform(["Test"], ["Test body"], [{}])
        x2 = restored.transform(["Test"], ["Test body"], [{}])
        np.testing.assert_array_equal(x1.toarray(), x2.toarray())

    def test_fit_transform_with_chinese(self):
        extractor = FeatureExtractor()
        subjects = ["会议通知", "免费中奖", "项目进度"]
        bodies = [
            "明天下午三点开会讨论项目进度",
            "恭喜您中了大奖，点击这里领取免费奖品！！！",
            "项目进展顺利，预计下周完成第一阶段",
        ]
        headers = [{}, {}, {}]

        x = extractor.fit_transform(subjects, bodies, headers)
        assert x.shape[0] == 3
        assert extractor.vocabulary_size > 0

    def test_mixed_language_fit(self):
        extractor = FeatureExtractor()
        subjects = ["Meeting 会议", "Free 免费", "Update 通知"]
        bodies = [
            "明天 meeting 讨论 project",
            "免费 free money click now",
            "项目 update 请 review",
        ]
        headers = [{}, {}, {}]

        x = extractor.fit_transform(subjects, bodies, headers)
        assert x.shape[0] == 3
        assert extractor.vocabulary_size > 0
