"""Centralized configuration for the spam detector service.

All tunable parameters, paths, and feature settings live here so they can be
changed in one place. Values can be overridden via environment variables
using the ``SPAM_`` prefix.
"""

from __future__ import annotations

import os
from typing import List


def _env(name: str, default: str) -> str:
    value = os.environ.get(f"SPAM_{name}")
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


DATA_DIR = _env("DATA_DIR", "data")
DB_PATH = _env("DB_PATH", "spam_predictions.db")
MODEL_PATH = _env("MODEL_PATH", "spam_model.joblib")

CLASSIFIER_TYPE = _env("CLASSIFIER_TYPE", "naive_bayes")

FEEDBACK_RETRAIN_THRESHOLD = _env_int("FEEDBACK_RETRAIN_THRESHOLD", 50)

NB_ALPHA = _env_float("NB_ALPHA", 1.0)

TFIDF_SUBLINEAR_TF = _env("TFIDF_SUBLINEAR_TF", "1") == "1"
TFIDF_NORM = _env("TFIDF_NORM", "l2")
TFIDF_MIN_DF = _env_int("TFIDF_MIN_DF", 1)
TFIDF_MAX_DF = _env_float("TFIDF_MAX_DF", 0.95)
TFIDF_NGRAM_MIN = _env_int("TFIDF_NGRAM_MIN", 1)
TFIDF_NGRAM_MAX = _env_int("TFIDF_NGRAM_MAX", 2)

SPAM_KEYWORDS: List[str] = [
    "free", "win", "winner", "prize", "guarantee", "guaranteed", "click",
    "urgent", "now", "congratulations", "lottery", "cash", "credit", "loan",
    "miracle", "amazing", "incredible", "risk", "selected", "limited",
    "offer", "discount", "deal", "money", "income", "profit", "bonus",
    "免费", "中奖", "恭喜", "优惠", "特价", "限时",
    "点击", "立即", "贷款", "信用卡", "发票", "代开",
    "赚钱", "兼职", "刷单", "提现", "转账", "密码",
    "账户", "验证", "红包", "返利", "促销", "豪礼",
    "百万", "千万", "大奖", "速抢", "不看后悔",
]

CAP_COUNT = 20.0
CAP_LENGTH = 5000.0
CAP_HEADERS = 20.0

VALID_LABELS = ("ham", "spam")
