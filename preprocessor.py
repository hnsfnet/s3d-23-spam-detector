"""Text preprocessing: cleaning, tokenization, and language detection.

Handles both Chinese and English text transparently. When ``jieba`` is
installed it is used for Chinese/mixed tokenisation; otherwise the module
falls back to English-only regex tokenisation so the service still works.
"""

from __future__ import annotations

import re
from typing import List

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def has_chinese(text: str) -> bool:
    """Return True if ``text`` contains any CJK characters."""
    return bool(_CJK_RE.search(text))


def tokenize(text: str) -> List[str]:
    """Tokenize text, handling both Chinese and English.

    * When ``jieba`` is available it is used for all text, which works for
      pure English, pure Chinese, and mixed text.
    * When jieba is not installed we fall back to an English-only regex
      tokeniser so the service remains functional (Chinese text will mostly
      come through as single-character tokens).

    Tokens are lower-cased and empty strings are filtered out.
    """
    lowered = text.lower()
    if JIEBA_AVAILABLE:
        tokens = list(jieba.cut(lowered, cut_all=False))
    else:
        tokens = _WORD_RE.findall(lowered)
    return [t.strip() for t in tokens if t.strip()]


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters.

    Keeps printable characters and standard whitespace; everything else is
    replaced with a space. Returns a single-line string with collapsed spaces.
    """
    cleaned = "".join(ch if ch.isprintable() or ch in "\n\r\t" else " " for ch in text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def make_document(subject: str, body: str) -> str:
    """Combine subject and body into a single document string."""
    return f"{subject}\n{body}".lower()
