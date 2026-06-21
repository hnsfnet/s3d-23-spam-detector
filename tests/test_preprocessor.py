"""Tests for the preprocessor module (text cleaning, tokenization, language detection)."""

from __future__ import annotations

import pytest

import preprocessor


class TestHasChinese:
    """Tests for the has_chinese language detection function."""

    def test_pure_english(self):
        assert preprocessor.has_chinese("Hello world, this is English.") is False

    def test_pure_chinese(self):
        assert preprocessor.has_chinese("你好世界，这是中文。") is True

    def test_mixed_text(self):
        assert preprocessor.has_chinese("Hello 你好 world 世界") is True

    def test_empty_string(self):
        assert preprocessor.has_chinese("") is False

    def test_cjk_punctuation_only(self):
        assert preprocessor.has_chinese("，。！？") is False

    def test_whitespace_only(self):
        assert preprocessor.has_chinese("   \n\t  ") is False


class TestTokenize:
    """Tests for the tokenize function."""

    def test_english_sentence_splits_on_spaces_and_punctuation(self):
        tokens = preprocessor.tokenize("Hello, world! How are you?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you" in tokens
        assert "," not in tokens
        assert "!" not in tokens
        assert "?" not in tokens

    def test_english_lowercase(self):
        tokens = preprocessor.tokenize("HELLO WORLD")
        assert all(t == t.lower() for t in tokens)

    def test_english_numbers_preserved(self):
        tokens = preprocessor.tokenize("Meeting at 10:30 AM room 42")
        assert "10" in tokens
        assert "30" in tokens
        assert "42" in tokens

    @pytest.mark.skipif(not preprocessor.JIEBA_AVAILABLE, reason="jieba not installed")
    def test_chinese_sentence(self):
        tokens = preprocessor.tokenize("我今天去超市买东西")
        assert len(tokens) > 1
        assert "我" in tokens
        assert "今天" in tokens
        assert "去" in tokens
        assert "超市" in tokens

    @pytest.mark.skipif(not preprocessor.JIEBA_AVAILABLE, reason="jieba not installed")
    def test_mixed_chinese_english(self):
        tokens = preprocessor.tokenize("我今天去超市 buy some milk")
        assert "我" in tokens
        assert "今天" in tokens
        assert "buy" in tokens
        assert "some" in tokens
        assert "milk" in tokens

    def test_empty_string(self):
        assert preprocessor.tokenize("") == []

    def test_whitespace_only(self):
        assert preprocessor.tokenize("   \n\t  ") == []

    def test_special_characters_filtered(self):
        tokens = preprocessor.tokenize("Hello!!!@@@###$$$World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_contractions(self):
        tokens = preprocessor.tokenize("don't won't can't")
        assert "don" in tokens
        assert "t" in tokens
        assert "won" in tokens
        assert "can" in tokens

    def test_hyphenated_words(self):
        tokens = preprocessor.tokenize("state-of-the-art technology")
        assert "state" in tokens
        assert "of" in tokens
        assert "the" in tokens
        assert "art" in tokens
        assert "technology" in tokens


class TestCleanText:
    """Tests for the clean_text normalisation function."""

    def test_strips_control_characters(self):
        raw = "Hello\x00World\x01!"
        cleaned = preprocessor.clean_text(raw)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "Hello World !" in cleaned

    def test_collapses_whitespace(self):
        raw = "  hello   \n\n  world   \t\t  "
        cleaned = preprocessor.clean_text(raw)
        assert cleaned == "hello world"

    def test_keeps_printable_characters(self):
        raw = "Hello! How are you? I'm fine, thank you."
        cleaned = preprocessor.clean_text(raw)
        assert cleaned == raw

    def test_keeps_newlines_tabs_carriage_returns(self):
        raw = "Line1\nLine2\tTabbed\r\nWindows"
        cleaned = preprocessor.clean_text(raw)
        assert cleaned == "Line1 Line2 Tabbed Windows"

    def test_removes_bell_characters(self):
        raw = "Ring the \x07 bell"
        cleaned = preprocessor.clean_text(raw)
        assert "\x07" not in cleaned
        assert "Ring the bell" in cleaned

    def test_handles_empty_string(self):
        assert preprocessor.clean_text("") == ""

    def test_html_tags_preserved_by_clean(self):
        raw = "<p>Hello <b>world</b></p>"
        cleaned = preprocessor.clean_text(raw)
        assert "<p>" in cleaned
        assert "</p>" in cleaned


class TestMakeDocument:
    """Tests for the make_document helper."""

    def test_combines_subject_and_body(self):
        doc = preprocessor.make_document("Hello", "World")
        assert doc == "hello\nworld"

    def test_lowercases_everything(self):
        doc = preprocessor.make_document("HELLO WORLD", "HOW ARE YOU")
        assert doc == "hello world\nhow are you"

    def test_handles_empty_subject(self):
        doc = preprocessor.make_document("", "Body only")
        assert doc == "\nbody only"

    def test_handles_empty_body(self):
        doc = preprocessor.make_document("Subject only", "")
        assert doc == "subject only\n"

    def test_handles_both_empty(self):
        doc = preprocessor.make_document("", "")
        assert doc == "\n"
