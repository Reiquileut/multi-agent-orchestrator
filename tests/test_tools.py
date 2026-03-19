"""Unit tests for orchestrator tools."""

import pytest
from src.tools.calculator import calculate
from src.tools.text_processing import summarize_text, extract_key_points


class TestCalculator:
    """Tests for the safe math calculator."""

    def test_basic_addition(self):
        result = calculate.invoke({"expression": "2 + 3"})
        assert result == "5"

    def test_complex_expression(self):
        result = calculate.invoke({"expression": "(10 + 5) * 2"})
        assert result == "30"

    def test_division(self):
        result = calculate.invoke({"expression": "100 / 4"})
        assert result == "25.0"

    def test_power(self):
        result = calculate.invoke({"expression": "2 ** 10"})
        assert result == "1024"

    def test_modulo(self):
        result = calculate.invoke({"expression": "17 % 5"})
        assert result == "2"

    def test_negative_numbers(self):
        result = calculate.invoke({"expression": "-5 + 3"})
        assert result == "-2"

    def test_division_by_zero(self):
        result = calculate.invoke({"expression": "10 / 0"})
        assert "Error" in result

    def test_invalid_expression(self):
        result = calculate.invoke({"expression": "import os"})
        assert "Error" in result

    def test_nested_parentheses(self):
        result = calculate.invoke({"expression": "((2 + 3) * (4 - 1)) / 5"})
        assert result == "3.0"


class TestSummarizeText:
    """Tests for the extractive summarizer."""

    def test_short_text_unchanged(self):
        text = "This is a short sentence. Another one here."
        result = summarize_text.invoke({"text": text, "max_sentences": 5})
        assert result == text

    def test_long_text_reduced(self):
        sentences = [f"Sentence number {i} is about topic {i % 3}." for i in range(20)]
        text = " ".join(sentences)
        result = summarize_text.invoke({"text": text, "max_sentences": 3})
        # Should be shorter than original
        assert len(result.split(". ")) <= len(text.split(". "))

    def test_empty_text(self):
        result = summarize_text.invoke({"text": "", "max_sentences": 5})
        assert result == ""


class TestExtractKeyPoints:
    """Tests for key point extraction."""

    def test_extracts_numeric_sentences(self):
        text = (
            "The company grew 50% in 2024. "
            "Weather was nice today. "
            "Revenue reached $1.2 billion last quarter. "
            "The cat sat on the mat."
        )
        result = extract_key_points.invoke({"text": text})
        assert "50%" in result or "$1.2 billion" in result

    def test_returns_bullets(self):
        text = "First important finding here. Second key metric of 25% growth. Third major result."
        result = extract_key_points.invoke({"text": text})
        assert result.startswith("•")

    def test_fallback_on_no_indicators(self):
        text = "The cat sat on a mat. The dog ran in the park. Birds flew over the trees."
        result = extract_key_points.invoke({"text": text})
        # Should return something (fallback)
        assert len(result) > 0
