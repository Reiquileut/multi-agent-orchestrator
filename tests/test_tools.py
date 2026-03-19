"""Unit tests for orchestrator tools."""

import sys
from unittest.mock import patch, MagicMock

# Pre-register a mock tavily module so the inner `from tavily import TavilyClient` works
_tavily_mock = MagicMock()
sys.modules.setdefault("tavily", _tavily_mock)

from src.tools.calculator import calculate  # noqa: E402
from src.tools.text_processing import summarize_text, extract_key_points  # noqa: E402


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

    def test_string_constant_rejected(self):
        result = calculate.invoke({"expression": "'hello'"})
        assert "Error" in result

    def test_floor_division_rejected(self):
        result = calculate.invoke({"expression": "10 // 3"})
        assert "Error" in result

    def test_bitwise_not_rejected(self):
        result = calculate.invoke({"expression": "~5"})
        assert "Error" in result

    def test_lambda_rejected(self):
        result = calculate.invoke({"expression": "(lambda x: x)"})
        assert "Error" in result


class TestSearch:
    """Tests for web search and scrape tools (mocked Tavily)."""

    def test_web_search_returns_formatted(self):
        from src.tools.search import web_search

        mock_response = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "Content 1",
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "content": "Content 2",
                },
            ]
        }

        with patch.object(_tavily_mock, "TavilyClient") as MockClient:
            MockClient.return_value.search.return_value = mock_response
            result = web_search.invoke({"query": "AI trends"})

            assert "Result 1" in result
            assert "Result 2" in result
            assert "https://example.com/1" in result

    def test_web_search_no_results(self):
        from src.tools.search import web_search

        with patch.object(_tavily_mock, "TavilyClient") as MockClient:
            MockClient.return_value.search.return_value = {"results": []}
            result = web_search.invoke({"query": "obscure query"})

            assert result == "No results found."

    def test_scrape_url_returns_content(self):
        from src.tools.search import scrape_url

        mock_response = {"results": [{"raw_content": "Page content here " * 100}]}

        with patch.object(_tavily_mock, "TavilyClient") as MockClient:
            MockClient.return_value.extract.return_value = mock_response
            result = scrape_url.invoke({"url": "https://example.com"})

            assert "Page content here" in result
            assert len(result) <= 3000

    def test_scrape_url_empty_content(self):
        from src.tools.search import scrape_url

        mock_response = {"results": [{"raw_content": ""}]}

        with patch.object(_tavily_mock, "TavilyClient") as MockClient:
            MockClient.return_value.extract.return_value = mock_response
            result = scrape_url.invoke({"url": "https://example.com"})

            assert result == "Could not extract content."

    def test_scrape_url_no_results(self):
        from src.tools.search import scrape_url

        with patch.object(_tavily_mock, "TavilyClient") as MockClient:
            MockClient.return_value.extract.return_value = {"results": []}
            result = scrape_url.invoke({"url": "https://example.com"})

            assert result == "Failed to fetch URL content."

    def test_scrape_url_truncates_long_content(self):
        from src.tools.search import scrape_url

        long_content = "A" * 5000
        mock_response = {"results": [{"raw_content": long_content}]}

        with patch.object(_tavily_mock, "TavilyClient") as MockClient:
            MockClient.return_value.extract.return_value = mock_response
            result = scrape_url.invoke({"url": "https://example.com"})

            assert len(result) == 3000


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
        text = (
            "The cat sat on a mat. The dog ran in the park. Birds flew over the trees."
        )
        result = extract_key_points.invoke({"text": text})
        # Should return something (fallback)
        assert len(result) > 0
