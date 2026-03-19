"""Unit tests for configuration and LLM initialization."""

import pytest
from unittest.mock import patch


class TestGetLlm:
    """Tests for the get_llm function."""

    def test_get_llm_openai(self):
        """get_llm should return ChatOpenAI when provider is openai."""
        with patch("src.config.settings") as mock_settings:
            mock_settings.llm_provider = "openai"
            mock_settings.llm_model = "gpt-4"
            mock_settings.openai_api_key = "test-key"

            from src.config import get_llm

            llm = get_llm(temperature=0)

            from langchain_openai import ChatOpenAI

            assert isinstance(llm, ChatOpenAI)

    def test_get_llm_anthropic(self):
        """get_llm should return ChatAnthropic when provider is anthropic."""
        try:
            import langchain_anthropic  # noqa: F401
        except ImportError:
            pytest.skip("langchain_anthropic not installed")

        with patch("src.config.settings") as mock_settings:
            mock_settings.llm_provider = "anthropic"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.anthropic_api_key = "test-key"

            from src.config import get_llm

            llm = get_llm(temperature=0)

            from langchain_anthropic import ChatAnthropic

            assert isinstance(llm, ChatAnthropic)

    def test_get_llm_invalid_provider(self):
        """get_llm should raise ValueError for unsupported provider."""
        with patch("src.config.settings") as mock_settings:
            mock_settings.llm_provider = "gemini"

            from src.config import get_llm

            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                get_llm(temperature=0)
