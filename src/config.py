"""Configuration and LLM initialization."""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    llm_provider: str = "openai"
    llm_model: str = "gpt-5.4-nano"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 120

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


def get_llm(temperature: float = 0):
    """Initialize the LLM based on configured provider."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
