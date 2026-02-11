"""Tests for the LLM client module."""

import json
import pytest
import asyncio

from llm.client import (
    FallbackClient,
    OpenAIClient,
    AnthropicClient,
    _extract_json,
    get_llm_client,
)


class TestExtractJson:
    """Tests for JSON extraction from LLM responses."""

    def test_plain_json(self):
        text = '{"intent": "run_agent", "agent": "EDAAgent"}'
        result = _extract_json(text)
        assert result["intent"] == "run_agent"
        assert result["agent"] == "EDAAgent"

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"intent": "help", "explanation": "showing help"}\n```'
        result = _extract_json(text)
        assert result["intent"] == "help"

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"intent": "general", "explanation": "hello"} that is it.'
        result = _extract_json(text)
        assert result["intent"] == "general"

    def test_invalid_json_returns_general(self):
        text = "This is not JSON at all."
        result = _extract_json(text)
        assert result["intent"] == "general"
        assert "This is not JSON" in result["explanation"]

    def test_empty_string(self):
        result = _extract_json("")
        assert result["intent"] == "general"


class TestFallbackClient:
    """Tests for the rule-based fallback LLM client."""

    @pytest.fixture
    def client(self):
        return FallbackClient()

    @pytest.mark.asyncio
    async def test_eda_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Run exploratory data analysis"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "EDAAgent"

    @pytest.mark.asyncio
    async def test_clean_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Clean my data please"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "DataCleanerAgent"

    @pytest.mark.asyncio
    async def test_train_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Train models on this dataset"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "ModelTrainerAgent"

    @pytest.mark.asyncio
    async def test_feature_engineering_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Engineer some features"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "FeatureEngineerAgent"

    @pytest.mark.asyncio
    async def test_visualize_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Show me some charts"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "DataVisualizerAgent"

    @pytest.mark.asyncio
    async def test_automl_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Run automl to find the best model"}]
        )
        assert result["intent"] == "run_agent"
        assert result["agent"] == "AutoMLAgent"

    @pytest.mark.asyncio
    async def test_pipeline_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "Run full analysis on everything"}]
        )
        assert result["intent"] == "run_pipeline"

    @pytest.mark.asyncio
    async def test_help_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "help me"}]
        )
        assert result["intent"] == "help"

    @pytest.mark.asyncio
    async def test_upload_intent(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "upload a new dataset"}]
        )
        assert result["intent"] == "upload_data"

    @pytest.mark.asyncio
    async def test_general_fallback(self, client):
        result = await client.chat_json(
            messages=[{"role": "user", "content": "what is the meaning of life"}]
        )
        assert result["intent"] == "general"

    @pytest.mark.asyncio
    async def test_chat_returns_string(self, client):
        result = await client.chat(
            messages=[{"role": "user", "content": "clean my data"}]
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestGetLLMClient:
    """Tests for the client factory function."""

    def test_no_env_returns_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        client = get_llm_client()
        assert isinstance(client, FallbackClient)

    def test_explicit_fallback(self):
        client = get_llm_client(provider=None)
        # Without env vars this should be fallback
        assert client is not None

    def test_unknown_provider_returns_fallback(self):
        client = get_llm_client(provider="nonexistent")
        assert isinstance(client, FallbackClient)
