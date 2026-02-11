"""Tests for LLM validation: key format checks, inference probe, error classification."""

import pytest

from llm.client import (
    OpenAIClient,
    AnthropicClient,
    OllamaClient,
    FallbackClient,
    validate_llm_client,
    LLMValidationResult,
    ValidationStatus,
    _validate_api_key_format,
)


# ---------------------------------------------------------------------------
# API key format validation (unit tests — no network)
# ---------------------------------------------------------------------------


class TestApiKeyFormatValidation:
    """Test _validate_api_key_format for known provider key patterns."""

    def test_openai_valid_key(self):
        assert _validate_api_key_format("openai", "sk-abc123def456ghi789jkl") is None

    def test_openai_empty_key(self):
        err = _validate_api_key_format("openai", "")
        assert err is not None
        assert "empty" in err.lower()

    def test_openai_missing_prefix(self):
        err = _validate_api_key_format("openai", "not-a-key-at-all-but-long-enough")
        assert err is not None
        assert "sk-" in err

    def test_openai_too_short(self):
        err = _validate_api_key_format("openai", "sk-short")
        assert err is not None
        assert "too short" in err.lower()

    def test_anthropic_valid_key(self):
        assert _validate_api_key_format("anthropic", "sk-ant-abc123def456ghi789jkl") is None

    def test_anthropic_empty_key(self):
        err = _validate_api_key_format("anthropic", "")
        assert err is not None
        assert "empty" in err.lower()

    def test_anthropic_missing_prefix(self):
        err = _validate_api_key_format("anthropic", "bad-key-without-ant-prefix-long")
        assert err is not None
        assert "sk-ant-" in err

    def test_anthropic_too_short(self):
        err = _validate_api_key_format("anthropic", "sk-ant-x")
        assert err is not None
        assert "too short" in err.lower()

    def test_ollama_has_no_key_validation(self):
        # Ollama doesn't use API keys — format check should pass anything
        assert _validate_api_key_format("ollama", "anything") is None

    def test_unknown_provider_passes(self):
        assert _validate_api_key_format("unknown", "any-key") is None


# ---------------------------------------------------------------------------
# Full validate_llm_client tests
# ---------------------------------------------------------------------------


class TestValidateLLMClient:
    """Test the full validation pipeline: presence, format, inference probe."""

    @pytest.mark.asyncio
    async def test_none_client_returns_misconfigured(self):
        result = await validate_llm_client(None)
        assert result.state == "misconfigured"
        assert result.status == "MISCONFIGURED"
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_fallback_client_returns_misconfigured(self):
        result = await validate_llm_client(FallbackClient())
        assert result.state == "misconfigured"
        assert result.status == "MISCONFIGURED"
        assert "rule-based" in result.message.lower() or "fallback" in result.message.lower()
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_openai_empty_key_returns_misconfigured(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIClient(api_key="", model="gpt-4o-mini")
        result = await validate_llm_client(client)
        assert result.state == "misconfigured"
        assert result.status == "MISCONFIGURED"
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_openai_bad_format_returns_auth_failed(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAIClient(api_key="not-a-key", model="gpt-4o-mini")
        result = await validate_llm_client(client)
        assert result.state == "auth_failed"
        assert result.status == "AUTH_FAILED"
        assert "sk-" in result.message
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_anthropic_bad_format_returns_auth_failed(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = AnthropicClient(api_key="bad-key", model="claude-3-5-sonnet-20240620")
        result = await validate_llm_client(client)
        assert result.state == "auth_failed"
        assert result.status == "AUTH_FAILED"
        assert "sk-ant-" in result.message
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_valid_probe_accepts_ok_response(self, monkeypatch):
        """When the LLM returns 'OK', validation should succeed."""
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            return "OK"

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "connected"
        assert result.status == "SUCCESS"
        assert result.is_ready
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_probe_rejects_wrong_response(self, monkeypatch):
        """If the LLM responds but doesn't say OK, validation fails."""
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            return "I am a language model and I cannot help with that."

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "invalid_response"
        assert result.status == "INVALID_RESPONSE"
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_probe_rejects_none_response(self, monkeypatch):
        """If the LLM returns None, validation fails."""
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            return None

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "invalid_response"
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_rate_limit_classification(self, monkeypatch):
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            raise RuntimeError("429 rate limit exceeded")

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "rate_limited"
        assert result.status == "RATE_LIMITED"
        assert not result.is_ready

    @pytest.mark.asyncio
    async def test_auth_error_classification(self, monkeypatch):
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            raise RuntimeError("401 Unauthorized: Invalid API key")

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "auth_failed"
        assert result.status == "AUTH_FAILED"

    @pytest.mark.asyncio
    async def test_network_error_classification(self, monkeypatch):
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            raise ConnectionError("Connection refused")

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "unavailable"
        assert result.status == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_model_not_found_classification(self, monkeypatch):
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            raise RuntimeError("model not found: gpt-4o-mini does not exist")

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "misconfigured"
        assert result.status == "MISCONFIGURED"

    @pytest.mark.asyncio
    async def test_quota_exceeded_classification(self, monkeypatch):
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            raise RuntimeError("insufficient_quota: You exceeded your current quota")

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "auth_failed"
        assert result.status == "AUTH_FAILED"
        assert "quota" in result.message.lower()

    @pytest.mark.asyncio
    async def test_server_error_classification(self, monkeypatch):
        client = OpenAIClient(api_key="sk-valid-key-that-is-long-enough", model="gpt-4o-mini")

        async def fake_chat(*args, **kwargs):
            raise RuntimeError("502 Bad Gateway - server error")

        monkeypatch.setattr(client, "chat", fake_chat)
        result = await validate_llm_client(client)
        assert result.state == "unavailable"


# ---------------------------------------------------------------------------
# ValidationResult serialization
# ---------------------------------------------------------------------------


class TestValidationResultSerialization:
    def test_to_dict_includes_status_and_is_ready(self):
        result = LLMValidationResult(
            state="connected",
            message="OK",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=42.0,
        )
        d = result.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["is_ready"] is True
        assert d["state"] == "connected"
        assert d["latency_ms"] == 42.0

    def test_to_dict_failed_state(self):
        result = LLMValidationResult(
            state="auth_failed",
            message="bad key",
            provider="openai",
            model="gpt-4o-mini",
        )
        d = result.to_dict()
        assert d["status"] == "AUTH_FAILED"
        assert d["is_ready"] is False
