"""
LLM Client - Provider-agnostic interface for OpenAI, Anthropic, and Ollama
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Send messages and return the assistant's text response."""
        pass

    @abstractmethod
    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Send messages and parse the response as JSON."""
        pass


class ValidationStatus:
    """Normalized validation status constants (uppercase for external API)."""
    SUCCESS = "SUCCESS"
    AUTH_FAILED = "AUTH_FAILED"
    RATE_LIMITED = "RATE_LIMITED"
    MISCONFIGURED = "MISCONFIGURED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"

    # Map internal state strings to normalized status
    _STATE_MAP = {
        "connected": "SUCCESS",
        "auth_failed": "AUTH_FAILED",
        "rate_limited": "RATE_LIMITED",
        "misconfigured": "MISCONFIGURED",
        "unavailable": "UNAVAILABLE",
        "invalid_response": "INVALID_RESPONSE",
    }

    @classmethod
    def from_state(cls, state: str) -> str:
        return cls._STATE_MAP.get(state, "MISCONFIGURED")


@dataclass
class LLMValidationResult:
    state: str
    message: str
    provider: str
    model: str
    latency_ms: Optional[float] = None

    @property
    def status(self) -> str:
        """Normalized uppercase status for external consumers."""
        return ValidationStatus.from_state(self.state)

    @property
    def is_ready(self) -> bool:
        """True only when the LLM has been validated with a real probe."""
        return self.state == "connected"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        d["is_ready"] = self.is_ready
        return d


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        client = self._get_client()
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = await client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        text = await self.chat(messages, system=system, temperature=temperature, max_tokens=max_tokens)
        return _extract_json(text)


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20240620"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            system=system or "",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text if response.content else ""

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        text = await self.chat(messages, system=system, temperature=temperature, max_tokens=max_tokens)
        return _extract_json(text)


class OllamaClient(LLMClient):
    """Ollama local model client."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        import aiohttp

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        text = await self.chat(messages, system=system, temperature=temperature, max_tokens=max_tokens)
        return _extract_json(text)


class FallbackClient(LLMClient):
    """Rule-based fallback when no LLM API key is configured.

    This keeps the platform functional without any external API dependency
    by using simple heuristics for intent detection and templated responses
    for result interpretation.
    """

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        return self._rule_based_response(last_msg)

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        last_msg = messages[-1]["content"] if messages else ""
        return self._rule_based_intent(last_msg)

    def _rule_based_intent(self, message: str) -> Dict[str, Any]:
        """Pattern-based intent detection as fallback."""
        msg = message.lower().strip()

        # Order matters: check specific intents before broad ones
        if any(w in msg for w in ["run all", "full analysis", "analyze everything", "complete analysis", "start analysis", "proceed", "run pipeline"]):
            return {"intent": "run_pipeline", "explanation": "Running the full data science pipeline."}
        if any(w in msg for w in ["help", "what can", "how do", "command"]):
            return {"intent": "help", "explanation": "Showing help information."}
        if any(w in msg for w in ["status", "progress", "where"]):
            return {"intent": "status", "explanation": "Showing current status."}
        if any(w in msg for w in ["clean", "preprocess", "missing", "duplicate", "outlier"]):
            return {"intent": "run_agent", "agent": "DataCleanerAgent", "action": "clean_data", "explanation": "Running data cleaning based on your request."}
        if any(w in msg for w in ["eda", "explor", "statistic", "profile", "distribut", "correlat", "describe", "analyze", "analyse"]):
            return {"intent": "run_agent", "agent": "EDAAgent", "action": "full_eda", "explanation": "Running exploratory data analysis."}
        if any(w in msg for w in ["feature", "engineer", "encod", "scal", "transform"]):
            return {"intent": "run_agent", "agent": "FeatureEngineerAgent", "action": "engineer_features", "explanation": "Running feature engineering."}
        if any(w in msg for w in ["automl", "auto ml", "best model", "recommend"]):
            return {"intent": "run_agent", "agent": "AutoMLAgent", "action": "auto_select_models", "explanation": "Running AutoML model selection."}
        if any(w in msg for w in ["train", "model", "predict", "classif", "regress", "fit"]):
            return {"intent": "run_agent", "agent": "ModelTrainerAgent", "action": "train_models", "explanation": "Training machine learning models."}
        if any(w in msg for w in ["forecast", "predict future", "time series", "prophet", "trend"]):
            return {"intent": "run_agent", "agent": "ForecastAgent", "action": "forecast", "explanation": "Running time series forecast."}
        if any(w in msg for w in ["insight", "finding", "discover", "narrative", "business"]):
            return {"intent": "run_agent", "agent": "InsightsAgent", "action": "generate_insights", "explanation": "Generating business insights."}
        if any(w in msg for w in ["export", "download report", "generate report", "html report", "markdown report"]):
            return {"intent": "run_agent", "agent": "ReportGeneratorAgent", "action": "generate_report", "explanation": "Generating exportable report."}
        if any(w in msg for w in ["visual", "chart", "plot", "graph", "draw"]):
            return {"intent": "run_agent", "agent": "DataVisualizerAgent", "action": "generate_visualizations", "explanation": "Generating visualizations."}
        if any(w in msg for w in ["dashboard", "report", "summary"]):
            return {"intent": "run_agent", "agent": "DashboardBuilderAgent", "action": "build_dashboard", "explanation": "Building dashboard."}
        if any(w in msg for w in ["upload", "load", "import", "open", "read", "data"]):
            return {"intent": "upload_data", "explanation": "Please upload a dataset using the sidebar."}

        return {"intent": "general", "explanation": "I can help with data cleaning, EDA, feature engineering, model training, and visualization. Try asking me to analyze your data or train models!"}

    def _rule_based_response(self, message: str) -> str:
        """Generate a simple text response."""
        intent = self._rule_based_intent(message)
        return intent.get("explanation", "I'm here to help with your data science tasks. Try uploading a dataset and asking me to analyze it!")


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM text response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last ``` lines
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"intent": "general", "explanation": text}


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    allow_fallback: bool = True,
) -> Optional[LLMClient]:
    """Factory function to create the appropriate LLM client.

    Auto-detects provider from environment variables if not specified.
    Falls back to rule-based client if no API keys are found.
    """
    if provider is None:
        if api_key or os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OLLAMA_MODEL"):
            provider = "ollama"
        else:
            if allow_fallback:
                logger.info("No LLM API key found. Using rule-based fallback client.")
                return FallbackClient()
            logger.info("No LLM API key found. Returning None (LLM disabled).")
            return None

    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-4o-mini")
    elif provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model or "claude-3-5-sonnet-20240620")
    elif provider == "ollama":
        return OllamaClient(
            model=model or os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        if allow_fallback:
            logger.warning(f"Unknown provider '{provider}', using fallback.")
            return FallbackClient()
        logger.warning(f"Unknown provider '{provider}', returning None.")
        return None


def _validate_api_key_format(provider: str, api_key: str) -> Optional[str]:
    """Validate API key format for known providers.

    Returns an error message if the key format is invalid, or None if acceptable.
    """
    if not api_key or not api_key.strip():
        return f"API key for {provider} is empty."

    key = api_key.strip()

    if provider == "openai":
        # OpenAI keys start with "sk-" and are at least 20 characters
        if not key.startswith("sk-"):
            return (
                "Invalid OpenAI API key format: must start with 'sk-'. "
                "Get a valid key at https://platform.openai.com/api-keys"
            )
        if len(key) < 20:
            return (
                "Invalid OpenAI API key format: key is too short. "
                "A valid key is at least 20 characters."
            )

    elif provider == "anthropic":
        # Anthropic keys start with "sk-ant-"
        if not key.startswith("sk-ant-"):
            return (
                "Invalid Anthropic API key format: must start with 'sk-ant-'. "
                "Get a valid key at https://console.anthropic.com/settings/keys"
            )
        if len(key) < 20:
            return (
                "Invalid Anthropic API key format: key is too short. "
                "A valid key is at least 20 characters."
            )

    return None


async def validate_llm_client(client: Optional[LLMClient]) -> LLMValidationResult:
    """Validate that the LLM client can authenticate and return a deterministic response.

    Validation steps (all must pass):
    1. Client must exist and not be a FallbackClient
    2. API key must be present (for cloud providers)
    3. API key format must be valid for the provider
    4. A real inference probe must succeed with the expected response

    This function never fakes success. If any step fails, the result reflects the failure.
    """
    if client is None:
        return LLMValidationResult(
            state="misconfigured",
            message="No LLM client configured. Please provide a valid API key.",
            provider="none",
            model="none",
        )

    if isinstance(client, FallbackClient):
        return LLMValidationResult(
            state="misconfigured",
            message=(
                "Fallback (rule-based) client cannot provide real intelligence. "
                "Please configure a valid LLM provider and API key."
            ),
            provider="fallback",
            model="n/a",
        )

    provider, model = _client_metadata(client)
    api_key = getattr(client, "api_key", None)

    # Step 1: API key presence check (cloud providers)
    if provider in {"openai", "anthropic"} and not api_key:
        return LLMValidationResult(
            state="misconfigured",
            message=f"Missing API key for {provider}. Please provide a valid API key.",
            provider=provider,
            model=model,
        )

    # Step 2: API key format validation (cloud providers)
    if provider in {"openai", "anthropic"} and api_key:
        format_error = _validate_api_key_format(provider, api_key)
        if format_error:
            return LLMValidationResult(
                state="auth_failed",
                message=format_error,
                provider=provider,
                model=model,
            )

    # Step 3: Real inference probe — deterministic, token-light
    start = time.monotonic()
    try:
        response = await client.chat(
            messages=[{"role": "user", "content": "Respond with token: OK"}],
            temperature=0.0,
            max_tokens=5,
        )
        latency_ms = (time.monotonic() - start) * 1000

        if response is None or "ok" not in response.lower():
            return LLMValidationResult(
                state="invalid_response",
                message=(
                    "LLM responded but did not return the expected validation token. "
                    "The model may be misconfigured or unavailable."
                ),
                provider=provider,
                model=model,
                latency_ms=latency_ms,
            )

        return LLMValidationResult(
            state="connected",
            message="LLM validated with deterministic probe.",
            provider=provider,
            model=model,
            latency_ms=latency_ms,
        )

    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        state, message = _classify_llm_error(exc)
        return LLMValidationResult(
            state=state,
            message=message,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
        )


def _client_metadata(client: LLMClient) -> Tuple[str, str]:
    if isinstance(client, OpenAIClient):
        return "openai", client.model
    if isinstance(client, AnthropicClient):
        return "anthropic", client.model
    if isinstance(client, OllamaClient):
        return "ollama", client.model
    return "unknown", "unknown"


def _classify_llm_error(exc: Exception) -> Tuple[str, str]:
    """Classify an LLM exception into a specific, actionable error category.

    Returns (state, user_facing_message) with explicit recovery guidance.
    """
    message = str(exc)
    lowered = message.lower()
    exc_type = type(exc).__name__

    # Authentication failures
    if any(s in lowered for s in [
        "unauthorized", "invalid api key", "invalid x-api-key",
        "authentication", "api key", "permission denied", "403", "401",
        "incorrect api key",
    ]):
        return (
            "auth_failed",
            f"Authentication failed: the API key was rejected by the provider. "
            f"Please verify your API key is correct and active. (Detail: {exc_type})"
        )

    # Rate limiting
    if any(s in lowered for s in ["rate limit", "rate_limit", "429", "too many requests"]):
        return (
            "rate_limited",
            "Rate limit reached. The provider is throttling requests. "
            "Wait a moment and try again, or check your plan's rate limits."
        )

    # Token/quota exhaustion
    if any(s in lowered for s in [
        "quota", "billing", "insufficient_quota", "exceeded",
        "budget", "payment", "credit",
    ]):
        return (
            "auth_failed",
            "Token quota exceeded or billing issue. "
            "Check your provider account for available credits or quota limits."
        )

    # Model not found / misconfigured
    if any(s in lowered for s in [
        "model not found", "does not exist", "invalid model",
        "model_not_found", "no such model",
    ]):
        return (
            "misconfigured",
            f"Model not found or unavailable. "
            f"Verify the model name is correct and accessible with your API key. (Detail: {message[:100]})"
        )

    # Network / connection errors
    if any(s in lowered for s in [
        "connection", "timeout", "timed out", "dns", "resolve",
        "unreachable", "refused", "network", "ssl", "eof",
    ]) or exc_type in ("ConnectionError", "TimeoutError", "OSError", "ClientConnectorError"):
        return (
            "unavailable",
            f"Network error: could not reach the LLM provider. "
            f"Check your internet connection and the provider's service status. (Detail: {exc_type})"
        )

    # Server errors
    if any(s in lowered for s in ["500", "502", "503", "504", "server error", "internal error"]):
        return (
            "unavailable",
            "The LLM provider returned a server error. "
            "This is likely a temporary issue — try again in a few moments."
        )

    # Catch-all with detail
    return (
        "unavailable",
        f"LLM provider error: {exc_type}: {message[:200]}. "
        f"Check your configuration and try again."
    )
