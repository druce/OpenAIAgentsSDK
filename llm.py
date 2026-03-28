#!/usr/bin/env python3
"""
LLM calling module with multi-vendor support, structured output, and batch processing.

Supports Anthropic, OpenAI, and Gemini models via a unified facade (LLMagent) that
preserves backward compatibility while delegating to vendor-specific agents internally.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, get_args, get_origin

import anthropic
import openai
import pandas as pd
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError, create_model
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import DEFAULT_CONCURRENCY, VENDOR_RPM_LIMITS

_logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Vendor enum and model registry (public, imported by prompts.py)
# ---------------------------------------------------------------------------


class Vendor(str, Enum):
    """Supported LLM vendors."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class LLMModel:
    """Model identity and capabilities."""

    model_id: str
    vendor: Vendor
    supports_logprobs: bool
    supports_reasoning: bool
    supports_temperature: bool
    default_max_tokens: int
    display_name: str


# --- Pre-defined model constants ---

CLAUDE_OPUS_MODEL = LLMModel(
    model_id="claude-opus-4-6",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=16384,
    display_name="Claude Opus 4.6",
)

CLAUDE_SONNET_MODEL = LLMModel(
    model_id="claude-sonnet-4-6",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=16384,
    display_name="Claude Sonnet 4.6",
)

CLAUDE_HAIKU_MODEL = LLMModel(
    model_id="claude-haiku-4-5-20251001",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=16384,
    display_name="Claude Haiku 4.5",
)

GPT5_MODEL = LLMModel(
    model_id="gpt-5",
    vendor=Vendor.OPENAI,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=False,
    default_max_tokens=16384,
    display_name="GPT-5",
)

GPT5_MINI_MODEL = LLMModel(
    model_id="gpt-5-mini",
    vendor=Vendor.OPENAI,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=False,
    default_max_tokens=16384,
    display_name="GPT-5 Mini",
)

GPT5_NANO_MODEL = LLMModel(
    model_id="gpt-5-nano",
    vendor=Vendor.OPENAI,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=False,
    default_max_tokens=16384,
    display_name="GPT-5 Nano",
)

GPT41_MODEL = LLMModel(
    model_id="gpt-4.1",
    vendor=Vendor.OPENAI,
    supports_logprobs=True,
    supports_reasoning=False,
    supports_temperature=True,
    default_max_tokens=4096,
    display_name="GPT-4.1",
)

GEMINI_FLASH_MODEL = LLMModel(
    model_id="gemini-3-flash-preview",
    vendor=Vendor.GEMINI,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=65536,
    display_name="Gemini 3.0 Flash",
)

GEMINI_PRO_MODEL = LLMModel(
    model_id="gemini-3.1-pro-preview",
    vendor=Vendor.GEMINI,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=65536,
    display_name="Gemini 3.1 Pro",
)

GLM5_MODEL = LLMModel(
    model_id="z-ai/glm-5",
    vendor=Vendor.OPENROUTER,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=False,
    default_max_tokens=32768,
    display_name="GLM-5",
)

SEED_MINI_MODEL = LLMModel(
    model_id="bytedance-seed/seed-2.0-mini",
    vendor=Vendor.OPENROUTER,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=16384,
    display_name="Seed 2.0 Mini",
)

KIMI_K25_MODEL = LLMModel(
    model_id="moonshotai/kimi-k2.5",
    vendor=Vendor.OPENROUTER,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=32768,
    display_name="Kimi K2.5",
)

HUNTER_ALPHA_MODEL = LLMModel(
    model_id="openrouter/hunter-alpha",
    vendor=Vendor.OPENROUTER,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=32000,
    display_name="Hunter Alpha",
)

MINIMAX_M27_MODEL = LLMModel(
    model_id="minimax/minimax-m2.7",
    vendor=Vendor.OPENROUTER,
    supports_logprobs=False,
    supports_reasoning=True,
    supports_temperature=True,
    default_max_tokens=32000,
    display_name="MiniMax M2.7",
)


MODEL_DICT: Dict[str, LLMModel] = {
    # Anthropic
    CLAUDE_OPUS_MODEL.model_id: CLAUDE_OPUS_MODEL,
    CLAUDE_SONNET_MODEL.model_id: CLAUDE_SONNET_MODEL,
    CLAUDE_HAIKU_MODEL.model_id: CLAUDE_HAIKU_MODEL,
    # OpenAI
    GPT5_MODEL.model_id: GPT5_MODEL,
    GPT5_MINI_MODEL.model_id: GPT5_MINI_MODEL,
    GPT41_MODEL.model_id: GPT41_MODEL,
    GPT5_NANO_MODEL.model_id: GPT5_NANO_MODEL,
    # Gemini
    GEMINI_FLASH_MODEL.model_id: GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL.model_id: GEMINI_PRO_MODEL,
    # OpenRouter
    GLM5_MODEL.model_id: GLM5_MODEL,
    SEED_MINI_MODEL.model_id: SEED_MINI_MODEL,
    KIMI_K25_MODEL.model_id: KIMI_K25_MODEL,
    HUNTER_ALPHA_MODEL.model_id: HUNTER_ALPHA_MODEL,
    MINIMAX_M27_MODEL.model_id: MINIMAX_M27_MODEL,
}


# ---------------------------------------------------------------------------
# Async token-bucket rate limiter (per vendor)
# ---------------------------------------------------------------------------


class AsyncTokenBucketRateLimiter:
    """Token-bucket rate limiter for async API calls.

    Refills at ``rpm / 60`` tokens per second.  Each ``acquire()`` consumes one
    token; callers sleep when the bucket is empty.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.max_tokens = float(rpm)
        self._tokens = float(rpm)  # start full
        self._rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.max_tokens, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self._tokens -= 1.0

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        pass


_vendor_rate_limiters: Dict[Vendor, AsyncTokenBucketRateLimiter] = {}


def _get_vendor_rate_limiter(vendor: Vendor) -> AsyncTokenBucketRateLimiter:
    """Return (or create) the singleton rate limiter for *vendor*."""
    if vendor not in _vendor_rate_limiters:
        rpm = VENDOR_RPM_LIMITS.get(vendor.value, 1000)
        _vendor_rate_limiters[vendor] = AsyncTokenBucketRateLimiter(rpm=rpm)
        _logger.info("Created rate limiter for %s (%d RPM)", vendor.value, rpm)
    return _vendor_rate_limiters[vendor]


# ---------------------------------------------------------------------------
# Model string resolution
# ---------------------------------------------------------------------------


def _resolve_model(model_id: str) -> LLMModel:
    """Resolve a model string to an LLMModel, using heuristics for unknown models."""
    if model_id in MODEL_DICT:
        return MODEL_DICT[model_id]

    # Prefix heuristic for unknown models
    if model_id.startswith("claude-"):
        vendor = Vendor.ANTHROPIC
    elif model_id.startswith("gemini-") or model_id.startswith("models/gemini-"):
        vendor = Vendor.GEMINI
    else:
        vendor = Vendor.OPENAI

    _logger.debug(f"Unknown model '{model_id}', inferring vendor={vendor.value}")
    return LLMModel(
        model_id=model_id,
        vendor=vendor,
        supports_logprobs=False,
        supports_reasoning=False,
        supports_temperature=True,
        default_max_tokens=(
            8192
            if vendor == Vendor.OPENAI
            else 8192 if vendor == Vendor.GEMINI else 16384
        ),
        display_name=model_id,
    )


# ---------------------------------------------------------------------------
# Reasoning effort mapping (int 0-10 scale used by vendor agents)
# ---------------------------------------------------------------------------

_ANTHROPIC_THINKING_MAP = {-1: 0, 0: 0, 2: 1024, 4: 2048, 6: 4096, 8: 8192, 10: 16384}
_OPENAI_REASONING_MAP = {
    -1: None,
    0: None,
    2: "low",
    4: "low",
    6: "medium",
    8: "high",
    10: "high",
}
_GEMINI_THINKING_MAP = {-1: 0, 0: 0, 2: 1024, 4: 2048, 6: 4096, 8: 8192, 10: 16384}

# Bridge between old string API ("low"/"medium"/"high") and new int API (0-10)
_EFFORT_STR_TO_INT = {"low": 2, "medium": 6, "high": 8}
_EFFORT_INT_TO_STR = {0: None, 2: "low", 4: "low", 6: "medium", 8: "high", 10: "high"}


def _convert_effort(effort) -> int:
    """Accept None, str, or int. Return int 0-10."""
    if effort is None:
        return 0
    if isinstance(effort, int):
        return effort
    return _EFFORT_STR_TO_INT.get(effort, 0)


def _validate_effort(effort: int) -> int:
    if effort == -1:
        return -1
    if effort < 0 or effort > 10:
        raise ValueError(
            f"reasoning_effort must be 0-10 (or -1 to disable), got {effort}"
        )
    return round(effort / 2) * 2


def _anthropic_thinking_tokens(effort: int) -> int:
    return _ANTHROPIC_THINKING_MAP[_validate_effort(effort)]


def _openai_reasoning_effort(effort: int) -> Optional[str]:
    return _OPENAI_REASONING_MAP[_validate_effort(effort)]


def _gemini_thinking_tokens(effort: int) -> int:
    return _GEMINI_THINKING_MAP[_validate_effort(effort)]


# ---------------------------------------------------------------------------
# Vendor agent base class and implementations (private)
# ---------------------------------------------------------------------------


class _VendorAgent(ABC):
    """Vendor-agnostic async LLM agent with structured output and retry logic."""

    _fatal_exceptions: Tuple[Type[Exception], ...] = ()
    _temporary_exceptions: Tuple[Type[Exception], ...] = ()

    def __init__(
        self,
        model: LLMModel,
        system_prompt: str,
        user_prompt: str,
        output_type: Optional[Type[BaseModel]] = None,
        reasoning_effort: int = 0,
        max_concurrency: int = 12,
        temperature: float = 0.0,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.output_type = output_type
        self.reasoning_effort = reasoning_effort
        if reasoning_effort < -1 or reasoning_effort > 10:
            raise ValueError(
                f"reasoning_effort must be 0-10 (or -1 to disable), got {reasoning_effort}"
            )
        self.max_concurrency = max_concurrency
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    async def _call_llm(
        self, system: str, user: str, output_schema: Optional[Dict[str, Any]]
    ) -> Any: ...

    @abstractmethod
    async def _parse_structured(
        self, raw: Any, output_type: Type[BaseModel]
    ) -> BaseModel: ...

    @abstractmethod
    async def _parse_logprobs(
        self, raw: Any, target_tokens: List[str]
    ) -> Dict[str, float]: ...

    @abstractmethod
    def _extract_text(self, raw: Any) -> str: ...

    def _format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(
                f"Missing prompt variable {e}. "
                f"Template expects: {[v[1] for v in __import__('string').Formatter().parse(template) if v[1]]}, "
                f"got: {list(variables.keys())}"
            ) from None

    _call_timeout: float = 900  # 15 minutes

    async def _call_with_retry(
        self, system: str, user: str, output_schema: Optional[Dict[str, Any]]
    ) -> Any:
        @retry(
            retry=retry_if_exception_type(
                (*self._temporary_exceptions, asyncio.TimeoutError)
            ),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=2, min=2, max=120),
            reraise=True,
        )
        async def _do_call():
            await _get_vendor_rate_limiter(self.model.vendor).acquire()
            return await asyncio.wait_for(
                self._call_llm(system, user, output_schema),
                timeout=self._call_timeout,
            )

        return await _do_call()

    async def prompt_dict(self, variables: Optional[Dict[str, Any]] = None) -> Any:
        user_text = self._format_prompt(self.user_prompt, variables or {})
        output_schema = None
        if self.output_type:
            output_schema = self.output_type.model_json_schema()
        raw = await self._call_with_retry(self.system_prompt, user_text, output_schema)
        if self.output_type:
            return await self._parse_structured(raw, self.output_type)
        return self._extract_text(raw)

    async def run_prompt_with_probs(
        self,
        variables: Optional[Dict[str, Any]] = None,
        target_tokens: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        target_tokens = target_tokens or []
        user_text = self._format_prompt(self.user_prompt, variables or {})
        if self.model.supports_logprobs:
            raw = await self._call_with_retry(self.system_prompt, user_text, None)
            return await self._parse_logprobs(raw, target_tokens)
        else:
            ConfidenceModel = create_model("ConfidenceModel", confidence=(float, ...))
            schema = ConfidenceModel.model_json_schema()
            raw = await self._call_with_retry(self.system_prompt, user_text, schema)
            parsed = await self._parse_structured(raw, ConfidenceModel)
            confidence = parsed.confidence
            result = {}
            if target_tokens:
                result[target_tokens[0]] = confidence
            return result


class _AnthropicAgent(_VendorAgent):
    """Anthropic Claude agent using tool_use for structured output."""

    _fatal_exceptions = (anthropic.AuthenticationError, anthropic.BadRequestError)
    _temporary_exceptions = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = anthropic.AsyncAnthropic()

    async def _call_llm(self, system, user, output_schema):
        kwargs = {
            "model": self.model.model_id,
            "max_tokens": self.model.default_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        thinking_tokens = _anthropic_thinking_tokens(self.reasoning_effort)
        if thinking_tokens > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_tokens}
        else:
            if self.model.supports_temperature:
                kwargs["temperature"] = self.temperature

        if output_schema and thinking_tokens <= 0:
            kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": "Return structured data",
                    "input_schema": output_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        response = await self._client.messages.create(**kwargs)
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Anthropic response truncated (stop_reason='max_tokens', "
                f"max_tokens={self.model.default_max_tokens}). "
                f"Increase default_max_tokens or reduce input size."
            )
        return response

    async def _parse_structured(self, raw, output_type):
        for block in raw.content:
            if block.type == "tool_use":
                inp = block.input
                # Claude occasionally serializes array fields as JSON strings;
                # coerce any top-level string values that look like JSON.
                _logger.debug(
                    f"tool_use input keys={list(inp.keys())}, "
                    f"results_list type={type(inp.get('results_list', None)).__name__}"
                )
                coerced = {}
                for k, v in inp.items():
                    if isinstance(v, str) and v[:1] in ("[", "{"):
                        try:
                            v = json.loads(v)
                        except Exception as e:
                            _logger.warning(
                                f"json.loads failed for field '{k}': {e}. "
                                f"Full string value:\n{v}"
                            )
                    coerced[k] = v
                return output_type.model_validate(coerced)
        text = self._extract_text(raw)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
        return output_type.model_validate_json(text)

    async def _parse_logprobs(self, raw, target_tokens):
        return {}

    def _extract_text(self, raw) -> str:
        for block in raw.content:
            if block.type == "text":
                return block.text
        return ""


def _make_openai_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively fix a JSON schema for OpenAI strict mode."""
    schema = schema.copy()
    if schema.get("type") == "object" and "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        schema["additionalProperties"] = False
        schema["properties"] = {
            k: _make_openai_strict_schema(v) for k, v in schema["properties"].items()
        }
    if "items" in schema:
        schema["items"] = _make_openai_strict_schema(schema["items"])
    for defs_key in ("$defs", "definitions"):
        if defs_key in schema:
            schema[defs_key] = {
                k: _make_openai_strict_schema(v) for k, v in schema[defs_key].items()
            }
    return schema


class _OpenAIAgent(_VendorAgent):
    """OpenAI agent using json_schema for structured output and native logprobs."""

    _fatal_exceptions = (openai.AuthenticationError, openai.BadRequestError)
    _temporary_exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = openai.AsyncOpenAI()

    async def _call_llm(self, system, user, output_schema):
        kwargs = {
            "model": self.model.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.model.supports_temperature:
            kwargs["temperature"] = self.temperature
        effort = _openai_reasoning_effort(self.reasoning_effort)
        if effort and self.model.supports_reasoning:
            kwargs["reasoning_effort"] = effort
        if output_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": _make_openai_strict_schema(output_schema),
                },
            }
        return await self._client.chat.completions.create(**kwargs)

    async def _parse_structured(self, raw, output_type):
        content = raw.choices[0].message.content
        return output_type.model_validate_json(content)

    async def _parse_logprobs(self, raw, target_tokens):
        probs = {}
        logprobs_data = raw.choices[0].logprobs
        if not logprobs_data or not logprobs_data.content:
            return probs
        for item in logprobs_data.content:
            for top in item.top_logprobs:
                if top.token in target_tokens:
                    probs[top.token] = math.exp(top.logprob)
        return probs

    def _extract_text(self, raw) -> str:
        return raw.choices[0].message.content


class _OpenRouterAgent(_OpenAIAgent):
    """OpenRouter agent — OpenAI-compatible API with a different base URL and reasoning mechanism.

    Uses json_object response format (not json_schema strict) since many OpenRouter
    models don't fully support strict JSON schema mode.  The schema is embedded in the
    system prompt so the model knows what structure to produce.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import os

        self._client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    async def _call_llm(self, system, user, output_schema):
        # Embed schema in system prompt since many OpenRouter models
        # don't support json_schema strict mode (e.g. GLM-5 returns empty strings).
        if output_schema:
            schema_str = json.dumps(output_schema, indent=2)
            system = (
                f"{system}\n\n"
                f"You MUST respond with valid JSON matching this exact schema:\n"
                f"```json\n{schema_str}\n```\n"
                f"Output ONLY the JSON object, no other text."
            )

        kwargs = {
            "model": self.model.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.model.supports_temperature:
            kwargs["temperature"] = self.temperature
        # OpenRouter uses extra_body for reasoning effort instead of reasoning_effort param
        effort_str = _EFFORT_INT_TO_STR.get(_validate_effort(self.reasoning_effort))
        if effort_str and self.model.supports_reasoning:
            kwargs["extra_body"] = {"reasoning": {"effort": effort_str}}
        if output_schema:
            kwargs["response_format"] = {"type": "json_object"}
        return await self._client.chat.completions.create(**kwargs)

    async def _parse_structured(self, raw, output_type):
        content = raw.choices[0].message.content
        if not content or not content.strip():
            finish = raw.choices[0].finish_reason if raw.choices else "unknown"
            raise RuntimeError(
                f"OpenRouter returned empty response (finish_reason={finish}). "
                f"Model may not support structured output for this request."
            )
        # Strip markdown fences if present
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
        # Happy path
        try:
            return output_type.model_validate_json(text)
        except Exception as original_err:
            # Handle schema-echo pattern: model prepends the JSON schema then appends actual data
            import json
            try:
                decoder = json.JSONDecoder()
                first_obj, end_idx = decoder.raw_decode(text)
                # Check if the first object looks like a schema echo
                if isinstance(first_obj, dict) and (
                    "$defs" in first_obj or "properties" in first_obj
                ):
                    remainder = text[end_idx:].strip()
                    if remainder:
                        return output_type.model_validate_json(remainder)
            except (json.JSONDecodeError, Exception):
                pass
            raise original_err


class _GeminiAgent(_VendorAgent):
    """Google Gemini agent using response_schema for structured output."""

    _fatal_exceptions = ()
    _temporary_exceptions = (genai_errors.ClientError, genai_errors.ServerError)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = genai.Client()

    async def _call_llm(self, system, user, output_schema):
        config_kwargs = {
            "system_instruction": system,
            "max_output_tokens": self.model.default_max_tokens,
        }
        if self.model.supports_temperature:
            config_kwargs["temperature"] = self.temperature
        thinking_tokens = _gemini_thinking_tokens(self.reasoning_effort)
        if thinking_tokens > 0:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_tokens
            )
        if output_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = output_schema
        config = genai_types.GenerateContentConfig(**config_kwargs)
        return await self._client.aio.models.generate_content(
            model=self.model.model_id,
            contents=user,
            config=config,
        )

    async def _parse_structured(self, raw, output_type):
        # Check for truncation before attempting validation
        if raw.candidates and raw.candidates[0].finish_reason == "MAX_TOKENS":
            raise RuntimeError(
                f"Gemini response truncated (finish_reason='MAX_TOKENS', "
                f"max_output_tokens={self.model.default_max_tokens}). "
                f"Increase default_max_tokens or reduce input size."
            )
        if raw.text is None:
            # Diagnose why Gemini returned no text
            details = []
            if hasattr(raw, "candidates") and raw.candidates:
                c = raw.candidates[0]
                details.append(f"finish_reason={c.finish_reason}")
                if hasattr(c, "safety_ratings") and c.safety_ratings:
                    details.append(f"safety_ratings={c.safety_ratings}")
            elif hasattr(raw, "candidates"):
                details.append("candidates=[]")
            if hasattr(raw, "prompt_feedback") and raw.prompt_feedback:
                details.append(f"prompt_feedback={raw.prompt_feedback}")
            detail_str = "; ".join(details) or "no diagnostic info available"
            raise RuntimeError(
                f"Gemini returned empty response (text=None). {detail_str}"
            )
        return output_type.model_validate_json(raw.text)

    async def _parse_logprobs(self, raw, target_tokens):
        return {}

    def _extract_text(self, raw) -> str:
        return raw.text


def _create_vendor_agent(
    model: LLMModel,
    system_prompt: str,
    user_prompt: str,
    output_type: Optional[Type[BaseModel]] = None,
    reasoning_effort: int = 0,
    max_concurrency: int = 12,
    temperature: float = 0.0,
) -> _VendorAgent:
    """Factory: create the right vendor agent subclass based on model vendor."""
    kwargs = dict(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_type=output_type,
        reasoning_effort=reasoning_effort,
        max_concurrency=max_concurrency,
        temperature=temperature,
    )
    if model.vendor == Vendor.ANTHROPIC:
        return _AnthropicAgent(**kwargs)
    elif model.vendor == Vendor.OPENAI:
        return _OpenAIAgent(**kwargs)
    elif model.vendor == Vendor.GEMINI:
        return _GeminiAgent(**kwargs)
    elif model.vendor == Vendor.OPENROUTER:
        return _OpenRouterAgent(**kwargs)
    else:
        raise ValueError(f"Unsupported vendor: {model.vendor}")


# ---------------------------------------------------------------------------
# Utility functions (public)
# ---------------------------------------------------------------------------


async def paginate_df_async(df: pd.DataFrame, chunk_size: int = 25):
    """Async generator for DataFrame pagination."""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]
        await asyncio.sleep(0)


async def paginate_list_async(lst, chunk_size: int = 25):
    """Async generator for list pagination."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
        await asyncio.sleep(0)


def _introspect_output_type(
    output_type: Type[BaseModel],
) -> tuple[Optional[str], Optional[str]]:
    """Introspect a Pydantic model to find list field and value field."""
    item_list_field = None
    value_field = None
    for field_name, field_info in output_type.model_fields.items():
        field_type = field_info.annotation
        origin = get_origin(field_type)
        if origin is list or origin is List:
            item_list_field = field_name
            args = get_args(field_type)
            if args and len(args) > 0:
                inner_type = args[0]
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    inner_fields = inner_type.model_fields
                    non_id_fields = [
                        name
                        for name in inner_fields.keys()
                        if name.lower() not in ("id", "index")
                    ]
                    if len(non_id_fields) == 1:
                        value_field = non_id_fields[0]
            break
    return item_list_field, value_field


# ---------------------------------------------------------------------------
# LangfuseClient — local prompt loader (Langfuse dependency removed)
# ---------------------------------------------------------------------------

# Global singleton
_global_langfuse_client: Optional["LangfuseClient"] = None


def get_langfuse_client(logger: Optional[logging.Logger] = None) -> "LangfuseClient":
    """Get or create singleton LangfuseClient (backed by local prompts.py)."""
    global _global_langfuse_client
    if _global_langfuse_client is None:
        _global_langfuse_client = LangfuseClient(logger=logger)
    return _global_langfuse_client


class LangfuseClient:
    """
    Prompt loader. Reads from local prompts.py (Langfuse dependency removed).

    Preserves the same public API so callers don't need changes.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or _logger
        if self.logger:
            self.logger.info("Initialized LangfuseClient (local prompt loader)")

    def get_prompt(self, prompt_name: str) -> tuple[str, str, str, str]:
        """
        Load prompt by name. Accepts 'newsagent/foo' or just 'foo'.

        Returns:
            Tuple of (system_prompt, user_prompt, model_id, reasoning_effort_str)
        """
        from prompts import load_prompt

        return load_prompt(prompt_name)

    def create_llm_agent(
        self,
        prompt_name: str,
        output_type: Type[BaseModel],
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> "LLMagent":
        """Create an LLMagent from a prompt name."""
        prompt_data = self.get_prompt(prompt_name)
        return LLMagent(
            system_prompt=prompt_data[0],
            user_prompt=prompt_data[1],
            output_type=output_type,
            model=prompt_data[2],
            reasoning_effort=prompt_data[3],
            verbose=verbose,
            logger=logger or self.logger,
        )


# ---------------------------------------------------------------------------
# LLMagent — public facade (preserves exact original API)
# ---------------------------------------------------------------------------


class LLMagent:
    """
    General-purpose LLM agent for making structured calls with flexible prompt templating.

    Supports Anthropic, OpenAI, and Gemini via internal vendor delegation.
    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        output_type: Type[BaseModel],
        model: str,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        reasoning_effort: Optional[str] = None,
        trace_enable: Optional[bool] = None,
        trace_tag_list: Optional[List[str]] = None,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.output_type = output_type
        self.model = model  # Keep as string for backward compat
        self.verbose = verbose
        self.logger = logger or _logger
        self.trace_tag = trace_tag_list

        # Validate and store reasoning_effort (as original string/None)
        if reasoning_effort is not None:
            valid_efforts = {"low", "medium", "high"}
            if reasoning_effort not in valid_efforts:
                raise ValueError(
                    f"reasoning_effort must be one of {valid_efforts}, got: {reasoning_effort}"
                )
        self.reasoning_effort = reasoning_effort

        # Resolve model and create vendor delegate
        self._llm_model = _resolve_model(model)
        effort_int = _convert_effort(reasoning_effort)

        self._delegate = _create_vendor_agent(
            model=self._llm_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=output_type,
            reasoning_effort=effort_int,
        )

        # Keep an OpenAI client for logprobs calls (backward compat path)
        self.openai_client = AsyncOpenAI()

        if self.verbose:
            self.logger.info(
                f"Initialized LLMagent: model={self.model}, vendor={self._llm_model.vendor.value}, "
                f"output_type={output_type.__name__ if output_type else None}, reasoning_effort={reasoning_effort}"
            )

    def _build_langfuse_metadata(self) -> Dict[str, Any]:
        """No-op (Langfuse removed). Returns empty dict."""
        return {}

    def _format_prompts(self, variables: Dict[str, Any]) -> str:
        """Format user prompt with variable substitution."""
        try:
            return self.user_prompt.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable in prompt template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompts: {e}")

    # --- Core prompt methods ---

    @retry(
        retry=retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                ValidationError,
                RuntimeError,
            )
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    async def prompt_dict(self, variables: Dict[str, Any]) -> Any:
        """Make a single LLM call with dictionary-based variable substitution."""
        user_message = self._format_prompts(variables).strip()
        if self.verbose:
            self.logger.info(f"User message: {user_message}")

        try:
            result = await self._delegate.prompt_dict(variables)
        except ValidationError as e:
            input_preview = {k: str(v)[:80] for k, v in variables.items()}
            self.logger.error(
                f"Pydantic validation failed (model={self.model}): {e}\n"
                f"  Input variables (first 80 chars): {input_preview}"
            )
            raise

        if self.verbose:
            self.logger.info(f"Result: {result}")
        return result

    @retry(
        retry=retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                ValidationError,
                RuntimeError,
            )
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    async def prompt_dict_chat(
        self, variables: Dict[str, Any], reasoning_effort: Optional[str] = None
    ) -> Any:
        """Make a single LLM call using the vendor-appropriate API."""
        user_message = self._format_prompts(variables).strip()
        if self.verbose:
            self.logger.info(f"User message: {user_message}")

        # Handle per-call reasoning effort override
        effective_reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        )

        if (
            effective_reasoning_effort is not None
            and not self._supports_reasoning_effort()
        ):
            self.logger.warning(
                f"reasoning_effort='{effective_reasoning_effort}' specified but model '{self.model}' "
                f"does not support this parameter. It will be ignored."
            )
            effective_reasoning_effort = None

        # Temporarily adjust delegate's reasoning effort if overridden
        original_effort = self._delegate.reasoning_effort
        if effective_reasoning_effort is not None:
            self._delegate.reasoning_effort = _convert_effort(
                effective_reasoning_effort
            )
            if self.verbose:
                self.logger.info(
                    f"Using reasoning_effort: {effective_reasoning_effort}"
                )

        try:
            result = await self._delegate.prompt_dict(variables)
        except ValidationError as e:
            input_preview = {k: str(v)[:80] for k, v in variables.items()}
            self.logger.error(
                f"Pydantic validation failed (model={self.model}): {e}\n"
                f"  Input variables (first 80 chars): {input_preview}"
            )
            raise
        finally:
            self._delegate.reasoning_effort = original_effort

        if self.verbose:
            self.logger.info(f"Result: {result}")
        return result

    async def run_prompt(self, reasoning_effort: Optional[str] = None, **kwargs) -> Any:
        """Make a single LLM call with keyword argument variable substitution."""
        return await self.prompt_dict_chat(kwargs, reasoning_effort=reasoning_effort)

    # --- Logprobs methods ---

    def _supports_logprobs(self) -> bool:
        """Check if the current model supports logprobs."""
        return self._llm_model.supports_logprobs

    def _supports_reasoning_effort(self) -> bool:
        """Check if the current model supports reasoning_effort."""
        return self._llm_model.supports_reasoning

    def _extract_token_probabilities(
        self, logprobs_data: Dict, target_tokens: List[str]
    ) -> Dict[str, float]:
        """Extract probabilities for specific target tokens from OpenAI logprobs response."""
        if not logprobs_data or getattr(logprobs_data, "content", None) is None:
            raise ValueError(
                "Invalid logprobs_data. Must contain 'content' key with non-None value."
            )
        first_token_logprobs = logprobs_data.content[0]
        if not hasattr(first_token_logprobs, "top_logprobs"):
            raise ValueError(
                "Invalid first_token_logprobs. Could not find 'top_logprobs' key or 'top_logprobs' is empty."
            )
        result = {}
        top_logprobs = first_token_logprobs.top_logprobs
        for target_token in target_tokens:
            found_prob = 0.0
            for token_info in top_logprobs:
                if token_info.token == target_token:
                    found_prob = math.exp(token_info.logprob)
                    break
            result[target_token] = found_prob
        return result

    @retry(
        retry=retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            )
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    async def prompt_dict_chat_probs(
        self, variables: Dict[str, Any], top_logprobs: int = 5
    ) -> Tuple[str, Dict]:
        """Make a single LLM call with logprobs enabled (no structured output)."""
        if not self._supports_logprobs():
            raise ValueError(
                f"Model '{self.model}' does not support logprobs. "
                f"Use an OpenAI model like gpt-4.1-mini."
            )

        user_message = self._format_prompts(variables).strip()
        if self.verbose:
            self.logger.info(f"User message (with logprobs): {user_message}")

        # Logprobs only supported natively on OpenAI, so use direct OpenAI client
        client = self.openai_client
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            await _get_vendor_rate_limiter(Vendor.OPENAI).acquire()
            api_params = {
                "model": self.model,
                "messages": messages,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
            response = await client.chat.completions.create(**api_params)
            message = response.choices[0].message
            if hasattr(message, "refusal") and message.refusal:
                self.logger.error(f"LLM refused request. User message: {user_message}")
                raise ValueError(f"LLM refused the request: {message.refusal}")
            response_text = message.content
            logprobs_data = response.choices[0].logprobs
        except BadRequestError as e:
            self.logger.error(f"BadRequestError: {e}")
            raise

        if self.verbose:
            self.logger.info(f"Response text: {response_text}")
            self.logger.info(f"Logprobs available: {logprobs_data is not None}")

        return response_text, logprobs_data

    async def run_prompt_with_probs(
        self, target_tokens: List[str] = ["1"], **kwargs
    ) -> Dict[str, float]:
        """Make a single LLM call and return probabilities for specific target tokens."""
        if self._supports_logprobs():
            # Native logprobs path (OpenAI)
            response_text, logprobs_data = await self.prompt_dict_chat_probs(kwargs)
            probabilities = self._extract_token_probabilities(
                logprobs_data, target_tokens
            )
        else:
            # Confidence fallback path (Anthropic/Gemini)
            probabilities = await self._delegate.run_prompt_with_probs(
                variables=kwargs, target_tokens=target_tokens
            )
        if self.verbose:
            self.logger.info(f"Token probabilities: {probabilities}")
        return probabilities

    # --- Batch processing methods ---

    async def prompt_batch(
        self,
        variables_list: List[Dict[str, Any]],
        batch_size: int = 25,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        retries: int = 3,
        item_list_field: str = "results_list",
        item_id_field: str = "",
        chat: bool = True,
    ) -> List[Any]:
        """Process a list of variable dictionaries using true batch calls."""
        if not variables_list:
            return []

        batches = [
            variables_list[i : i + batch_size]
            for i in range(0, len(variables_list), batch_size)
        ]

        sem = asyncio.Semaphore(max_concurrency)
        if self.verbose:
            self.logger.info(
                f"Processing {len(batches)} batches with concurrency {max_concurrency}"
            )

        async def _process_batch(
            batch_idx: int, batch_variables: List[Dict[str, Any]]
        ) -> tuple[int, List[Any]]:
            last_exc = None
            for attempt in range(retries):
                try:
                    async with sem:
                        if chat:
                            result = await self.prompt_dict_chat(
                                {"input_str": str(batch_variables)}
                            )
                        else:
                            result = await self.prompt_dict(
                                {"input_str": str(batch_variables)}
                            )
                        batch_results = result
                        if item_id_field:
                            sent_ids = [
                                var.get(item_id_field) for var in batch_variables
                            ]
                            received_ids = []
                            for result in batch_results:
                                if hasattr(result, item_id_field):
                                    received_ids.append(getattr(result, item_id_field))
                                elif (
                                    isinstance(result, dict) and item_id_field in result
                                ):
                                    received_ids.append(result[item_id_field])
                                else:
                                    raise ValueError(
                                        f"Result missing required ID field '{item_id_field}': {result}"
                                    )
                            sent_set = set(sent_ids)
                            received_set = set(received_ids)
                            if sent_set != received_set:
                                missing_ids = sent_set - received_set
                                extra_ids = received_set - sent_set
                                error_msg = f"ID mismatch in batch {batch_idx}:"
                                if missing_ids:
                                    error_msg += f" Missing IDs: {missing_ids}"
                                if extra_ids:
                                    error_msg += f" Extra IDs: {extra_ids}"
                                raise ValueError(error_msg)
                        return batch_idx, batch_results
                except Exception as e:
                    last_exc = e
                    self.logger.warning(
                        f"Batch {batch_idx} attempt {attempt + 1}/{retries} failed: {e}"
                    )
                    if attempt < retries - 1:
                        await asyncio.sleep(2**attempt)
            raise last_exc or RuntimeError(
                f"Unknown error processing batch {batch_idx}"
            )

        tasks = [
            asyncio.create_task(_process_batch(i, batch))
            for i, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*tasks)

        if item_list_field:
            flattened_results = []
            flattened_success = False
            for batch_idx, results in sorted(batch_results, key=lambda x: x[0]):
                if hasattr(results, item_list_field):
                    flattened_results.extend(getattr(results, item_list_field))
                else:
                    break
                flattened_success = True
            if flattened_success:
                if len(flattened_results) != len(variables_list):
                    raise ValueError(
                        f"Result count mismatch: expected {len(variables_list)}, got {len(flattened_results)}"
                    )
                else:
                    return flattened_results
            else:
                return batch_results
        else:
            return batch_results

    async def filter_dataframe_chunk(
        self,
        input_df: pd.DataFrame,
        input_vars: Optional[Dict[str, Any]] = None,
        item_list_field: str = "results_list",
        item_id_field: str = "id",
        retries: int = 3,
        chat: bool = True,
    ) -> Any:
        """Process a single DataFrame chunk asynchronously."""
        expected_count = len(input_df)
        last_exc = None

        for attempt in range(retries):
            try:
                input_text = input_df.to_json(orient="records", indent=2)
                input_dict = {"input_text": input_text}
                if input_vars is not None:
                    input_dict.update(input_vars)

                self.logger.info(
                    f"Sending chunk of {expected_count} items to {self.model}"
                )

                if chat:
                    result = await self.prompt_dict_chat(input_dict)
                else:
                    result = await self.prompt_dict(input_dict)

                if item_list_field:
                    if hasattr(result, item_list_field):
                        result_list = getattr(result, item_list_field)
                        if isinstance(result_list, list):
                            received_count = len(result_list)
                            if received_count != expected_count:
                                error_msg = f"Item count mismatch: expected {expected_count}, got {received_count}"
                                self.logger.warning(
                                    f"Attempt {attempt + 1}/{retries}: {error_msg}"
                                )
                                if attempt < retries - 1:
                                    await asyncio.sleep(2**attempt)
                                    continue
                                else:
                                    raise ValueError(error_msg)

                            if item_id_field and item_id_field in input_df.columns:
                                sent_ids = input_df[item_id_field].tolist()
                                received_ids = []
                                for item in result_list:
                                    if hasattr(item, item_id_field):
                                        received_ids.append(
                                            getattr(item, item_id_field)
                                        )
                                    elif (
                                        isinstance(item, dict) and item_id_field in item
                                    ):
                                        received_ids.append(item[item_id_field])
                                    else:
                                        error_msg = f"Result item missing required ID field '{item_id_field}': {item}"
                                        self.logger.warning(
                                            f"Attempt {attempt + 1}/{retries}: {error_msg}"
                                        )
                                        if attempt < retries - 1:
                                            await asyncio.sleep(2**attempt)
                                            continue
                                        else:
                                            raise ValueError(error_msg)

                                if sent_ids != received_ids:
                                    sent_set = set(sent_ids)
                                    received_set = set(received_ids)
                                    missing_ids = sent_set - received_set
                                    extra_ids = received_set - sent_set
                                    if missing_ids or extra_ids:
                                        error_msg = "ID presence mismatch:"
                                        if missing_ids:
                                            error_msg += f" Missing IDs: {missing_ids}"
                                        if extra_ids:
                                            error_msg += f" Extra IDs: {extra_ids}"
                                    else:
                                        error_msg = f"ID order mismatch: sent {sent_ids} != received {received_ids}"
                                    self.logger.warning(
                                        f"Attempt {attempt + 1}/{retries}: {error_msg}"
                                    )
                                    if attempt < retries - 1:
                                        await asyncio.sleep(2**attempt)
                                        continue
                                    else:
                                        raise ValueError(error_msg)
                        else:
                            raise ValueError(
                                f"Field '{item_list_field}' is not a list: {type(result_list)}"
                            )
                    else:
                        raise ValueError(
                            f"Result missing required field '{item_list_field}': {result}"
                        )

                received_count = "?"
                if item_list_field and hasattr(result, item_list_field):
                    result_list = getattr(result, item_list_field)
                    if isinstance(result_list, list):
                        received_count = len(result_list)
                self.logger.info(
                    f"Received chunk: {received_count} items returned, validation passed"
                )
                return result

            except (ValidationError, RetryError) as e:
                last_exc = e
                self.logger.error(
                    f"Pydantic validation failed in filter_dataframe_chunk: {e}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except asyncio.TimeoutError as e:
                last_exc = e
                self.logger.error(f"Timeout error in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except (ConnectionError, TimeoutError) as e:
                last_exc = e
                self.logger.error(
                    f"Network/timeout error in filter_dataframe_chunk: {str(e)}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except ValueError as e:
                last_exc = e
                self.logger.error(f"Invalid data in filter_dataframe_chunk: {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
            except Exception as e:
                last_exc = e
                self.logger.error(
                    f"Unexpected error in filter_dataframe_chunk: {str(e)}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        raise last_exc or RuntimeError(f"Unknown error after {retries} attempts")

    async def _process_indexed_chunk(
        self,
        chunk_idx: int,
        chunk_df: pd.DataFrame,
        sem: asyncio.Semaphore,
        input_vars: Optional[Dict[str, Any]] = None,
        item_list_field: str = "results_list",
        item_id_field: str = "id",
        retries: int = 3,
        chat: bool = True,
    ) -> tuple[int, Any]:
        """Process a single chunk and return with its index for order preservation."""
        async with sem:
            result = await self.filter_dataframe_chunk(
                chunk_df,
                input_vars=input_vars,
                item_list_field=item_list_field,
                item_id_field=item_id_field,
                retries=retries,
                chat=chat,
            )
            return chunk_idx, result

    async def filter_dataframe_batch(
        self,
        input_df: pd.DataFrame,
        input_vars: Optional[Dict[str, Any]] = None,
        item_list_field: str = "results_list",
        item_id_field: str = "id",
        retries: int = 3,
        chunk_size: int = 25,
        return_series: bool = False,
        value_field: str = "output",
        chat: bool = True,
        return_probabilities: bool = False,
        target_tokens: List[str] = None,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        **kwargs,
    ) -> Any:
        """Process a DataFrame in chunks asynchronously with concurrent calls."""
        if input_df.empty:
            return []

        sem = asyncio.Semaphore(max_concurrency)

        # Handle probability extraction mode
        if return_probabilities:
            if target_tokens is None:
                target_tokens = ["1"]

            if not self._supports_logprobs():
                raise ValueError(
                    f"Model '{self.model}' does not support logprobs required for probability extraction"
                )

            self.logger.info(
                f"Processing {len(input_df)} probability requests "
                f"(model={self.model}, concurrency={max_concurrency})"
            )

            async def _process_row_with_sem(row):
                async with sem:
                    row_vars = row.to_dict()
                    if input_vars:
                        row_vars.update(input_vars)
                    return await self.run_prompt_with_probs(
                        target_tokens=target_tokens, **row_vars
                    )

            tasks = [_process_row_with_sem(row) for _, row in input_df.iterrows()]
            prob_dicts = await asyncio.gather(*tasks)
            probabilities = [
                prob_dict.get(target_tokens[0], 0.0) for prob_dict in prob_dicts
            ]
            self.logger.info(f"Completed {len(probabilities)} probability requests")
            return pd.Series(probabilities, index=input_df.index)

        # Create chunks
        chunks = []
        async for chunk in paginate_df_async(input_df, chunk_size):
            chunks.append(chunk)

        if not chunks:
            return []

        self.logger.info(
            f"Processing {len(input_df)} items in {len(chunks)} chunks "
            f"(chunk_size={chunk_size}, model={self.model}, concurrency={max_concurrency})"
        )

        tasks = [
            self._process_indexed_chunk(
                i,
                chunk,
                sem,
                input_vars=input_vars,
                item_list_field=item_list_field,
                item_id_field=item_id_field,
                retries=retries,
                chat=chat,
                **kwargs,
            )
            for i, chunk in enumerate(chunks)
        ]

        try:
            indexed_results = await asyncio.gather(*tasks)
            sorted_results = sorted(indexed_results, key=lambda x: x[0])
        except Exception as e:
            self.logger.error(f"Error in filter_dataframe_batch: {e}")
            raise

        self.logger.info(
            f"All {len(chunks)} chunks completed for {len(input_df)} items"
        )

        if item_list_field:
            try:
                all_items = []
                for chunk_idx, chunk_result in sorted_results:
                    if hasattr(chunk_result, item_list_field):
                        result_list = getattr(chunk_result, item_list_field)
                        if isinstance(result_list, list):
                            all_items.extend(result_list)
                        else:
                            self.logger.error(
                                f"Field '{item_list_field}' is not a list: {type(result_list)}"
                            )
                            return [result for _, result in sorted_results]
                    else:
                        self.logger.error(
                            f"Result missing field '{item_list_field}': {chunk_result}"
                        )
                        return [result for _, result in sorted_results]

                if return_series:
                    values = [getattr(item, value_field) for item in all_items]
                    return pd.Series(values, index=input_df.index)

                if sorted_results and hasattr(sorted_results[0][1], item_list_field):
                    first_result = sorted_results[0][1]
                    concatenated_result = first_result.__class__(
                        **{
                            **{
                                k: v
                                for k, v in first_result.__dict__.items()
                                if k != item_list_field
                            },
                            item_list_field: all_items,
                        }
                    )
                    return concatenated_result
                else:
                    return all_items

            except Exception as e:
                self.logger.error(f"Error concatenating results: {e}")
                return [result for _, result in sorted_results]
        else:
            if return_series:
                self.logger.warning(
                    "return_series=True but no item_list_field specified, returning raw results"
                )
            return [result for _, result in sorted_results]

    async def filter_dataframe(
        self, input_df: pd.DataFrame, value_field: str = "output", **kwargs
    ) -> pd.Series:
        """Process DataFrame and return values as Series for direct column assignment."""
        result = await self.filter_dataframe_batch(input_df, **kwargs)

        if isinstance(result, pd.Series):
            return result

        if hasattr(result, "results_list"):
            values = [getattr(item, value_field) for item in result.results_list]
        elif isinstance(result, list):
            values = [getattr(item, value_field) for item in result]
        else:
            raise ValueError(
                f"Unexpected result format from filter_dataframe_batch: {type(result)}"
            )

        if len(values) != len(input_df):
            raise ValueError(
                f"Value count mismatch: expected {len(input_df)}, got {len(values)}"
            )

        return pd.Series(values, index=input_df.index)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def run_prompt_on_dataframe(
    input_df: pd.DataFrame,
    prompt_name: str,
    output_type: Type[BaseModel],
    value_field: Optional[str] = None,
    item_list_field: Optional[str] = None,
    item_id_field: str = "id",
    chunk_size: int = 25,
    max_concurrency: int = DEFAULT_CONCURRENCY,
    return_probabilities: bool = False,
    target_tokens: Optional[List[str]] = None,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> pd.Series:
    """
    Convenience function to run a prompt on a DataFrame.

    Automatically fetches prompt from local prompts.py, creates LLMagent,
    introspects output_type, runs inference, and returns pandas Series.
    """
    logger = logger or _logger

    lf_client = get_langfuse_client(logger=logger)
    system_prompt, user_prompt, model, reasoning_effort = lf_client.get_prompt(
        prompt_name
    )

    if item_list_field is None or value_field is None:
        detected_list_field, detected_value_field = _introspect_output_type(output_type)
        if item_list_field is None:
            item_list_field = detected_list_field or "results_list"
        if value_field is None:
            value_field = detected_value_field
            if value_field is None and not return_probabilities:
                raise ValueError(
                    f"Could not auto-detect value_field for {output_type.__name__}. "
                    f"Please specify explicitly."
                )

    agent = LLMagent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_type=output_type,
        model=model,
        reasoning_effort=reasoning_effort,
        verbose=verbose,
        logger=logger,
    )

    result_series = await agent.filter_dataframe(
        input_df,
        value_field=value_field,
        item_list_field=item_list_field,
        item_id_field=item_id_field,
        chunk_size=chunk_size,
        max_concurrency=max_concurrency,
        return_probabilities=return_probabilities,
        target_tokens=target_tokens,
        **kwargs,
    )

    return result_series
