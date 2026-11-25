"""Thin provider-agnostic LLM client (Anthropic / OpenAI / Gemini) using plain HTTPX.

Returns parsed JSON plus token usage. Structured output strategy per provider:
  * Anthropic: forced tool call whose input_schema is our JSON Schema (most reliable)
  * OpenAI:    response_format = json_object
  * Gemini:    responseMimeType = application/json
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from app.config import Settings

log = logging.getLogger("llm")
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class LLMError(Exception):
    """Provider unavailable / misconfigured / rate-limited."""


class LLMOutputError(LLMError):
    """The model answered but not with usable JSON."""


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    latency_s: float


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("Model did not return JSON.") from None
        try:
            obj = json.loads(text[start : end + 1])
        except ValueError:
            raise LLMOutputError("Model returned malformed or truncated JSON.") from None
    if not isinstance(obj, dict):
        raise LLMOutputError("Model JSON was not an object.")
    return obj


class LLMClient:
    def __init__(self, provider: str, model: str, settings: Settings, max_output_tokens: int = 6000):
        if provider not in {"anthropic", "openai", "gemini"}:
            raise LLMError(f"Unsupported LLM provider '{provider}'.")
        self.provider, self.model, self.max_output_tokens = provider, model, max_output_tokens
        self._key = settings.api_key_for(provider)

    async def complete_json(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        if not self._key:
            raise LLMError(f"No API key configured for provider '{self.provider}'.")
        started = time.perf_counter()
        call = {"anthropic": self._anthropic, "openai": self._openai, "gemini": self._gemini}[self.provider]
        async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10)) as client:
            data, tin, tout = await self._with_retries(lambda: call(client, system, user, schema))
        return LLMResult(data, tin, tout, self.provider, self.model, time.perf_counter() - started)

    async def _with_retries(self, fn, attempts: int = 3):
        delay = 1.0
        for i in range(attempts):
            try:
                return await fn()
            except httpx.TimeoutException:
                err: Exception = LLMError("LLM request timed out.")
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code not in RETRY_STATUS:
                    # Never echo response bodies verbatim into logs/clients: could contain request echoes.
                    raise LLMError(f"LLM provider returned HTTP {code}.") from None
                err = LLMError(f"LLM provider returned HTTP {code}.")
            except httpx.HTTPError:
                err = LLMError("Could not reach the LLM provider.")
            if i == attempts - 1:
                raise err
            await asyncio.sleep(delay)
            delay *= 2
        raise LLMError("LLM call failed.")  # pragma: no cover

    # ------------------------------------------------------------ providers
    async def _anthropic(self, client, system, user, schema):
        body = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema:
            body["tools"] = [{"name": "submit_analysis", "description": "Submit the completed website analysis.",
                              "input_schema": schema}]
            body["tool_choice"] = {"type": "tool", "name": "submit_analysis"}
        r = await client.post("https://api.anthropic.com/v1/messages", json=body,
                              headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        j = r.json()
        if j.get("stop_reason") == "max_tokens":
            raise LLMOutputError("Model output was truncated (max_tokens reached).")
        blocks = j.get("content", [])
        tool = next((b for b in blocks if b.get("type") == "tool_use"), None)
        data = tool["input"] if tool else _parse_json("".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
        u = j.get("usage", {})
        return data, u.get("input_tokens", 0), u.get("output_tokens", 0)

    async def _openai(self, client, system, user, schema):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_output_tokens,
        }
        r = await client.post("https://api.openai.com/v1/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {self._key}"})
        r.raise_for_status()
        j = r.json()
        choice = j["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMOutputError("Model output was truncated (max tokens reached).")
        u = j.get("usage", {})
        return _parse_json(choice["message"]["content"] or ""), u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    async def _gemini(self, client, system, user, schema):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2,
                                 "maxOutputTokens": self.max_output_tokens},
        }
        r = await client.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                              json=body, headers={"x-goog-api-key": self._key})
        r.raise_for_status()
        j = r.json()
        cand = (j.get("candidates") or [{}])[0]
        if cand.get("finishReason") == "MAX_TOKENS":
            raise LLMOutputError("Model output was truncated (max tokens reached).")
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        u = j.get("usageMetadata", {})
        return _parse_json(text), u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
