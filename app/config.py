"""Central configuration. All secrets come from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
    "gemini": "gemini-2.5-flash-lite",
}

# List prices in USD per 1M tokens (input, output) used ONLY for the cost estimate
# returned in `meta`. Verified 2026-11-18 – see README "LLM comparison" for sources.
# Update these when providers change pricing.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),     # prompts <= 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    fallback_provider: str | None
    fallback_model: str | None
    max_content_chars: int
    fetch_timeout: int
    max_download_bytes: int
    extra_pages: int
    verify_competitors: bool
    cache_ttl: int
    rate_limit_per_minute: int
    api_access_key: str | None

    def api_key_for(self, provider: str) -> str | None:
        env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
        return os.getenv(env) or None


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    fb_provider = (os.getenv("FALLBACK_LLM_PROVIDER") or "").strip().lower() or None
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, ""),
        fallback_provider=fb_provider,
        fallback_model=(os.getenv("FALLBACK_LLM_MODEL") or DEFAULT_MODELS.get(fb_provider or "", "")) or None,
        max_content_chars=_int("MAX_CONTENT_CHARS", 20000),
        fetch_timeout=_int("FETCH_TIMEOUT_SECONDS", 12),
        max_download_bytes=_int("MAX_DOWNLOAD_BYTES", 2_000_000),
        extra_pages=_int("EXTRA_PAGES", 3),
        verify_competitors=_bool("VERIFY_COMPETITORS", True),
        cache_ttl=_int("CACHE_TTL_SECONDS", 3600),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 10),
        api_access_key=os.getenv("API_ACCESS_KEY") or None,
    )


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICING_USD_PER_MTOK.get(model)
    if not price:
        return None
    return round(input_tokens * price[0] / 1e6 + output_tokens * price[1] / 1e6, 6)
