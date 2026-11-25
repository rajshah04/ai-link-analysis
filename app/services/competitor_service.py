"""Competitor post-processing.

The LLM *suggests* competitors (we have no market-research API). To avoid presenting
hallucinated companies as fact, each suggestion is labelled and, where a website is
given, checked by fetching its homepage: reachable AND mentions the company name.
"""
from __future__ import annotations

import asyncio
import re

from app.config import Settings
from app.schemas import BrandPositioning, CompetitorEntry, ComparisonRow
from app.services.website_extractor import WebsiteFetchError, extract_page, safe_get
from app.utils.validators import UrlValidationError


def _name_tokens(name: str) -> list[str]:
    stop = {"inc", "llc", "ltd", "co", "company", "corp", "the", "and", "technologies", "technology", "group", "solutions"}
    return [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) >= 3 and t not in stop]


async def _verify_one(c: CompetitorEntry, settings: Settings) -> None:
    if not c.website:
        c.source, c.verified = "ai_suggestion", False
        c.verification_note = "No website provided; unverified AI suggestion."
        return
    try:
        url, html = await safe_get(c.website, settings, max_bytes=400_000)
    except (WebsiteFetchError, UrlValidationError) as exc:
        c.source, c.verified = "ai_suggestion", False
        c.verification_note = f"Could not verify website: {getattr(exc, 'message', str(exc))}"
        return
    page = extract_page(html, url)
    haystack = " ".join([page.title, page.meta_description, " ".join(page.og.values()), page.text[:3000]]).lower()
    tokens = _name_tokens(c.name)
    if tokens and any(t in haystack for t in tokens):
        c.source, c.verified = "ai_suggestion_verified_by_site_fetch", True
        c.verification_note = f"Website reachable and mentions the company name (page title: {page.title[:80] or 'n/a'})."
    else:
        c.source, c.verified = "ai_suggestion", False
        c.verification_note = "Website reachable but does not clearly match the company name; treat as unverified."


async def verify_competitors(competitors: list[CompetitorEntry], settings: Settings) -> None:
    if not settings.verify_competitors:
        for c in competitors:
            c.verification_note = "Verification disabled; AI-generated suggestion."
        return
    await asyncio.gather(*[_verify_one(c, settings) for c in competitors[:6]], return_exceptions=True)


def build_comparison_table(brand_name: str, pos: BrandPositioning, competitors: list[CompetitorEntry]) -> list[ComparisonRow]:
    """One block of rows per competitor (mirrors the table in the brief)."""
    rows: list[ComparisonRow] = []
    for c in competitors[:3]:
        rows += [
            ComparisonRow(attribute="Brand Name", your_brand=brand_name, competitor=c.name, competitor_name=c.name),
            ComparisonRow(attribute="Main Service", your_brand=pos.main_service, competitor=c.main_service or "n/a", competitor_name=c.name),
            ComparisonRow(attribute="Target Market", your_brand=pos.target_market, competitor=c.target_market or "n/a", competitor_name=c.name),
            ComparisonRow(attribute="Key Strength", your_brand=pos.key_strength, competitor="; ".join(c.strengths[:2]) or "n/a", competitor_name=c.name),
            ComparisonRow(attribute="Opportunity", your_brand=pos.opportunity, competitor="; ".join(c.weaknesses_or_gaps[:2]) or "n/a", competitor_name=c.name),
        ]
    return rows
