"""Pydantic models: API request/response and the structure the LLM must return."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
_PLATFORM_ALIASES = {
    "google": "Google", "google search": "Google",
    "facebook": "Facebook", "fb": "Facebook", "meta": "Facebook",
    "instagram": "Instagram", "ig": "Instagram",
    "linkedin": "LinkedIn", "linked in": "LinkedIn",
    "x": "X", "twitter": "X", "x (twitter)": "X", "x/twitter": "X", "x (formerly twitter)": "X",
}


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ----------------------------------------------------------------- request
class LinkAnalysisRequest(BaseModel):
    url: str = Field(..., examples=["https://example.com"], description="Website URL to analyse")


# ------------------------------------------------------- LLM output models
class Brand(_LLMModel):
    name: str
    category: str
    summary: str


class CompetitorEntry(_LLMModel):
    name: str
    website: Optional[str] = Field(None, description="Competitor domain/URL if you are confident, else null")
    main_service: str = ""
    target_market: str = ""
    strengths: list[str] = []
    weaknesses_or_gaps: list[str] = []
    confidence: Literal["high", "medium", "low"] = "medium"
    # Filled in by competitor_service (not by the LLM):
    source: str = "ai_suggestion"
    verified: bool = False
    verification_note: Optional[str] = None


class BrandPositioning(_LLMModel):
    main_service: str
    target_market: str
    key_strength: str
    opportunity: str


class ComparisonRow(BaseModel):
    attribute: str
    your_brand: str
    competitor: str
    competitor_name: str


class CompetitorAnalysis(_LLMModel):
    brand_overview: str
    main_category: str
    main_products_services: list[str] = []
    brand_positioning: BrandPositioning
    competitors: list[CompetitorEntry] = []
    strengths: list[str] = Field([], description="Your brand's competitive advantages")
    opportunities: list[str] = Field([], description="Ways to improve market positioning")
    comparison_table: list[ComparisonRow] = []  # built in code, not by the LLM


class SearchQuery(_LLMModel):
    query: str
    platforms: list[Platform] = Field(..., min_length=1)
    intent: str
    business_relevance: str
    target_audience: str

    @field_validator("platforms", mode="before")
    @classmethod
    def _normalise_platforms(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = [v]
        out: list[str] = []
        for item in v or []:
            key = str(item).strip().lower()
            canon = _PLATFORM_ALIASES.get(key)
            if canon and canon not in out:
                out.append(canon)
        return out


class SupportingContent(_LLMModel):
    type: Literal["service_landing_page", "industry_page", "faq", "blog_topic", "location_page", "supporting_article"]
    title: str
    rationale: str


class SeoImprovements(_LLMModel):
    primary_keyword: str
    secondary_keywords: list[str] = []
    long_tail_keywords: list[str] = []
    business_keywords: list[str] = []
    recommended_title: str
    title_structure: str = ""
    meta_description: str
    supporting_content: list[SupportingContent] = []
    # Filled in from the real page by the extractor (not by the LLM):
    current_title: Optional[str] = None
    current_meta_description: Optional[str] = None
    title_length: Optional[int] = None
    meta_description_length: Optional[int] = None


class TopService(_LLMModel):
    name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source: Literal["website", "inferred"] = Field(
        ..., description="'website' only if the service is explicitly present in the provided content"
    )
    evidence: Optional[str] = Field(None, description="Short paraphrase of where on the site it appears")


class AudienceSegment(_LLMModel):
    audience: str
    business_type: str
    requirement: str


class TargetAudience(_LLMModel):
    primary_audience: list[str] = []
    secondary_audience: list[str] = []
    industries: list[str] = []
    business_types: list[str] = []
    customer_profiles: list[str] = []
    geographic_focus: Optional[str] = None
    customer_needs: list[str] = []
    customer_challenges: list[str] = []
    customer_interests: list[str] = []
    recommended_segments: list[AudienceSegment] = []


class LLMAnalysis(_LLMModel):
    """Exactly what the LLM must return (validated before we trust it)."""
    brand: Brand
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(..., min_length=5)
    seo_improvements: SeoImprovements
    top_services: list[TopService] = Field(..., min_length=1)
    target_audience: TargetAudience

    @field_validator("target_search_queries")
    @classmethod
    def _five_queries(cls, v: list[SearchQuery]) -> list[SearchQuery]:
        return v[:5]

    @field_validator("top_services")
    @classmethod
    def _five_services(cls, v: list[TopService]) -> list[TopService]:
        return v[:5]


# ---------------------------------------------------------------- response
class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery]
    seo_improvements: SeoImprovements
    top_services: list[TopService]
    target_audience: TargetAudience


class Meta(BaseModel):
    provider: str
    model: str
    pages_analyzed: list[str]
    cached: bool = False
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Optional[float] = None
    used_fallback_model: bool = False
    warnings: list[str] = []
    disclaimers: list[str] = []


class LinkAnalysisResponse(BaseModel):
    success: Literal[True] = True
    website: str
    brand: Brand
    analysis: Analysis
    meta: Meta


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody


def inline_schema(model: type[BaseModel]) -> dict:
    """JSON Schema with $ref/$defs inlined (some LLM APIs do not support $ref)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(i) for i in node]
        return node

    return resolve(schema)
