"""Measure latency, token use, cost and structural quality of candidate LLMs on real websites.

    python -m scripts.benchmark --urls https://stripe.com https://www.shopify.com \
        --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5 openai:gpt-5-mini gemini:gemini-2.5-flash-lite \
        --runs 3

Needs the matching API keys in the environment/.env. Produces benchmark_results.json and a Markdown
table you can paste into the README. 'Accuracy' cannot be computed automatically: the script writes a
manual-review sheet (docs/accuracy_review_template.md) with a 1-5 rubric for you to fill in per model.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from pydantic import ValidationError

from app.config import estimate_cost_usd, get_settings
from app.schemas import LLMAnalysis, inline_schema
from app.services.analysis_service import SYSTEM_PROMPT, build_user_prompt
from app.services.llm_service import LLMClient, LLMError
from app.services.website_extractor import fetch_site


def structure_score(a: LLMAnalysis) -> float:
    """0-1 completeness / rule-compliance check (NOT a measure of factual accuracy)."""
    checks = [
        len(a.competitor_analysis.competitors) >= 3,
        len({p for q in a.target_search_queries for p in q.platforms}) == 5,
        len(a.top_services) == 5,
        any(s.source == "website" for s in a.top_services),
        len(a.seo_improvements.recommended_title) <= 65,
        140 <= len(a.seo_improvements.meta_description) <= 165,
        len(a.seo_improvements.long_tail_keywords) >= 3,
        len(a.seo_improvements.supporting_content) >= 5,
        len(a.target_audience.recommended_segments) >= 3,
        bool(a.target_audience.customer_needs),
    ]
    return sum(checks) / len(checks)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="provider:model")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    settings, schema = get_settings(), inline_schema(LLMAnalysis)

    sites = {u: await fetch_site(u, settings) for u in args.urls}
    rows, raw = [], []
    for spec in args.models:
        provider, model = spec.split(":", 1)
        client = LLMClient(provider, model, settings)
        lat, costs, scores, ok, tin, tout = [], [], [], 0, [], []
        total = 0
        for url, site in sites.items():
            prompt = build_user_prompt(site)
            for _ in range(args.runs):
                total += 1
                try:
                    res = await client.complete_json(SYSTEM_PROMPT, prompt, schema)
                    parsed = LLMAnalysis.model_validate(res.data)  # first-try validity, no repair
                except (LLMError, ValidationError) as exc:
                    raw.append({"model": spec, "url": url, "error": str(exc)[:300]})
                    continue
                ok += 1
                lat.append(res.latency_s); tin.append(res.input_tokens); tout.append(res.output_tokens)
                costs.append(estimate_cost_usd(model, res.input_tokens, res.output_tokens) or 0)
                scores.append(structure_score(parsed))
                raw.append({"model": spec, "url": url, "latency_s": res.latency_s, "input_tokens": res.input_tokens,
                            "output_tokens": res.output_tokens, "output": res.data})
        med = lambda xs: round(statistics.median(xs), 2) if xs else None
        rows.append({"model": spec, "valid_json_rate": f"{ok}/{total}", "median_latency_s": med(lat),
                     "p_max_latency_s": round(max(lat), 2) if lat else None, "median_input_tokens": med(tin),
                     "median_output_tokens": med(tout), "mean_cost_usd": round(statistics.mean(costs), 5) if costs else None,
                     "mean_structure_score": round(statistics.mean(scores), 2) if scores else None})

    Path("benchmark_results.json").write_text(json.dumps({"summary": rows, "runs": raw}, indent=2))
    print("\n| Model | Valid JSON (1st try) | Median latency (s) | Median in/out tokens | Mean cost / analysis | Structure score |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['valid_json_rate']} | {r['median_latency_s']} | {r['median_input_tokens']} / {r['median_output_tokens']} | ${r['mean_cost_usd']} | {r['mean_structure_score']} |")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/accuracy_review_template.md").write_text(
        "# Manual accuracy review (1-5 per cell)\n\nOpen benchmark_results.json and score each model's output per site.\n\n"
        "| Site | Model | Brand/services match site | Competitors real & relevant | Queries realistic | SEO usable | Audience specific | Total |\n|---|---|---|---|---|---|---|---|\n"
        + "".join(f"| {u} | {r['model']} | | | | | | |\n" for u in args.urls for r in rows))
    print("\nWrote benchmark_results.json and docs/accuracy_review_template.md")


if __name__ == "__main__":
    asyncio.run(main())
