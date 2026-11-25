"""Offline tests: no network and no real LLM calls (fetching and the LLM are stubbed)."""
import asyncio
import ipaddress

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import get_settings
from app.schemas import LLMAnalysis
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMError, LLMResult, _parse_json
from app.services.website_extractor import PageContent, SiteContent, WebsiteFetchError, extract_page
from app.utils.validators import UrlValidationError, is_public_ip, normalize_url, resolve_public_ips

HTML = """<html lang="en"><head><title>Acme Web Studio | Websites for Small Business</title>
<meta name="description" content="We build affordable websites."><meta property="og:site_name" content="Acme Web Studio">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Web Studio"}</script></head>
<body><nav><a href="/services">Services</a><a href="/about">About us</a></nav>
<div class="cookie-banner">We use cookies</div>
<h1>Websites that grow your business</h1><p>Custom web design and e-commerce development.</p>
<p>Custom web design and e-commerce development.</p><script>var x=1</script>
<h2>SEO Services</h2><p>Rank higher on Google with our SEO packages.</p><footer>Copyright</footer></body></html>"""


def good_llm_json():
    q = lambda i, p: {"query": f"query {i}", "platforms": p, "intent": "Service Discovery",
                      "business_relevance": "r", "target_audience": "SMBs"}
    return {
        "brand": {"name": "Acme Web Studio", "category": "Web Development", "summary": "Builds websites."},
        "competitor_analysis": {
            "brand_overview": "o", "main_category": "Web Development", "main_products_services": ["Web design"],
            "brand_positioning": {"main_service": "Web design", "target_market": "SMB", "key_strength": "Price", "opportunity": "Visibility"},
            "competitors": [{"name": n, "website": None, "main_service": "Web", "target_market": "Enterprise",
                             "strengths": ["Brand"], "weaknesses_or_gaps": ["Price"], "confidence": "medium"}
                            for n in ("Beta Digital", "Gamma Web", "Delta Studio")],
            "strengths": ["Affordable"], "opportunities": ["Local SEO"]},
        "target_search_queries": [q(1, ["Google"]), q(2, ["Facebook"]), q(3, ["Instagram"]),
                                  q(4, ["LinkedIn"]), q(5, ["Twitter"])],  # alias must normalise to X
        "seo_improvements": {"primary_keyword": "web design", "recommended_title": "Affordable Web Design | Acme",
                             "meta_description": "We build affordable websites for small business.",
                             "supporting_content": [{"type": "faq", "title": "t", "rationale": "r"}]},
        "top_services": [{"name": f"S{i}", "description": "d", "target_industry": "i", "target_customer": "c",
                          "main_benefit": "b", "source": "website"} for i in range(6)],  # 6 -> truncated to 5
        "target_audience": {"primary_audience": ["SMB owners"]},
    }


class FakeClient:
    provider, model = "anthropic", "claude-haiku-5-5"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def complete_json(self, system, user, schema=None):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResult(r, 6000, 2500, self.provider, self.model, 0.1)


async def fake_fetch(url, settings):
    home = extract_page(HTML, url)
    return SiteContent(url=url, home=home, extra_pages=[])


def make_service(replies, fallback=None):
    s = get_settings()
    object.__setattr__(s, "verify_competitors", False)
    return AnalysisService(s, primary=FakeClient(replies), fallback=fallback, fetcher=fake_fetch)


# ----------------------------------------------------------------- validators
@pytest.mark.parametrize("bad", ["", "   ", "ftp://example.com", "javascript:alert(1)", "http://user:pw@example.com",
                                 "http://localhost", "http://intranet", "http://x.internal", "http://example.com:22",
                                 "https://exa mple.com", "file:///etc/passwd"])
def test_rejects_bad_urls(bad):
    with pytest.raises(UrlValidationError):
        normalize_url(bad)


def test_normalises_bare_domain():
    assert normalize_url("Example.com") == "https://example.com/"


@pytest.mark.parametrize("ip,public", [("127.0.0.1", False), ("10.1.2.3", False), ("192.168.0.1", False),
                                       ("172.16.5.5", False), ("169.254.169.254", False), ("100.64.0.1", False),
                                       ("::1", False), ("fe80::1", False), ("fd00::1", False),
                                       ("::ffff:127.0.0.1", False), ("0.0.0.0", False),
                                       ("8.8.8.8", True), ("93.184.216.34", True)])
def test_ip_classification(ip, public):
    assert is_public_ip(ipaddress.ip_address(ip)) is public


@pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254", "10.0.0.5", "[::1]"])
def test_private_literals_blocked(host):
    with pytest.raises(UrlValidationError):
        asyncio.run(resolve_public_ips(host, 80))


def test_public_literal_allowed():
    assert asyncio.run(resolve_public_ips("8.8.8.8", 443)) == ["8.8.8.8"]


# ------------------------------------------------------------------ extractor
def test_extract_page_cleans_and_dedupes():
    p = extract_page(HTML, "https://acme.test/")
    assert p.title.startswith("Acme Web Studio")
    assert p.meta_description == "We build affordable websites."
    assert p.text.count("Custom web design and e-commerce development.") == 1
    assert "cookies" not in p.text and "var x" not in p.text and "Copyright" not in p.text
    assert "Services" in p.nav_labels and "SEO Services" in p.headings
    assert any("/services" in u for u in p.candidate_links)
    assert p.json_ld and p.json_ld[0]["name"] == "Acme Web Studio"


def test_parse_json_variants():
    assert _parse_json('```json\n{"a":1}\n```') == {"a": 1}
    assert _parse_json('Here you go: {"a": 2} thanks') == {"a": 2}
    with pytest.raises(LLMError):
        _parse_json('{"a": ')


# --------------------------------------------------------------- schema logic
def test_schema_normalises_and_truncates():
    m = LLMAnalysis.model_validate(good_llm_json())
    assert [p for q in m.target_search_queries for p in q.platforms][-1] == "X"
    assert len(m.top_services) == 5


# -------------------------------------------------------------------- service
def test_service_happy_path_and_cache():
    svc = make_service([good_llm_json()])
    r = asyncio.run(svc.analyze("acme.test"))
    assert r.success and r.brand.name == "Acme Web Studio"
    assert r.analysis.seo_improvements.current_title.startswith("Acme")
    assert r.analysis.competitor_analysis.competitors[0].verified is False
    assert len(r.analysis.competitor_analysis.comparison_table) == 15  # 3 competitors x 5 rows
    assert r.meta.input_tokens == 6000 and r.meta.estimated_cost_usd == pytest.approx(0.00185, rel=1e-3)
    r2 = asyncio.run(svc.analyze("https://acme.test/"))
    assert r2.meta.cached and svc.primary.calls == 1


def test_repair_retry_on_invalid_json():
    bad = good_llm_json()
    del bad["top_services"]
    svc = make_service([bad, good_llm_json()])
    r = asyncio.run(svc.analyze("https://acme.test"))
    assert svc.primary.calls == 2 and r.meta.input_tokens == 12000


def test_fallback_model_used_when_primary_fails():
    fb = FakeClient([good_llm_json()])
    fb.model = "claude-sonnet-5-5"
    svc = make_service([LLMError("down")], fallback=fb)
    r = asyncio.run(svc.analyze("https://acme.test"))
    assert r.meta.used_fallback_model and r.meta.model == "claude-sonnet-5-5"


# ------------------------------------------------------------------------ API
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.delenv("API_ACCESS_KEY", raising=False)
    main.app.dependency_overrides[main.get_service] = lambda: make_service([good_llm_json()])
    yield TestClient(main.app, raise_server_exceptions=False)
    main.app.dependency_overrides.clear()


def test_endpoint_success(client):
    r = client.post("/link-analysis", json={"url": "https://acme.test"})
    assert r.status_code == 200
    j = r.json()
    assert j["success"] is True and set(j["analysis"]) == {
        "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}
    assert len(j["analysis"]["target_search_queries"]) == 5


def test_endpoint_invalid_and_blocked_urls(client):
    for url, code in [("not a url", "INVALID_URL"), ("http://127.0.0.1/admin", "URL_NOT_ALLOWED"),
                      ("http://169.254.169.254/latest/meta-data", "URL_NOT_ALLOWED")]:
        r = client.post("/link-analysis", json={"url": url})
        assert r.status_code == 400 and r.json() == {"success": False, "error": {"code": code, "message": r.json()["error"]["message"]}}


def test_endpoint_bad_body(client):
    r = client.post("/link-analysis", json={"nope": 1})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_REQUEST"


def test_endpoint_website_errors_are_graceful(client):
    async def boom(url, settings):
        raise WebsiteFetchError("WEBSITE_BLOCKED", "blocked", 422)
    main.app.dependency_overrides[main.get_service] = lambda: AnalysisService(
        get_settings(), primary=FakeClient([]), fetcher=boom)
    r = client.post("/link-analysis", json={"url": "https://acme.test"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "WEBSITE_BLOCKED"


def test_endpoint_api_key_and_rate_limit(monkeypatch):
    main.app.dependency_overrides[main.get_service] = lambda: make_service([good_llm_json()] * 5)
    c = TestClient(main.app, raise_server_exceptions=False)
    monkeypatch.setenv("API_ACCESS_KEY", "secret")
    assert c.post("/link-analysis", json={"url": "https://acme.test"}).status_code == 401
    assert c.post("/link-analysis", json={"url": "https://acme.test"}, headers={"X-API-Key": "secret"}).status_code == 200
    monkeypatch.delenv("API_ACCESS_KEY")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    main._hits.clear()
    codes = [c.post("/link-analysis", json={"url": "https://acme.test"}).status_code for _ in range(2)]
    assert codes == [200, 429]
    main.app.dependency_overrides.clear()
