"""Safe website fetching (SSRF-hardened) and content extraction."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.utils.validators import UrlValidationError, normalize_url, resolve_public_ips

USER_AGENT = "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://github.com/your-user/ai-link-analysis)"
MAX_REDIRECTS = 5
PAGE_HINTS = ("about", "service", "solution", "product", "offer", "what-we-do", "pricing", "industr",
              "company", "capabilit", "work", "portfolio", "case-stud", "feature")
NOISE_ATTR = re.compile(r"cookie|consent|gdpr|banner|popup|modal|newsletter|subscribe", re.I)


class WebsiteFetchError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


@dataclass
class PageContent:
    url: str
    title: str = ""
    meta_description: str = ""
    og: dict = field(default_factory=dict)
    lang: str = ""
    headings: list[str] = field(default_factory=list)
    nav_labels: list[str] = field(default_factory=list)
    json_ld: list[dict] = field(default_factory=list)
    text: str = ""
    candidate_links: list[str] = field(default_factory=list)


@dataclass
class SiteContent:
    url: str
    home: PageContent
    extra_pages: list[PageContent]
    warnings: list[str] = field(default_factory=list)

    @property
    def pages_analyzed(self) -> list[str]:
        return [self.home.url] + [p.url for p in self.extra_pages]


# --------------------------------------------------------------- fetching
async def safe_get(url: str, settings: Settings, max_bytes: int | None = None) -> tuple[str, str]:
    """GET `url` and return (final_url, html). Raises WebsiteFetchError / UrlValidationError.

    Connects to the validated IP address (not the hostname) so DNS rebinding cannot
    redirect the request to an internal host. TLS SNI/verification still use the hostname.
    """
    max_bytes = max_bytes or settings.max_download_bytes
    current = normalize_url(url)
    timeout = httpx.Timeout(settings.fetch_timeout, connect=min(6, settings.fetch_timeout))
    headers_base = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                    "Accept-Language": "en;q=0.9,*;q=0.5"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(current)
            host = parts.hostname or ""
            default_port = 443 if parts.scheme == "https" else 80
            port = parts.port or default_port
            ips = await resolve_public_ips(host, port)  # raises if any address is private
            host_header = host if port == default_port else f"{host}:{port}"
            last_exc: Exception | None = None
            response_info = None
            for ip in ips[:3]:
                ip_host = f"[{ip}]" if ":" in ip else ip
                pinned = urlunsplit((parts.scheme, f"{ip_host}:{port}", parts.path or "/", parts.query, ""))
                ext = {"sni_hostname": host} if parts.scheme == "https" else {}
                try:
                    async with client.stream("GET", pinned, headers={**headers_base, "Host": host_header},
                                             extensions=ext) as resp:
                        if resp.status_code in (301, 302, 303, 307, 308):
                            loc = resp.headers.get("location")
                            if not loc:
                                raise WebsiteFetchError("BAD_REDIRECT", "Website sent an invalid redirect.")
                            response_info = ("redirect", urljoin(current, loc))
                        else:
                            response_info = ("final", await _read_response(resp, max_bytes))
                    break
                except httpx.TimeoutException:
                    raise WebsiteFetchError("WEBSITE_TIMEOUT", "The website took too long to respond.", 504) from None
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                    last_exc = exc
                    continue
                except httpx.HTTPError as exc:
                    raise WebsiteFetchError("WEBSITE_UNREACHABLE", f"Could not fetch the website ({type(exc).__name__}).") from None
            if response_info is None:
                raise WebsiteFetchError("WEBSITE_UNREACHABLE",
                                        f"Could not connect to the website ({type(last_exc).__name__}).")
            kind, payload = response_info
            if kind == "redirect":
                current = normalize_url(payload)  # re-validates scheme/host/port of the new hop
                continue
            return current, payload
    raise WebsiteFetchError("TOO_MANY_REDIRECTS", "The website redirected too many times.")


async def _read_response(resp: httpx.Response, max_bytes: int) -> str:
    status = resp.status_code
    if status in (401, 403, 429) or status == 999:
        raise WebsiteFetchError("WEBSITE_BLOCKED",
                                f"The website refused automated access (HTTP {status}); it may use bot protection or require login.", 422)
    if status == 404 or status == 410:
        raise WebsiteFetchError("WEBSITE_NOT_FOUND", "The page was not found (HTTP 404).", 422)
    if status >= 400:
        raise WebsiteFetchError("WEBSITE_ERROR", f"The website returned HTTP {status}.", 502)
    ctype = resp.headers.get("content-type", "").lower()
    if ctype and not any(t in ctype for t in ("text/html", "application/xhtml", "text/plain", "xml")):
        raise WebsiteFetchError("UNSUPPORTED_CONTENT", f"URL does not return an HTML page (content-type: {ctype.split(';')[0]}).", 422)
    chunks, size = [], 0
    async for chunk in resp.aiter_bytes():
        size += len(chunk)
        chunks.append(chunk)
        if size >= max_bytes:  # truncate, do not fail: the head of the page is what matters
            break
    body = b"".join(chunks)[:max_bytes]
    enc = resp.charset_encoding or "utf-8"
    try:
        return body.decode(enc, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


# ------------------------------------------------------------- extraction
def _clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def extract_page(html: str, url: str) -> PageContent:
    soup = BeautifulSoup(html, "html.parser")
    page = PageContent(url=url)
    page.title = _clean(soup.title.string if soup.title else "")
    page.lang = (soup.html.get("lang", "") if soup.html else "")[:10]

    def meta(**attrs) -> str:
        tag = soup.find("meta", attrs=attrs)
        return _clean(tag.get("content")) if tag else ""

    page.meta_description = meta(name="description")
    for key in ("og:title", "og:description", "og:site_name", "og:type"):
        val = meta(property=key)
        if val:
            page.og[key] = val

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and item.get("@type"):
                keep = {k: item[k] for k in ("@type", "name", "description", "url", "areaServed", "address", "sameAs", "knowsAbout")
                        if k in item and isinstance(item[k], (str, list, dict))}
                serialized = json.dumps(keep)
                page.json_ld.append(keep if len(serialized) <= 1200 else {"@type": item.get("@type"), "name": item.get("name")})

    # internal links worth reading + nav labels (service names often live in navigation)
    base_host = (urlsplit(url).hostname or "").removeprefix("www.")
    scored: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        label = _clean(a.get_text(" "))
        href = urljoin(url, a["href"]).split("#")[0]
        sp = urlsplit(href)
        if sp.scheme not in ("http", "https") or (sp.hostname or "").removeprefix("www.") != base_host:
            continue
        hay = (sp.path + " " + label).lower()
        score = sum(1 for h in PAGE_HINTS if h in hay)
        if score and sp.path.strip("/") and href.rstrip("/") != url.rstrip("/"):
            scored[href] = max(scored.get(href, 0), score)
    page.candidate_links = [u for u, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:8]

    for nav in soup.find_all("nav")[:2]:
        for a in nav.find_all("a"):
            t = _clean(a.get_text(" "))
            if 2 < len(t) < 40 and t not in page.nav_labels:
                page.nav_labels.append(t)
    page.nav_labels = page.nav_labels[:30]

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "form", "button", "input", "select"]):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": NOISE_ATTR}) + soup.find_all(attrs={"id": NOISE_ATTR}):
        tag.decompose()
    for tag in soup(["nav", "footer"]):  # already captured nav labels; footer is mostly boilerplate
        tag.decompose()

    for h in soup.find_all(["h1", "h2", "h3"]):
        t = _clean(h.get_text(" "))
        if 2 < len(t) < 160 and t not in page.headings:
            page.headings.append(t)
    page.headings = page.headings[:40]

    seen, lines = set(), []
    for raw in soup.get_text("\n").split("\n"):
        line = _clean(raw)
        if len(line) < 3 or line.lower() in seen:
            continue
        seen.add(line.lower())
        lines.append(line)
    page.text = "\n".join(lines)
    return page


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind("\n")] if "\n" in cut[limit // 2:] else cut


async def fetch_site(url: str, settings: Settings) -> SiteContent:
    """Fetch the homepage and a few high-value internal pages; return cleaned content."""
    import asyncio

    final_url, html = await safe_get(url, settings)
    home = extract_page(html, final_url)
    warnings: list[str] = []

    if len(home.text) < 150 and not home.meta_description and not home.title:
        raise WebsiteFetchError(
            "INSUFFICIENT_CONTENT",
            "The page has almost no readable text (it may be rendered by JavaScript, empty, or blocked).", 422)
    if len(home.text) < 400:
        warnings.append("Very little text was found on the page (possibly JavaScript-rendered); analysis may be shallow.")

    async def _extra(u: str) -> PageContent | None:
        try:
            fu, h = await safe_get(u, settings, max_bytes=800_000)
            return extract_page(h, fu)
        except (WebsiteFetchError, UrlValidationError):
            return None

    extras: list[PageContent] = []
    if settings.extra_pages > 0 and home.candidate_links:
        results = await asyncio.gather(*[_extra(u) for u in home.candidate_links[: settings.extra_pages]])
        extras = [p for p in results if p and len(p.text) > 150]

    # Budget the characters: homepage gets the lion's share
    budget = settings.max_content_chars
    home.text = _truncate(home.text, int(budget * 0.55) if extras else budget)
    if extras:
        per_page = int(budget * 0.45) // len(extras)
        for p in extras:
            p.text = _truncate(p.text, per_page)
    return SiteContent(url=final_url, home=home, extra_pages=extras, warnings=warnings)
