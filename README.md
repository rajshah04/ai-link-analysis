# AI-Powered Website Link Analysis API

FastAPI service that fetches a website, extracts its real content, and uses an LLM to return five structured sections:
**competitor vs brand**, **top 5 search queries (Google / Facebook / Instagram / LinkedIn / X)**, **SEO improvements**,
**top 5 services**, and **target audience**.

* `POST /link-analysis` – the analysis endpoint · `GET /docs` – Swagger UI · `GET /health`
* Provider-agnostic: Anthropic, OpenAI or Gemini via environment variables (developed and tested with **Gemini**)
* SSRF-hardened fetching, strict output validation, graceful JSON errors, caching, optional fallback model

| | URL |
|---|---|
| Live demo | `TODO: https://your-domain.com` |
| API endpoint | `TODO: https://your-domain.com/link-analysis` |
| API docs | `TODO: https://your-domain.com/docs` |

## Quick start

```bash
git clone https://github.com/rajshah04/ai-link-analysis.git && cd ai-link-analysis
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

Edit `.env` before starting. Example for Gemini (get a key at https://aistudio.google.com/apikey):

```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite
FALLBACK_LLM_PROVIDER=
FALLBACK_LLM_MODEL=
GEMINI_API_KEY=your_key_here
```
Leave the two `FALLBACK_*` lines empty unless you have a key for the fallback provider. **Never commit `.env`** (it is git-ignored).
Anthropic and OpenAI work the same way (`LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`, or `openai` + `OPENAI_API_KEY`).
Check `http://localhost:4000/health` – it should show `"llm_key_configured": true`.

Open http://localhost:4000/docs. Run tests (offline, no API key needed): `python -m pytest -q` (47 tests).

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```
On Windows PowerShell use `curl.exe` instead of `curl`.

### Sample requests, responses and evidence
Real responses (produced by `python -m scripts.capture_samples`, model `gemini-3.1-flash-lite`):

* [`docs/sample_response_crest_dxora_com.json`](docs/sample_response_crest_dxora_com.json) – `https://crest.dxora.com`
* [`docs/sample_response_www_easycater_co_in.json`](docs/sample_response_www_easycater_co_in.json) – `https://www.easycater.co.in` (JavaScript-rendered page → thin-content warning)
* [`docs/sample_response_www_youtube_com.json`](docs/sample_response_www_youtube_com.json) – `https://www.youtube.com` (JavaScript-heavy; weak sample)
* [`docs/error_cases.txt`](docs/error_cases.txt) – invalid URL, private IP, cloud-metadata IP, `localhost`, non-existent domain → clean JSON errors
* Screenshots: [`docs/screenshot_docs.png`](docs/screenshot_docs.png), [`docs/screenshot_response.png`](docs/screenshot_response.png), [`docs/screenshot_response_2.png`](docs/screenshot_response_2.png), [`docs/screenshot_server_log.png`](docs/screenshot_server_log.png)

## How it works

1. **Validate URL** (`utils/validators.py`): http/https only, no credentials, ports 80/443/8080/8443, blocks `localhost`/`.internal`/`.local`, and rejects private/loopback/link-local/CGNAT/metadata IPs (IPv4, IPv6, IPv4-mapped), including IP literals, before any network call.
2. **Fetch safely** (`services/website_extractor.py`): resolves DNS itself, requires *every* address to be public, then **connects to the validated IP** (TLS SNI/verification still use the hostname) so DNS rebinding can't swap in an internal host. Every redirect hop is re-validated; 5 redirects max; 2 MB cap; timeouts; content-type check.
3. **Extract**: title, meta, Open Graph, JSON-LD, nav labels, headings, cleaned & de-duplicated body text (scripts/styles/cookie banners/footers removed), plus up to 3 high-value internal pages (about/services/products/pricing…). Total text capped (`MAX_CONTENT_CHARS`, default 20 000 chars).
4. **LLM analysis** (`services/llm_service.py`, `analysis_service.py`): scraped text is passed as clearly delimited *untrusted data* (prompt-injection guard), together with today's date so the model does not use stale years. Anthropic uses a forced tool call with our JSON Schema; OpenAI and Gemini use JSON mode.
5. **Validate**: output is parsed into Pydantic models (exactly 5 queries / ≤5 services, platform names normalised, e.g. "Twitter"→"X"). Invalid → one automatic "repair" retry with the validation errors → then the fallback model, if configured.
6. **Enrich** (`services/competitor_service.py`): competitors are checked and a comparison table is built (below). Real `current_title` / `current_meta_description` from the page are added next to the recommended ones.

### Honesty guardrails (from the brief)
* **Competitors are AI suggestions, not research.** Each is labelled `source`, and when a website is given, its homepage is fetched: `verified=true` only means the site was **reachable and mentions the company name** – it does not prove the company is a direct competitor. JavaScript-rendered homepages may stay `verified=false` even for real companies (observed with Swiggy).
* **No invented search volumes** – none are returned; the response carries `meta.disclaimers` saying so.
* Services carry `source: "website" | "inferred"` (+ `evidence`) so inferred recommendations are never passed off as stated services.
* Each response includes `meta` with the real `latency_ms`, token counts, model used and an estimated cost.

### Error handling
All errors use `{"success": false, "error": {"code", "message"}}` – never a stack trace.

| HTTP | code | Meaning |
|---|---|---|
| 400 | `INVALID_URL`, `URL_NOT_ALLOWED` | malformed URL / private or disallowed target (SSRF) |
| 422 | `INVALID_REQUEST`, `DNS_FAILURE`, `WEBSITE_BLOCKED` (401/403/429 from site), `WEBSITE_NOT_FOUND`, `UNSUPPORTED_CONTENT`, `INSUFFICIENT_CONTENT` (JS-only/empty page) | request or site can't be analysed |
| 429 | `RATE_LIMITED` | per-IP limit (`RATE_LIMIT_PER_MINUTE`) |
| 401 | `UNAUTHORIZED` | only if `API_ACCESS_KEY` is set and header `X-API-Key` is wrong |
| 502/504 | `WEBSITE_UNREACHABLE`, `WEBSITE_TIMEOUT`, `WEBSITE_ERROR`, `LLM_UNAVAILABLE` | upstream failure |
| 503 | `LLM_NOT_CONFIGURED` | no API key on the server |

### Known limitations
* Pages rendered purely by JavaScript return little text (reported as a `meta.warnings` entry or `INSUFFICIENT_CONTENT`); no headless browser is used. Observed on `easycater.co.in`: the analysis relied mostly on the page title and meta description, so its services are generic features rather than a real service catalogue.
* Some sites are slow: in one run `easycater.co.in` exceeded the default 12 s fetch limit and returned a clean `WEBSITE_TIMEOUT` (it succeeded on later runs). Raise `FETCH_TIMEOUT_SECONDS` if needed.
* `robots.txt` is not consulted. Sites with bot protection will return `WEBSITE_BLOCKED`.
* Cache and rate limiter are in-process (single instance). Use Redis for multi-instance deployments.
* The rate limiter keys on the client IP. Behind a proxy (Render, Railway…) run uvicorn with `--proxy-headers --forwarded-allow-ips='*'` (already set in the Dockerfile) so users are not all counted as one IP.
* Competitor names come from the LLM's general knowledge and may be outdated or only adjacent to the business; only the site check above adds evidence.

## Deployment (public demo)
`Dockerfile` included (honours `$PORT`). On Render / Railway / Fly.io: create a Web Service from the repo, set `LLM_PROVIDER`, `LLM_MODEL`, the API key as a **secret env var**, and ideally `API_ACCESS_KEY` + a low `RATE_LIMIT_PER_MINUTE` so a public demo can't drain your credits. If you set `API_ACCESS_KEY`, tell reviewers the key to send as header `X-API-Key` (or leave it unset for open testing with a low rate limit).

---

# LLM Platform & Model Research

> **Method & honesty note.** Prices were collected on **18 Nov 2026**. Anthropic prices come from Anthropic's official pricing page.
> OpenAI and Google prices come from third-party price trackers that cite the official pages, so re-check them before relying on them
> (the Gemini 3.1 Flash-Lite price is from such a tracker). Vendors list newer models that were **not** evaluated.
> **Only Gemini 3.1 Flash-Lite was actually run.** Every other speed and accuracy cell is an expectation, not a result.
> No model was formally benchmarked for factual accuracy; accuracy for the measured model was judged by manually reading outputs (see "Measured results").

## A. Comparison

| Provider / model | Input $/1M | Output $/1M | Context | JSON support | Speed | Accuracy | Est. cost / analysis* |
|---|---|---|---|---|---|---|---|
| **Google Gemini 3.1 Flash-Lite** (`gemini-3.1-flash-lite`) | 0.25 | 1.50 | 1M | `responseMimeType: application/json` | **Measured:** median ≈ 17 s end-to-end, range 9.7–23.7 s (8 runs) | **Manually reviewed** on 2 sites – see Measured results | **$0.0040 (measured avg)** |
| Google Gemini 2.5 Flash-Lite (`gemini-2.5-flash-lite`) | 0.10 | 0.40 | 1M | same | Not measured – returned HTTP 404 for my key (see note) | Not measured | $0.0012 |
| Anthropic Claude Haiku 5.5 (`claude-haiku-5-5`) | 0.10 (≤100K prompt; 0.50 above) | 0.50 (2.50 above) | >100K (tiered pricing above 100K) | Forced tool-use with JSON Schema | Not measured | Not measured | $0.0014 |
| Anthropic Claude Sonnet 5.5 (`claude-sonnet-5-5`) | 2.00 | 10.00 | 1M | Forced tool-use with JSON Schema | Not measured (expected slower than Haiku) | Not measured (expected highest here) | $0.028 |
| OpenAI GPT-5 mini (`gpt-5-mini`) | 0.25 | 2.00 | 400K | JSON mode (Structured Outputs also available) | Not measured | Not measured | $0.0052 |
| OpenAI GPT-5 nano (`gpt-5-nano`) | 0.05 | 0.40 | 400K | JSON mode | Not measured | Not measured | $0.0010 |

\*Estimated rows assume 3,000 input / 2,200 output tokens (typical of my measured Gemini runs; other providers tokenize differently, so real counts will differ). The Gemini 3.1 Flash-Lite figure is the measured average. The Anthropic and OpenAI rows were integrated and unit-tested but **not run against their live APIs**. Batch APIs give ~50% off but are asynchronous, so they don't suit a request/response endpoint.

**Note on Gemini 2.5 Flash-Lite:** on 20 Nov 2026 Google's model-list endpoint showed `gemini-2.5-flash-lite`, but `generateContent` returned HTTP 404 for my key. Listing a model does not guarantee it is usable on a given account; verify with a test call.

## B. Selection
The workload is *read a few thousand tokens of messy web text → emit ~2K tokens of strictly structured JSON*. That favours a small, cheap model with reliable JSON output, because output tokens dominate the cost. At list prices the small models differ by only fractions of a cent per analysis, so **price alone does not decide; output quality and JSON reliability must.**

I recommend **Gemini 3.1 Flash-Lite** as the default because it is the model I actually ran end-to-end: all 8 successful runs returned schema-valid JSON with all five sections, at about $0.004 per analysis, and its 1M-token context easily fits the capped website text.
On paper Claude Haiku 5.5 is cheaper (≈$0.0014) and its forced tool-use gives the strongest structured-output guarantee of the integrations here, so it is the first model I would benchmark against Gemini – but I have not tested it, so I do not claim it is better.
I also do not claim any model is "most accurate": the competitor and SEO sections are where a larger model may justify a much higher price, and that needs a side-by-side test with `scripts/benchmark.py`.

## C. Cost optimisation (implemented unless noted)
* Strip scripts/styles/nav/footer/cookie banners; de-duplicate lines; hard cap on characters (`MAX_CONTENT_CHARS`) – the biggest saving. Measured inputs were only ~1.2–4K tokens.
* Fetch only 3 relevant internal pages, with a per-page budget, instead of crawling.
* One LLM call for all five sections (no per-section calls); one repair retry only on invalid JSON.
* Small model by default; optionally **escalate to a larger model only on failure** (`FALLBACK_LLM_*`), not for every request.
* In-memory cache keyed by normalised/final URL (`CACHE_TTL_SECONDS`) – repeat requests cost $0.
* Competitor verification uses plain HTTP fetches, not extra LLM calls.
* Per-IP rate limit and optional API key protect spend on a public demo.
* *Not implemented / next steps:* provider prompt caching for the static system prompt; Redis cache; Batch API for offline bulk jobs.

## D. Final recommendation
| Item | Value |
|---|---|
| Recommended provider / model | Google **Gemini 3.1 Flash-Lite** (`gemini-3.1-flash-lite`) – the model I measured |
| Input tokens (measured) | 1,166–3,995 per analysis, depending on how much text the site has |
| Output tokens (measured) | 1,865–2,305 per analysis |
| Cost per analysis (measured) | $0.0031–$0.0042, average ≈ **$0.0040** (≈ $4 per 1,000 analyses), from real token counts × list price $0.25 / $1.50 per 1M |
| Response time (measured) | Median ≈ **17 s**, range 9.7–23.7 s over 8 successful runs; includes fetching the site (and up to 3 internal pages) plus the LLM call. Varies considerably between runs |
| Fallback | Another Gemini model (e.g. `gemini-3.5-flash-lite`) or a different provider via `FALLBACK_LLM_PROVIDER/MODEL` – **not tested** |
| Balancing approach | Cheap model first → validate → escalate only on failure; cache; cap input. Track `meta.estimated_cost_usd`, `meta.latency_ms` and the fallback rate; benchmark Haiku 5.5 / GPT-5 mini against it before changing the default |

### Measured results
Measured in November 2026 with model `gemini-3.1-flash-lite`, from the `meta` block of real API responses (local machine, home internet; latency includes fetching the site and the LLM call; `cached=false` on every run). Final run, after the prompt fixes described below:

| Website | latency_ms | input_tokens | output_tokens | estimated_cost_usd | Valid JSON first try? | Notes |
|---|---|---|---|---|---|---|
| crest.dxora.com | 12,968 | 3,103 | 2,305 | 0.004233 | Yes (inferred from single-call token counts) | Brand, category and services match the page; 3 competitors, no warnings; only the homepage was analysed |
| easycater.co.in | 23,746 | 1,211 | 2,110 | 0.003468 | Yes (inferred) | JavaScript-rendered page → thin-content warning; services are mostly generic features |
| youtube.com | 20,140 | 3,995 | 2,155 | 0.004231 | Yes (inferred) | JavaScript-heavy page; not manually reviewed |

All 8 successful runs (3 batches): latency 9,681 / 12,968 / 15,648 / 15,966 / 18,130 / 19,294 / 20,140 / 23,746 ms (median ≈ 17.0 s). One earlier attempt on `easycater.co.in` hit `WEBSITE_TIMEOUT` (the site was slow to respond) and was returned as a clean error. "Valid JSON first try" is **inferred**, not directly measured: no fallback model was configured, and token counts look like single calls (an internal repair retry would roughly double input tokens), so a retry is unlikely but I cannot rule it out. `scripts/benchmark.py` measures this directly.

**Manual accuracy review.** I read the CREST and EasyCater responses in full (first run) and compared them with the evidence quotes and page titles/meta descriptions in the output; I cannot independently see the live sites beyond what the extractor captured. Brand names, categories and summaries were consistent with each site's own title and meta description, and every service was labelled `website` with an evidence quote. Problems found:
1. Only 2 competitors were returned for both sites (the brief asks for ≥3), because my prompt told the model to give fewer if unsure.
2. One search query contained a stale year ("2025") because the model did not know the current date.
3. EasyCater is JavaScript-rendered, so its services were mostly generic features (payments, offers) and one query/page idea ("corporate catering") came from the brand name rather than the site.
4. Competitor "verification" is weak: Semrush/HubSpot were marked verified for CREST (real, but adjacent rather than direct competitors), while Swiggy stayed unverified because its JavaScript homepage does not show its name in text.

Fixes: the prompt now asks for at least 3 competitors and passes today's date. After the fix, the CREST response lists three companies (Semrush, HubSpot, BrightLocal) with no competitor warning, and no sample response contains "2025". Problems 3 and 4 are limitations of fetching without a headless browser and are documented above rather than fixed. I did not repeat the full manual review on the final-run files beyond these checks.

To compare more models: `python -m scripts.benchmark --urls <sites> --models gemini:gemini-3.1-flash-lite gemini:gemini-3.5-flash-lite --runs 2` (add other providers if you have keys). It reports first-try valid-JSON rate, median latency, tokens, cost and a *structural* completeness score, and writes `docs/accuracy_review_template.md` for a manual 1–5 rubric.

---

## Project layout
```
app/main.py                        FastAPI app, error handlers, rate limit/auth guard
app/config.py                      env-based settings + model price table
app/schemas.py                     request/response + LLM output models
app/services/website_extractor.py  SSRF-safe fetch + content extraction
app/services/llm_service.py        Anthropic / OpenAI / Gemini client, retries
app/services/analysis_service.py   prompt, validation/repair, fallback, cache
app/services/competitor_service.py competitor verification + comparison table
app/utils/validators.py            URL validation / SSRF rules
scripts/benchmark.py               latency / cost / structure benchmark
scripts/capture_samples.py         saves real sample responses + error cases to docs/
tests/                             47 offline tests (incl. local-server fetch tests)
Dockerfile                         container deployment
```