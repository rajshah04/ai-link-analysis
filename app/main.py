"""FastAPI entrypoint.  Run:  uvicorn app.main:app --host 0.0.0.0 --port 4000"""
from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict, deque
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import get_settings
from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisError, AnalysisService
from app.services.website_extractor import WebsiteFetchError
from app.utils.validators import UrlValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(
    title="AI-Powered Website Link Analysis API",
    version="1.0.0",
    description="Analyses a website URL with an LLM and returns competitor, search-query, SEO, services and audience insights.",
)


@lru_cache
def get_service() -> AnalysisService:
    return AnalysisService(get_settings())


_hits: dict[str, deque] = defaultdict(deque)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "error": {"code": code, "message": message}})


async def guard(request: Request, x_api_key: str | None = Header(default=None, include_in_schema=False)) -> None:
    """Optional shared-secret auth + simple per-IP rate limit (protects LLM spend on a public demo)."""
    s = get_settings()
    if s.api_access_key and not (x_api_key and hmac.compare_digest(x_api_key, s.api_access_key)):
        raise AnalysisError("UNAUTHORIZED", "Missing or invalid X-API-Key header.", 401)
    if s.rate_limit_per_minute > 0:
        ip = request.client.host if request.client else "unknown"
        now, q = time.time(), _hits[ip]
        while q and q[0] < now - 60:
            q.popleft()
        if len(q) >= s.rate_limit_per_minute:
            raise AnalysisError("RATE_LIMITED", "Too many requests. Please wait a minute and retry.", 429)
        q.append(now)


@app.exception_handler(UrlValidationError)
async def _url_error(_: Request, exc: UrlValidationError):
    return _error(422 if exc.code == "DNS_FAILURE" else 400, exc.code, exc.message)


@app.exception_handler(WebsiteFetchError)
async def _fetch_error(_: Request, exc: WebsiteFetchError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(AnalysisError)
async def _analysis_error(_: Request, exc: AnalysisError):
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    return _error(422, "INVALID_REQUEST", 'Request body must be JSON like {"url": "https://example.com"}.')


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    log.exception("Unhandled error")  # stack trace stays in server logs only
    return _error(500, "INTERNAL_ERROR", "An unexpected error occurred.")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
async def health():
    s = get_settings()
    return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model, "llm_key_configured": bool(s.api_key_for(s.llm_provider))}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    tags=["analysis"],
    summary="Analyse a website URL",
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               429: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(guard)],
)
async def link_analysis(body: LinkAnalysisRequest, service: AnalysisService = Depends(get_service)):
    return await service.analyze(body.url)
