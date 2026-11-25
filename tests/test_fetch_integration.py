"""Exercises the real HTTP fetch path against a local server (DNS check patched to allow 127.0.0.1)."""
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.config import get_settings
from app.services import website_extractor as we
from app.utils.validators import UrlValidationError

PAGE = "<html><head><title>Local Shop</title></head><body><h1>Handmade candles</h1>" + "<p>We sell candles and soap. </p>" * 20 + \
       '<a href="/about">About</a></body></html>'


class H(BaseHTTPRequestHandler):
    seen_hosts = []

    def do_GET(self):
        H.seen_hosts.append(self.headers.get("Host"))
        if self.path == "/redir-private":
            self.send_response(302); self.send_header("Location", "http://169.254.169.254/latest"); self.end_headers(); return
        if self.path == "/redir-ok":
            self.send_response(301); self.send_header("Location", "/final"); self.end_headers(); return
        if self.path == "/forbidden":
            self.send_response(403); self.end_headers(); return
        if self.path == "/pdf":
            self.send_response(200); self.send_header("Content-Type", "application/pdf"); self.end_headers(); self.wfile.write(b"%PDF"); return
        body = PAGE.encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a): pass


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 8080), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield srv
    srv.shutdown()


@pytest.fixture(autouse=True)
def allow_loopback(monkeypatch):
    async def fake_resolve(host, port):
        return ["127.0.0.1"]
    monkeypatch.setattr(we, "resolve_public_ips", fake_resolve)


def test_fetch_pins_ip_and_sends_original_host(server):
    final, html = asyncio.run(we.safe_get("http://shop.example.com:8080/", get_settings()))
    assert "Handmade candles" in html and final == "http://shop.example.com:8080/"
    assert H.seen_hosts[-1] == "shop.example.com:8080"


def test_follows_safe_redirect(server):
    final, _ = asyncio.run(we.safe_get("http://shop.example.com:8080/redir-ok", get_settings()))
    assert final.endswith("/final")


def test_redirect_to_metadata_ip_blocked(server):
    with pytest.raises(UrlValidationError):
        asyncio.run(we.safe_get("http://shop.example.com:8080/redir-private", get_settings()))


def test_403_and_non_html_are_graceful(server):
    with pytest.raises(we.WebsiteFetchError) as e:
        asyncio.run(we.safe_get("http://shop.example.com:8080/forbidden", get_settings()))
    assert e.value.code == "WEBSITE_BLOCKED"
    with pytest.raises(we.WebsiteFetchError) as e:
        asyncio.run(we.safe_get("http://shop.example.com:8080/pdf", get_settings()))
    assert e.value.code == "UNSUPPORTED_CONTENT"


def test_fetch_site_end_to_end(server):
    site = asyncio.run(we.fetch_site("http://shop.example.com:8080/", get_settings()))
    assert site.home.title == "Local Shop" and "candles" in site.home.text.lower()


def test_unreachable_port_graceful(server, monkeypatch):
    with pytest.raises(we.WebsiteFetchError) as e:
        asyncio.run(we.safe_get("http://shop.example.com:8443/", get_settings()))
    assert e.value.code == "WEBSITE_UNREACHABLE"
