"""URL validation and SSRF protection.

Defence in depth:
1. Only http/https, no credentials in the URL, limited ports.
2. Hostname is resolved by us and EVERY resolved address must be a public IP.
3. The HTTP client later connects to the *validated IP* (see website_extractor),
   so a DNS-rebinding attack cannot swap in a private address between check and use.
4. Every redirect hop is re-validated.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443, 8080, 8443}
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".lan", ".home.arpa")
BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}
MAX_URL_LENGTH = 2048


class UrlValidationError(ValueError):
    """Raised for malformed URLs or URLs that point at disallowed destinations."""

    def __init__(self, message: str, code: str = "INVALID_URL"):
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_url(raw: str) -> str:
    """Return a clean absolute http(s) URL or raise UrlValidationError."""
    if not isinstance(raw, str) or not raw.strip():
        raise UrlValidationError("URL is required.")
    raw = raw.strip()
    if len(raw) > MAX_URL_LENGTH:
        raise UrlValidationError("URL is too long.")
    if any(c.isspace() for c in raw) or any(ord(c) < 32 for c in raw):
        raise UrlValidationError("URL must not contain whitespace or control characters.")
    if "://" not in raw:
        raw = "https://" + raw  # allow "example.com"
    try:
        parts = urlsplit(raw)
        port = parts.port  # may raise ValueError for bad ports
    except ValueError:
        raise UrlValidationError("URL is malformed.") from None
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UrlValidationError("Only http and https URLs are supported.")
    if not parts.hostname:
        raise UrlValidationError("URL has no hostname.")
    if parts.username or parts.password:
        raise UrlValidationError("URLs containing credentials are not allowed.")
    host = parts.hostname.lower().rstrip(".")
    if host in BLOCKED_HOSTS or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise UrlValidationError("This hostname is not allowed.", "URL_NOT_ALLOWED")
    if port is not None and port not in ALLOWED_PORTS:
        raise UrlValidationError(f"Port {port} is not allowed.", "URL_NOT_ALLOWED")
    try:  # IP literals are rejected immediately if they are not public (fail fast, before any I/O)
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None and not is_public_ip(literal):
        raise UrlValidationError("URL points to a private or internal address.", "URL_NOT_ALLOWED")
    # A bare name with no dot and not an IP literal (e.g. "intranet") is never a public website.
    if "." not in host and ":" not in host:
        raise UrlValidationError("URL must contain a valid domain name.")
    netloc = parts.netloc.rsplit("@", 1)[-1].lower()
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_public_ip(ip.ipv4_mapped)
        if ip.sixtofour is not None and not is_public_ip(ip.sixtofour):
            return False
        if ip.teredo is not None:
            return False
    return ip.is_global and not ip.is_multicast


async def resolve_public_ips(host: str, port: int) -> list[str]:
    """Resolve `host` and return its addresses; raise if ANY of them is non-public."""
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
        infos_ips = [str(literal)]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            raise UrlValidationError("The domain could not be resolved.", "DNS_FAILURE") from None
        infos_ips = list(dict.fromkeys(i[4][0] for i in infos))
    if not infos_ips:
        raise UrlValidationError("The domain could not be resolved.", "DNS_FAILURE")
    for addr in infos_ips:
        if not is_public_ip(ipaddress.ip_address(addr.split("%")[0])):
            raise UrlValidationError("URL resolves to a private or internal address.", "URL_NOT_ALLOWED")
    return infos_ips
