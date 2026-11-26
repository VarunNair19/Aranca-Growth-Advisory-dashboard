import ipaddress
import re
import socket
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import build_opener, HTTPRedirectHandler, Request
from urllib.robotparser import RobotFileParser

USER_AGENT = "ArancaCompetitiveIntelBot/1.0 (+https://www.aranca.com)"
_ROBOTS_CACHE: dict[str, RobotFileParser] = {}


def canonicalize_url(raw_url: str) -> str:
    """Strips UTM tracking parameters and normalizes URLs for exact deduplication."""
    parsed = urlparse(raw_url.strip())
    clean_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref"))
    ]
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            "",
            urlencode(clean_query),
            "",
        )
    )


def is_safe_public_url(raw_url: str) -> bool:
    """SSRF Guard: Ensures URL uses http/https and resolves strictly to public IP addresses."""
    try:
        parsed = urlparse(raw_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        hostname = parsed.hostname.lower()
        if hostname in {"localhost", "0.0.0.0", "::1"}:
            return False

        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip_str = entry[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_reserved
                or ip_obj.is_multicast
                or ip_obj.is_unspecified
                or getattr(ip_obj, "is_site_local", False)
            ):
                return False
        return True
    except Exception:
        return False


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_safe_public_url(
    raw_url: str,
    max_bytes: int = 2_000_000,
    timeout_seconds: float = 10.0,
) -> tuple[str | None, str | None]:
    """Fetches an HTTP(S) resource while validating the URL and every redirect hop."""
    current_url = raw_url.strip()

    for _ in range(6):
        if not is_safe_public_url(current_url):
            return None, None

        request = Request(
            current_url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            },
        )
        try:
            response = build_opener(_NoRedirectHandler()).open(
                request,
                timeout=timeout_seconds,
            )
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get("Location")
                if not location:
                    return None, None
                current_url = urljoin(current_url, location)
                continue
            return None, None
        except Exception:
            return None, None

        try:
            status = getattr(response, "status", response.getcode())
            if status < 200 or status >= 300:
                return None, None
            content = response.read(max_bytes + 1)
            if len(content) > max_bytes:
                return None, None
            charset = response.headers.get_content_charset() or "utf-8"
            return content.decode(charset, errors="replace"), current_url
        except Exception:
            return None, None
        finally:
            response.close()

    return None, None


def is_allowed_by_robots(url: str) -> bool:
    """Checks robots.txt politely before scraping a company webpage."""
    if not is_safe_public_url(url):
        return False
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _ROBOTS_CACHE:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        robots_text, _ = fetch_safe_public_url(
            f"{base}/robots.txt",
            max_bytes=500_000,
            timeout_seconds=5.0,
        )
        if robots_text is None:
            return True
        rp.parse(robots_text.splitlines())
        _ROBOTS_CACHE[base] = rp
    return _ROBOTS_CACHE[base].can_fetch(USER_AGENT, url)


def wrap_untrusted_text(text: str, max_chars: int = 3500) -> str:
    """Isolates scraped content in XML delimiters to mitigate indirect prompt injection."""
    cleaned = re.sub(r"</?untrusted_source_data>", "", text or "", flags=re.IGNORECASE)
    cleaned = cleaned.strip()[:max_chars]
    return f"<untrusted_source_data>\n{cleaned}\n</untrusted_source_data>"


def sanitize_excel_cell(value: object) -> object:
    """Prevents CSV/Excel formula injection (=, +, -, @) on exported strings."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value
