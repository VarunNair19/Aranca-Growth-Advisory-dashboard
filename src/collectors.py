from datetime import datetime, timedelta, timezone
from io import BytesIO
import re
import time
from typing import Any, Optional
from urllib.parse import quote_plus, urlparse

from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from ddgs import DDGS
import feedparser
from pypdf import PdfReader
from trafilatura import extract

from src.config import get_logger, settings
from src.schemas import RawArticle, RegionType
from src.security import fetch_safe_public_url, is_allowed_by_robots, is_safe_public_url

logger = get_logger(__name__)

_REGION_LOCALE: dict[str, tuple[str, str, str]] = {
    "US": ("en-US", "US", "US:en"),
    "KSA": ("en-SA", "SA", "SA:en"),
    "UAE": ("en-AE", "AE", "AE:en"),
    "India": ("en-IN", "IN", "IN:en"),
    "Japan": ("en-JP", "JP", "JP:en"),
}

_DDGS_REGION: dict[str, str] = {
    "US": "us-en",
    "KSA": "sa-en",
    "UAE": "ae-en",
    "India": "in-en",
    "Japan": "jp-en",
}


def _strip_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _parse_iso_date(raw_date: Optional[str], fallback_today: bool = True) -> str:
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not raw_date:
        return today_str if fallback_today else ""
    try:
        dt = date_parser.parse(str(raw_date).strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        parsed_str = dt.strftime("%Y-%m-%d")
        if parsed_str > today_str:
            return today_str
        return parsed_str
    except Exception:
        return today_str if fallback_today else ""


def _build_google_rss_url(query: str, region: str) -> str:
    hl, gl, ceid = _REGION_LOCALE.get(region, ("en-US", "US", "US:en"))
    encoded_q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded_q}&hl={hl}&gl={gl}&ceid={ceid}"


def fetch_google_news_rss(
    competitor: dict[str, Any],
    since_date: Optional[str] = None,
    lookback_days: int = settings.baseline_lookback_days,
    max_items: int = 8,
) -> list[RawArticle]:
    """Fetches competitor news items from Google News RSS filtered by date."""
    company = competitor["name"]
    region: RegionType = competitor["region"]  # type: ignore[assignment]
    query = competitor.get("rss_query") or f'"{company}" consulting OR advisory OR strategy'

    cutoff_date = (
        since_date
        if since_date
        else (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    )
    rss_url = _build_google_rss_url(query, region)
    articles: list[RawArticle] = []

    try:
        feed = feedparser.parse(rss_url)
        for entry in getattr(feed, "entries", [])[: max_items * 2]:
            link = (getattr(entry, "link", "") or "").strip()
            title = _strip_html(getattr(entry, "title", "") or "")
            summary = _strip_html(getattr(entry, "summary", "") or "")
            raw_published = (
                getattr(entry, "published", None)
                or getattr(entry, "updated", None)
                or None
            )
            pub_date = _parse_iso_date(raw_published, fallback_today=True)

            if not link or not title:
                continue
            if pub_date < cutoff_date:
                continue

            articles.append(
                RawArticle(
                    company=company,
                    region=region,
                    title=title,
                    snippet=summary[:1200] if summary else title,
                    source_url=link,
                    published_date=pub_date,
                    source_type="rss",
                )
            )
            if len(articles) >= max_items:
                break

        logger.info(
            "Google RSS fetched %d articles for company=%s region=%s (cutoff=%s)",
            len(articles),
            company,
            region,
            cutoff_date,
        )
    except Exception:
        logger.exception("Google News RSS failed for company=%s region=%s", company, region)

    return articles


def fetch_ddgs_news(
    competitor: dict[str, Any],
    since_date: Optional[str] = None,
    lookback_days: int = settings.baseline_lookback_days,
    max_items: int = 5,
) -> list[RawArticle]:
    """Fetches supplemental competitor news via DuckDuckGo (ddgs)."""
    company = competitor["name"]
    region: RegionType = competitor["region"]  # type: ignore[assignment]
    ddgs_reg = _DDGS_REGION.get(region, "wt-wt")
    cutoff_date = (
        since_date
        if since_date
        else (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    )

    timelimit = "m" if lookback_days <= 35 else None
    query = f'"{company}" (consulting OR advisory OR M&A OR strategy OR partner)'
    articles: list[RawArticle] = []

    try:
        raw_results = DDGS().news(
            query=query,
            region=ddgs_reg,
            safesearch="moderate",
            timelimit=timelimit,
            max_results=max_items * 2,
        )
        for item in raw_results or []:
            link = (item.get("url") or item.get("href") or "").strip()
            title = _strip_html(item.get("title") or "")
            body = _strip_html(item.get("body") or item.get("excerpt") or "")
            pub_date = _parse_iso_date(item.get("date"), fallback_today=True)

            if not link or not title:
                continue
            if pub_date < cutoff_date:
                continue

            articles.append(
                RawArticle(
                    company=company,
                    region=region,
                    title=title,
                    snippet=body[:1200] if body else title,
                    source_url=link,
                    published_date=pub_date,
                    source_type="search",
                )
            )
            if len(articles) >= max_items:
                break

        logger.info(
            "DDGS news fetched %d articles for company=%s region=%s (cutoff=%s)",
            len(articles),
            company,
            region,
            cutoff_date,
        )
    except Exception:
        logger.exception("DDGS news search failed for company=%s region=%s", company, region)

    return articles


def fetch_safe_webpage_text(url: str, max_chars: int = 4000) -> str:
    """Safely fetches and extracts main text from a public URL after SSRF & robots checks."""
    if not is_safe_public_url(url):
        logger.warning("Blocked unsafe or non-public URL in collector: %s", url)
        return ""
    if not is_allowed_by_robots(url):
        logger.info("Skipped URL disallowed by robots.txt: %s", url)
        return ""

    try:
        if settings.scrape_delay_sec > 0:
            time.sleep(settings.scrape_delay_sec)
        downloaded, final_url = fetch_safe_public_url(url)
        if not downloaded:
            return ""
        if final_url and not is_safe_public_url(final_url):
            return ""
        text = extract(downloaded, include_comments=False, include_tables=True) or ""
        return re.sub(r"\s+", " ", text).strip()[:max_chars]
    except Exception:
        logger.exception("Webpage extraction failed for url=%s", url)
        return ""


def fetch_aranca_growth_page(url: str = "https://www.aranca.com/growth-advisory.php") -> Optional[str]:
    """Scrapes Aranca's live Growth Advisory page text safely."""
    extracted = fetch_safe_webpage_text(url, max_chars=6000)
    if extracted:
        logger.info("Extracted %d characters from live Aranca Growth Advisory page", len(extracted))
        return extracted
    logger.warning("Live Aranca page extraction returned empty; caller will use profile JSON fallback")
    return None


def extract_uploaded_document_text(file_name: str, file_bytes: bytes, max_chars: int = 8000) -> str:
    """Validates and extracts text from uploaded PDF, TXT, or MD documents."""
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > settings.max_upload_bytes:
        raise ValueError(f"File exceeds maximum allowed size ({settings.max_upload_bytes // (1024 * 1024)} MB).")

    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF header.")
        reader = PdfReader(BytesIO(file_bytes))
        pages_text = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(pages_text)
    elif lower_name.endswith((".txt", ".md")):
        text = file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported file type. Please upload a .pdf, .txt, or .md file.")

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("No readable text could be extracted from the uploaded file.")

    logger.info("Extracted %d chars from uploaded file=%s", min(len(cleaned), max_chars), file_name)
    return cleaned[:max_chars]


def _is_google_news_wrapper(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return "news.google.com" in host


def collect_competitor_articles(
    competitors: list[dict[str, Any]],
    since_date: Optional[str] = None,
    lookback_days: int = settings.baseline_lookback_days,
    max_articles: int = settings.max_articles_per_run,
) -> list[RawArticle]:
    """Collects candidate articles across active competitors using RSS + DDGS and enriches direct URLs."""
    collected: list[RawArticle] = []
    seen_links: set[str] = set()

    for comp in competitors:
        if len(collected) >= max_articles:
            logger.warning("Reached max_articles_per_run cap (%d); stopping collection.", max_articles)
            break

        rss_items = fetch_google_news_rss(
            competitor=comp,
            since_date=since_date,
            lookback_days=lookback_days,
            max_items=6,
        )
        ddgs_items = fetch_ddgs_news(
            competitor=comp,
            since_date=since_date,
            lookback_days=lookback_days,
            max_items=4,
        )

        comp_items: list[RawArticle] = []
        for item in rss_items + ddgs_items:
            if item.source_url in seen_links:
                continue
            seen_links.add(item.source_url)
            comp_items.append(item)
            if len(collected) + len(comp_items) >= max_articles:
                break

        enriched_count = 0
        for idx, item in enumerate(comp_items):
            if enriched_count >= 2 or _is_google_news_wrapper(item.source_url):
                continue
            full_text = fetch_safe_webpage_text(item.source_url, max_chars=2500)
            if full_text and len(full_text) > len(item.snippet) + 80:
                comp_items[idx] = item.model_copy(update={"snippet": full_text})
                enriched_count += 1

        collected.extend(comp_items)

    logger.info("Total raw candidate articles collected this run: %d", len(collected))
    return collected