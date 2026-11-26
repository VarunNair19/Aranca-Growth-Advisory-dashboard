import argparse
import json
from pathlib import Path
from typing import Any

from src.collectors import fetch_aranca_growth_page
from src.config import get_logger, settings
from src.database import IntelDatabase
from src.embeddings import embed_single
from src.graph import run_competitor_scan
from src.llm_client import LLMClient
from src.schemas import IntelEvent
from src.security import canonicalize_url

logger = get_logger(__name__)

_SAMPLE_BASELINE_EVENTS: list[dict[str, Any]] = [
    {
        "company": "Arthur D. Little",
        "region": "KSA",
        "title": "Arthur D. Little Expands Riyadh Advisory Practice for Industrial & Vision 2030 Transformation",
        "tracking_pillar": "Business Updates",
        "aranca_segment": "Business Transformation & PMO Advisory",
        "published_date": "2026-10-14",
        "source_url": "https://www.adlittle.com/en/media",
        "executive_summary": "Arthur D. Little expanded its Riyadh presence to support industrial localization and transformation programs aligned with Saudi Vision 2030 priorities.",
        "strategic_implication": "Increases competition for industrial transformation and PMO mandates in KSA; Aranca should highlight intelligence-backed feasibility and commercial due diligence.",
        "materiality_score": 4,
        "review_status": "APPROVED",
    },
    {
        "company": "Praxis Global Alliance",
        "region": "India",
        "title": "Praxis Global Alliance Launches Consumer & Digital GTM Growth Practice in India",
        "tracking_pillar": "Deals & Engagements",
        "aranca_segment": "Market Share Expansion & Growth Acceleration",
        "published_date": "2026-10-22",
        "source_url": "https://www.praxisga.com",
        "executive_summary": "Praxis Global Alliance expanded go-to-market and consumer growth advisory capabilities for mid-market and PE-backed enterprises in India.",
        "strategic_implication": "Signals rising demand for mid-market GTM and market-entry work in India where Aranca's primary research and benchmarking offer a cost-effective complement.",
        "materiality_score": 4,
        "review_status": "APPROVED",
    },
    {
        "company": "RedSeer Strategy Consultants",
        "region": "UAE",
        "title": "RedSeer Publishes GCC Digital Commerce & Cross-Border Expansion Study",
        "tracking_pillar": "Digital Presence",
        "aranca_segment": "International Market Entry & Expansion Strategy",
        "published_date": "2026-11-02",
        "source_url": "https://redseer.com",
        "executive_summary": "RedSeer released new market intelligence on GCC digital commerce adoption and cross-border expansion across UAE and neighboring Gulf markets.",
        "strategic_implication": "Demonstrates strong appetite for regional market-entry and VoC studies in the UAE; useful benchmark for Aranca's GCC growth research.",
        "materiality_score": 3,
        "review_status": "APPROVED",
    },
    {
        "company": "IGPI (Industrial Growth Platform)",
        "region": "Japan",
        "title": "IGPI Advises Japanese Manufacturing & Cross-Border M&A Restructuring",
        "tracking_pillar": "M&A Activity",
        "aranca_segment": "Mergers & Acquisitions Strategy & Due Diligence",
        "published_date": "2026-11-09",
        "source_url": "https://igpi.co.jp/en/",
        "executive_summary": "IGPI expanded cross-border M&A and corporate growth advisory support for Japanese industrial and manufacturing enterprises.",
        "strategic_implication": "Highlights active cross-border deal and portfolio restructuring activity in Japan where Aranca can support commercial due diligence and target screening.",
        "materiality_score": 4,
        "review_status": "APPROVED",
    },
    {
        "company": "L.E.K. Consulting",
        "region": "US",
        "title": "L.E.K. Consulting Expands Life Sciences & Commercial Due Diligence Capabilities",
        "tracking_pillar": "Business Updates",
        "aranca_segment": "Mergers & Acquisitions Strategy & Due Diligence",
        "published_date": "2026-11-15",
        "source_url": "https://www.lek.com",
        "executive_summary": "L.E.K. Consulting broadened commercial due diligence and biopharma growth strategy coverage for private equity and corporate buyers in the US.",
        "strategic_implication": "Reinforces PE commercial diligence competition in the US while creating white-space opportunities for Aranca in pre-deal market sizing and competitor benchmarking.",
        "materiality_score": 4,
        "review_status": "APPROVED",
    },
]


def load_json_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required seed file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def seed_aranca_profile(db: IntelDatabase, fetch_live: bool = True) -> None:
    profile = load_json_file(settings.aranca_profile_path)
    structured_text = json.dumps(profile, indent=2)
    db.upsert_aranca_context(
        source_type="profile_json",
        source_label="Aranca Growth Advisory Profile (JSON)",
        content_text=structured_text,
    )

    if fetch_live:
        source_url = profile.get("source_url", "https://www.aranca.com/growth-advisory.php")
        live_text = fetch_aranca_growth_page(source_url)
        if live_text:
            db.upsert_aranca_context(
                source_type="live_webpage",
                source_label=f"Live Practice Page ({source_url})",
                content_text=live_text,
            )
        else:
            logger.info("Using structured aranca_profile.json fallback because live page text was empty.")


def seed_competitors(db: IntelDatabase) -> list[dict[str, Any]]:
    competitors = load_json_file(settings.competitors_path)
    if not isinstance(competitors, list) or not competitors:
        raise ValueError("data/competitors.json must contain a non-empty list of competitors.")
    db.sync_competitors(competitors)
    return competitors


def seed_sample_events(db: IntelDatabase) -> int:
    inserted = 0
    for item in _SAMPLE_BASELINE_EVENTS:
        canon_url = canonicalize_url(item["source_url"])
        if not canon_url:
            continue
        event = IntelEvent(**item)
        emb_text = (
            f"{event.company} ({event.region}) | {event.tracking_pillar} | "
            f"{event.aranca_segment} | {event.title} | {event.executive_summary}"
        )
        emb = embed_single(emb_text)
        eid = db.insert_event(event=event, canonical_url=canon_url, embedding=emb)
        if eid is not None:
            inserted += 1
    logger.info("Seeded %d baseline sample events into SQLite", inserted)
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed SQLite database for Aranca Competitive Intel Agent.")
    parser.add_argument(
        "--skip-live-aranca",
        action="store_true",
        help="Skip live scraping of Aranca's Growth Advisory page during seeding.",
    )
    parser.add_argument(
        "--scan-baseline",
        action="store_true",
        help="Run a live baseline competitor scan over baseline_lookback_days.",
    )
    parser.add_argument(
        "--seed-samples",
        action="store_true",
        help="Insert starter baseline sample events across US, KSA, UAE, India, and Japan.",
    )
    args = parser.parse_args()

    logger.info("Starting database initialization and seeding at %s", settings.db_path)
    db = IntelDatabase(db_path=settings.db_path)

    seed_aranca_profile(db=db, fetch_live=not args.skip_live_aranca)
    competitors = seed_competitors(db=db)
    logger.info("Loaded %d active competitors.", len(competitors))

    if args.seed_samples:
        seed_sample_events(db=db)

    if args.scan_baseline:
        llm = LLMClient()
        summary = run_competitor_scan(
            db=db,
            llm=llm,
            run_type="BASELINE_SCAN",
            lookback_days=settings.baseline_lookback_days,
            force_baseline=True,
        )
        logger.info("Baseline scan summary: %s", summary)

    logger.info("Database seeding completed successfully.")


if __name__ == "__main__":
    main()