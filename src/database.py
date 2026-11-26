import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import get_logger, settings
from src.schemas import IntelEvent

logger = get_logger(__name__)

_FTS5_RESERVED = {"AND", "OR", "NOT", "NEAR"}


class IntelDatabase:
    """SQLite storage layer with WAL mode, FTS5 full-text index, vector storage, and retention policy."""

    def __init__(self, db_path: Path = settings.db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("Initialized IntelDatabase at %s", self.db_path)

    def _connect(self, read_only: bool = False) -> sqlite3.Connection:
        if read_only and not self.db_path.exists():
            self._init_schema()

        if read_only:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect(read_only=False) as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")

                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS intel_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company TEXT NOT NULL,
                        region TEXT NOT NULL,
                        title TEXT NOT NULL,
                        tracking_pillar TEXT NOT NULL,
                        aranca_segment TEXT NOT NULL,
                        published_date TEXT NOT NULL,
                        source_url TEXT NOT NULL,
                        canonical_url TEXT NOT NULL UNIQUE,
                        additional_sources_json TEXT NOT NULL DEFAULT '[]',
                        executive_summary TEXT NOT NULL,
                        strategic_implication TEXT NOT NULL,
                        materiality_score INTEGER NOT NULL DEFAULT 3,
                        review_status TEXT NOT NULL DEFAULT 'APPROVED',
                        embedding_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_date ON intel_events(published_date);
                    CREATE INDEX IF NOT EXISTS idx_events_region ON intel_events(region);
                    CREATE INDEX IF NOT EXISTS idx_events_company ON intel_events(company);
                    CREATE INDEX IF NOT EXISTS idx_events_status ON intel_events(review_status);

                    CREATE VIRTUAL TABLE IF NOT EXISTS intel_fts USING fts5(
                        title,
                        company,
                        region,
                        tracking_pillar,
                        aranca_segment,
                        executive_summary,
                        strategic_implication,
                        content='intel_events',
                        content_rowid='event_id'
                    );

                    CREATE TRIGGER IF NOT EXISTS intel_events_ai AFTER INSERT ON intel_events BEGIN
                        INSERT INTO intel_fts(
                            rowid, title, company, region, tracking_pillar,
                            aranca_segment, executive_summary, strategic_implication
                        ) VALUES (
                            new.event_id, new.title, new.company, new.region, new.tracking_pillar,
                            new.aranca_segment, new.executive_summary, new.strategic_implication
                        );
                    END;

                    CREATE TRIGGER IF NOT EXISTS intel_events_ad AFTER DELETE ON intel_events BEGIN
                        INSERT INTO intel_fts(
                            intel_fts, rowid, title, company, region, tracking_pillar,
                            aranca_segment, executive_summary, strategic_implication
                        ) VALUES (
                            'delete', old.event_id, old.title, old.company, old.region, old.tracking_pillar,
                            old.aranca_segment, old.executive_summary, old.strategic_implication
                        );
                    END;

                    CREATE TRIGGER IF NOT EXISTS intel_events_au AFTER UPDATE ON intel_events BEGIN
                        INSERT INTO intel_fts(
                            intel_fts, rowid, title, company, region, tracking_pillar,
                            aranca_segment, executive_summary, strategic_implication
                        ) VALUES (
                            'delete', old.event_id, old.title, old.company, old.region, old.tracking_pillar,
                            old.aranca_segment, old.executive_summary, old.strategic_implication
                        );
                        INSERT INTO intel_fts(
                            rowid, title, company, region, tracking_pillar,
                            aranca_segment, executive_summary, strategic_implication
                        ) VALUES (
                            new.event_id, new.title, new.company, new.region, new.tracking_pillar,
                            new.aranca_segment, new.executive_summary, new.strategic_implication
                        );
                    END;

                    CREATE TABLE IF NOT EXISTS seen_urls (
                        canonical_url TEXT PRIMARY KEY,
                        first_seen_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS aranca_context (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_type TEXT NOT NULL,
                        source_label TEXT NOT NULL,
                        content_text TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS competitors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        region TEXT NOT NULL,
                        peer_tier TEXT NOT NULL DEFAULT 'Boutique (<300 consultants)',
                        hq TEXT NOT NULL DEFAULT '',
                        focus TEXT NOT NULL DEFAULT '',
                        rss_query TEXT NOT NULL DEFAULT '',
                        website TEXT NOT NULL DEFAULT '',
                        active INTEGER NOT NULL DEFAULT 1
                    );

                    CREATE TABLE IF NOT EXISTS run_logs (
                        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_type TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        articles_fetched INTEGER NOT NULL DEFAULT 0,
                        events_inserted INTEGER NOT NULL DEFAULT 0,
                        duplicates_skipped INTEGER NOT NULL DEFAULT 0,
                        llm_calls_used INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'RUNNING',
                        notes TEXT NOT NULL DEFAULT ''
                    );
                    """
                )
        except Exception:
            logger.exception("Failed to initialize SQLite schema at %s", self.db_path)
            raise

    def sync_competitors(self, competitors: list[dict[str, Any]]) -> None:
        try:
            with self._connect(read_only=False) as conn:
                for comp in competitors:
                    conn.execute(
                        """
                        INSERT INTO competitors (name, region, peer_tier, hq, focus, rss_query, website, active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            region = excluded.region,
                            peer_tier = excluded.peer_tier,
                            hq = excluded.hq,
                            focus = excluded.focus,
                            rss_query = excluded.rss_query,
                            website = excluded.website,
                            active = excluded.active
                        """,
                        (
                            comp["name"],
                            comp["region"],
                            comp.get("peer_tier", "Boutique (<300 consultants)"),
                            comp.get("hq", ""),
                            comp.get("focus", ""),
                            comp.get("rss_query", ""),
                            comp.get("website", ""),
                            1 if comp.get("active", True) else 0,
                        ),
                    )
            logger.info("Synced %d competitor records into SQLite", len(competitors))
        except Exception:
            logger.exception("Failed to sync competitors")
            raise

    def get_active_competitors(self) -> list[dict[str, Any]]:
        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(
                    "SELECT * FROM competitors WHERE active = 1 ORDER BY region, name"
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to fetch active competitors")
            return []

    def is_url_seen(self, canonical_url: str) -> bool:
        try:
            with self._connect(read_only=True) as conn:
                row = conn.execute(
                    "SELECT 1 FROM seen_urls WHERE canonical_url = ?", (canonical_url,)
                ).fetchone()
                return row is not None
        except Exception:
            logger.exception("Failed to check seen_urls for %s", canonical_url)
            return False

    def mark_url_seen(self, canonical_url: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect(read_only=False) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO seen_urls (canonical_url, first_seen_at) VALUES (?, ?)",
                    (canonical_url, now_iso),
                )
        except Exception:
            logger.exception("Failed to mark URL seen: %s", canonical_url)

    def insert_event(
        self,
        event: IntelEvent,
        canonical_url: str,
        embedding: Optional[list[float]] = None,
    ) -> Optional[int]:
        now_iso = datetime.now(timezone.utc).isoformat()
        emb_json = json.dumps(embedding or [])
        sources_json = json.dumps(event.additional_sources or [])
        try:
            with self._connect(read_only=False) as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO intel_events (
                        company, region, title, tracking_pillar, aranca_segment,
                        published_date, source_url, canonical_url, additional_sources_json,
                        executive_summary, strategic_implication, materiality_score,
                        review_status, embedding_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.company,
                        event.region,
                        event.title,
                        event.tracking_pillar,
                        event.aranca_segment,
                        event.published_date,
                        event.source_url,
                        canonical_url,
                        sources_json,
                        event.executive_summary,
                        event.strategic_implication,
                        event.materiality_score,
                        event.review_status,
                        emb_json,
                        now_iso,
                    ),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO seen_urls (canonical_url, first_seen_at) VALUES (?, ?)",
                    (canonical_url, now_iso),
                )
                if cur.rowcount > 0:
                    event_id = int(cur.lastrowid)
                    logger.info(
                        "Inserted intel_event id=%d company=%s region=%s pillar=%s status=%s",
                        event_id,
                        event.company,
                        event.region,
                        event.tracking_pillar,
                        event.review_status,
                    )
                    return event_id

                logger.debug("Skipped duplicate canonical_url=%s", canonical_url)
                return None
        except Exception:
            logger.exception("Failed to insert intel_event for company=%s", event.company)
            raise

    def append_additional_source(self, event_id: int, new_url: str) -> bool:
        try:
            with self._connect(read_only=False) as conn:
                row = conn.execute(
                    "SELECT additional_sources_json FROM intel_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if not row:
                    logger.warning("append_additional_source: event_id=%d not found", event_id)
                    return False
                sources: list[str] = json.loads(row["additional_sources_json"] or "[]")
                if new_url not in sources:
                    sources.append(new_url)
                    conn.execute(
                        "UPDATE intel_events SET additional_sources_json = ? WHERE event_id = ?",
                        (json.dumps(sources), event_id),
                    )
                    logger.info("Appended secondary source to event_id=%d: %s", event_id, new_url)
                return True
        except Exception:
            logger.exception("Failed to append additional source for event_id=%d", event_id)
            return False

    def get_recent_candidate_embeddings(
        self, days_back: int = 90
    ) -> list[tuple[int, str, str, np.ndarray]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        results: list[tuple[int, str, str, np.ndarray]] = []
        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(
                    """
                    SELECT event_id, company, title, embedding_json
                    FROM intel_events
                    WHERE published_date >= ? AND review_status != 'REJECTED'
                    """,
                    (cutoff,),
                ).fetchall()
                for r in rows:
                    vec = json.loads(r["embedding_json"] or "[]")
                    if vec:
                        results.append(
                            (r["event_id"], r["company"], r["title"], np.array(vec, dtype=np.float32))
                        )
            logger.debug("Loaded %d candidate embeddings from last %d days", len(results), days_back)
        except Exception:
            logger.exception("Failed to load recent candidate embeddings")
        return results

    def query_events(
        self,
        regions: Optional[list[str]] = None,
        company: Optional[str] = None,
        pillar: Optional[str] = None,
        days_back: int = 90,
        include_pending: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        clauses = ["published_date >= ?"]
        params: list[Any] = [cutoff]

        if include_pending:
            clauses.append("review_status IN ('APPROVED', 'PENDING_REVIEW')")
        else:
            clauses.append("review_status = 'APPROVED'")

        if regions:
            placeholders = ",".join("?" for _ in regions)
            clauses.append(f"region IN ({placeholders})")
            params.extend(regions)

        if company and company != "All":
            clauses.append("company = ?")
            params.append(company)

        if pillar and pillar != "All":
            clauses.append("tracking_pillar = ?")
            params.append(pillar)

        sql = f"""
            SELECT * FROM intel_events
            WHERE {' AND '.join(clauses)}
            ORDER BY published_date DESC, materiality_score DESC, event_id DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(sql, params).fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    d["additional_sources"] = json.loads(d.pop("additional_sources_json") or "[]")
                    d.pop("embedding_json", None)
                    out.append(d)
                return out
        except Exception:
            logger.exception("Failed querying intel_events")
            return []

    def fts_search_ids(self, keywords: str, limit: int = 20) -> list[int]:
        raw_tokens = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in keywords).split()
        tokens = [w for w in raw_tokens if len(w) > 1 and w.upper() not in _FTS5_RESERVED]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens[:8])
        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(
                    "SELECT rowid FROM intel_fts WHERE intel_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, limit),
                ).fetchall()
                return [int(r["rowid"]) for r in rows]
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 MATCH failed for query=%r: %s", fts_query, exc)
            return []
        except Exception:
            logger.exception("Unexpected error during FTS5 search")
            return []

    def update_review_status(self, event_id: int, status: str) -> None:
        if status not in {"APPROVED", "PENDING_REVIEW", "REJECTED"}:
            raise ValueError(f"Invalid review status: {status}")
        try:
            with self._connect(read_only=False) as conn:
                conn.execute(
                    "UPDATE intel_events SET review_status = ? WHERE event_id = ?",
                    (status, event_id),
                )
            logger.info("Updated event_id=%d review_status=%s", event_id, status)
        except Exception:
            logger.exception("Failed updating review_status for event_id=%d", event_id)
            raise

    def purge_old_events(self, older_than_days: int, confirmed: bool = False) -> int:
        if not confirmed:
            logger.warning("purge_old_events blocked because confirmed=False")
            return 0
        if older_than_days < 30:
            raise ValueError("older_than_days must be at least 30 days")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).strftime("%Y-%m-%d")
        try:
            with self._connect(read_only=False) as conn:
                cur = conn.execute("DELETE FROM intel_events WHERE published_date < ?", (cutoff,))
                deleted = cur.rowcount
            logger.warning(
                "Purged %d intel_events older than %d days (cutoff=%s)",
                deleted,
                older_than_days,
                cutoff,
            )
            return deleted
        except Exception:
            logger.exception("Failed purging old events older than %d days", older_than_days)
            raise

    def get_last_run_date(self) -> Optional[str]:
        try:
            with self._connect(read_only=True) as conn:
                row = conn.execute(
                    """
                    SELECT started_at FROM run_logs
                    WHERE status = 'COMPLETED'
                    ORDER BY run_id DESC LIMIT 1
                    """
                ).fetchone()
                return row["started_at"][:10] if row and row["started_at"] else None
        except Exception:
            logger.exception("Failed fetching last completed run date")
            return None

    def start_run_log(self, run_type: str) -> int:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect(read_only=False) as conn:
                cur = conn.execute(
                    "INSERT INTO run_logs (run_type, started_at, status) VALUES (?, ?, 'RUNNING')",
                    (run_type, now_iso),
                )
                run_id = int(cur.lastrowid)
            logger.info("Started run_log run_id=%d run_type=%s", run_id, run_type)
            return run_id
        except Exception:
            logger.exception("Failed starting run_log for run_type=%s", run_type)
            raise

    def finish_run_log(
        self,
        run_id: int,
        articles_fetched: int,
        events_inserted: int,
        duplicates_skipped: int,
        llm_calls_used: int,
        status: str = "COMPLETED",
        notes: str = "",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect(read_only=False) as conn:
                conn.execute(
                    """
                    UPDATE run_logs
                    SET finished_at = ?, articles_fetched = ?, events_inserted = ?,
                        duplicates_skipped = ?, llm_calls_used = ?, status = ?, notes = ?
                    WHERE run_id = ?
                    """,
                    (
                        now_iso,
                        articles_fetched,
                        events_inserted,
                        duplicates_skipped,
                        llm_calls_used,
                        status,
                        notes,
                        run_id,
                    ),
                )
            logger.info(
                "Finished run_log run_id=%d status=%s fetched=%d inserted=%d dupes=%d llm_calls=%d",
                run_id,
                status,
                articles_fetched,
                events_inserted,
                duplicates_skipped,
                llm_calls_used,
            )
        except Exception:
            logger.exception("Failed finishing run_log run_id=%d", run_id)
            raise

    def upsert_aranca_context(self, source_type: str, source_label: str, content_text: str) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect(read_only=False) as conn:
                existing = conn.execute(
                    "SELECT id FROM aranca_context WHERE source_type = ? AND source_label = ?",
                    (source_type, source_label),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE aranca_context SET content_text = ?, updated_at = ? WHERE id = ?",
                        (content_text, now_iso, existing["id"]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO aranca_context (source_type, source_label, content_text, updated_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (source_type, source_label, content_text, now_iso),
                    )
            logger.info("Upserted aranca_context source_type=%s label=%s", source_type, source_label)
        except Exception:
            logger.exception("Failed upserting aranca_context for label=%s", source_label)
            raise

    def get_aranca_context_summary(self, max_chars: int = 3500) -> str:
        try:
            with self._connect(read_only=True) as conn:
                rows = conn.execute(
                    "SELECT source_label, content_text FROM aranca_context ORDER BY id DESC"
                ).fetchall()
                parts = [f"[{r['source_label']}]\n{r['content_text']}" for r in rows]
                return "\n\n".join(parts)[:max_chars]
        except Exception:
            logger.exception("Failed fetching aranca_context summary")
            return ""