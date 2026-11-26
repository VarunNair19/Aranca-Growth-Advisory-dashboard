# Aranca Growth Advisory — Competitive Intelligence Platform

An AI-assisted Competitive Intelligence (CI) platform built for **Aranca Growth Advisory** to monitor competitor moves across **US, KSA, UAE, India, and Japan**, screen and extract structured signals, support human-in-the-loop (HITL) review, and generate Practice Leader (PL) briefings and commercial ideation.

## Key Features

* **Multi-Region Watchlist & Feed Collection:** Tracks active competitors across `US`, `KSA`, `UAE`, `India`, and `Japan` via Google News RSS and DuckDuckGo news (`ddgs`), with safe URL validation (`robots.txt` and SSRF checks) and selective article enrichment via `trafilatura`.

* **LangGraph Ingestion Pipeline (`collect` → `gate_and_dedupe` → `extract_and_store`):**

  * **URL & Semantic Deduplication:** Canonicalizes URLs, checks `seen_urls`, generates sentence-transformer embeddings (`all-MiniLM-L6-v2`), and merges near-duplicate stories into existing events.

  * **Batch Gating & Materiality Scoring:** Screens fresh articles for relevance and strategic materiality (`1/5` to `5/5`).

  * **Structured Extraction & Grounding Check:** Extracts `tracking_pillar`, `aranca_segment`, `executive_summary`, and `strategic_implication`, routing low-confidence, lower-materiality, or unverified-number items to `PENDING_REVIEW`.

* **SQLite + WAL + FTS5 + Hybrid Retrieval:** Stores events, active competitors, run logs, and Aranca context in SQLite with Write-Ahead Logging (WAL) and an FTS5 full-text index.

* **Flask Web Interface:**

  * **Signal Radar (`/`):** Filter competitor records by competitor, region, tracking pillar, and FTS5 keyword search, inspect expandable cards, and export signals to a formula-safe CSV.

  * **Review & Approvals (`/approvals`):** Approve, hold (`PENDING_REVIEW`), or reject extracted signals and uploaded documents.

  * **Executive Briefing (`/briefing`):** Run **Hybrid Q&A** (FTS5 keyword search + cosine similarity + Reciprocal Rank Fusion) or generate a **Daily Executive Brief**.

  * **Upload & Watchlist (`/manage`):** Add or update competitors in the watchlist and upload `.pdf`, `.txt`, or `.md` documents.

## System Architecture & Workflow

```
Watchlist (competitors table)
        │
        ▼
[1. Collect] ──▶ Google News RSS + DDGS + safe webpage enrichment (trafilatura)
        │
        ▼
[2. Gate & Dedupe] ──▶ Canonical URL check + Batch LLM Gating + Embedding cosine similarity
        │
        ▼
[3. Extract & Store] ──▶ Structured LLM Extraction + Grounding check + SQLite (WAL + FTS5 + Embeddings)
        │
        ▼
[4. Review & Serve] ──▶ Flask UI (/ , /approvals , /briefing , /manage) + Hybrid Q&A + Daily Brief
```

### How the Modules Fit Together

* **Ingestion & Security (`src/collectors.py`, `src/security.py`):** Fetches candidate coverage per active competitor, enforces SSRF and `robots.txt` checks, extracts full article text where safe, parses uploaded `.pdf`, `.txt`, and `.md` files, wraps untrusted text against prompt injection, and sanitizes CSV cells.

* **Orchestration & LLM Routing (`src/graph.py`, `src/llm_client.py`, `src/prompts.py`, `src/schemas.py`):** Coordinates the LangGraph scan workflow, enforces Pydantic schemas, checks numeric grounding, and powers **Hybrid Q&A** and **Daily Briefs** using a primary Gemini model with a Groq fallback (`openai/gpt-oss-120b`).

* **Storage & Views (`src/database.py`, `app.py`, `templates/*.html`):** Manages SQLite WAL/FTS5 persistence and serves the Signal Radar, Approvals queue, Executive Briefing workspace, and Watchlist/Upload manager.

## Project Structure

```
├── app.py                  # Flask routes, CSV export, scan/seed triggers, upload & watchlist handlers
├── scripts/
│   └── seed_db.py          # Starter baseline event seeding helper for quick smoke testing
├── src/
│   ├── collectors.py       # RSS, DDGS, safe webpage text, and PDF/TXT/MD document extraction
│   ├── config.py           # Environment configuration and logging setup
│   ├── database.py         # SQLite schema, WAL/FTS5 triggers, CRUD, and query helpers
│   ├── embeddings.py       # Sentence-transformer embeddings, cosine similarity, and RRF fusion
│   ├── graph.py            # LangGraph scan workflow, Hybrid Q&A, and Daily Brief synthesis
│   ├── llm_client.py       # Primary Gemini client with retries/circuit breaker + Groq fallback
│   ├── prompts.py          # System instructions and prompt builders
│   ├── schemas.py          # Pydantic models for articles, gating, extraction, and briefings
│   └── security.py         # SSRF/robots checks, URL canonicalization, prompt wrapping, CSV escaping
└── templates/
    ├── base.html
    ├── radar.html
    ├── approvals.html
    ├── briefing.html
    └── manage.html
```

## Setup & Running Locally

1. **Create a virtual environment and install dependencies:**

   ```
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Create a `.env` file in the project root (do not commit `.env` to Git):**

   ```
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   
   PRIMARY_LLM_MODEL=gemini-3.5-flash-lite
   FAST_GATE_MODEL=gemini-3.5-flash-lite
   FALLBACK_LLM_MODEL=openai/gpt-oss-120b
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. **Run the Flask application:**

   ```
   python app.py
   ```

   Open `http://127.0.0.1:5000` in your browser.

## Hosting & GitHub Sharing

* **Local / Internal Team Setup:** Ideal for running locally or on a small internal VM because it uses local SQLite storage and local sentence-transformer embeddings (`all-MiniLM-L6-v2`) with no external vector database required.

* **Cloud Deployment:** When deploying to a cloud platform (such as Render, Railway, or a Linux VM), run with a production WSGI server (e.g., `gunicorn app:app`), configure environment variables in the hosting dashboard, and attach persistent disk storage so SQLite records and logs persist across restarts or redeploys.

* **Repository Hygiene:** Keep `.env`, `.venv/`, `*.db*`, `*.log`, and `__pycache__/` in `.gitignore`, provide a `.env.example` with placeholder keys, and include sample outputs or UI snapshots in `docs/` when sharing the repository on GitHub.

## Current Advantages, Limitations & Roadmap

### Advantages

* **Safety & Auditability:** Includes SSRF/robots checks, prompt-injection wrapping, numeric grounding checks, HITL approval status, and run logging.

* **Provider Fallback:** Automatic fallback to Groq (`openai/gpt-oss-120b`) helps keep runs completing when the primary Gemini endpoint experiences spikes or limits.

* **Hybrid Retrieval:** Combines lexical FTS5 search with dense embedding similarity and Reciprocal Rank Fusion (RRF).
- **Expandable Signal Radar (`/`):** Filter stored competitor records across competitor, region, tracking pillar, and FTS5 keyword search, with clickable cards to inspect summaries, strategic implications, and source links plus an `Expand All` / `Collapse All` toggle.
- **Broad Watchlist Scanning & Hybrid Q&A:** Scans log active watchlist competitors and support baseline lookback windows for newly added firms, while Hybrid Q&A widens retrieval across stored events when filtered matches are limited.
- **URL State Tracking (`PROCESSING`, `COMPLETED`, `IGNORED`, `RETRYABLE`):** Tracks article processing states so transient failures can retry on later scans, irrelevant items are marked `IGNORED`, duplicates and stored signals are marked `COMPLETED`, existing rows migrate safely as `COMPLETED`, and stale `PROCESSING` entries can be reclaimed after crashes.
- **Explicit Gate-Failure Handling:** When structured gating returns no result, the pipeline keeps `gate_failed` separate from `materiality_score` instead of assigning a passing score, routing extracted items to `PENDING_REVIEW` without crashing the scan.
- **SSRF Redirect Validation:** Validates redirect targets on every hop and blocks private, loopback, and link-local addresses while keeping public redirects working.
### Limitations & Data Scope Notes

* **Starter Seeding vs. Rich Ideation:** Starter sample events provide a small baseline for testing UI and retrieval; broader cross-region ideation across `US`, `KSA`, `UAE`, `India`, and `Japan` improves as you run wider scans, approve `PENDING_REVIEW` items, and upload reference documents.

* **Single-Node SQLite & Local Embeddings:** Well-suited for local or small-team workflows, while larger multi-user deployments may eventually prefer managed PostgreSQL/pgvector and external job scheduling.

### Planned Next Steps

* **Watchlist Deactivation / Soft-Delete:** Add a control on **Upload & Watchlist** (`/manage`) to deactivate (`active = 0`) competitors so firms can be paused from future scans while keeping historical signals and run logs intact.

* **Logs Viewer in the Web UI:** Add a dedicated logs view/tab in the Flask interface to inspect recent scan run logs and filtered application log entries directly from the browser.

* **Admin Add-Ons & Operational Controls:** Add admin utilities such as Aranca context editor/management, scan lookback/quota controls, database retention/maintenance actions, and competitor watchlist coverage views.

* **Broader Scan & Retrieval Coverage:** Expand multi-competitor scan coverage and Hybrid Q&A retrieval across active watchlist firms so newly added competitors and multi-region signals surface consistently for Practice Leader ideation.

* **Model & Fallback Resilience:** Keep primary Gemini model settings and Groq fallback (`openai/gpt-oss-120b`) tuned for consistent structured extraction, Q&A, and daily briefing synthesis.