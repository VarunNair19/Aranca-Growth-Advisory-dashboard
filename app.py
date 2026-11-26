import csv
import io
import os
from pathlib import Path
import re
from urllib.parse import urlparse
import warnings

from dotenv import load_dotenv
from flask import Flask, Response, flash, redirect, render_template, request

from scripts.seed_db import seed_sample_events
from src.collectors import extract_uploaded_document_text
from src.config import get_logger, settings
from src.database import IntelDatabase
from src.embeddings import embed_single
from src.graph import generate_daily_brief, run_competitor_scan, run_hybrid_qa
from src.schemas import ARANCA_SEGMENTS, IntelEvent, PillarType, RegionType, TRACKING_PILLARS
from src.security import canonicalize_url, sanitize_excel_cell

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r".*huggingface_hub.*cache-system uses symlinks.*")

logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "aranca-local-dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes

REGIONS: list[RegionType] = ["US", "KSA", "UAE", "India", "Japan"]
db = IntelDatabase(db_path=settings.db_path)


def _is_safe_http_url(url: str) -> bool:
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _sanitize_filename(raw_name: str) -> str:
    base = Path(raw_name or "uploaded_doc").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return cleaned[:120] or "uploaded_doc"


def _is_valid_iso_date(date_str: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", (date_str or "").strip()))


@app.context_processor
def inject_globals():
    all_events = db.query_events(days_back=3650, include_pending=True, limit=500)
    pending_count = sum(1 for e in all_events if e.get("review_status") == "PENDING_REVIEW")
    approved_count = sum(1 for e in all_events if e.get("review_status") == "APPROVED")
    gemini_ready = bool(os.getenv("GEMINI_API_KEY") or settings.gemini_api_key)
    last_run_date = db.get_last_run_date()
    return {
        "regions": REGIONS,
        "pillars": list(TRACKING_PILLARS),
        "segments": ARANCA_SEGMENTS,
        "pending_count": pending_count,
        "approved_count": approved_count,
        "gemini_ready": gemini_ready,
        "last_run_date": last_run_date,
        "is_safe_http_url": _is_safe_http_url,
    }


@app.get("/")
def radar():
    sel_comp = request.args.get("company", "All").strip() or "All"
    sel_reg = request.args.get("region", "All").strip() or "All"
    sel_pillar = request.args.get("pillar", "All").strip() or "All"
    search_q = request.args.get("q", "").strip()[:200]

    events = db.query_events(
        regions=None if sel_reg == "All" else [sel_reg],
        company=sel_comp,
        pillar=sel_pillar,
        days_back=3650,
        include_pending=True,
        limit=100,
    )

    if search_q:
        matched_ids = set(db.fts_search_ids(search_q, limit=150))
        events = [e for e in events if int(e["event_id"]) in matched_ids]

    competitors = db.get_active_competitors()
    return render_template(
        "radar.html",
        active_tab="radar",
        events=events,
        competitors=competitors,
        sel_comp=sel_comp,
        sel_reg=sel_reg,
        sel_pillar=sel_pillar,
        search_q=search_q,
    )


@app.get("/export/signals.csv")
def export_signals_csv():
    events = db.query_events(days_back=3650, include_pending=True, limit=500)
    buf = io.StringIO()
    writer = csv.writer(buf)
    headers = [
        "event_id",
        "company",
        "region",
        "tracking_pillar",
        "aranca_segment",
        "published_date",
        "review_status",
        "materiality_score",
        "title",
        "executive_summary",
        "strategic_implication",
        "source_url",
    ]
    writer.writerow([sanitize_excel_cell(h) for h in headers])
    for ev in events:
        row_vals = [
            ev.get("event_id", ""),
            ev.get("company", ""),
            ev.get("region", ""),
            ev.get("tracking_pillar", ""),
            ev.get("aranca_segment", ""),
            ev.get("published_date", ""),
            ev.get("review_status", ""),
            ev.get("materiality_score", ""),
            ev.get("title", ""),
            ev.get("executive_summary", ""),
            ev.get("strategic_implication", ""),
            ev.get("source_url", ""),
        ]
        writer.writerow([sanitize_excel_cell(v) for v in row_vals])

    return Response(
        buf.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=aranca_intel_signals.csv"},
    )


@app.post("/scan")
def trigger_scan():
    force_baseline = request.form.get("force_baseline", "1") == "1"
    last_date = db.get_last_run_date()
    effective_lookback = max(settings.baseline_lookback_days, 30)
    summary = run_competitor_scan(
        db=db,
        run_type="BASELINE_SCAN" if force_baseline else "INCREMENTAL_SCAN",
        lookback_days=effective_lookback,
        force_baseline=force_baseline,
    )
    if summary.get("status") == "COMPLETED":
        scope_note = f"full {effective_lookback}-day baseline window" if force_baseline else f"since last run ({last_date or 'baseline'})"
        flash(
            f"Scan complete ({scope_note}): fetched {summary.get('articles_fetched', 0)}, "
            f"inserted {summary.get('events_inserted', 0)}, "
            f"duplicates skipped {summary.get('duplicates_skipped', 0)}. "
            f"0 fetched usually means no new items appeared in the window or feed sources returned none.",
            "success",
        )
    else:
        flash(f"Scan finished with status {summary.get('status')}: {summary.get('error', '')}", "warning")
    return redirect("/")


@app.post("/seed")
def trigger_seed():
    added = seed_sample_events(db)
    if added > 0:
        flash(f"Added {added} starter baseline event(s).", "success")
    else:
        flash("Starter baseline events are already in the database.", "info")
    return redirect("/")


@app.get("/approvals")
def approvals():
    status_filter = request.args.get("status", "PENDING_REVIEW")
    all_events = db.query_events(days_back=3650, include_pending=True, limit=300)
    if status_filter in {"APPROVED", "PENDING_REVIEW"}:
        filtered = [e for e in all_events if e.get("review_status") == status_filter]
    else:
        filtered = all_events

    return render_template(
        "approvals.html",
        active_tab="approvals",
        events=filtered,
        status_filter=status_filter,
    )


@app.post("/approvals/<int:event_id>")
def update_approval(event_id: int):
    new_status = request.form.get("status", "")
    if new_status in {"APPROVED", "PENDING_REVIEW", "REJECTED"}:
        db.update_review_status(event_id, new_status)
        flash(f"Event #{event_id} marked as {new_status}.", "success")
    return redirect("/approvals")


@app.route("/briefing", methods=["GET", "POST"])
def briefing():
    question = "Key competitor moves in KSA, UAE, and India"
    region_scope = "All"
    qa_answer = None
    cited_events = []
    daily_brief = None

    if request.method == "POST":
        action = request.form.get("action", "qa")
        question = (request.form.get("question") or "").strip()[:300]
        region_scope = request.form.get("region", "All")

        if action == "qa" and question:
            qa_answer, cited_events = run_hybrid_qa(question=question, db=db, days_back=90)
        elif action == "brief":
            events_for_brief = db.query_events(
                regions=None if region_scope == "All" else [region_scope],
                days_back=3650,
                include_pending=False,
                limit=50,
            )
            daily_brief = generate_daily_brief(events=events_for_brief)
            if daily_brief is None:
                flash("No approved events available in this region scope yet.", "info")

    return render_template(
        "briefing.html",
        active_tab="briefing",
        question=question,
        region_scope=region_scope,
        qa_answer=qa_answer,
        cited_events=cited_events,
        daily_brief=daily_brief,
    )


@app.route("/manage", methods=["GET", "POST"])
def manage():
    if request.method == "POST":
        form_type = request.form.get("form_type")
        if form_type == "competitor":
            comp_name = (request.form.get("company") or "").strip()[:120]
            comp_reg = request.form.get("region", "US")
            comp_query = (request.form.get("rss_query") or "").strip()[:240]
            if not comp_name:
                flash("Please enter a competitor name.", "warning")
            elif any(ch in comp_query for ch in ("\n", "\r", "\x00")):
                flash("Search query cannot contain line breaks.", "warning")
            else:
                db.sync_competitors(
                    [
                        {
                            "name": comp_name,
                            "region": comp_reg,
                            "rss_query": comp_query or f'"{comp_name}" advisory OR strategy',
                            "active": True,
                        }
                    ]
                )
                flash(f"Saved {comp_name} ({comp_reg}) to watchlist.", "success")
            return redirect("/manage")

        if form_type == "upload":
            uploaded = request.files.get("doc_file")
            comp_name = ((request.form.get("company") or "").strip() or "Competitor")[:120]
            reg: RegionType = request.form.get("region", "US")  # type: ignore[assignment]
            pillar: PillarType = request.form.get("pillar", "Business Updates")  # type: ignore[assignment]
            pub_date = (request.form.get("published_date") or "").strip()

            if not uploaded or not uploaded.filename:
                flash("Please choose a .pdf, .txt, or .md file.", "warning")
            elif not _is_valid_iso_date(pub_date):
                flash("Please enter a valid date in YYYY-MM-DD format.", "warning")
            else:
                safe_name = _sanitize_filename(uploaded.filename)
                try:
                    doc_text = extract_uploaded_document_text(safe_name, uploaded.read())
                    src_url = f"https://local-upload.aranca.local/{safe_name}"
                    event = IntelEvent(
                        company=comp_name,
                        region=reg,
                        title=f"Uploaded Document: {safe_name}",
                        tracking_pillar=pillar,
                        aranca_segment=ARANCA_SEGMENTS[4],
                        published_date=pub_date,
                        source_url=src_url,
                        additional_sources=[],
                        executive_summary=doc_text[:600],
                        strategic_implication="Uploaded document captured for practice review.",
                        materiality_score=3,
                        review_status="PENDING_REVIEW",
                    )
                    emb = embed_single(f"{event.company}: {event.title} - {event.executive_summary}")
                    eid = db.insert_event(event=event, canonical_url=canonicalize_url(src_url), embedding=emb)
                    if eid is not None:
                        flash(f"Uploaded '{safe_name}' as Event #{eid} (queued for Approval).", "success")
                    else:
                        flash(f"'{safe_name}' was already stored (duplicate skipped).", "info")
                except ValueError as exc:
                    flash(f"Upload rejected: {exc}", "warning")
            return redirect("/manage")

    competitors = db.get_active_competitors()
    return render_template("manage.html", active_tab="manage", competitors=competitors)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)