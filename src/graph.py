import re
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
import numpy as np

from src.collectors import collect_competitor_articles
from src.config import get_logger, settings
from src.database import IntelDatabase
from src.embeddings import cosine_similarities, embed_single, embed_texts, reciprocal_rank_fusion
from src.llm_client import LLMClient
from src.prompts import (
    EXTRACTION_SYSTEM_INSTRUCTION,
    GATE_SYSTEM_INSTRUCTION,
    QA_SYSTEM_INSTRUCTION,
    build_batch_gate_prompt,
    build_daily_brief_prompt,
    build_extraction_prompt,
    build_qa_prompt,
    build_self_query_prompt,
)
from src.schemas import (
    ARANCA_SEGMENTS,
    BatchGateOutput,
    DailyBriefOutput,
    IntelEvent,
    LLMAnalysisOutput,
    RawArticle,
    SelfQueryFilter,
)
from src.security import canonicalize_url, wrap_untrusted_text

logger = get_logger(__name__)

_NUMBER_TOKEN_RE = re.compile(r"(?:\$|€|£|SAR|AED|INR|JPY)?\s*\d+(?:[.,]\d+)?%?")


class ScanState(TypedDict):
    run_type: str
    lookback_days: int
    force_baseline: bool
    run_id: int
    since_date: Optional[str]
    raw_articles: list[RawArticle]
    gated_articles: list[tuple[RawArticle, int, bool]]
    inserted_count: int
    duplicates_skipped: int
    errors: list[str]


def _normalize_segment(raw_segment: str) -> str:
    if raw_segment in ARANCA_SEGMENTS:
        return raw_segment
    lowered = (raw_segment or "").lower()
    for seg in ARANCA_SEGMENTS:
        if any(word in lowered for word in seg.lower().split()[:2]):
            return seg
    return "Market Intelligence & Competitive Benchmarking"


def _has_ungrounded_numbers(generated_text: str, source_text: str) -> bool:
    """Checks whether numeric claims in generated text appear anywhere in the source text."""
    gen_nums = {m.strip().lower() for m in _NUMBER_TOKEN_RE.findall(generated_text or "")}
    if not gen_nums:
        return False
    src_lower = (source_text or "").lower()
    for token in gen_nums:
        digits_only = re.sub(r"[^\d.]", "", token)
        if not digits_only or len(digits_only) == 4 and digits_only.startswith(("202", "201")):
            continue
        if token not in src_lower and digits_only not in src_lower:
            return True
    return False


def _build_event_embedding_text(event: IntelEvent) -> str:
    return (
        f"{event.company} ({event.region}) | {event.tracking_pillar} | "
        f"{event.aranca_segment} | {event.title} | {event.executive_summary}"
    )


def _node_collect(state: ScanState, db: IntelDatabase) -> dict[str, Any]:
    competitors = db.get_active_competitors()
    if not competitors:
        logger.warning("[Scan #%d | Collect] No active competitors found in watchlist; skipping collection.", state["run_id"])
        return {"raw_articles": [], "since_date": None}

    since_date = None if state["force_baseline"] else db.get_last_run_date()
    effective_lookback = state["lookback_days"] if state["force_baseline"] else settings.baseline_lookback_days

    comp_names = [c["name"] for c in competitors]
    logger.info(
        "[Scan #%d | Collect] Starting feed collection across %d active competitors (%s) | since_date=%s lookback_days=%d cap=%d.",
        state["run_id"],
        len(competitors),
        ", ".join(comp_names[:25]),
        since_date or "None (baseline lookback)",
        effective_lookback,
        settings.max_articles_per_run,
    )

    articles = collect_competitor_articles(
        competitors=competitors,
        since_date=since_date,
        lookback_days=effective_lookback,
        max_articles=settings.max_articles_per_run,
    )
    logger.info(
        "[Scan #%d | Collect] Complete: fetched=%d raw articles (since_date=%s).",
        state["run_id"],
        len(articles),
        since_date,
    )
    return {"raw_articles": articles, "since_date": since_date}


def _node_gate_and_dedupe(
    state: ScanState,
    db: IntelDatabase,
    llm: LLMClient,
) -> dict[str, Any]:
    raw_articles = state.get("raw_articles", [])
    if not raw_articles:
        logger.info("[Scan #%d | Gate & Dedupe] No raw articles to screen.", state["run_id"])
        return {"gated_articles": [], "duplicates_skipped": 0}

    fresh_candidates: list[RawArticle] = []
    dupes_skipped = 0

    for art in raw_articles:
        canon_url = canonicalize_url(art.source_url)
        if not canon_url or db.is_url_seen(canon_url):
            dupes_skipped += 1
            continue
        fresh_candidates.append(art)

    if not fresh_candidates:
        logger.info(
            "[Scan #%d | Gate & Dedupe] Skipped all %d fetched articles (already seen by URL).",
            state["run_id"],
            len(raw_articles),
        )
        return {"gated_articles": [], "duplicates_skipped": dupes_skipped}

    logger.info(
        "[Scan #%d | Gate & Dedupe] Screening %d fresh articles (skipped %d already-seen URLs).",
        state["run_id"],
        len(fresh_candidates),
        dupes_skipped,
    )

    relevant_scored: list[tuple[RawArticle, int, bool]] = []
    batch_size = 12

    for start in range(0, len(fresh_candidates), batch_size):
        chunk = fresh_candidates[start : start + batch_size]
        lines = []
        for idx, item in enumerate(chunk):
            safe_title = wrap_untrusted_text(item.title, max_chars=250)
            safe_snip = wrap_untrusted_text(item.snippet, max_chars=500)
            lines.append(
                f"[{idx}] Company={item.company} | Region={item.region} | Date={item.published_date}\n"
                f"Title: {safe_title}\nSnippet: {safe_snip}"
            )

        gate_prompt = build_batch_gate_prompt(lines)
        gate_res = llm.generate_structured(
            prompt=gate_prompt,
            response_schema=BatchGateOutput,
            use_fast_model=True,
            system_instruction=GATE_SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

        if not gate_res:
            for item in chunk:
                relevant_scored.append((item, 3, False))
            continue

        decision_map = {d.index: d for d in gate_res.decisions}
        for idx, item in enumerate(chunk):
            dec = decision_map.get(idx)
            if dec and dec.is_relevant and dec.materiality_score >= 2:
                relevant_scored.append((item, dec.materiality_score, True))
            elif dec:
                db.mark_url_seen(canonicalize_url(item.source_url))
                logger.debug(
                    "[Scan #%d | Gate] Filtered out company=%s title=%r reason=%s",
                    state["run_id"],
                    item.company,
                    item.title[:80],
                    dec.reason,
                )
            else:
                logger.warning(
                    "[Scan #%d | Gate] Missing decision for company=%s title=%r; routing to review.",
                    state["run_id"],
                    item.company,
                    item.title[:80],
                )
                relevant_scored.append((item, 3, False))

    existing_candidates = db.get_recent_candidate_embeddings(days_back=settings.baseline_lookback_days)
    deduped_items: list[tuple[RawArticle, int, bool]] = []
    semantic_dupes = 0

    if relevant_scored:
        candidate_texts = [
            f"{art.company}: {art.title} - {art.snippet[:300]}"
            for art, _, _ in relevant_scored
        ]
        candidate_vecs = embed_texts(candidate_texts)

        for (art, score, gate_succeeded), vec in zip(relevant_scored, candidate_vecs):
            same_company_pool = [
                (eid, emb)
                for eid, comp, _, emb in existing_candidates
                if comp.lower() == art.company.lower()
            ]
            if gate_succeeded and same_company_pool and vec:
                sims = cosine_similarities(vec, [emb for _, emb in same_company_pool])
                best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
                best_sim = sims[best_idx]
                best_eid = same_company_pool[best_idx][0]

                if best_sim >= settings.dedupe_similarity_threshold:
                    source_url = canonicalize_url(art.source_url)
                    if db.append_additional_source(best_eid, source_url):
                        db.mark_url_seen(source_url)
                        dupes_skipped += 1
                        semantic_dupes += 1
                        logger.info(
                            "[Scan #%d | Dedupe] Merged duplicate for %s into event_id=%d (similarity=%.3f).",
                            state["run_id"],
                            art.company,
                            best_eid,
                            best_sim,
                        )
                        continue
                    logger.warning(
                        "[Scan #%d | Dedupe] Could not merge source for %s; continuing to extraction.",
                        state["run_id"],
                        art.company,
                    )

            deduped_items.append((art, score, gate_succeeded))

    logger.info(
        "[Scan #%d | Gate & Dedupe] Complete: fresh=%d, candidates_for_extraction=%d, semantic_dupes=%d, ready_for_extraction=%d.",
        state["run_id"],
        len(fresh_candidates),
        len(relevant_scored),
        semantic_dupes,
        len(deduped_items),
    )
    return {"gated_articles": deduped_items, "duplicates_skipped": dupes_skipped}


def _node_extract_and_store(
    state: ScanState,
    db: IntelDatabase,
    llm: LLMClient,
) -> dict[str, Any]:
    gated = state.get("gated_articles", [])
    if not gated:
        logger.info("[Scan #%d | Extract & Store] No gated articles to extract.", state["run_id"])
        return {"inserted_count": 0}

    logger.info(
        "[Scan #%d | Extract & Store] Extracting structured insights & Aranca ideation for %d articles.",
        state["run_id"],
        len(gated),
    )

    aranca_ctx = db.get_aranca_context_summary(max_chars=2500)
    inserted = 0
    approved_cnt = 0
    pending_cnt = 0

    for art, gate_score, gate_succeeded in gated:
        canon_url = canonicalize_url(art.source_url)
        prompt = build_extraction_prompt(
            company=art.company,
            region=art.region,
            title=art.title,
            published_date=art.published_date,
            snippet=art.snippet,
            aranca_context=aranca_ctx,
        )

        analysis = llm.generate_structured(
            prompt=prompt,
            response_schema=LLMAnalysisOutput,
            use_fast_model=False,
            system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
            temperature=0.1,
        )

        if not analysis:
            logger.warning(
                "[Scan #%d | Extract] Skipped article (LLM returned no valid output) company=%s title=%r",
                state["run_id"],
                art.company,
                art.title[:80],
            )
            continue

        segment = _normalize_segment(analysis.aranca_segment)
        source_blob = f"{art.title} {art.snippet}"
        combined_insight = f"{analysis.executive_summary} {analysis.strategic_implication}"
        ungrounded_num = _has_ungrounded_numbers(combined_insight, source_blob)

        if ungrounded_num:
            logger.info(
                "[Scan #%d | Grounding] Unverified numeric claim detected for %s (%r); routing to PENDING_REVIEW.",
                state["run_id"],
                art.company,
                art.title[:70],
            )

        review_status = (
            "APPROVED"
            if gate_succeeded and analysis.confidence >= 0.65 and gate_score >= 3 and not ungrounded_num
            else "PENDING_REVIEW"
        )

        event = IntelEvent(
            company=art.company,
            region=art.region,
            title=art.title.strip(),
            tracking_pillar=analysis.tracking_pillar,
            aranca_segment=segment,
            published_date=art.published_date,
            source_url=art.source_url,
            additional_sources=[],
            executive_summary=analysis.executive_summary.strip(),
            strategic_implication=analysis.strategic_implication.strip(),
            materiality_score=gate_score,
            review_status=review_status,  # type: ignore[arg-type]
        )

        emb = embed_single(_build_event_embedding_text(event))
        event_id = db.insert_event(event=event, canonical_url=canon_url, embedding=emb)
        if event_id is not None:
            inserted += 1
            if review_status == "APPROVED":
                approved_cnt += 1
            else:
                pending_cnt += 1

    logger.info(
        "[Scan #%d | Extract & Store] Complete: stored=%d new signals (APPROVED=%d, PENDING_REVIEW=%d).",
        state["run_id"],
        inserted,
        approved_cnt,
        pending_cnt,
    )
    return {"inserted_count": inserted}


def build_scan_graph(db: IntelDatabase, llm: LLMClient):
    graph = StateGraph(ScanState)
    graph.add_node("collect", lambda s: _node_collect(s, db))
    graph.add_node("gate_and_dedupe", lambda s: _node_gate_and_dedupe(s, db, llm))
    graph.add_node("extract_and_store", lambda s: _node_extract_and_store(s, db, llm))

    graph.set_entry_point("collect")
    graph.add_edge("collect", "gate_and_dedupe")
    graph.add_edge("gate_and_dedupe", "extract_and_store")
    graph.add_edge("extract_and_store", END)
    return graph.compile()


def run_competitor_scan(
    db: IntelDatabase,
    llm: Optional[LLMClient] = None,
    run_type: str = "INCREMENTAL_SCAN",
    lookback_days: int = settings.baseline_lookback_days,
    force_baseline: bool = False,
) -> dict[str, Any]:
    client = llm or LLMClient()
    client.reset_call_counter()
    run_id = db.start_run_log(run_type=run_type)
    logger.info(
        "[Scan #%d | Start] run_type=%s force_baseline=%s lookback_days=%d",
        run_id,
        run_type,
        force_baseline,
        lookback_days,
    )
    app = build_scan_graph(db=db, llm=client)

    initial_state: ScanState = {
        "run_type": run_type,
        "lookback_days": lookback_days,
        "force_baseline": force_baseline,
        "run_id": run_id,
        "since_date": None,
        "raw_articles": [],
        "gated_articles": [],
        "inserted_count": 0,
        "duplicates_skipped": 0,
        "errors": [],
    }

    try:
        final_state = app.invoke(initial_state)
        articles_fetched = len(final_state.get("raw_articles", []))
        events_inserted = int(final_state.get("inserted_count", 0))
        dupes_skipped = int(final_state.get("duplicates_skipped", 0))
        llm_calls = client.calls_made

        db.finish_run_log(
            run_id=run_id,
            articles_fetched=articles_fetched,
            events_inserted=events_inserted,
            duplicates_skipped=dupes_skipped,
            llm_calls_used=llm_calls,
            status="COMPLETED",
            notes=f"since_date={final_state.get('since_date')}",
        )
        logger.info(
            "[Scan #%d | Completed] fetched=%d, inserted=%d, duplicates_skipped=%d, llm_calls=%d.",
            run_id,
            articles_fetched,
            events_inserted,
            dupes_skipped,
            llm_calls,
        )
        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "articles_fetched": articles_fetched,
            "events_inserted": events_inserted,
            "duplicates_skipped": dupes_skipped,
            "llm_calls_used": llm_calls,
        }
    except Exception as exc:
        logger.exception("[Scan #%d | Failed] Unexpected error during competitor scan: %s", run_id, exc)
        db.finish_run_log(
            run_id=run_id,
            articles_fetched=0,
            events_inserted=0,
            duplicates_skipped=0,
            llm_calls_used=client.calls_made,
            status="FAILED",
            notes=str(exc)[:300],
        )
        return {
            "run_id": run_id,
            "status": "FAILED",
            "error": str(exc),
            "llm_calls_used": client.calls_made,
        }


def run_hybrid_qa(
    question: str,
    db: IntelDatabase,
    llm: Optional[LLMClient] = None,
    days_back: int = 90,
) -> tuple[str, list[dict[str, Any]]]:
    logger.info("[Hybrid Q&A | Start] question=%r default_days_back=%d", question[:120], days_back)
    client = llm or LLMClient()
    parsed_filter = client.generate_structured(
        prompt=build_self_query_prompt(question=question, default_days_back=days_back),
        response_schema=SelfQueryFilter,
        use_fast_model=True,
        system_instruction=GATE_SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    eff_regions = parsed_filter.regions if (parsed_filter and parsed_filter.regions) else None
    eff_company = parsed_filter.company if (parsed_filter and parsed_filter.company) else None
    eff_pillar = parsed_filter.pillar if (parsed_filter and parsed_filter.pillar) else None
    eff_days = parsed_filter.days_back if parsed_filter else days_back
    sem_query = parsed_filter.semantic_query if (parsed_filter and parsed_filter.semantic_query) else question

    candidate_rows = db.query_events(
        regions=eff_regions,
        company=eff_company,
        pillar=eff_pillar,
        days_back=eff_days,
        include_pending=False,
        limit=200,
    )
    if len(candidate_rows) < 5:
        broad_rows = db.query_events(
            days_back=3650,
            include_pending=False,
            limit=300,
        )
        seen_ids = {int(r["event_id"]) for r in candidate_rows}
        for r in broad_rows:
            if int(r["event_id"]) not in seen_ids:
                candidate_rows.append(r)
                seen_ids.add(int(r["event_id"]))

    if not candidate_rows:
        candidate_rows = db.query_events(days_back=3650, include_pending=True, limit=300)

    if not candidate_rows:
        logger.info("[Hybrid Q&A | Done] No competitive intelligence events available in database.")
        return "No competitive intelligence events are stored in the database yet.", []

    active_comps = db.get_active_competitors()
    if active_comps:
        active_names = {c["name"].strip().lower() for c in active_comps if c.get("name")}
        watchlist_matched = [
            r for r in candidate_rows if (r.get("company") or "").strip().lower() in active_names
        ]
        if watchlist_matched:
            candidate_rows = watchlist_matched

    by_id = {int(r["event_id"]): r for r in candidate_rows}
    fts_ids = [eid for eid in db.fts_search_ids(sem_query, limit=25) if eid in by_id]

    q_vec = embed_single(sem_query)
    cand_vecs = [
        np.array(
            embed_single(f"{r['company']} {r['title']} {r['executive_summary']}"),
            dtype=np.float32,
        )
        for r in candidate_rows[:40]
    ]
    vec_ids_scored = sorted(
        zip([int(r["event_id"]) for r in candidate_rows[:40]], cosine_similarities(q_vec, cand_vecs)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    vec_ids = [eid for eid, _ in vec_ids_scored[:25]]

    fused_ids = reciprocal_rank_fusion(fts_ranked_ids=fts_ids, vector_ranked_ids=vec_ids, top_n=8)
    top_events = [by_id[eid] for eid in fused_ids if eid in by_id] or candidate_rows[:6]

    logger.info(
        "[Hybrid Q&A | Retrieval] candidates=%d, fts_hits=%d, fused_top=%d, active_watchlist_count=%d, regions=%s, company=%s.",
        len(candidate_rows),
        len(fts_ids),
        len(top_events),
        len(active_comps),
        eff_regions,
        eff_company,
    )

    context_blocks = [
        f"[Source {idx}] Date: {ev['published_date']} | Company: {ev['company']} | "
        f"Region: {ev['region']} | Pillar: {ev['tracking_pillar']} | Segment: {ev['aranca_segment']}\n"
        f"Title: {ev['title']}\n"
        f"Summary: {ev['executive_summary']}\n"
        f"Strategic Implication: {ev['strategic_implication']}\n"
        f"URL: {ev['source_url']}"
        for idx, ev in enumerate(top_events, start=1)
    ]

    answer_prompt = build_qa_prompt(question=question, context_blocks=context_blocks)

    answer_text = client.generate_text(
        prompt=answer_prompt,
        use_fast_model=False,
        system_instruction=QA_SYSTEM_INSTRUCTION,
        temperature=0.1,
    )
    if answer_text:
        logger.info("[Hybrid Q&A | Completed] Generated response (%d chars) across %d cited signals.", len(answer_text), len(top_events))
        return answer_text, top_events

    logger.warning("[Hybrid Q&A | Fallback] LLM synthesis unavailable; returning structured source summary.")
    fallback_lines = [
        f"[Source {idx}] {ev['company']} ({ev['region']}, {ev['published_date']}) — "
        f"{ev['executive_summary']} | Strategic Implication: {ev['strategic_implication']}"
        for idx, ev in enumerate(top_events[:5], start=1)
    ]
    return "\n".join(fallback_lines), top_events


def generate_daily_brief(
    events: list[dict[str, Any]],
    llm: Optional[LLMClient] = None,
) -> Optional[DailyBriefOutput]:
    if not events:
        logger.info("[Daily Brief | Skipped] No events provided for briefing.")
        return None
    logger.info("[Daily Brief | Start] Synthesizing executive brief from %d approved signals.", min(len(events), 25))
    client = llm or LLMClient()
    event_lines = [
        f"- [{ev['published_date']}] {ev['company']} ({ev['region']}, {ev['tracking_pillar']}): "
        f"{ev['executive_summary']} | Implication: {ev['strategic_implication']}"
        for ev in events[:25]
    ]
    brief = client.generate_structured(
        prompt=build_daily_brief_prompt(event_lines=event_lines),
        response_schema=DailyBriefOutput,
        use_fast_model=False,
        system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
        temperature=0.2,
    )
    if brief:
        logger.info(
            "[Daily Brief | Completed] Generated brief with %d regional highlights and %d practice actions.",
            len(brief.regional_highlights),
            len(brief.recommended_pl_actions),
        )
    else:
        logger.warning("[Daily Brief | Failed] Could not generate structured DailyBriefOutput.")
    return brief