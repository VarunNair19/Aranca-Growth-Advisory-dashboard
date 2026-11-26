from src.schemas import ARANCA_SEGMENTS, TRACKING_PILLARS
from src.security import wrap_untrusted_text

GATE_SYSTEM_INSTRUCTION = (
    "You are a precision screening filter for Aranca's Growth Advisory competitive intelligence pipeline. "
    "Evaluate each candidate item strictly from its title and snippet."
)

EXTRACTION_SYSTEM_INSTRUCTION = (
    "You are a senior competitive intelligence and strategy analyst supporting Aranca's Growth Advisory practice "
    "(covering US, KSA, UAE, India, and Japan). Extract only facts supported by the provided source text, "
    "and translate them into practical business insights, whitespace angles, and client pitch opportunities for Aranca. "
    "Never invent figures, deal values, dates, or claims absent from the source."
)

QA_SYSTEM_INSTRUCTION = (
    "You are a competitive intelligence and growth strategy advisor for Aranca Growth Advisory Practice Leaders. "
    "Answer strictly using the retrieved intelligence sources provided, highlighting competitor moves, commercial/client "
    "opportunities, and practical business implications for Aranca. Cite every claim inline as [Source N]. "
    "If the retrieved sources do not cover part of the question, state that gap explicitly."
)


def build_batch_gate_prompt(formatted_items: list[str]) -> str:
    joined = "\n---\n".join(formatted_items)
    return (
        "Score each numbered item for relevance to boutique/mid-sized growth, strategy, commercial, "
        "M&A, feasibility, or market intelligence consultancies.\n"
        "Set is_relevant=true only when the item reports a concrete business move, deal/mandate, "
        "M&A/partnership, leadership or practice expansion, research/report release, or strategic shift.\n"
        "Set is_relevant=false for generic fluff, unrelated topics, or paywalled/empty noise.\n\n"
        f"{joined}"
    )


def build_extraction_prompt(
    company: str,
    region: str,
    title: str,
    published_date: str,
    snippet: str,
    aranca_context: str,
) -> str:
    pillars_text = ", ".join(TRACKING_PILLARS)
    segments_text = ", ".join(ARANCA_SEGMENTS)
    safe_aranca = wrap_untrusted_text(aranca_context, max_chars=2500)
    safe_source = wrap_untrusted_text(
        f"Headline: {title}\nPublished Date: {published_date}\nSnippet/Body: {snippet}",
        max_chars=3000,
    )

    return (
        f"Tracked Competitor: {company}\n"
        f"Target Region: {region}\n"
        f"Allowed Tracking Pillars: [{pillars_text}]\n"
        f"Allowed Aranca Growth Advisory Segments: [{segments_text}]\n\n"
        f"Reference Practice Context:\n{safe_aranca}\n\n"
        f"Source Material to Analyze:\n{safe_source}\n\n"
        "Produce:\n"
        "1. A concise, factual executive_summary grounded strictly in the source material.\n"
        "2. A practical strategic_implication for Aranca's Growth Advisory team covering: "
        "(a) competitive impact in the region/segment, (b) potential client pitch or whitespace angle for Aranca "
        "(e.g., primary research, market entry/feasibility, commercial due diligence, pricing/GTM, or benchmarking), "
        "and (c) a concrete next step—without inventing facts or numbers not present in the source."
    )


def build_self_query_prompt(question: str, default_days_back: int) -> str:
    safe_q = wrap_untrusted_text(question, max_chars=800)
    return (
        "Convert the user's question into structured search filters for a competitive intelligence database.\n"
        "Valid regions: US, KSA, UAE, India, Japan.\n"
        "Valid pillars: Business Updates, Deals & Engagements, M&A Activity, Digital Presence.\n"
        f"Use days_back={default_days_back} unless a specific timeframe is requested.\n"
        f"User Question: {safe_q}"
    )


def build_qa_prompt(question: str, context_blocks: list[str]) -> str:
    safe_q = wrap_untrusted_text(question, max_chars=800)
    sources_text = "\n\n".join(context_blocks)
    return (
        f"User Question: {safe_q}\n\n"
        f"Retrieved Intelligence Sources:\n{sources_text}\n\n"
        "Provide a structured, insight-driven answer with inline [Source N] citations grounded solely in these sources. "
        "Include key competitor takeaways, commercial/client ideation angles for Aranca Growth Advisory, "
        "and practical next steps where supported by the sources."
    )


def build_daily_brief_prompt(event_lines: list[str]) -> str:
    joined = "\n".join(event_lines)
    return (
        "Synthesize an executive brief and business ideation summary for Aranca's Growth Advisory Practice Leaders "
        "across US, KSA, UAE, India, and Japan using only the events below.\n"
        "Emphasize cross-regional competitor patterns, whitespace/client pitch angles, and concrete actions "
        "Aranca practice leaders can take.\n\n"
        f"{joined}"
    )