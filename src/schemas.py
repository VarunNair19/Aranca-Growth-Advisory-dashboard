from typing import Literal, Optional

from pydantic import BaseModel, Field

RegionType = Literal["US", "KSA", "UAE", "India", "Japan"]
PillarType = Literal[
    "Business Updates",
    "Deals & Engagements",
    "M&A Activity",
    "Digital Presence",
]
ReviewStatusType = Literal["APPROVED", "PENDING_REVIEW", "REJECTED"]

TRACKING_PILLARS: list[PillarType] = [
    "Business Updates",
    "Deals & Engagements",
    "M&A Activity",
    "Digital Presence",
]

ARANCA_SEGMENTS: list[str] = [
    "Market Share Expansion & Growth Acceleration",
    "Adjacency & Business Diversification Strategy",
    "Sales Effectiveness & Commercial Excellence",
    "Feasibility Study & Long Range Business Planning",
    "Market Intelligence & Competitive Benchmarking",
    "B2B Voice of Customer & Customer Analytics",
    "Mergers & Acquisitions Strategy & Due Diligence",
    "New Product Development & Innovation Strategy",
    "Business Transformation & PMO Advisory",
    "International Market Entry & Expansion Strategy",
]


class RawArticle(BaseModel):
    company: str
    region: RegionType
    title: str
    snippet: str
    source_url: str
    published_date: str  # Strict ISO YYYY-MM-DD
    source_type: Literal["rss", "search", "website_diff"]


class GateItemScore(BaseModel):
    index: int = Field(description="Index of the candidate article in the batch.")
    is_relevant: bool = Field(
        description="True only if concrete business update, deal, M&A, leadership move, or offering launch."
    )
    materiality_score: int = Field(
        ge=1, le=5, description="1=generic fluff, 3=borderline signal, 5=major strategic move."
    )
    reason: str = Field(description="One-sentence justification.")


class BatchGateOutput(BaseModel):
    decisions: list[GateItemScore]


class LLMAnalysisOutput(BaseModel):
    """Structured fields populated by LLM (URL and date are locked deterministically outside LLM)."""

    tracking_pillar: PillarType
    aranca_segment: str = Field(
        description="Closest matching Aranca Growth Advisory practice area."
    )
    executive_summary: str = Field(
        description="Concise 2-3 sentence factual summary in English."
    )
    strategic_implication: str = Field(
        description="Actionable 'So What?' insight for Aranca Practice Leaders, referencing historical context if relevant."
    )
    confidence: float = Field(ge=0.0, le=1.0)


class IntelEvent(BaseModel):
    event_id: Optional[int] = None
    company: str
    region: RegionType
    title: str
    tracking_pillar: PillarType
    aranca_segment: str
    published_date: str
    source_url: str
    additional_sources: list[str] = Field(default_factory=list)
    executive_summary: str
    strategic_implication: str
    materiality_score: int = Field(ge=1, le=5)
    review_status: ReviewStatusType = "APPROVED"


class SelfQueryFilter(BaseModel):
    """LLM-extracted structured filters for natural language Q&A over SQLite."""

    regions: Optional[list[RegionType]] = None
    company: Optional[str] = None
    pillar: Optional[PillarType] = None
    days_back: int = Field(default=90, ge=1, le=365)
    semantic_query: str = Field(description="Core topic for vector similarity ranking.")


class DailyBriefOutput(BaseModel):
    headline_takeaway: str
    regional_highlights: list[str]
    recommended_pl_actions: list[str]