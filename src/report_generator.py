from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import re
from typing import Any, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.config import get_logger
from src.schemas import DailyBriefOutput
from src.security import is_safe_public_url

logger = get_logger(__name__)

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_cell(value: Any) -> Any:
    """Prevents spreadsheet formula injection and strips unsafe control characters."""
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value

    text = _CONTROL_CHAR_RE.sub("", str(value))
    if text and text[0] in _FORMULA_PREFIXES:
        text = f"'{text}"
    return text


def _style_header_row(ws, max_col: int) -> None:
    header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    for col_idx in range(1, max_col + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 24
    ws.freeze_panes = "A2"


def _auto_fit_columns(ws, max_width: int = 55, min_width: int = 12) -> None:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        longest = 0
        for cell in col:
            val_str = "" if cell.value is None else str(cell.value)
            first_line = max((len(line) for line in val_str.splitlines()), default=0)
            if first_line > longest:
                longest = first_line
        ws.column_dimensions[col_letter].width = max(min_width, min(longest + 4, max_width))


def build_intel_excel_report(
    events: list[dict[str, Any]],
    brief: Optional[DailyBriefOutput] = None,
    filters_summary: str = "All Regions | Last 90 Days",
) -> bytes:
    """Builds a multi-sheet .xlsx competitive intelligence workbook and returns raw bytes."""
    try:
        wb = Workbook()

        # Sheet 1: Executive Brief
        ws_brief = wb.active
        ws_brief.title = "Executive Brief"
        ws_brief.append(["Section", "Details"])
        _style_header_row(ws_brief, max_col=2)

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ws_brief.append(["Generated At", _sanitize_cell(generated_at)])
        ws_brief.append(["Active Filters", _sanitize_cell(filters_summary)])
        ws_brief.append(["Total Events Included", len(events)])

        if brief:
            ws_brief.append(["Headline Takeaway", _sanitize_cell(brief.headline)])
            ws_brief.append(["Regional Highlights", _sanitize_cell(brief.regional_highlights)])
            ws_brief.append(["Recommended Actions", _sanitize_cell(brief.recommended_actions)])
        else:
            ws_brief.append([
                "Executive Summary",
                _sanitize_cell("Structured competitor intelligence events exported from SQLite."),
            ])

        for row in ws_brief.iter_rows(min_row=2, max_row=ws_brief.max_row, min_col=1, max_col=2):
            row[0].font = Font(bold=True, size=10)
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        _auto_fit_columns(ws_brief, max_width=75, min_width=18)

        # Sheet 2: Competitive Intel Events
        ws_events = wb.create_sheet(title="Competitive Intel")
        headers = [
            "Date",
            "Region",
            "Competitor",
            "Tracking Pillar",
            "Aranca Segment",
            "Materiality (1-5)",
            "Status",
            "Headline",
            "Executive Summary",
            "Strategic Implication",
            "Source URL",
            "Additional Sources",
        ]
        ws_events.append(headers)
        _style_header_row(ws_events, max_col=len(headers))

        for row_idx, ev in enumerate(events, start=2):
            add_sources = ev.get("additional_sources") or []
            add_sources_str = "; ".join(str(u) for u in add_sources if u)
            raw_url = str(ev.get("source_url") or "").strip()

            values = [
                _sanitize_cell(ev.get("published_date", "")),
                _sanitize_cell(ev.get("region", "")),
                _sanitize_cell(ev.get("company", "")),
                _sanitize_cell(ev.get("tracking_pillar", "")),
                _sanitize_cell(ev.get("aranca_segment", "")),
                int(ev.get("materiality_score", 3)),
                _sanitize_cell(ev.get("review_status", "APPROVED")),
                _sanitize_cell(ev.get("title", "")),
                _sanitize_cell(ev.get("executive_summary", "")),
                _sanitize_cell(ev.get("strategic_implication", "")),
                _sanitize_cell(raw_url),
                _sanitize_cell(add_sources_str),
            ]
            ws_events.append(values)

            if is_safe_public_url(raw_url):
                url_cell = ws_events.cell(row=row_idx, column=11)
                url_cell.hyperlink = raw_url
                url_cell.font = Font(color="0563C1", underline="single", size=10)

            for cell in ws_events[row_idx]:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        _auto_fit_columns(ws_events, max_width=50, min_width=12)

        # Sheet 3: Regional Breakdown
        ws_reg = wb.create_sheet(title="Regional Breakdown")
        ws_reg.append(["Region", "Approved Event Count", "Top Pillars"])
        _style_header_row(ws_reg, max_col=3)

        region_counts = Counter(str(ev.get("region", "Unknown")) for ev in events)
        region_pillars: dict[str, Counter] = {}
        for ev in events:
            reg = str(ev.get("region", "Unknown"))
            pil = str(ev.get("tracking_pillar", "General"))
            region_pillars.setdefault(reg, Counter())[pil] += 1

        for reg in ["US", "KSA", "UAE", "India", "Japan"]:
            cnt = region_counts.get(reg, 0)
            top_pil = ", ".join(
                f"{p} ({c})" for p, c in region_pillars.get(reg, Counter()).most_common(3)
            )
            ws_reg.append([_sanitize_cell(reg), cnt, _sanitize_cell(top_pil)])

        for row in ws_reg.iter_rows(min_row=2, max_row=ws_reg.max_row, min_col=1, max_col=3):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        _auto_fit_columns(ws_reg, max_width=45, min_width=14)

        buf = BytesIO()
        wb.save(buf)
        data = buf.getvalue()
        logger.info("Generated Excel workbook (%d bytes) with %d events", len(data), len(events))
        return data
    except Exception:
        logger.exception("Failed generating Excel intelligence report")
        raise